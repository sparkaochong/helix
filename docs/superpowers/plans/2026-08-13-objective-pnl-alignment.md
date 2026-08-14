# Objective–P&L Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hit-label gini GP fitness with fully vectorised production-Top4 D+2-close net portfolio P&L, retain IC/gini/hit only as training-only monitors, and publish a reproducible root-cause and validation report.

**Architecture:** Add one shared economic-objective module that precomputes canonical backtest costs and performs stable fixed-Top-K selection without replacement. Feed that objective into GP fit/selection while keeping direction explicit in the expression, then reuse it in training-only factor reports and the alignment audit. The backtest engine and label builder remain behaviourally unchanged.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, DEAP, Pydantic, pytest, Ruff, PyArrow/Parquet.

---

## File Map

- Create `helix/eval/objective.py`: vectorised net-return target, fixed-Top-K portfolio series, and summaries.
- Create `tests/test_objective.py`: arithmetic, costs, ties, masks, no replacement, and vectorisation contracts.
- Modify `helix/gp/fitness.py`: replace hit-gini fitness with Top-K net P&L.
- Create `tests/test_gp_fitness.py`: direction, label independence, coverage, and monotonic fitness tests.
- Modify `helix/gp/engine.py`: economic context wiring, two-component DEAP ordering, positive selection-P&L gate, and `sign=+1` for new factors.
- Modify `tests/test_gp.py`, `tests/test_neutralize.py`, and `tests/test_pipeline_smoke.py`: economic-target search fixtures and engine contracts.
- Modify `helix/splits.py` and `tests/test_splits.py`: D+2-complete search-window helper.
- Modify `helix/pipeline.py`, `helix/pipeline_events.py`, and `scripts/mine_argus.py`: pass D0 candidate masks, D+2 gross returns, dates, costs, and D+2-complete rows.
- Modify `tests/test_training_masks.py` and `tests/test_event_table.py`: caller-level no-future and evaluation-window tests.
- Create `scripts/objective_pnl_alignment.py`: training-only diagnosis, 30-factor validation, and report generation.
- Create `tests/test_objective_pnl_alignment.py`: report calendar, regimes, correlations, roles, and rendering tests.
- Create `docs/risk/objective_pnl_alignment.md`: generated evidence report.
- Modify `docs/factor-governance.md`: close D1 and document the new objective/sign/monitoring rules.
- Modify `configs/default.yaml` and `configs/argus_neutral.yaml`: describe Top4 objective ownership and remove the active gini-penalty claim.

### Task 1: Vectorised Economic Objective

**Files:**
- Create: `helix/eval/objective.py`
- Create: `tests/test_objective.py`

- [ ] **Step 1: Write failing objective arithmetic tests**

Create tests that compare the vectorised result with a looped reference, verify stable
tie selection, and pin failed selections as cash rather than deeper-name replacement:

```python
from __future__ import annotations

import numpy as np
import pytest

from helix.config import BacktestConfig
from helix.eval.backtest import _cost_rates, _net_returns
from helix.eval.objective import (
    cost_adjusted_returns,
    daily_top_k_portfolio,
    summarize_objective,
)


def test_vectorized_top_k_matches_looped_reference():
    rng = np.random.default_rng(7)
    score = rng.normal(size=(9, 13))
    gross = rng.normal(0.001, 0.04, size=score.shape)
    mask = rng.random(score.shape) > 0.2
    gross[2, np.argsort(-score[2])[:1]] = np.nan
    dates = np.array([f"202401{day:02d}" for day in range(2, 11)])
    cfg = BacktestConfig(top_k=4, slippage_bps=10.0)
    net = cost_adjusted_returns(gross, dates, cfg)

    actual = daily_top_k_portfolio(score, net, mask, top_k=cfg.top_k, overlap=2)

    expected_returns = []
    expected_executed = []
    for row in range(score.shape[0]):
        candidates = np.flatnonzero(mask[row] & np.isfinite(score[row]))
        if len(candidates) < cfg.top_k:
            expected_returns.append(np.nan)
            expected_executed.append(0)
            continue
        picked = candidates[np.argsort(-score[row, candidates], kind="stable")[: cfg.top_k]]
        selected = net[row, picked]
        finite = np.isfinite(selected)
        expected_returns.append(np.where(finite, selected, 0.0).sum() / cfg.top_k / 2)
        expected_executed.append(int(finite.sum()))

    np.testing.assert_allclose(actual.portfolio_return, expected_returns, equal_nan=True)
    np.testing.assert_array_equal(actual.executed, expected_executed)


def test_selected_nan_stays_cash_and_is_not_replaced():
    score = np.array([[4.0, 3.0, 2.0, 1.0]])
    net = np.array([[0.04, np.nan, 0.50, 0.60]])
    result = daily_top_k_portfolio(
        score, net, np.ones_like(score, dtype=bool), top_k=2, overlap=2
    )
    assert result.portfolio_return[0] == pytest.approx(0.04 / 2 / 2)
    assert result.executed[0] == 1


def test_stable_tie_break_matches_first_columns():
    score = np.ones((1, 4))
    net = np.array([[0.01, 0.02, 0.90, 1.00]])
    result = daily_top_k_portfolio(
        score, net, np.ones_like(score, dtype=bool), top_k=2, overlap=1
    )
    assert result.portfolio_return[0] == pytest.approx(0.015)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `.venv/bin/pytest tests/test_objective.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'helix.eval.objective'`.

- [ ] **Step 3: Implement the minimal vectorised objective module**

Create `helix/eval/objective.py` with these exact public types and functions:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BacktestConfig
from .backtest import _cost_rates, _net_returns


@dataclass(frozen=True)
class TopKPortfolio:
    portfolio_return: np.ndarray
    executed: np.ndarray


def _validate_aligned(*arrays: np.ndarray) -> tuple[int, int]:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("objective arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("objective arrays must be two-dimensional")
    return shape


def cost_adjusted_returns(
    gross_return: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
) -> np.ndarray:
    gross = np.asarray(gross_return, dtype=np.float64)
    if gross.ndim != 2 or len(dates) != gross.shape[0]:
        raise ValueError("dates must align with two-dimensional gross returns")
    date_digits = np.asarray([str(value).replace("-", "") for value in dates])
    if np.any(date_digits[1:] <= date_digits[:-1]):
        raise ValueError("objective dates must be strictly increasing")
    buy, sells = _cost_rates(config, date_digits)
    return _net_returns(gross, buy, sells[:, None])


def daily_top_k_portfolio(
    score: np.ndarray,
    net_return: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    top_k: int,
    overlap: int,
) -> TopKPortfolio:
    scores = np.asarray(score, dtype=np.float64)
    net = np.asarray(net_return, dtype=np.float64)
    candidates = np.asarray(candidate_mask, dtype=bool)
    rows, _ = _validate_aligned(scores, net, candidates)
    if top_k <= 0 or overlap <= 0:
        raise ValueError("top_k and overlap must be positive")
    eligible = candidates & np.isfinite(scores)
    enough = eligible.sum(axis=1) >= top_k
    order = np.argsort(np.where(eligible, -scores, np.inf), axis=1, kind="stable")
    picked = order[:, :top_k]
    selected = np.take_along_axis(net, picked, axis=1)
    finite = np.isfinite(selected)
    portfolio = np.where(finite, selected, 0.0).sum(axis=1) / top_k / overlap
    return TopKPortfolio(
        portfolio_return=np.where(enough, portfolio, np.nan),
        executed=np.where(enough, finite.sum(axis=1), 0),
    )


def summarize_objective(series: TopKPortfolio, top_k: int) -> dict[str, float]:
    usable = np.isfinite(series.portfolio_return)
    values = series.portfolio_return[usable]
    if values.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"), "ir": float("nan"),
            "positive_rate": float("nan"), "execution_rate": float("nan"),
            "coverage": 0.0, "n_days": 0.0,
        }
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        "mean": float(values.mean()),
        "std": std,
        "ir": float(values.mean() / std) if std > 0 else float("nan"),
        "positive_rate": float((values > 0).mean()),
        "execution_rate": float(series.executed[usable].sum() / (values.size * top_k)),
        "coverage": float(values.size / len(series.portfolio_return)),
        "n_days": float(values.size),
    }
```

- [ ] **Step 4: Add validation and cost-equivalence tests**

Append parameterised tests for K=1/4/10, pre/post stamp-cut dates, exact equality with
`_cost_rates`/`_net_returns`, insufficient candidates, shape errors, non-increasing
dates, input immutability, and summary coverage.

- [ ] **Step 5: Run objective tests and verify GREEN**

Run: `.venv/bin/pytest tests/test_objective.py -q`

Expected: all objective tests pass.

- [ ] **Step 6: Commit the objective primitive**

```bash
git add helix/eval/objective.py tests/test_objective.py
git commit -m "feat: add vectorized top-k pnl objective"
```

### Task 2: Replace GP Fitness with Net P&L

**Files:**
- Modify: `helix/gp/fitness.py`
- Create: `tests/test_gp_fitness.py`

- [ ] **Step 1: Write failing fitness direction tests**

Construct `EvalContext` directly and pin the new API and behaviours:

```python
import numpy as np
import pytest

from helix.eval.objective import cost_adjusted_returns
from helix.gp.fitness import EvalContext, INVALID, score_values


def context(net_returns: np.ndarray, mask: np.ndarray | None = None) -> EvalContext:
    candidate = np.ones_like(net_returns, dtype=bool) if mask is None else mask
    return EvalContext(
        field_arrays=[],
        net_returns=net_returns,
        candidate_mask=candidate,
        fit_rows=slice(0, 3),
        sel_rows=slice(3, 5),
        top_k=2,
        overlap=1,
        min_coverage=0.5,
    )


def test_fitness_rewards_the_production_long_direction():
    returns = np.tile(np.array([-0.04, -0.02, 0.02, 0.04]), (5, 1))
    values = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))
    good = score_values(values, context(returns), n_nodes=3)
    bad = score_values(-values, context(returns), n_nodes=3)
    assert good.fitness > bad.fitness
    assert good.fit_net_return > 0
    assert bad.fit_net_return < 0
    assert good.sign == 1.0 == bad.sign


def test_node_count_cannot_reverse_pnl_ordering():
    returns = np.tile(np.array([-0.02, 0.00, 0.01, 0.03]), (5, 1))
    high = np.tile(np.array([1.0, 2.0, 3.0, 5.0]), (5, 1))
    low = np.tile(np.array([1.0, 2.0, 4.0, 3.0]), (5, 1))
    assert score_values(high, context(returns), 40).fitness > score_values(
        low, context(returns), 1
    ).fitness
```

Also add tests for wrong shapes, defined fraction, residual-basis handling, fit/selection
coverage, and cache behaviour in `evaluate`.

- [ ] **Step 2: Run the fitness tests and verify RED**

Run: `.venv/bin/pytest tests/test_gp_fitness.py -q`

Expected: tests fail because `EvalContext` still requires `y/mask` and `FactorScore`
lacks net-return fields.

- [ ] **Step 3: Rewrite `EvalContext` and `FactorScore`**

Replace target-specific context fields with:

```python
@dataclass
class EvalContext:
    field_arrays: list[np.ndarray]
    net_returns: np.ndarray
    candidate_mask: np.ndarray
    fit_rows: slice
    sel_rows: slice
    top_k: int
    overlap: int
    min_coverage: float = 0.4
    min_defined_fraction: float = 0.2
    basis: np.ndarray | None = None
    _cache: dict[str, FactorScore] = field(default_factory=dict, repr=False)


@dataclass
class FactorScore:
    fitness: float
    sign: float
    fit_net_return: float
    fit_net_ir: float
    sel_net_return: float
    sel_net_ir: float
    coverage: float
    n_nodes: int
```

`as_dict()` must publish these exact names. `_invalid()` returns `INVALID`, `sign=1.0`,
NaN net metrics, zero coverage, and the supplied node count.

- [ ] **Step 4: Implement P&L-only `score_values`**

Validate the expression array against `ctx.net_returns.shape`, apply defined-fraction
and residualisation guards using `ctx.candidate_mask`, calculate one full
`daily_top_k_portfolio`, summarize fit and selection slices separately, reject
insufficient fit coverage, and set:

```python
fitness = 10_000.0 * fit_stats["mean"]
```

Never call `daily_gini`, never take `abs`, and always set `sign=1.0`.

- [ ] **Step 5: Run focused fitness and neutralisation tests**

Run: `.venv/bin/pytest tests/test_gp_fitness.py tests/test_neutralize.py -q`

Expected: new fitness tests pass; existing neutralisation tests may fail only at old
constructor/caller sites that Task 3 will migrate.

- [ ] **Step 6: Commit fitness replacement**

```bash
git add helix/gp/fitness.py tests/test_gp_fitness.py
git commit -m "fix: align gp fitness with net pnl"
```

### Task 3: Wire Economic Fitness Through the GP Engine

**Files:**
- Modify: `helix/gp/engine.py`
- Modify: `tests/test_gp.py`
- Modify: `tests/test_neutralize.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Convert the planted-signal fixture to realised returns**

Replace the binary-only fixture in `tests/test_gp.py` with `alpha` plus a realised return
grid where large alpha has positive D+2 return. Call `run_search` with
`gross_returns`, `candidate_mask`, dates, `BacktestConfig(top_k=4, slippage_bps=0)`,
and label offsets 1/2. Assert retained factors have positive `sel_net_return`, contain
`alpha`, and store `sign=1.0`.

- [ ] **Step 2: Add engine selection and lexicographic tests**

Pin creator weights/values so P&L is first and smaller node count breaks exact ties.
Use a minimal fake hall of fame to verify `_select_factors` rejects
`sel_net_return <= 0`, sorts positive survivors by selection P&L, and emits no
hit-gini fallback.

- [ ] **Step 3: Run engine tests and verify RED**

Run: `.venv/bin/pytest tests/test_gp.py tests/test_neutralize.py tests/test_pipeline_smoke.py -q`

Expected: failures at the legacy positional `run_search` target/mask signature and old
`sel_gini` assertions.

- [ ] **Step 4: Change `make_context` and `run_search` signatures**

Use this contract:

```python
def run_search(
    fields: dict[str, np.ndarray],
    field_names: list[str],
    gross_returns: np.ndarray,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    cfg: GPConfig,
    backtest_cfg: BacktestConfig,
    entry_offset: int,
    touch_offset: int,
    embargo_days: int,
    pset: gp.PrimitiveSetTyped | None = None,
    kind: str = "panel",
    basis: np.ndarray | None = None,
) -> SearchResult:
```

`make_context` validates all shapes, precomputes net returns once with
`cost_adjusted_returns`, derives `overlap`, and keeps the current 80%/embargo/selection
split and 20-selection-date minimum.

- [ ] **Step 5: Change DEAP and survivor ordering**

Create `HelixFitness` with `weights=(1.0, -1.0)`. Assign every individual:

```python
score = evaluate(ind, toolbox, ctx)
ind.fitness.values = (score.fitness, score.n_nodes)
```

Log the first component only. In `_select_factors`, require finite strictly positive
`sel_net_return`, sort by `(-sel_net_return, n_nodes)`, persist `sign=1.0`, and return
hall-of-fame tuples `(expression, fit_net_return, sel_net_return)`.

- [ ] **Step 6: Migrate all GP tests and run GREEN**

Update neutralisation and smoke fixtures with gross returns/dates/backtest config. Run:

`.venv/bin/pytest tests/test_gp.py tests/test_gp_fitness.py tests/test_neutralize.py tests/test_pipeline_smoke.py -q`

Expected: all targeted GP tests pass.

- [ ] **Step 7: Commit engine wiring**

```bash
git add helix/gp/engine.py tests/test_gp.py tests/test_neutralize.py tests/test_pipeline_smoke.py
git commit -m "feat: evolve factors on production pnl"
```

### Task 4: Enforce D+2-Complete Training Boundaries and Update Evaluators

**Files:**
- Modify: `helix/splits.py`
- Modify: `tests/test_splits.py`
- Modify: `helix/pipeline.py`
- Modify: `helix/pipeline_events.py`
- Modify: `scripts/mine_argus.py`
- Modify: `tests/test_training_masks.py`
- Modify: `tests/test_event_table.py`

- [ ] **Step 1: Write failing complete-outcome-window tests**

Add to `tests/test_splits.py`:

```python
from helix.splits import complete_outcome_window


def test_complete_outcome_window_drops_boundary_decisions():
    assert complete_outcome_window(slice(0, 649), horizon=2) == slice(0, 647)
    assert complete_outcome_window(slice(10, 20), horizon=3) == slice(10, 17)


def test_complete_outcome_window_rejects_too_short_windows():
    with pytest.raises(ValueError, match="outcome-complete"):
        complete_outcome_window(slice(0, 2), horizon=2)
```

- [ ] **Step 2: Run split tests and verify RED**

Run: `.venv/bin/pytest tests/test_splits.py -q`

Expected: import fails because `complete_outcome_window` does not exist.

- [ ] **Step 3: Implement the slice helper**

Add to `helix/splits.py`:

```python
def complete_outcome_window(rows: slice, horizon: int) -> slice:
    start = 0 if rows.start is None else rows.start
    if rows.stop is None or horizon < 1 or rows.stop - start <= horizon:
        raise ValueError("window is too short to build an outcome-complete slice")
    return slice(start, rows.stop - horizon)
```

- [ ] **Step 4: Write caller-level boundary and mask tests**

Pin these behaviours:

- panel `mine` passes `prepared.universe`, not `labels.valid`, as candidate mask;
- panel gross return is NaN where `labels.valid` is false;
- event mining passes `panel.occupied` as D0 candidates;
- both paths end objective rows two dates before nominal training end;
- event feature screening/basis uses the same 647 rows; and
- rows after the training boundary cannot alter any reported fit/selection/full-training
  metric.

- [ ] **Step 5: Migrate pipeline callers**

In `pipeline.py`, compute:

```python
rows = complete_outcome_window(search_window(len(panel.dates), cfg.split), cfg.label.touch_offset)
gross = np.where(
    labels.valid[rows],
    labels.exit_price[rows] / labels.entry_price[rows] - 1.0,
    np.nan,
)
```

Pass `prepared.universe[rows]`, D0 dates, `cfg.backtest`, and label offsets to
`run_search`. Apply the same window to liquidity sampling.

In both event paths, shorten the nominal fraction slice before feature screening and
use `label_d2_return`, `panel.occupied`, panel dates, `cfg.backtest`, and cfg label
offsets. Do not derive a new label.

- [ ] **Step 6: Replace post-search evaluation with training blocks**

Update panel/event factor evaluation to report `fit`, `selection`, and `training_full`
only. Each retained factor report must contain:

- production Top4 objective summary;
- supplemental Top10 objective summary;
- hit-label IC/ICIR, gini, Top-K hit/base/lift;
- peak-return IC/ICIR; and
- close-return IC/ICIR.

Compute auxiliary monitors only here, never inside `score_values`. Name the economic
roles `production_objective` and `supplemental_top10`.

- [ ] **Step 7: Run pipeline and boundary tests**

Run:

`.venv/bin/pytest tests/test_splits.py tests/test_training_masks.py tests/test_event_table.py tests/test_pipeline_smoke.py -q`

Expected: all pass, including exact 647-date routing.

- [ ] **Step 8: Commit boundary and evaluator changes**

```bash
git add helix/splits.py helix/pipeline.py helix/pipeline_events.py scripts/mine_argus.py tests/test_splits.py tests/test_training_masks.py tests/test_event_table.py tests/test_pipeline_smoke.py
git commit -m "fix: keep pnl fitness inside complete training outcomes"
```

### Task 5: Build the Training-Only Alignment Audit

**Files:**
- Create: `scripts/objective_pnl_alignment.py`
- Create: `tests/test_objective_pnl_alignment.py`

- [ ] **Step 1: Write failing calendar and role tests**

Test constants `TRAIN_START="2022-01-04"`, `TRAIN_END="2024-09-04"`,
`OBJECTIVE_D0_END="2024-09-02"`, production K from `BacktestConfig(top_k=4)`, and
supplemental K=10. Assert `validate_training_frame` rejects any D0 later than objective
end and `validate_forward_exit` rejects any exit later than training end.

- [ ] **Step 2: Write failing statistical-helper tests**

Use tiny frames with sentinel future returns so a leak would visibly change the result:

```python
def test_market_regime_is_trailing_and_uses_fixed_thresholds():
    returns = pd.Series(
        [0.01] * 20 + [-0.01] * 20,
        index=pd.date_range("2024-01-02", periods=40, freq="B").strftime("%Y-%m-%d"),
    )
    result = market_regime(returns)
    assert result.iloc[18] == "unavailable"
    assert result.iloc[19] == "bull"
    assert result.iloc[-1] == "bear"


def test_correlation_reports_pearson_and_spearman_pvalues():
    result = correlation(np.arange(10.0), np.arange(10.0))
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson_pvalue"] < 0.05
    assert result["spearman_pvalue"] < 0.05


def test_monthly_table_keeps_each_month_separate():
    daily = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"],
            "hit_ic": [0.1, 0.2, 0.3, 0.4],
            "production_net": [0.01, 0.02, -0.01, -0.02],
        }
    )
    result = evaluate_months(daily)
    assert result["month"].tolist() == ["2024-01", "2024-02"]
    assert result["production_net"].tolist() == pytest.approx([0.015, -0.015])


def test_horizon_filter_drops_only_exits_after_train_end():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-09-02", "2024-09-03", "2024-09-04"],
            "exit_date": ["2024-09-04", "2024-09-05", "2024-09-06"],
            "gross_return": [0.01, 9.0, 9.0],
        }
    )
    result = validate_forward_exit(frame, train_end=TRAIN_END)
    assert result["gross_return"].tolist() == [0.01]


def test_top10_cannot_change_production_acceptance():
    production = {"pearson": 0.4, "pearson_pvalue": 0.01,
                  "spearman": 0.3, "spearman_pvalue": 0.02}
    assert list(inspect.signature(alignment_decision).parameters) == ["production"]
    assert alignment_decision(production) == "PASS"


def test_library_acceptance_uses_fit_against_selection():
    table = pd.DataFrame(
        {"fit_net": [1.0, 2.0, 3.0, 4.0], "selection_net": [2.0, 4.0, 6.0, 8.0]}
    )
    result = library_alignment_statistics(table)
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
```

- [ ] **Step 3: Run audit tests and verify RED**

Run: `.venv/bin/pytest tests/test_objective_pnl_alignment.py -q`

Expected: module import fails.

- [ ] **Step 4: Implement strict loaders and reusable calculations**

The script must expose:

```python
validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame
market_regime(market_daily_return: pd.Series) -> pd.Series
correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]
validate_forward_exit(frame: pd.DataFrame, train_end: str) -> pd.DataFrame
evaluate_horizons(daily_by_horizon: dict[int, pd.DataFrame]) -> pd.DataFrame
evaluate_months(daily: pd.DataFrame) -> pd.DataFrame
evaluate_regimes(daily: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame
evaluate_score_quintiles(frame: pd.DataFrame) -> pd.DataFrame
evaluate_execution_feasibility(frame: pd.DataFrame) -> dict[str, float]
library_alignment_statistics(table: pd.DataFrame) -> dict[str, float]
alignment_decision(
    production: dict[str, float]
) -> str
evaluate_library_alignment(
    factor_values: dict[str, np.ndarray],
    gross_return: np.ndarray,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
    fit_rows: slice,
    selection_rows: slice,
) -> tuple[pd.DataFrame, dict[str, float]]
render_report(payload: dict[str, object]) -> str
run_audit(
    input_path: Path,
    formal_library: Path,
    multi_library: Path,
    n40_library: Path,
    market_cache: Path,
    report_path: Path,
) -> dict[str, object]
```

Load only the three factor libraries' common 70 fields plus labels and keys. Filter
Parquet reads to the training range before panel construction. Reuse factor replay,
canonical IC/gini functions, `helix.eval.objective`, and the existing G3 price lookup
for D+1..D+10 adjusted returns. Assert every resolved exit is no later than
`TRAIN_END`.

- [ ] **Step 5: Implement root-cause measurements**

Publish distinct fields for:

- `score_return_cross_sectional_ic`;
- `daily_hit_ic_to_portfolio_pnl_correlation`;
- Top4 and Top10 gross/net returns;
- candidate-pool gross and Top-K excess;
- hit/miss close payoff and peak-to-close giveback;
- score-quintile hit/peak/close monotonicity;
- turnover by score bucket;
- reconstructed D+1 limit-open share;
- bottom turnover/market-value decile concentration; and
- D+1..D+10 results by whole window, month, and trailing-20-day bull/neutral/bear regime.

Rank root causes by measured bps/coverage contribution, not prose order.

- [ ] **Step 6: Implement 30-factor acceptance**

Replay formal (1), `argus_multi` (17), and `argus_n40` (12) factors with their stored
signs. Use the exact 517 fit dates, 5 embargo dates, and 125 selection dates. Compare
fit production Top4 mean net P&L with selection Top4 mean net P&L using Pearson and
Spearman. Pass only when both coefficients are positive and both two-sided p-values are
below 0.05. Calculate Top10 separately without reading it in the decision function.

- [ ] **Step 7: Run audit tests and verify GREEN**

Run: `.venv/bin/pytest tests/test_objective_pnl_alignment.py -q`

Expected: all helper, leakage, and rendering tests pass.

- [ ] **Step 8: Commit the audit implementation**

```bash
git add scripts/objective_pnl_alignment.py tests/test_objective_pnl_alignment.py
git commit -m "feat: add training-only objective pnl audit"
```

### Task 6: Generate Evidence and Close Governance D1

**Files:**
- Create: `docs/risk/objective_pnl_alignment.md`
- Modify: `docs/factor-governance.md`
- Modify: `configs/default.yaml`
- Modify: `configs/argus_neutral.yaml`

- [ ] **Step 1: Run the real training-only audit**

Run:

```bash
.venv/bin/python scripts/objective_pnl_alignment.py \
  --input data/raw/argus_quant_working.parquet \
  --formal-library data/artifacts/argus/event_factors.json \
  --multi-library data/artifacts/argus_multi/event_factors.json \
  --n40-library data/artifacts/argus_n40/event_factors.json \
  --market-cache data/artifacts/g3_style_market.parquet \
  --report docs/risk/objective_pnl_alignment.md
```

Expected metadata: 649 formal training dates, 647 D+2-complete objective D0 dates,
last objective D0 `2024-09-02`, production K=4, supplemental K=10, 30 factors, and no
source/exit date beyond `2024-09-04`.

- [ ] **Step 2: Audit generated acceptance values**

Run a read-only Python assertion that production Pearson/Spearman coefficients and
p-values equal the machine-readable payload, are positive/significant, and that Top10
fields are absent from the acceptance-decision input. Expected values are close to the
pre-design evidence `r=0.857423`, `rho=0.722803`; source-code reproduction is
authoritative.

- [ ] **Step 3: Update governance objective and sign rules**

Replace the §4 fitness formula with production Top4 mean D+2-close net portfolio P&L
in bps, complexity as exact-tie secondary order, `sign=+1` for newly mined factors, and
IC/gini/hit monitoring-only language. Add the D+2-complete boundary and K/cost alignment
to global invariants.

- [ ] **Step 4: Close D1 with measured evidence**

Change D1 status to **已关闭** and include the confirmed target/payoff mismatch,
production Top4 fit→selection correlations/p-values, supplemental Top10 role, and a
link to `risk/objective_pnl_alignment.md`. Leave D5/D6/D13/D14/D15 unchanged.

- [ ] **Step 5: Update configuration comments**

In both YAML files, state that `backtest.top_k` is also GP fitness K and that all listed
costs feed both fitness and backtest. Remove comments saying `complexity_penalty` is
subtracted from absolute gini; if the Pydantic field remains for compatibility, label
it deprecated and inactive.

- [ ] **Step 6: Run report/document tests**

Run:

`.venv/bin/pytest tests/test_objective_pnl_alignment.py tests/test_gp_fitness.py tests/test_gp.py -q`

Expected: pass with report file present and governance assertions satisfied.

- [ ] **Step 7: Commit evidence and governance**

```bash
git add docs/risk/objective_pnl_alignment.md docs/factor-governance.md configs/default.yaml configs/argus_neutral.yaml
git commit -m "docs: close objective pnl alignment gap"
```

### Task 7: Full Verification and Final Diff Audit

**Files:**
- Verify: entire repository

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/pytest`

Expected: zero failures.

- [ ] **Step 2: Run repository-wide Ruff**

Run: `.venv/bin/ruff check .`

Expected: `All checks passed!`.

- [ ] **Step 3: Re-run the audit from source data**

Run the Task 6 command again and require a zero diff in
`docs/risk/objective_pnl_alignment.md`.

- [ ] **Step 4: Re-run the vectorised performance benchmark**

Benchmark 50 calls at shape `(647, 1990)` and 23% occupancy using five repeats for the
new `daily_top_k_portfolio` and current `daily_gini`. Record medians in the final handoff
and require the new median to be no greater than the gini median. Do not include factor
expression evaluation or one-time cost precomputation in either timed block.

- [ ] **Step 5: Prove the protected modules did not change behaviour**

Run:

```bash
git diff 52d037a -- helix/eval/backtest.py helix/labels/touch_label.py
```

Expected: empty diff. Also run `tests/test_backtest.py` and `tests/test_labels.py` as
direct behavioural evidence.

- [ ] **Step 6: Review requirements and repository state**

Check every design section against the diff, run `git diff --check`, inspect
`git status --short`, and confirm no OOS statistic appears in the alignment report or
acceptance code.

- [ ] **Step 7: Commit any verification-only corrections, then report evidence**

If verification required a correction, commit only its focused files with a specific
message and rerun Steps 1–5. Otherwise leave the verified commits unchanged and report
the exact pytest count, Ruff result, acceptance coefficients/p-values, training bounds,
and remaining governance debts.
