# Training Performance Confidence-Interval Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible ten-seed moving-block-bootstrap confidence-interval audit of formal `gp_000` training-window performance under the canonical realistic-exit backtest.

**Architecture:** Extract the existing circular date bootstrap into a pure NumPy module, then add a standalone experiment orchestrator that aligns the event factor to an observed market panel and calls `run_backtest` once. Resolve chronological exits before vectorised whole-date resampling; render the measured decision into the risk report and governance status without changing production defaults.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyArrow, Pydantic configuration, pytest, Ruff, existing Helix factor/event/backtest modules.

---

## File map

- Create `helix/eval/bootstrap.py`: reusable seed validation, circular block indices,
  vectorised performance replicates, and percentile summaries.
- Modify `scripts/g3_style_ablation.py`: import the shared bootstrap functions while
  preserving its public import surface and existing output.
- Create `scripts/performance_ci_bootstrap.py`: training-boundary validation, factor and
  market alignment, canonical backtest execution, bootstrap aggregation, decision,
  report rendering, CLI, and artifact output.
- Create `tests/test_performance_ci_bootstrap.py`: numerical, boundary, configuration,
  decision, and report contracts.
- Modify `docs/factor-governance.md`: measured §7.5 status only.
- Create `docs/risk/performance_ci_bootstrap.md`: generated experiment evidence.

### Task 1: Shared circular bootstrap numerical layer

**Files:**
- Create: `helix/eval/bootstrap.py`
- Create: `tests/test_performance_ci_bootstrap.py`

- [ ] **Step 1: Write failing index and validation tests**

Add tests that import the not-yet-created functions and specify this API:

```python
from helix.eval.bootstrap import (
    bootstrap_performance_metrics,
    circular_block_bootstrap_indices,
    summarize_bootstrap_distribution,
)


def test_circular_indices_are_seeded_and_preserve_blocks():
    seeds = (7, 13, 42)
    first = circular_block_bootstrap_indices(11, 4, seeds)
    second = circular_block_bootstrap_indices(11, 4, seeds)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 11)
    for row in first:
        chunks = np.split(row[:8], 2)
        assert all(np.all(np.diff(chunk) % 11 == 1) for chunk in chunks)


@pytest.mark.parametrize(
    ("n_dates", "block_length", "seeds"),
    [(0, 4, (1, 2)), (10, 0, (1, 2)), (10, 4, (1,)), (10, 4, (1, 1))],
)
def test_circular_indices_reject_invalid_contracts(n_dates, block_length, seeds):
    with pytest.raises(ValueError):
        circular_block_bootstrap_indices(n_dates, block_length, seeds)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_performance_ci_bootstrap.py -k circular
```

Expected: collection fails because `helix.eval.bootstrap` does not exist.

- [ ] **Step 3: Implement seed validation and index generation**

Implement these exact interfaces in `helix/eval/bootstrap.py`:

```python
def validate_bootstrap_seeds(seeds: Sequence[int], *, minimum: int = 2) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if len(values) < minimum or len(set(values)) != len(values):
        raise ValueError(f"bootstrap requires at least {minimum} unique seeds")
    return values


def circular_block_bootstrap_indices(
    n_dates: int,
    block_length: int,
    seeds: Sequence[int],
) -> np.ndarray:
    if n_dates <= 0 or block_length <= 0:
        raise ValueError("n_dates and block_length must be positive")
    seed_values = validate_bootstrap_seeds(seeds)
    block_count = int(np.ceil(n_dates / block_length))
    offsets = np.arange(block_length, dtype=np.intp)
    rows = []
    for seed in seed_values:
        starts = np.random.default_rng(seed).integers(0, n_dates, size=block_count)
        rows.append(((starts[:, None] + offsets) % n_dates).reshape(-1)[:n_dates])
    return np.stack(rows)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Write failing vectorised metric and summary tests**

Cover daily CAGR/Sharpe/win rate, trade-sum/count weighting, sample standard deviation,
linear percentiles, and non-finite/shape errors. Compare every replicate to
`summarize_portfolio_returns` and explicit selected-trade concatenation.

- [ ] **Step 6: Implement vectorised metrics and summaries**

Use these interfaces and output shapes:

```python
PERFORMANCE_METRICS = ("cagr", "sharpe", "day_win_rate", "mean_trade_return_net")


def bootstrap_performance_metrics(
    daily_returns: np.ndarray,
    trade_return_sum: np.ndarray,
    trade_count: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    # Validate one-dimensional aligned inputs and a two-dimensional bounded index.
    sampled = daily_returns[indices]
    final_equity = np.prod(1.0 + sampled, axis=1)
    volatility = sampled.std(axis=1, ddof=1)
    sampled_trade_count = trade_count[indices].sum(axis=1)
    return {
        "cagr": np.where(final_equity > 0, final_equity ** (252 / sampled.shape[1]) - 1, -1),
        "sharpe": np.divide(sampled.mean(axis=1) * np.sqrt(252), volatility),
        "day_win_rate": (sampled > 0).mean(axis=1),
        "mean_trade_return_net": np.divide(
            trade_return_sum[indices].sum(axis=1), sampled_trade_count
        ),
    }


def summarize_bootstrap_distribution(
    values: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | list[float]]]:
    # For each finite vector return mean, sample std, ci_low, ci_high, and values.
```

- [ ] **Step 7: Run module tests and Ruff**

```bash
.venv/bin/pytest -q tests/test_performance_ci_bootstrap.py
.venv/bin/ruff check helix/eval/bootstrap.py tests/test_performance_ci_bootstrap.py
```

Expected: zero failures and zero Ruff findings.

- [ ] **Step 8: Commit the shared layer**

```bash
git add helix/eval/bootstrap.py tests/test_performance_ci_bootstrap.py
git commit -m "feat: add vectorized performance bootstrap"
```

### Task 2: Make G3 consume the shared bootstrap implementation

**Files:**
- Modify: `scripts/g3_style_ablation.py:180-215`
- Test: `tests/test_g3_style_ablation.py:263-287`

- [ ] **Step 1: Change tests to exercise the shared implementation**

Update the G3 tests so their direct imports come from `helix.eval.bootstrap`; adapt the
expected index call from one integer seed to a one-element row selected from the
multi-seed matrix. Preserve G3's minimum-three-seed governance validation separately.

- [ ] **Step 2: Run the G3 bootstrap tests and verify RED**

```bash
.venv/bin/pytest -q tests/test_g3_style_ablation.py -k bootstrap
```

Expected: failures because the old callback summary and single-seed index signature are
still local to the script.

- [ ] **Step 3: Import shared helpers and add a compatibility adapter**

Delete the duplicate index implementation. Keep G3's callback-oriented report shape
through a small adapter that calls the shared multi-seed index generator and loops only
over the ten expensive arm evaluations:

```python
def bootstrap_metric_summary(n_dates, block_length, seeds, metric):
    seed_values = validate_seed_contract(seeds)
    index_rows = circular_block_bootstrap_indices(n_dates, block_length, seed_values)
    runs = [metric(index) for index in index_rows]
    return summarize_metric_runs(runs)
```

The pure distribution aggregation belongs in `helix.eval.bootstrap`; do not duplicate
mean/std code.

- [ ] **Step 4: Run all G3 tests**

```bash
.venv/bin/pytest -q tests/test_g3_style_ablation.py
```

Expected: all pass with unchanged report contracts.

- [ ] **Step 5: Commit the reuse change**

```bash
git add scripts/g3_style_ablation.py tests/test_g3_style_ablation.py
git commit -m "refactor: share circular bootstrap helpers"
```

### Task 3: Standalone training performance experiment

**Files:**
- Create: `scripts/performance_ci_bootstrap.py`
- Modify: `tests/test_performance_ci_bootstrap.py`

- [ ] **Step 1: Write failing boundary, decision, configuration, and report tests**

Specify these public contracts:

- `complete_training_decision_dates(calendar: Sequence[str], train_start: str,
  train_end: str, horizon: int) -> np.ndarray` returns the sorted complete-outcome D0
  dates.
- `realistic_backtest_config(config: Config) -> BacktestConfig` returns a copy with
  realistic exit enabled after asserting Top4/close and preserving every cost field.
- `aggregate_performance_ci(result: BacktestResult, indices: np.ndarray) -> dict`
  returns deterministic metrics plus the vectorised bootstrap distribution.
- `performance_ci_decision(sharpe_ci_low: float) -> str` returns `LIFT_DOWNGRADE` only
  for a finite value strictly greater than zero.
- `render_report(payload: dict[str, object]) -> str` renders the complete evidence and
  reproduction command.

Tests must prove that the formal calendar ends at D0 `2024-09-02`, production costs and
Top4 are unchanged while realistic exit becomes true, zero is not sufficient to lift
the downgrade, trade means use full resampled days, and the report contains the lower
bound plus exact cache-only command.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
.venv/bin/pytest -q tests/test_performance_ci_bootstrap.py -k 'training or config or decision or report or aggregate'
```

Expected: import failures for the new script contracts.

- [ ] **Step 3: Implement input and alignment helpers**

The script must:

- load and validate the one-factor event library;
- read only the factor fields and identifiers for complete-D+2 D0 dates;
- call `build_event_panel` and `compute_factors`;
- load the exchange calendar through `load_open_dates`;
- load/fetch observed market rows through `load_or_fetch_market`;
- call `build_exit_panel`, then align scores/occupancy into its fixed coordinates;
- call `build_touch_label` and canonical `run_backtest` with the realistic config copy.

Reject date mismatches, missing market caches in `--cache-only` mode, non-Top4 config,
non-close exits, any market date after `TRAIN_END`, or a backtest that does not produce
exactly the complete-decision calendar.

- [ ] **Step 4: Implement vectorised aggregation and rendering**

Derive per-D0 resolved trade sums/counts from `BacktestResult.trades`, call the shared
bootstrap module, attach canonical deterministic summary values, and render:

```text
| 指标 | 确定性全样本 | 10 种子均值 | 样本标准差 | 95% CI |
```

Include per-seed values, boundary/config/execution audits, decision, limitations, and
the command:

```bash
.venv/bin/python scripts/performance_ci_bootstrap.py --cache-only \
  --seeds 7,13,42,101,211,307,419,523,631,743 \
  --block-length 20
```

- [ ] **Step 5: Run experiment-script tests and Ruff**

```bash
.venv/bin/pytest -q tests/test_performance_ci_bootstrap.py
.venv/bin/ruff check scripts/performance_ci_bootstrap.py tests/test_performance_ci_bootstrap.py
```

Expected: zero failures and zero Ruff findings.

- [ ] **Step 6: Commit the orchestrator**

```bash
git add scripts/performance_ci_bootstrap.py tests/test_performance_ci_bootstrap.py
git commit -m "feat: add training performance CI experiment"
```

### Task 4: Run the real experiment and publish governance evidence

**Files:**
- Create: `docs/risk/performance_ci_bootstrap.md`
- Modify: `docs/factor-governance.md:532-547`
- Runtime artifact: `data/artifacts/performance_ci_bootstrap.json` (ignored)

- [ ] **Step 1: Populate missing observed market cache and run the experiment**

```bash
.venv/bin/python scripts/performance_ci_bootstrap.py \
  --seeds 7,13,42,101,211,307,419,523,631,743 \
  --block-length 20
```

Expected: exit 0, a 647-date deterministic result, ten finite values per metric, report
and JSON artifact written, and an explicit `LIFT_DOWNGRADE` or `KEEP_DOWNGRADE` result.

- [ ] **Step 2: Re-run cache-only and compare hashes**

Run the same command with `--cache-only`, hash the report and JSON before/after, and
require identical hashes. Any mismatch is a reproducibility defect.

- [ ] **Step 3: Update §7.5 with measured evidence**

Replace the stale three-seed sentence with the exact ten-seed Sharpe interval, lower
bound, report link, and current status. Preserve the strict `> 0` release criterion and
state that in-sample evidence does not close D6 or establish OOS profitability.

- [ ] **Step 4: Check report and governance consistency**

Use a small read-only Python check to parse the JSON and assert the exact Sharpe lower
bound and decision strings occur in both Markdown files.

- [ ] **Step 5: Commit measured evidence**

```bash
git add docs/risk/performance_ci_bootstrap.md docs/factor-governance.md
git commit -m "docs: publish performance CI bootstrap audit"
```

### Task 5: Full verification and handoff

**Files:** all changed files

- [ ] **Step 1: Run the complete test suite**

```bash
.venv/bin/pytest
```

Expected: zero failures.

- [ ] **Step 2: Run the complete linter and whitespace checks**

```bash
.venv/bin/ruff check .
git diff --check main...HEAD
```

Expected: zero Ruff findings and no whitespace errors.

- [ ] **Step 3: Audit scope and repository state**

```bash
git status --short --branch
git diff --stat main...HEAD
git log --oneline main..HEAD
```

Expected: only the design/plan, shared bootstrap, G3 reuse, experiment/tests, report,
and governance files differ; the branch worktree has no uncommitted files.
