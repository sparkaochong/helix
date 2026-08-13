# G3 Style Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the formal `gp_000`, remove the prescribed daily style exposures, compare raw and neutral arms inside the fixed training window, and publish a mechanical G3 GO/NO-GO decision.

**Architecture:** Add one small vectorised numerical module and one standalone experiment orchestrator. The orchestrator keeps training and OOS frames separate, derives point-in-time style inputs from a local Tushare cache, reuses the canonical factor/metric/backtest cost functions, and publishes deterministic plus ten-seed moving-block-bootstrap results.

**Tech Stack:** Python 3.10+, NumPy, pandas, PyArrow, SciPy, Tushare Pro, pytest, Ruff.

---

## File Map

- Create `tests/test_style_neutralize.py`: numerical contract for batched daily style residualisation.
- Create `tests/test_g3_style_ablation.py`: window isolation, style construction, bootstrap, metrics, decision, and report contracts.
- Create `helix/eval/style_neutralize.py`: fully vectorised daily design construction and rank-safe pseudoinverse residualisation.
- Create `scripts/g3_style_ablation.py`: data/cache orchestration, factor replay, evaluation, report generation, and CLI.
- Create `docs/risk/g3_style_ablation.md`: generated experiment report.
- Modify `docs/factor-governance.md`: update D7 only; preserve D13 byte-for-byte.
- Modify `.gitignore`: ignore `/uv.lock` and the local worktree directory; do not track the existing lock file.

## Task 1: Write the complete failing numerical test suite

**Files:**

- Create: `tests/test_style_neutralize.py`
- Create: `tests/test_g3_style_ablation.py`

- [ ] **Step 1: Write all core neutralisation tests before production code**

Import the wished-for API below in `tests/test_style_neutralize.py`:

```python
from helix.eval.style_neutralize import build_style_design, style_residualize
```

Create eight named tests: `test_residual_is_orthogonal_to_each_same_day_style_column`,
`test_changing_a_later_date_cannot_change_an_earlier_residual`,
`test_missing_rows_stay_nan_and_do_not_enter_the_regression`,
`test_absent_and_collinear_industries_are_rank_safe`,
`test_fully_explained_factor_is_nan_not_rankable_float_noise`,
`test_inputs_are_not_mutated`, `test_batched_result_matches_daily_lstsq_reference`, and
`test_shapes_and_industry_levels_are_validated`. The orthogonality test constructs two
dates with different coefficients and verifies `X[t].T @ residual[t] == 0` separately.
The isolation test changes every factor/style/industry value on date 1 and asserts date
0 is bitwise unchanged. The reference test is the only test allowed to loop over dates;
production code may not.

- [ ] **Step 2: Write all experiment-contract tests before the script exists**

Use pure functions from the wished-for script API:

```python
from scripts.g3_style_ablation import (
    TRAIN_END,
    TRAIN_START,
    bootstrap_metric_summary,
    circular_block_bootstrap_indices,
    compute_trailing_styles,
    decide_go,
    forward_return_panel,
    render_report,
    split_evaluation_windows,
    validate_seed_contract,
)
```

Create twelve named tests covering the exact behaviours in their names:
`test_training_window_is_exact_and_oos_is_returned_separately`,
`test_training_decision_receives_no_oos_values`,
`test_forward_horizon_drops_exits_after_window_end`,
`test_trailing_styles_use_d0_and_exactly_19_prior_sessions`,
`test_trailing_styles_do_not_bridge_missing_stock_sessions`,
`test_at_least_three_unique_seeds_are_required`,
`test_circular_block_bootstrap_is_reproducible_and_keeps_whole_dates`,
`test_bootstrap_summary_uses_sample_standard_deviation`,
`test_go_requires_strict_icir_p95_exceedance_and_same_nonzero_return_sign`,
`test_nonfinite_decision_inputs_are_no_go`,
`test_report_has_training_tables_decision_decay_reproduction_and_oos_appendix`, and
`test_oos_numbers_cannot_change_rendered_decision`.

- [ ] **Step 3: Run the complete new test set and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_style_neutralize.py tests/test_g3_style_ablation.py -q
```

Expected: collection fails because `helix.eval.style_neutralize` and
`scripts.g3_style_ablation` do not exist. This is the required RED evidence for every
new public function; do not create either production file before this run.

- [ ] **Step 4: Commit the tests only**

```bash
git add tests/test_style_neutralize.py tests/test_g3_style_ablation.py
git commit -m "test: define G3 style ablation contracts"
```

## Task 2: Implement vectorised daily style neutralisation

**Files:**

- Create: `helix/eval/style_neutralize.py`
- Test: `tests/test_style_neutralize.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add local-only paths to `.gitignore`**

Append the exact root-anchored entries:

```gitignore
/uv.lock
/.worktrees/
```

Verify `git status --short` no longer displays `uv.lock`.

- [ ] **Step 2: Implement the public numerical API without date/stock loops**

The module exposes:

```python
def build_style_design(
    continuous: np.ndarray,
    industry: np.ndarray,
    mask: np.ndarray,
    *,
    industry_levels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(design, valid)` for intercept + z-scored styles + L-1 dummies."""


def style_residualize(
    factor: np.ndarray,
    continuous: np.ndarray,
    industry: np.ndarray,
    mask: np.ndarray,
    *,
    industry_levels: np.ndarray | None = None,
    min_residual_fraction: float = 1e-6,
) -> np.ndarray:
    """Batched per-date pseudoinverse residual; NaN outside complete observations."""
```

Implementation rules:

```python
valid = (
    mask
    & np.isfinite(factor)
    & np.isfinite(continuous).all(axis=2)
    & np.isfinite(industry)
)
```

Compute cross-sectional means/standard deviations by masked reductions over axis 1,
zero constant continuous directions, create all but one fixed industry dummy, and zero
every design row outside `valid`. Form every daily Gram matrix in one batched multiply,
apply `np.linalg.pinv(gram, rcond=EPS, hermitian=True)` to the full batch, and project
with two further batched multiplies. This remains rank-safe when an absent industry
dummy appears before a later populated dummy. Return NaN for dates whose residual norm is
negligible relative to the demeaned factor norm. Validate dimensional consistency,
nonempty fixed industry levels, and a positive finite residual fraction.

- [ ] **Step 3: Run numerical tests and verify GREEN**

```bash
.venv/bin/pytest tests/test_style_neutralize.py -q
```

Expected: all tests pass with no warnings.

- [ ] **Step 4: Run existing neutralisation regression tests**

```bash
.venv/bin/pytest tests/test_neutralize.py tests/test_style_neutralize.py -q
```

Expected: both old GP neutralisation and new style neutralisation suites pass.

- [ ] **Step 5: Commit the numerical module and ignore rule**

```bash
git add .gitignore helix/eval/style_neutralize.py
git commit -m "feat: add vectorized daily style neutralization"
```

## Task 3: Implement the reproducible experiment orchestrator

**Files:**

- Create: `scripts/g3_style_ablation.py`
- Test: `tests/test_g3_style_ablation.py`

- [ ] **Step 1: Implement constants and strict validation**

Define these fixed defaults:

```python
TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TARGET = "label_d2_hit_8pct"
FORMAL_FACTOR = "gp_000"
DEFAULT_SEEDS = (7, 13, 42, 101, 211, 307, 419, 523, 631, 743)
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 20)
DEFAULT_BLOCK_LENGTH = 20
DEFAULT_TOP_K = 10
```

`split_evaluation_windows` sorts and deduplicates keys, returns independent copies for
`TRAIN_START <= trade_date <= TRAIN_END` and `trade_date > TRAIN_END`, and rejects an
incomplete primary calendar. `validate_seed_contract` requires at least three distinct
integer seeds. Keep the two frames separate throughout `run_experiment`; call
`decide_go` before constructing or rendering OOS results.

- [ ] **Step 2: Implement point-in-time style construction and cache coverage**

`compute_trailing_styles` accepts long market rows with
`trade_date, stock_code, pct_chg, total_mv, turnover_rate_f`. It uses stock-grouped,
20-observation rolling operations with `min_periods=20`; momentum is
`prod(1 + pct_chg/100) - 1`, volatility is sample standard deviation of `pct_chg/100`,
mean turnover is the rolling mean, and `log_total_mv = log(total_mv)` on D0. Reject
duplicate keys and nonpositive market cap.

Build/download helpers use the repository's `TushareSource` and cache only the required
columns. They fetch a market-wide date once, write atomically, skip cached dates, and
include at least 19 sessions before the first D0. SW2021 membership is aligned with
vectorised merge/filter logic on `in_date/out_date`; overlapping memberships on a date
are rejected.

- [ ] **Step 3: Replay exactly the formal factor**

Load `FactorLibrary`, require `kind == "event"`, require exactly one factor named
`gp_000`, project only its `field_names` plus labels/prices from the Parquet scan, pack
with `build_event_panel`, and call `compute_factors`. Apply no extra sign: the library
evaluator already applies the recorded sign. Record the library SHA-256 in outputs.

- [ ] **Step 4: Implement canonical metrics and Top10 backtest**

Use `helix.eval.ic.daily_ic/summarize_ic`,
`helix.eval.metrics.daily_gini/precision_at_k/lift_at_k`, and the cost semantics from
`helix.eval.backtest`. Add a small public wrapper in the script only when the existing
helper is private; do not modify the engine. The main D+2 book uses existing
`label_px_d1_open` and `label_px_d2_close`, applies historical commission, transfer,
stamp, and 10 bps per-side slippage multiplicatively, divides daily capital by Top10
and overlap two, and never replaces an invalid selected name.

Return a metric dictionary whose required keys are exactly:

```python
REQUIRED_METRICS = (
    "ic_mean",
    "icir",
    "gini",
    "top10_hit_rate",
    "base_rate",
    "lift",
    "net_return",
    "net_per_trade",
    "cagr",
    "sharpe",
    "max_drawdown",
    "n_days",
    "coverage",
)
```

- [ ] **Step 5: Implement decay and forward-window truncation**

`forward_return_panel` aligns each D0 event with D+1 raw open and D+h raw close,
adjusted via same-day factors or the equivalent pct-change chain, and returns NaN when
the entry/exit is missing or the exit exceeds the evaluated window end. Decay reports
IC mean, ICIR, and Top10 net return per trade for each requested horizon and arm.

- [ ] **Step 6: Implement seeded moving-block robustness**

`circular_block_bootstrap_indices` uses `np.random.default_rng(seed)`, samples starting
date positions, expands complete circular blocks, and truncates to `n_dates`.
`bootstrap_metric_summary` evaluates each arm on each date-index replicate and reports
mean plus `std(ddof=1)` for every finite scalar. Preserve the date order emitted by the
blocks for cumulative metrics.

- [ ] **Step 7: Implement the mechanical decision and Markdown publication**

`decide_go` accepts only the training neutral ICIR, p95 threshold, raw net-per-trade,
and neutral net-per-trade. It returns NO-GO for non-finite inputs, equality to p95, zero
return, or opposite signs. It has no OOS parameter.

Read p95 by recomputing the 0.95 quantile of `icir` in
`data/artifacts/placebo_ic_distribution.parquet` after validating its training metadata.
`render_report` creates the required executive decision, exact rule inputs, deterministic
comparison, seed mean ± standard deviation, orthogonality audit, decay curve, OOS
appendix, limitations, and reproduction sections. Publish Markdown and JSON atomically.

- [ ] **Step 8: Run all script tests and verify GREEN**

```bash
.venv/bin/pytest tests/test_g3_style_ablation.py -q
```

Expected: all tests pass, including the test proving OOS changes cannot alter decision.

- [ ] **Step 9: Run both new suites together**

```bash
.venv/bin/pytest tests/test_style_neutralize.py tests/test_g3_style_ablation.py -q
```

Expected: all new tests pass with no warnings.

- [ ] **Step 10: Commit the experiment script**

```bash
git add scripts/g3_style_ablation.py
git commit -m "feat: add reproducible G3 style ablation"
```

## Task 4: Build cache, run the experiment, and audit the report

**Files:**

- Create: `docs/risk/g3_style_ablation.md`
- Create ignored: `data/artifacts/g3_style_ablation.json`
- Create ignored: `data/artifacts/g3_style_market.parquet`
- Create ignored: `data/artifacts/g3_sw2021_members.parquet`

- [ ] **Step 1: Run refresh mode to fill the point-in-time cache**

```bash
.venv/bin/python scripts/g3_style_ablation.py \
  --refresh-style-cache \
  --seeds 7,13,42,101,211,307,419,523,631,743 \
  --bootstrap-block-length 20 \
  --top-k 10 \
  --horizons 1,2,3,5,10,20
```

Expected: cache coverage includes the 19-session lookback, every training D0, and the
available OOS appendix range; output reports missingness instead of silently imputing.

- [ ] **Step 2: Rerun cache-only mode to prove reproducibility**

```bash
.venv/bin/python scripts/g3_style_ablation.py \
  --cache-only \
  --seeds 7,13,42,101,211,307,419,523,631,743 \
  --bootstrap-block-length 20 \
  --top-k 10 \
  --horizons 1,2,3,5,10,20
```

Expected: exit 0 and identical deterministic, bootstrap, decay, and decision values in
the JSON artifact.

- [ ] **Step 3: Run a read-only result audit**

Use a short Python assertion command to verify:

- report metadata says `2022-01-04` through `2024-09-04` and 649 dates;
- ten unique seeds are present;
- p95 equals a fresh linear quantile from the saved placebo distribution;
- decision inputs equal the training deterministic metrics;
- no OOS field is named in the decision input object;
- both arms contain every requested metric;
- every decay exit is bounded by its window end; and
- the Markdown contains the same GO/NO-GO token as JSON.

- [ ] **Step 4: Commit the generated report**

```bash
git add docs/risk/g3_style_ablation.md
git commit -m "docs: publish G3 style ablation results"
```

## Task 5: Update governance and complete verification

**Files:**

- Modify: `docs/factor-governance.md`
- Verify: all changed files

- [ ] **Step 1: Update D7 and preserve D13**

Replace only D7's `量级` and `状态` cells with the measured raw-to-neutral changes and
the GO/NO-GO result, linking `risk/g3_style_ablation.md`. Do not edit D13. Before and
after editing, capture the exact D13 line and assert the strings are identical.

- [ ] **Step 2: Run targeted tests**

```bash
.venv/bin/pytest tests/test_style_neutralize.py tests/test_g3_style_ablation.py -q
```

Expected: all pass.

- [ ] **Step 3: Run the full test suite**

```bash
.venv/bin/pytest -q
```

Expected: zero failures and zero errors.

- [ ] **Step 4: Run Ruff over source, scripts, and tests**

```bash
.venv/bin/ruff check helix scripts tests
```

Expected: `All checks passed!`.

- [ ] **Step 5: Run diff and requirement audits**

```bash
git diff --check
git status --short
rg -n "D7|D13|GO|NO-GO|2022-01-04|2024-09-04|样本外" \
  docs/factor-governance.md docs/risk/g3_style_ablation.md
```

Expected: no whitespace errors; only scoped source/test/docs changes; `uv.lock` absent
from status; D13 unchanged; report contains the strict training/OOS boundary and result.

- [ ] **Step 6: Commit governance closure**

```bash
git add docs/factor-governance.md
git commit -m "docs: close D7 with style ablation evidence"
```

- [ ] **Step 7: Repeat fresh completion verification after the final commit**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check helix scripts tests
git status --short
```

Expected: full tests pass, Ruff reports no violations, and the worktree is clean.
