# D+2 Realistic Exit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, same-day-data-only exit resolver that defers D+2 exits through suspensions and sealed limit-down sessions, then quantify the bias on the current aligned Top4 backtest.

**Architecture:** Keep D0 ranking, no-replacement selection, transaction costs, and fixed-overlap cohort accounting unchanged. Split D+1 entry observability from D+2 label observability, resolve each entered position serially against actual daily `down_limit` and OHLC data, and derive both baseline and realistic returns from one trade ledger. A thin reproducible report script rebuilds the current ARGUS Top4 holdings and reuses the engine resolver.

**Tech Stack:** Python 3.11, NumPy, pandas, Pydantic, PyArrow, Tushare, pytest, Ruff.

---

### Task 1: Pin entry observability and all exit boundaries with failing tests

**Files:**
- Modify: `tests/test_labels.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Add a label test proving D+2 suspension does not erase a valid D+1 entry**

Add to `tests/test_labels.py`:

```python
def test_d2_suspension_preserves_d1_entry_observability(cfg):
    trading = np.ones((6, 1))
    trading[3, 0] = 0.0
    panel = make_panel(is_trading=trading)

    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)

    assert labels.entry_valid[1, 0]
    assert labels.entry_price[1, 0] == pytest.approx(10.0)
    assert not labels.valid[1, 0]
    assert np.isnan(labels.y[1, 0])
```

- [ ] **Step 2: Add a realistic-exit panel fixture**

Add `Panel` to the imports in `tests/test_backtest.py` and define a fixture helper whose
fields include `open`, `high`, `close`, `open_hfq`, `close_hfq`, `down_limit`,
`limit_price_observed`, and `is_trading`. Dates and codes must align exactly with the
prediction/label arrays.

```python
def make_exit_panel(n_dates=8, n_codes=1, **overrides):
    shape = (n_dates, n_codes)
    fields = {
        "open": np.full(shape, 10.0),
        "high": np.full(shape, 10.0),
        "close": np.full(shape, 10.0),
        "open_hfq": np.full(shape, 10.0),
        "close_hfq": np.full(shape, 10.0),
        "down_limit": np.full(shape, 9.0),
        "limit_price_observed": np.ones(shape),
        "is_trading": np.ones(shape),
    }
    fields.update(overrides)
    return Panel(
        dates=np.array([f"2024010{i}" for i in range(1, n_dates + 1)]),
        codes=np.array([f"00000{j}.SZ" for j in range(n_codes)]),
        fields={k: np.asarray(v, dtype=float) for k, v in fields.items()},
    )
```

- [ ] **Step 3: Add four required resolver boundary tests**

Add tests which call the public `resolve_realistic_exit` function directly:

```python
def test_three_sealed_limit_down_days_exit_at_fourth_day_open(cfg):
    panel = make_exit_panel()
    for row in (2, 3, 4):
        panel.fields["open"][row, 0] = 9.0
        panel.fields["high"][row, 0] = 9.0
        panel.fields["close"][row, 0] = 9.0
    panel.fields["open"][5, 0] = 9.5
    panel.fields["open_hfq"][5, 0] = 9.5

    resolved = resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.exit_index == 5
    assert resolved.exit_price == pytest.approx(9.5)
    assert resolved.exit_session == "open"
    assert resolved.d2_limit_down
    assert resolved.delay_days == 3
    assert resolved.holding_days == 5
    assert resolved.exit_price / 10.0 - 1.0 == pytest.approx(-0.05)


def test_intraday_opening_that_reseals_at_close_keeps_deferring(cfg):
    panel = make_exit_panel()
    panel.fields["open"][2:4, 0] = 9.0
    panel.fields["close"][2:4, 0] = 9.0
    panel.fields["high"][2, 0] = 9.0
    panel.fields["high"][3, 0] = 9.6
    panel.fields["open"][4, 0] = 9.4
    panel.fields["open_hfq"][4, 0] = 9.4

    resolved = resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.exit_index == 4
    assert resolved.exit_session == "open"
    assert resolved.delay_days == 2


def test_d2_suspension_defers_to_next_tradable_open(cfg):
    panel = make_exit_panel()
    panel.fields["is_trading"][2, 0] = 0.0
    panel.fields["open"][3, 0] = 9.4
    panel.fields["open_hfq"][3, 0] = 9.4

    resolved = resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.exit_index == 3
    assert resolved.exit_price == pytest.approx(9.4)
    assert resolved.exit_session == "open"
    assert resolved.encountered_suspension
    assert resolved.delay_days == 1


def test_unresolved_position_at_panel_end_has_no_fabricated_price(cfg):
    panel = make_exit_panel(n_dates=4)
    panel.fields["open"][2:, 0] = 9.0
    panel.fields["high"][2:, 0] = 9.0
    panel.fields["close"][2:, 0] = 9.0

    resolved = resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.unresolved_at_end
    assert resolved.exit_index is None
    assert np.isnan(resolved.exit_price)
```

The first case uses D2/D3/D4 raw OHLC equal to each day's observed `down_limit`, then a
D5 open above its limit. Assert `exit_session == "open"`, `delay_days == 3`,
`holding_days == 5`, and the exact adjusted-price return. The reseal case uses
`open == close == down_limit` and `high > down_limit`; assert it is skipped. The
suspension case sets D2 `is_trading=0` and asserts D3 open execution. The terminal case
asserts `unresolved_at_end`, `exit_index is None`, and `exit_price` is NaN.

- [ ] **Step 4: Add exact per-symbol limit and integration tests**

Add the per-symbol limit test:

```python
def test_each_symbol_uses_its_actual_daily_limit(cfg):
    down = np.column_stack([np.full(6, 9.0), np.full(6, 8.0)])
    panel = make_exit_panel(n_dates=6, n_codes=2, down_limit=down)
    panel.fields["open"][2] = [9.0, 9.0]
    panel.fields["high"][2] = [9.0, 9.0]
    panel.fields["close"][2] = [9.0, 9.0]
    panel.fields["open"][3, 0] = 9.3
    panel.fields["open_hfq"][3, 0] = 9.3

    first = resolve_realistic_exit(panel, 0, 0, cfg)
    second = resolve_realistic_exit(panel, 0, 1, cfg)

    assert first.d2_limit_down
    assert first.exit_index == 3
    assert not second.d2_limit_down
    assert second.exit_index == 2
    assert second.exit_session == "d2_close"
```

Extend `make_labels` with an `entry_valid` argument, then add integration assertions:

```python
def test_unresolved_realistic_exit_keeps_the_slot_in_cash(cfg):
    panel = make_exit_panel(n_dates=4)
    panel.fields["open"][2:, 0] = 9.0
    panel.fields["high"][2:, 0] = 9.0
    panel.fields["close"][2:, 0] = 9.0
    labels = make_labels(
        y=np.zeros((4, 1)),
        entry=np.full((4, 1), 10.0),
        exit_price=np.full((4, 1), 9.0),
        entry_valid=np.ones((4, 1), dtype=bool),
    )
    predictions = np.full((4, 1), np.nan)
    predictions[0, 0] = 1.0
    candidates = np.zeros((4, 1), dtype=bool)
    candidates[0, 0] = True

    result = run_backtest(
        predictions, labels, candidates, panel.dates, cfg,
        free(top_k=1, enable_realistic_exit=True), panel,
    )

    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.0)
    assert result.summary["unresolved_at_end"] == 1.0
    assert result.trades["unresolved_at_end"].iloc[0]


def test_switch_off_ignores_realistic_market_data(cfg):
    labels = make_labels([[1.0]], [[10.0]], [[11.0]])
    predictions = np.array([[1.0]])
    candidates = np.ones((1, 1), dtype=bool)
    dates = np.array(["20240101"])
    config = free(top_k=1)

    without_panel = run_backtest(predictions, labels, candidates, dates, cfg, config)
    with_panel = run_backtest(
        predictions, labels, candidates, dates, cfg, config, make_exit_panel()
    )

    pd.testing.assert_frame_equal(without_panel.daily, with_panel.daily)
    assert without_panel.summary == with_panel.summary
```

- [ ] **Step 5: Run the new tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_labels.py tests/test_backtest.py -q
```

Expected: failures because `LabelSet.entry_valid`, `resolve_realistic_exit`,
`enable_realistic_exit`, and realistic trade accounting do not exist yet. Fix only test
syntax or fixture mistakes until failures are caused by those missing behaviors.

### Task 2: Preserve actual limit provenance and D+1 entry state

**Files:**
- Modify: `helix/data/panel.py`
- Modify: `helix/labels/touch_label.py`
- Modify: `tests/test_labels.py`

- [ ] **Step 1: Record whether `stk_limit` was observed**

In `build_panel`, compute the finite source mask before filling gaps:

```python
observed = np.isfinite(up) & np.isfinite(down)
panel.add("limit_price_observed", observed.astype(np.float32))
```

If the table is entirely absent, add an all-false mask before applying rule-based fallback.

- [ ] **Step 2: Split entry eligibility from label validity**

Extend `LabelSet` with optional `entry_valid` and expose a backward-compatible property
used by realistic mode:

```python
entry_valid: np.ndarray | None = None

@property
def executable_entry(self) -> np.ndarray:
    return self.valid if self.entry_valid is None else self.entry_valid
```

In `build_touch_label`, build `entry_valid` from the D0 universe, D+1 trading state,
finite positive D+1 adjusted open, and the existing D+1 limit-up exclusion. Build
`valid = entry_valid & touch_tradable & finite(D+2 high)`. Store `entry_price` and
`target_price` under `entry_valid`, while `exit_price` remains valid only where the D+2
close exists.

- [ ] **Step 3: Run label tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_labels.py -q
```

Expected: all label tests pass, including the new D+2 suspension case.

### Task 3: Implement the serial realistic-exit resolver

**Files:**
- Modify: `helix/eval/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Add the result type and validation helpers**

Add an immutable `ExitResolution` containing:

```python
exit_index: int | None
exit_price: float
exit_session: str
d2_limit_down: bool
encountered_suspension: bool
missing_limit_data: bool
delay_days: int | None
holding_days: int
unresolved_at_end: bool
```

Add helpers that require `open`, `high`, `close`, `open_hfq`, `close_hfq`,
`down_limit`, `limit_price_observed`, and `is_trading`, and compare prices with
`np.isclose(..., rtol=0.0, atol=limit_price_eps)`.

- [ ] **Step 2: Implement D+2 handling**

Implement:

```python
def resolve_realistic_exit(
    panel: Panel, d0_index: int, stock_index: int, label_cfg: LabelConfig
) -> ExitResolution:
```

At D+2, defer on suspension/missing bar/missing observed limit or an all-day sealed
limit-down (`high` and `close` both equal the actual limit). Otherwise return D+2
`close_hfq` with session `d2_close`.

- [ ] **Step 3: Implement D+3 onward serial handling**

For each later panel row in order: skip suspension or missing actual limits; exit at
`open_hfq` only when raw open is above limit plus epsilon; otherwise exit at `close_hfq`
only when raw close is above limit plus epsilon. Ignore intraday highs for execution.
Return unresolved without a price at the panel boundary.

- [ ] **Step 4: Run resolver tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_backtest.py -k 'limit_down or reseal or suspension or unresolved or per_symbol' -q
```

Expected: all resolver-focused tests pass.

### Task 4: Integrate the switch, aligned ledger, and requested metrics

**Files:**
- Modify: `helix/config.py`
- Modify: `configs/default.yaml`
- Modify: `configs/argus_neutral.yaml`
- Modify: `helix/eval/backtest.py`
- Modify: `helix/pipeline.py`
- Modify: `tests/test_backtest.py`
- Modify: `tests/test_pipeline_masks.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add the opt-in switch**

Add `enable_realistic_exit: bool = False` to `BacktestConfig` and both YAML files.
Document that `exit_rule=target` cannot combine with realistic exit; model validation or
`run_backtest` must reject that combination because the requested correction applies to
D+2 close exits only.

- [ ] **Step 2: Add a trade ledger without changing default selection**

Extend `BacktestResult` with `trades: pd.DataFrame = field(default_factory=pd.DataFrame)`.
Add `panel: Panel | None = None` as the final `run_backtest` parameter. Keep the existing
default-mode branch byte-for-byte equivalent in selection and return calculations.
Realistic mode uses `labels.executable_entry` only after the D0 Top-K list is fixed and
creates one row per entered position.

- [ ] **Step 3: Derive aligned baseline and realistic daily returns from one ledger**

For resolved comparable trades, calculate both D+2 baseline and actual exit returns.
For missing D+2 baselines and terminal unresolved trades, leave both matched return
columns at zero and retain the Top-K denominator. Aggregate each D0 basket as
`sum(net_return) / top_k / overlap`. Do not rank or select a second time.

- [ ] **Step 4: Add requested metrics**

Add helpers for CAGR, day win rate, positive-day mean / absolute negative-day mean,
longest consecutive negative-day run, Sharpe, and maximum drawdown. Populate realistic
summary keys plus limit-down share, mean delay, mean loss, D+2 suspension count, and
terminal unresolved count. Keep all pre-existing summary keys and values unchanged when
the switch is disabled.

- [ ] **Step 5: Pass the panel through the pipeline only for realistic mode**

Update `pipeline.backtest` to pass `prepared.panel`. Update mocks and smoke tests for the
optional keyword without changing existing mask assertions.

- [ ] **Step 6: Run integration tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_backtest.py tests/test_pipeline_smoke.py tests/test_pipeline_masks.py -q
```

Expected: all tests pass. The compatibility assertion must show exact equality with the
existing switch-off baseline.

### Task 5: Build the reproducible ARGUS risk-quantification command

**Files:**
- Create: `scripts/d2_limit_down_bias.py`
- Create: `tests/test_d2_limit_down_bias.py`

- [ ] **Step 1: Write failing tests for fixed Top4 extraction and matched aggregation**

Construct a two-day scored frame where one Top4 candidate is D+1-unfillable and rank 5
would be fillable. Assert the output contains only the remaining names from the original
Top4 and never rank 5. Construct a miniature trade ledger and assert baseline/realistic
metrics use identical dates and denominators.

- [ ] **Step 2: Run the report tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_d2_limit_down_bias.py -q
```

Expected: import or missing-function failure from the new report module.

- [ ] **Step 3: Implement current-holding reconstruction**

Reuse `feature_columns`, `unfillable_mask`, and `fit_and_score` from the existing ARGUS
scripts. Use the existing split (`2024-09-04`, three-day embargo), seeds `7,13,42`, and
classification ranker. Select exactly the highest four scores before applying D+1
fillability; never fill from rank 5.

- [ ] **Step 4: Implement targeted market-data caching**

Fetch daily bars, adjustment factors, and actual `stk_limit` rows through the repository's
configured Tushare client for the required test-period trade dates. Persist only required
columns under ignored `data/raw/d2_exit_cache/`. Validate uniqueness of
`(trade_date, ts_code)`, actual limit coverage, and alignment between event D+1 prices and
cached adjusted prices before trusting the data.

- [ ] **Step 5: Reuse the engine resolver and emit structured results**

Call `resolve_realistic_exit` for every entered Top4 holding. Write a machine-readable
trade ledger and JSON summary under ignored `data/artifacts/`, including per-seed metrics,
means, standard deviations, limit-down samples, suspensions, and unresolved positions.

- [ ] **Step 6: Run report tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_d2_limit_down_bias.py -q
```

Expected: all report helper tests pass.

### Task 6: Generate and verify the Markdown risk report

**Files:**
- Create: `docs/risk/d2_limit_down_bias.md`

- [ ] **Step 1: Run the actual aligned backtest**

Run the report script against `data/raw/argus_quant_working.parquet` with Top4,
no-replacement selection, seeds `7,13,42`, and current statutory costs/slippage. Record
the exact command and data cutoff in the generated report.

- [ ] **Step 2: Render the required comparison**

Include a baseline-versus-realistic table for CAGR, maximum drawdown, day win rate,
profit/loss ratio, maximum consecutive losses, and Sharpe. Include limit-down sample
share, mean delay, mean realized loss, suspension cases, and terminal unresolved cases.

- [ ] **Step 3: Audit the report against its JSON source**

Programmatically or manually compare every displayed scalar with the generated JSON.
Ensure percentages use consistent denominators and explain the matched-sample rule.

### Task 7: Full verification and handoff

**Files:**
- Verify all modified files

- [ ] **Step 1: Run the full test suite**

Run:

```bash
.venv/bin/pytest -q
```

Expected: zero failed tests.

- [ ] **Step 2: Run lint checks**

Run:

```bash
.venv/bin/ruff check .
```

Expected: zero lint errors.

- [ ] **Step 3: Verify default-off compatibility from a saved fixture**

Run the existing `tests/test_backtest.py` default-mode tests and compare the baseline
summary produced before and after the feature. Confirm the new switch is false in both
checked-in YAML files.

- [ ] **Step 4: Review the final diff and requirements**

Run `git diff --check`, inspect `git diff --stat`, and map every user requirement to a
test or report section. Do not commit overlapping pre-existing user changes in
`backtest.py`, label, pipeline, or their tests; report them separately from this task's
new lines.
