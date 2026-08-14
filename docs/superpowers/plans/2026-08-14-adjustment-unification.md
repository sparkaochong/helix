# Adjustment Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a fail-closed, point-in-time HFQ contract across data loading, price-derived features, labels, and backtests while preserving the existing `gp_000` event table as an explicitly unverified legacy baseline.

**Architecture:** Add a serializable lineage object to `Panel`, construct HFQ prices only by exact same-stock/same-date factor joins, and propagate an immutable adjustment stamp into labels and backtests. Governed event loading requires a manifest that maps every used field to four real audit columns; the current table lacks that evidence and remains accessible only to a dedicated fixed-score comparison script.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, pandas, PyArrow, pytest, Ruff, Git, GitHub CLI.

---

## File map

| File | Responsibility |
| --- | --- |
| `helix/data/price_lineage.py` | Typed panel lineage, version digest, validation, adjustment stamp |
| `helix/data/panel.py` | Exact HFQ construction and lineage-aware cache serialization |
| `helix/features/base_fields.py` | HFQ price features; raw inputs limited to limit-state checks |
| `helix/labels/touch_label.py` | HFQ D+1/D+2 labels and stamp propagation |
| `helix/eval/backtest.py` | Fail-closed validation before return accounting |
| `helix/data/event_lineage.py` | Event manifest and four-column audit validation |
| `scripts/adjustment_unification_baseline.py` | Fixed-score legacy-before/HFQ-after comparison |
| `docs/risk/adjustment_unification_fix.md` | Repair report and metric table |

### Task 1: Add serializable point-in-time price lineage

**Files:**
- Create: `helix/data/price_lineage.py`
- Modify: `helix/data/panel.py:25-159`
- Modify: `helix/pipeline.py:33-75`
- Create: `tests/test_price_lineage.py`
- Modify: `tests/test_panel_cache_upgrade.py`

- [ ] **Step 1: Write failing exact-date and cache tests**

Create `tests/test_price_lineage.py`:

```python
def test_adjusted_prices_use_same_stock_and_trade_date() -> None:
    dates = np.array(["20240102", "20240103"])
    fields, lineage = build_adjusted_price_fields(
        daily_frame(), adj_frame(), dates, np.array(["000001.SZ"])
    )
    np.testing.assert_allclose(fields["open_hfq"][:, 0], [10.0, 10.0])
    assert lineage["open_hfq"].source_date.tolist() == dates.tolist()
    assert lineage["open_hfq"].price_basis == "hfq"
    assert lineage["open_hfq"].adj_factor_version.startswith(
        "raw-times-same-day-adj-v1:"
    )


def test_neighboring_session_factor_is_rejected() -> None:
    with pytest.raises(PriceLineageError, match="20240103.*000001.SZ.*same-date"):
        build_adjusted_price_fields(
            daily_frame(), shifted_adj_frame(),
            np.array(["20240102", "20240103"]), np.array(["000001.SZ"]),
        )


def test_panel_cache_round_trips_lineage(tmp_path) -> None:
    panel = governed_panel()
    path = tmp_path / "panel.npz"
    panel.save(path)
    loaded = Panel.load(path)
    stamp = loaded.require_adjusted_prices(("open_hfq", "close_hfq"), "test")
    assert stamp.price_basis == "hfq"
    assert stamp.adj_factor_version == panel.price_lineage["open_hfq"].adj_factor_version
```

The helpers use raw close `10 -> 5`, factor `1 -> 2`, and a shifted failure factor dated `20240104`. Extend `tests/test_panel_cache_upgrade.py` to prove a legacy cache loads without lineage and forces `pipeline.prepare` to rebuild rather than invent metadata.

- [ ] **Step 2: Run tests and verify the API is absent**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_price_lineage.py tests/test_panel_cache_upgrade.py -v
```

Expected: import/collection failure for `build_adjusted_price_fields` or `PriceLineageError`.

- [ ] **Step 3: Implement the lineage model**

Create `helix/data/price_lineage.py` with these complete interfaces:

```python
HFQ_BASIS = "hfq"
ADJUSTMENT_ALGORITHM = "raw-times-same-day-adj-v1"


class PriceLineageError(ValueError):
    """A governed price field has missing or invalid lineage."""


@dataclass(frozen=True)
class AdjustmentStamp:
    price_basis: str
    adj_factor_version: str


@dataclass(frozen=True)
class PriceLineage:
    source_date: np.ndarray
    as_of_time: np.ndarray
    price_basis: str
    adj_factor_version: str

    def __post_init__(self) -> None:
        source = np.asarray(self.source_date).astype(str)
        as_of = np.asarray(self.as_of_time).astype(str)
        if source.ndim != 1 or as_of.shape != source.shape:
            raise PriceLineageError("source_date and as_of_time must be aligned 1-D arrays")
        if not self.adj_factor_version:
            raise PriceLineageError("adj_factor_version must be non-empty")
        object.__setattr__(self, "source_date", source)
        object.__setattr__(self, "as_of_time", as_of)


def adjustment_factor_version(frame: pd.DataFrame) -> str:
    ordered = frame[["trade_date", "ts_code", "adj_factor"]].copy()
    ordered["trade_date"] = ordered["trade_date"].astype(str)
    ordered["ts_code"] = ordered["ts_code"].astype(str)
    ordered = ordered.sort_values(["trade_date", "ts_code"], kind="stable")
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    return f"{ADJUSTMENT_ALGORITHM}:{digest}"


def make_hfq_lineage(dates: np.ndarray, version: str) -> PriceLineage:
    source = np.asarray(dates).astype(str)
    compact = [value.replace("-", "") for value in source]
    as_of = np.asarray([
        f"{value[:4]}-{value[4:6]}-{value[6:]}T15:00:00+08:00" for value in compact
    ])
    return PriceLineage(source, as_of, HFQ_BASIS, version)


def require_hfq_lineage(dates, lineage, fields, purpose) -> AdjustmentStamp:
    expected = np.asarray(dates).astype(str)
    missing = [name for name in fields if name not in lineage]
    if missing:
        raise PriceLineageError(f"{purpose}: missing price lineage for {missing}")
    versions = set()
    for name in fields:
        meta = lineage[name]
        if meta.price_basis != HFQ_BASIS:
            raise PriceLineageError(
                f"{purpose}: {name} price_basis={meta.price_basis!r}, expected 'hfq'"
            )
        if not np.array_equal(meta.source_date, expected):
            mismatch = int(np.flatnonzero(meta.source_date != expected)[0])
            raise PriceLineageError(
                f"{purpose}: {name} source_date mismatch at {expected[mismatch]}"
            )
        as_of_dates = np.asarray([value[:10].replace("-", "") for value in meta.as_of_time])
        expected_dates = np.asarray([value.replace("-", "") for value in expected])
        if not np.array_equal(as_of_dates, expected_dates):
            raise PriceLineageError(f"{purpose}: {name} as_of_time is not date-local")
        versions.add(meta.adj_factor_version)
    if len(versions) != 1:
        raise PriceLineageError(f"{purpose}: inconsistent adj_factor_version values")
    return AdjustmentStamp(HFQ_BASIS, versions.pop())
```

- [ ] **Step 4: Add exact construction and Panel persistence**

Add `price_lineage: dict[str, PriceLineage]` to `Panel`, an optional keyword-only lineage to `Panel.add`, and `require_adjusted_prices`. Implement:

```python
def build_adjusted_price_fields(daily, adj, dates, codes):
    adj = _deduplicate_or_fail(adj, ("trade_date", "ts_code"), "adj_factor")
    factor = _pivot(adj, "adj_factor", dates, codes)
    raw_close = _pivot(daily, "close", dates, codes)
    invalid = np.isfinite(raw_close) & (~np.isfinite(factor) | (factor <= 0))
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise PriceLineageError(
            f"{dates[row]} {codes[column]} has no positive same-date adj_factor"
        )
    metadata = make_hfq_lineage(dates, adjustment_factor_version(adj))
    fields = {"adj_factor": factor}
    lineage = {}
    for name in PRICE_COLUMNS:
        fields[f"{name}_hfq"] = _pivot(daily, name, dates, codes) * factor
        lineage[f"{name}_hfq"] = metadata
    return fields, lineage
```

Serialize lineage under fixed-width NumPy keys `__lineage__<field>__<attribute>` and exclude those keys from `Panel.fields` during load. Keep `allow_pickle=False`. In `pipeline.prepare`, require lineage after cache load; on `PriceLineageError`, rebuild panel and features.

`build_panel` must call `build_adjusted_price_fields` once and attach every returned lineage record. `Panel.slice_dates` slices `source_date` and `as_of_time` by the identical date slice; `Panel.select_codes` retains the field-level record unchanged because its audit arrays are date-aligned, not stock-aligned.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_price_lineage.py tests/test_panel_cache_upgrade.py -v
git add helix/data/price_lineage.py helix/data/panel.py helix/pipeline.py \
  tests/test_price_lineage.py tests/test_panel_cache_upgrade.py
git commit -m "feat: enforce point-in-time adjusted price lineage"
```

Expected: all focused tests pass.

### Task 2: Compute all price-derived base features from HFQ

**Files:**
- Modify: `helix/features/base_fields.py:20-84`
- Create: `tests/test_base_fields_adjustment.py`
- Modify: `tests/test_pipeline_smoke.py:35-75`

- [ ] **Step 1: Write failing split and missing-lineage tests**

Use a three-date split panel where raw close is `10 -> 5`, factor is `1 -> 2`, and HFQ close is `10 -> 10`:

```python
fields = compute_base_fields(governed_split_panel())
assert fields["ret1"][1, 0] == pytest.approx(0.0)
assert fields["gap"][1, 0] == pytest.approx(0.0)
assert fields["intraday"][1, 0] == pytest.approx(0.0)

panel = governed_split_panel()
panel.price_lineage.clear()
with pytest.raises(PriceLineageError, match="compute_base_fields.*missing price lineage"):
    compute_base_fields(panel)
```

- [ ] **Step 2: Verify current failure**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_base_fields_adjustment.py -v
```

Expected: `gap` is about `-0.5` or missing lineage is accepted.

- [ ] **Step 3: Gate and replace feature formulas**

Require `open/high/low/close_hfq` at function entry. Replace price math with:

```python
ret1 = ops.div(close_h, ops.delay(close_h, 1)) - 1.0
hl_h = high_h - low_h
"gap": ops.div(open_h, ops.delay(close_h, 1)) - 1.0,
"intraday": ops.div(close_h, open_h) - 1.0,
"hl_range": ops.div(hl_h, ops.delay(close_h, 1)),
"close_pos": ops.div(close_h - low_h, hl_h),
"upper_shadow": ops.div(high_h - np.maximum(open_h, close_h), ops.delay(close_h, 1)),
"lower_shadow": ops.div(np.minimum(open_h, close_h) - low_h, ops.delay(close_h, 1)),
"open_gap_mean5": ops.ts_mean(ops.div(open_h, ops.delay(close_h, 1)) - 1.0, 5),
```

Keep only the allowed raw limit-state formulas:

```python
"to_up_limit": ops.div(up_limit - raw_close, raw_close),
"limitup_cnt20": ops.ts_sum((raw_close >= up_limit - 0.001).astype(np.float64), 20),
```

Remove raw `open/high/low/pre_close` from feature math. Attach explicit synthetic lineage in `tests/test_pipeline_smoke.py`.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_base_fields_adjustment.py tests/test_pipeline_smoke.py -v
git add helix/features/base_fields.py tests/test_base_fields_adjustment.py \
  tests/test_pipeline_smoke.py
git commit -m "fix: compute price features from governed HFQ prices"
```

Expected: all tests pass.

### Task 3: Carry the HFQ contract through labels

**Files:**
- Modify: `helix/labels/touch_label.py:25-96`
- Modify: `tests/test_labels.py`
- Modify: `tests/test_shared_entry_check.py`
- Modify: `tests/test_pipeline_masks.py`

- [ ] **Step 1: Add failing label contract tests**

```python
def test_label_fails_closed_without_hfq_lineage(cfg):
    panel = make_panel()
    panel.price_lineage.clear()
    with pytest.raises(PriceLineageError, match="build_touch_label.*missing price lineage"):
        build_touch_label(panel, np.ones(panel.shape, dtype=bool), cfg)


def test_label_carries_adjustment_stamp(cfg):
    labels = build_touch_label(make_panel(), np.ones((6, 1), dtype=bool), cfg)
    assert labels.adjustment == AdjustmentStamp("hfq", TEST_ADJ_VERSION)
```

Modify `make_panel` to attach explicit lineage to `open_hfq`, `high_hfq`, and `close_hfq`.

- [ ] **Step 2: Run and observe failure**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest tests/test_labels.py -v
```

Expected: `LabelSet` has no `adjustment` or the ungoverned panel is accepted.

- [ ] **Step 3: Require and propagate the stamp**

Add required `adjustment: AdjustmentStamp` to `LabelSet`. At `build_touch_label` entry:

```python
adjustment = panel.require_adjusted_prices(
    ("open_hfq", "high_hfq", "close_hfq"), "build_touch_label"
)
```

Pass it to `LabelSet`. Do not add a default. Keep `entry_is_fillable` unchanged, so raw open remains isolated to fillability. Add explicit stamps to manual test `LabelSet` objects and panel lineage to `tests/test_shared_entry_check.py`.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_labels.py tests/test_shared_entry_check.py tests/test_pipeline_masks.py -v
git add helix/labels/touch_label.py tests/test_labels.py \
  tests/test_shared_entry_check.py tests/test_pipeline_masks.py
git commit -m "feat: carry adjusted price lineage through labels"
```

Expected: all tests pass.

### Task 4: Fail closed before backtest return accounting

**Files:**
- Modify: `helix/eval/backtest.py:248-456`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write the backtest boundary tests**

```python
def test_backtest_rejects_raw_basis_labels(cfg):
    labels = make_labels(adjustment=AdjustmentStamp("raw", "legacy"))
    predictions, candidates, dates = one_trade_inputs(labels)
    with pytest.raises(PriceLineageError, match="run_backtest.*price_basis='raw'"):
        run_backtest(predictions, labels, candidates, dates, cfg, BacktestConfig())


def test_raw_price_cannot_change_executed_hfq_return(cfg):
    panel, labels, predictions, candidates = governed_trade_fixture()
    first = run_backtest(predictions, labels, candidates, panel.dates, cfg, BacktestConfig())
    panel.fields["close"] *= 0.5
    second = run_backtest(predictions, labels, candidates, panel.dates, cfg, BacktestConfig())
    assert second.summary["mean_trade_return_net"] == pytest.approx(
        first.summary["mean_trade_return_net"]
    )
```

Retain separate existing tests proving raw open can change limit-up fillability.

- [ ] **Step 2: Verify raw-basis labels are accepted today**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest tests/test_backtest.py -v
```

Expected: the raw-basis rejection test fails.

- [ ] **Step 3: Add label and panel gates**

Implement:

```python
def _require_adjusted_label_prices(labels: LabelSet, purpose: str) -> None:
    if labels.adjustment.price_basis != "hfq":
        raise PriceLineageError(
            f"{purpose}: label price_basis={labels.adjustment.price_basis!r}, expected 'hfq'"
        )
    if not labels.adjustment.adj_factor_version:
        raise PriceLineageError(f"{purpose}: label adj_factor_version is empty")
```

Call it at the start of `run_backtest`. For realistic exits, require `open_hfq` and `close_hfq` from the panel and assert its stamp equals `labels.adjustment`. Do not change `_net_returns`, `_cost_rates`, shortlist selection, or raw limit comparisons. Update all test factories with explicit stamps and matching panel lineage.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest tests/test_backtest.py -v
git add helix/eval/backtest.py tests/test_backtest.py
git commit -m "fix: require governed HFQ prices in backtests"
```

Expected: all tests pass.

### Task 5: Govern new event inputs and remove raw formal accounting

**Files:**
- Create: `helix/data/event_lineage.py`
- Modify: `helix/data/event_table.py:180-260`
- Modify: `helix/pipeline_events.py:25-145`
- Modify: `scripts/mine_argus.py:30-130`
- Modify: `scripts/backtest_argus.py:80-340`
- Modify: `tests/test_event_table.py`
- Modify: `tests/test_backtest_argus.py`

- [ ] **Step 1: Write failing governed-loader tests**

Add missing-manifest, missing-audit-column, and raw-basis cases to `tests/test_event_table.py`. The success fixture writes actual audit columns and this manifest:

```json
{
  "schema_version": 1,
  "fields": {
    "stock_intra_amp_d0": {
      "source_date": "stock_intra_amp_d0__source_date",
      "as_of_time": "stock_intra_amp_d0__as_of_time",
      "price_basis": "stock_intra_amp_d0__price_basis",
      "adj_factor_version": "stock_intra_amp_d0__adj_factor_version",
      "horizon": 0
    }
  }
}
```

The principal assertion is:

```python
with pytest.raises(EventLineageError, match="lineage manifest is required"):
    load_event_panel(
        path,
        label_columns=["label_d2_return_hfq"],
        feature_columns=["stock_intra_amp_d0"],
    )
```

- [ ] **Step 2: Run and observe legacy acceptance**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest tests/test_event_table.py -v
```

Expected: import/signature failure because no governed manifest exists.

- [ ] **Step 3: Implement manifest parsing and validation**

Create `helix/data/event_lineage.py`:

```python
class EventLineageError(ValueError):
    """An event field lacks required price-basis evidence."""


@dataclass(frozen=True)
class EventAuditColumns:
    source_date: str
    as_of_time: str
    price_basis: str
    adj_factor_version: str
    horizon: int


def load_event_lineage(path: Path | None) -> dict[str, EventAuditColumns]:
    if path is None:
        raise EventLineageError("event lineage manifest is required")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("fields"), dict):
        raise EventLineageError("event lineage manifest must use schema_version=1")
    return {name: EventAuditColumns(**value) for name, value in payload["fields"].items()}


def validate_event_fields(frame, manifest, fields, *, train_end=None) -> None:
    versions = set()
    decision = pd.to_datetime(frame["trade_date"], errors="raise")
    for field in fields:
        if field not in manifest:
            raise EventLineageError(f"{field}: four-column lineage entry is missing")
        audit = manifest[field]
        columns = (audit.source_date, audit.as_of_time, audit.price_basis, audit.adj_factor_version)
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise EventLineageError(f"{field}: audit column is missing: {missing}")
        basis = set(frame[audit.price_basis].dropna().astype(str))
        if basis != {"hfq"}:
            raise EventLineageError(f"{field}: price_basis values are {sorted(basis)}")
        source = pd.to_datetime(frame[audit.source_date], errors="raise")
        as_of = pd.to_datetime(frame[audit.as_of_time], errors="raise", utc=True)
        if (as_of.dt.date != source.dt.date).any():
            raise EventLineageError(f"{field}: as_of_time is not source-date local")
        if audit.horizon == 0 and not source.equals(decision):
            raise EventLineageError(f"{field}: feature source_date must equal trade_date")
        if audit.horizon > 0 and (source <= decision).any():
            raise EventLineageError(f"{field}: outcome source_date must be after trade_date")
        if train_end is not None and audit.horizon > 0:
            cutoff = pd.Timestamp(train_end)
            if (source > cutoff).any():
                raise EventLineageError(f"{field}: outcome source_date crosses training cutoff")
        field_versions = set(frame[audit.adj_factor_version].dropna().astype(str))
        if len(field_versions) != 1 or "" in field_versions:
            raise EventLineageError(f"{field}: adj_factor_version must be one non-empty value")
        versions.update(field_versions)
    if len(versions) != 1:
        raise EventLineageError("event fields use inconsistent adj_factor_version values")
```

Make `load_event_panel` accept keyword-only `lineage_path`, validate every formal feature/label, and exclude audit columns from feature discovery. Keep `build_event_panel` lineage-neutral because legacy audit replay is an explicit, in-memory comparison rather than a formal loader.

For feature fields, additionally require `source_date == trade_date` row by row and `as_of_time <= D0 15:00:00+08:00`. For D+1/D+2 outcome fields, require source dates after D0, require their declared horizon to match the manifest entry, and reject any source/as-of date later than the supplied training cutoff. These checks are parameters of the validator, not name-based guesses.

- [ ] **Step 4: Switch formal targets to HFQ names**

Use:

```python
PRIMARY_TARGET = "label_d2_peak_return_hfq"
BINARY_TARGET = "label_d2_hit_8pct_hfq"
RETURN_TARGET = "label_d2_return_hfq"
DEFAULT_LABELS = (
    PRIMARY_TARGET, BINARY_TARGET, RETURN_TARGET,
    "label_px_d1_open_hfq", "label_px_d2_high_hfq", "label_px_d2_close_hfq",
)
```

Require a lineage path in `pipeline_events.load`; replace raw return targets with `RETURN_TARGET`.

- [ ] **Step 5: Gate streaming mining and migrate script accounting**

Add required `--lineage` to both scripts. In `scripts/backtest_argus.py` define:

```python
ENTRY_HFQ = "label_px_d1_open_hfq"
HIGH_HFQ = "label_px_d2_high_hfq"
EXIT_HFQ = "label_px_d2_close_hfq"
RETURN_HFQ = "label_d2_return_hfq"
```

All hit/PnL ratios use those fields. Raw `label_px_d1_open` remains only in `REQUIRED_COLUMNS` for `unfillable_mask`. Add a test proving raw open changes cannot alter `gross_returns` when fillability is unchanged.

- [ ] **Step 6: Verify and commit**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_event_table.py tests/test_backtest_argus.py -v
git add helix/data/event_lineage.py helix/data/event_table.py helix/pipeline_events.py \
  scripts/mine_argus.py scripts/backtest_argus.py tests/test_event_table.py \
  tests/test_backtest_argus.py
git commit -m "feat: fail closed on unverified event price lineage"
```

Expected: all tests pass.

### Task 6: Generate the fixed-score `gp_000` comparison

**Files:**
- Create: `scripts/adjustment_unification_baseline.py`
- Create: `tests/test_adjustment_unification_baseline.py`
- Create: `docs/risk/adjustment_unification_fix.md`

- [ ] **Step 1: Write failing comparison and tolerance tests**

Use a four-stock aligned frame to prove both rows share the same selected-score digest. Add:

```python
table = pd.DataFrame({
    "price_basis": ["raw", "hfq"],
    "net_per_trade": [
        EXPECTED_HFQ_NET_PER_TRADE - EXPECTED_NET_DELTA,
        EXPECTED_HFQ_NET_PER_TRADE,
    ],
})
validate_expected_impact(table)
table.loc[1, "net_per_trade"] = 0.01
with pytest.raises(AssertionError, match="historical audit tolerance"):
    validate_expected_impact(table)
```

- [ ] **Step 2: Verify the module is absent**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_adjustment_unification_baseline.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement the dedicated legacy adapter**

Reuse alignment/IC/Top-K functions from `gp000_loss_attribution.py`; never call its output writer. Freeze:

```python
EXPECTED_RAW_IC = -0.0627748063907745
EXPECTED_HFQ_IC = -0.062899974234733
EXPECTED_RAW_NET_PER_TRADE = -0.005455654320765759
EXPECTED_HFQ_NET_PER_TRADE = -0.005233397934459387
EXPECTED_NET_DELTA = 0.0002222563863063718
EXPECTED_RAW_CAGR = -0.5517349330358576
EXPECTED_HFQ_CAGR = -0.5385714016648523
EXPECTED_RAW_SHARPE = -1.4420300457461805
EXPECTED_HFQ_SHARPE = -1.3882776746645582
EXPECTED_D0_DATES = 647
ABS_TOLERANCE = 1e-12
```

Implement:

```python
def compare_fixed_scores(aligned, config, *, min_ic_samples=30):
    selected = _top_k_selected_rows(aligned, config)
    digest = _hash_frame(selected, ["trade_date", "stock_code", "factor_score"])
    rows = []
    for basis, column in (("raw", "raw_return"), ("hfq", "hfq_return")):
        _, score, target, mask = event_grids(aligned, "factor_score", column)
        ic = summarize_ic(daily_ic(score, target, mask, min_samples=min_ic_samples))
        metrics, _ = evaluate_top_k_book(
            aligned.assign(gross_return=aligned[column]), config, gross=False, overlap=2
        )
        rows.append({
            "price_basis": basis, "d2_close_ic": ic["ic_mean"],
            "net_per_trade": metrics["mean_trade_return"], "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"], "final_equity": metrics["final_equity"],
            "n_days": int(metrics["n_days"]), "selected_score_digest": digest,
        })
    return pd.DataFrame(rows)
```

Assert 647 D+2-complete dates, exit no later than `2024-09-04`, identical score digests, frozen values within tolerance, and negative before/after returns. JSON output includes `legacy_unverified_lineage: true` and `historical_reports_rewritten: false`.

- [ ] **Step 4: Render and run only the new report**

The report contains contract summary, D+2 boundary, score digest, before/after IC, Top4 net/trade, CAGR, Sharpe, final equity, limitations, and links to D10 plus the old audit. Run:

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_adjustment_unification_baseline.py -v
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/python \
  scripts/adjustment_unification_baseline.py \
  --input /Users/aochong/code/helix/data/raw/argus_quant_working.parquet \
  --library /Users/aochong/code/helix/data/artifacts/argus/event_factors.json \
  --price-cache /Users/aochong/code/helix/data/raw/d2_exit_cache \
  --config configs/default.yaml \
  --report docs/risk/adjustment_unification_fix.md
```

Expected: HFQ net/trade `-0.005233397934459387`, delta `+0.0002222563863063718`, 647 dates, loss unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/adjustment_unification_baseline.py \
  tests/test_adjustment_unification_baseline.py docs/risk/adjustment_unification_fix.md
git commit -m "docs: publish adjustment unification baseline"
```

### Task 7: Close D10 and add a four-node split regression

**Files:**
- Modify: `docs/factor-governance.md:556-568`
- Create: `tests/test_adjustment_unification_contract.py`

- [ ] **Step 1: Write the full-chain split test**

Create a governed panel containing one 2-for-1 split, use zero costs and Top1, then run the real feature, label, and backtest entry points:

```python
def test_split_is_smoothed_across_features_labels_and_backtest() -> None:
    panel = split_event_panel_with_lineage()
    universe = np.ones(panel.shape, dtype=bool)
    fields = compute_base_fields(panel)
    labels = build_touch_label(panel, universe, LabelConfig())
    predictions = fixed_predictions(panel.shape)
    result = run_backtest(
        predictions, labels, universe, panel.dates,
        LabelConfig(), zero_cost_backtest(top_k=1),
    )
    assert fields["ret1"][SPLIT_DAY, SPLIT_STOCK] == pytest.approx(0.0)
    assert labels.entry_price[DECISION_DAY, SPLIT_STOCK] == pytest.approx(10.0)
    assert labels.exit_price[DECISION_DAY, SPLIT_STOCK] == pytest.approx(10.0)
    assert result.summary["mean_trade_return_net"] == pytest.approx(0.0)
    assert not labels.valid[-2:].any()
```

Add a companion test that removes one lineage record and asserts failure at the first affected node.

- [ ] **Step 2: Run the integration test**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_adjustment_unification_contract.py -v
```

Expected after explicit fixture wiring: all tests pass.

- [ ] **Step 3: Close D10 without changing historical numbers**

Keep `1,425`, `48`, `+0.022226%`, `-0.5233%`, and `-53.86%`. Replace the status text with:

```markdown
**修复完成（2026-08-14）**：新链路已建立四元血缘 fail-closed 合同，因子、标签、成交计价与收益统一使用点时 HFQ；raw 仅用于涨跌停/可成交性。旧 event/`gp_000` 保留为不可追溯 legacy 基线，不回溯背书或改写结论。固定分数复算仍为小幅修正且不逆转亏损。见[修复说明](risk/adjustment_unification_fix.md)与[专项审计](risk/gp000_loss_attribution.md)。
```

- [ ] **Step 4: Verify references and commit**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_adjustment_unification_contract.py tests/test_gp000_loss_attribution.py -v
rg -n 'D10|修复完成|legacy|fail-closed|0.022226|-0.5233' \
  docs/factor-governance.md docs/risk/adjustment_unification_fix.md
git add docs/factor-governance.md tests/test_adjustment_unification_contract.py
git commit -m "docs: close D10 adjustment mismatch"
```

Expected: tests pass and both reports link to the D10 governance entry.

### Task 8: Verify, review, push, and create the PR

**Files:**
- Review: every path returned by `git diff --name-only main...HEAD`

- [ ] **Step 1: Run complete quality gates**

```bash
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/pytest
PYTHONPATH=$PWD /Users/aochong/code/helix/.venv/bin/ruff check .
```

Expected: at least 514 tests plus the new tests pass, the existing single skip remains, zero failures, and Ruff prints `All checks passed!`.

- [ ] **Step 2: Check the raw-price boundary and diff hygiene**

```bash
rg -n 'panel\.f64\("(open|high|low|close|pre_close)"\)' helix/features helix/labels
rg -n 'label_px_d2_close.*label_px_d1_open|label_d2_return' \
  helix scripts/backtest_argus.py scripts/mine_argus.py
git diff --check main...HEAD
```

Expected: raw reads remain only in limit/fillability/execution-state code, formal event PnL uses HFQ-suffixed outcomes, and diff check is silent.

- [ ] **Step 3: Invoke completion verification and code review**

Use `verification-before-completion`, then `requesting-code-review` against `main`. Resolve every actionable Important-or-higher finding and rerun focused plus full gates.

- [ ] **Step 4: Inspect final repository state**

```bash
git status --short --branch
git log --oneline --decorate main..HEAD
git diff --stat main...HEAD
git diff -- docs/factor-governance.md docs/risk/adjustment_unification_fix.md
```

Expected: clean worktree and focused commits for lineage, features, labels/backtest, event governance, metrics, and D10.

- [ ] **Step 5: Push and create the PR**

Create `/tmp/adjustment-unification-pr.md` using `apply_patch` with this exact body:

```markdown
## Summary
- enforce four-field, fail-closed HFQ lineage across data, features, labels, and backtests
- confine raw prices to limit/fillability decisions and block unverified event inputs
- preserve legacy gp_000 scores and publish a fixed-training-window before/after comparison
- close governance gap D10 without rewriting historical audit results

## Metrics
- D+2 close IC: -0.0627748064 -> -0.0628999742
- Top4 net/trade: -0.545565% -> -0.523340% (+0.022226pp)
- CAGR: -55.1735% -> -53.8571%
- Sharpe: -1.4420 -> -1.3883
- loss conclusion unchanged

## Verification
- full pytest: passing
- ruff check .: passing
```

Then run:

```bash
git push -u origin fix/adjustment-unification
gh pr create --base main --head fix/adjustment-unification \
  --title "fix: unify point-in-time adjusted price baseline" \
  --body-file /tmp/adjustment-unification-pr.md
```

Remove only that exact temporary file after PR creation. Do not merge the PR.

- [ ] **Step 6: Report delivery evidence**

Return the branch, commit range, PR URL, exact pytest pass/skip count, Ruff result, and before/after metric table. State that the PR is open but not merged and that no historical `gp_000` report or experiment conclusion was rewritten.
