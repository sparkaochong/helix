# Placebo IC/Gini Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training-only, cross-sectional label-permutation calibration pipeline that produces empirical IC/ICIR/gini thresholds, screens the formal and supplemental factor libraries without mixing their roles, and records the result in configuration and governance documentation.

**Architecture:** `helix/eval/placebo.py` contains pure NumPy/pandas calibration and screening functions. `scripts/calibrate_placebo.py` is the only data-loading and artifact-writing layer: it predicate-filters the Parquet input at `2024-09-04`, replays saved factors with the existing factor engine, runs the formal null, and applies those thresholds to all three libraries while keeping the supplemental scopes ineligible. The existing GP, G2, and G3 modules remain untouched.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, Pydantic v2, PyYAML, pytest, Ruff.

---

## File map

- Create `tests/test_placebo_calibration.py`: all nine specification test categories, written before production code.
- Create `helix/eval/placebo.py`: binary-label validation, vectorized daily null metrics, empirical distribution, quantiles, true metrics, classification, and admission decision.
- Create `scripts/calibrate_placebo.py`: fixed-cutoff filtered loading, factor replay, formal/supplement isolation, Parquet outputs, Markdown report, and CLI.
- Modify `helix/config.py`: typed `factor_admission.placebo_threshold` configuration.
- Modify `configs/default.yaml`: exact p99 values produced by the real calibration run.
- Create `docs/risk/placebo_ic_calibration.md`: generated formal report.
- Modify `docs/factor-governance.md`: replace provisional G1 values and close D12.
- Generate ignored local artifacts `data/artifacts/placebo_ic_distribution.parquet` and `data/artifacts/placebo_factor_screening.parquet`.

### Task 1: Write the complete calibration test contract first

**Files:**
- Create: `tests/test_placebo_calibration.py`
- Reference: `docs/superpowers/specs/2026-08-13-placebo-ic-calibration-design.md`
- Reference: `helix/eval/ic.py`
- Reference: `helix/eval/metrics.py`

- [ ] **Step 1: Add all nine categories of tests before creating production modules**

Use imports inside the tests so pytest collects the file before `helix.eval.placebo` exists. Define small helpers for two-date factor/label grids, then add tests with these exact behavioral assertions:

```python
def test_cross_sectional_permutations_preserve_daily_class_counts():
    from helix.eval.placebo import permute_binary_labels

    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    out = permute_binary_labels(y, 1000, np.random.default_rng(7))
    assert out.shape == (1000, 5)
    np.testing.assert_array_equal(out.sum(axis=1), np.full(1000, 2))


def test_placebo_generation_never_moves_labels_between_dates():
    from helix.eval.placebo import iter_cross_sectional_permutations

    y = np.array([[1.0, 0.0, 0.0, np.nan], [1.0, 1.0, 0.0, 0.0]])
    mask = np.isfinite(y)
    blocks = list(iter_cross_sectional_permutations(
        y, mask, 100, np.random.default_rng(8)
    ))
    assert [block[2].shape[1] for block in blocks] == [3, 4]
    np.testing.assert_array_equal(blocks[0][2].sum(axis=1), np.ones(100))
    np.testing.assert_array_equal(blocks[1][2].sum(axis=1), np.full(100, 2))


def test_permutations_are_reproducible_by_seed():
    from helix.eval.placebo import permute_binary_labels

    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    a = permute_binary_labels(y, 50, np.random.default_rng(11))
    b = permute_binary_labels(y, 50, np.random.default_rng(11))
    c = permute_binary_labels(y, 50, np.random.default_rng(12))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
```

Add a parity test that feeds one generated permutation into the optimized daily kernel and compares it with `daily_ic` and `daily_gini` at `min_samples=1`. Add the known-quantile test:

```python
def test_quantiles_use_numpy_linear_method():
    from helix.eval.placebo import metric_quantiles

    distribution = pd.DataFrame({
        "permutation_id": np.arange(5),
        "ic_mean": np.arange(5, dtype=float),
        "icir": np.arange(5, dtype=float) * 2,
        "gini": np.arange(5, dtype=float) * 3,
    })
    out = metric_quantiles(distribution).set_index("metric")
    assert out.loc["ic_mean", "p95"] == pytest.approx(3.8)
    assert out.loc["ic_mean", "p99"] == pytest.approx(3.96)
    assert out.loc["ic_mean", "p999"] == pytest.approx(3.996)
```

Add admission tests proving strict `>` on all three metrics, a filtered-loading test whose post-cutoff extreme rows cannot affect the returned frame or derived metric, a formal/supplement isolation test proving changed supplemental metrics do not change formal quantiles and supplemental rows remain ineligible, and Pydantic validation tests for negative/non-finite metrics, invalid quantile, and reversed dates. Add forced-error tests for non-binary labels, single-class dates, empty formal factors, and non-finite final placebo statistics.

- [ ] **Step 2: Run the complete new test file and record the expected red state**

Run:

```bash
.venv/bin/pytest tests/test_placebo_calibration.py -q
```

Expected: collected tests fail because `helix.eval.placebo`, `scripts.calibrate_placebo`, and the new configuration models do not exist. Confirm failures are caused by the missing feature contract rather than syntax or fixture errors.

- [ ] **Step 3: Commit only the red tests**

```bash
git add tests/test_placebo_calibration.py
git commit -m "test: define placebo calibration contract"
```

### Task 2: Implement the pure placebo calibration core

**Files:**
- Create: `helix/eval/placebo.py`
- Test: `tests/test_placebo_calibration.py`

- [ ] **Step 1: Implement binary validation and within-date permutation**

Provide these public interfaces:

```python
QUANTILES = {"p95": 0.95, "p99": 0.99, "p999": 0.999}
METRICS = ("ic_mean", "icir", "gini")


def permute_binary_labels(
    labels: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return `(P, N)` bool labels with the exact input positive count in every row."""


def iter_cross_sectional_permutations(
    labels: np.ndarray,
    mask: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
):
    """Yield `(date_index, valid_positions, permuted_labels)` one date at a time."""
```

Validate `n_permutations > 0`, finite labels are exactly 0/1, and every included date has both classes. Generate one `(P, N)` random-key matrix per date and select exactly `k` positions per row with `np.argpartition`; never loop over permutations.

- [ ] **Step 2: Implement the vectorized rank-sum daily kernel**

Provide:

```python
def placebo_daily_metrics(
    factors: np.ndarray,
    permuted_labels: np.ndarray,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(daily_ic, daily_gini)`, both shaped `(P, K)`."""
```

Rank the `N × K` factor block once with the same `cs_rank` and `cs_rank_ordinal` operators used by `daily_ic` and `daily_gini`. Use matrix multiplication between the `(P, N)` positive mask and factor rank matrices to obtain all positive rank sums. Compute Spearman correlation as Pearson correlation between factor ranks and binary labels; this is identical to correlation with binary-label ranks because the latter is a positive affine transform. Compute AUC/gini with the same ordinal-rank formula as `daily_gini`. Apply the per-factor finite mask and return NaN for fewer than `min_samples` or a single class.

- [ ] **Step 3: Implement empirical distribution and true factor metrics**

Provide:

```python
def placebo_distribution(
    factors: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 20260813,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Return one row per permutation with max absolute formal-factor metrics."""


def factor_metrics(
    factor_names: list[str],
    factors: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Compute signed and absolute true-label IC, ICIR, and mean daily gini."""
```

`placebo_distribution` accumulates IC sums, squared sums, counts, and gini sums in `(P, K)` arrays, then takes the maximum absolute statistic across formal factors per permutation. Raise if any final value is non-finite. `factor_metrics` must call existing `daily_ic`, `summarize_ic`, `daily_gini`, and `summarize_daily` directly for every real factor.

- [ ] **Step 4: Implement quantiles, levels, screening, and admission**

Provide:

```python
def metric_quantiles(distribution: pd.DataFrame) -> pd.DataFrame:
    """Return rows `metric` and columns p95/p99/p999 using method='linear'."""


def passes_placebo_threshold(
    metrics: Mapping[str, float], thresholds: Mapping[str, float]
) -> bool:
    return all(float(metrics[name]) > float(thresholds[name]) for name in METRICS)


def screen_factor_metrics(
    metrics: pd.DataFrame,
    quantiles: pd.DataFrame,
    scope: str,
    library_path: str,
) -> pd.DataFrame:
    """Add per-metric levels, weakest overall level, eligibility, and eviction advice."""
```

Use level order `低于随机水平 < 超 p95 < 超 p99 < 超 p99.9`. For `scope == "formal"`, eligibility comes from strict p99 admission and `suggest_evict` is its inverse. For supplemental scopes, force `candidate_eligible=False` and nullable `suggest_evict`, while preserving the counterfactual levels.

- [ ] **Step 5: Run focused tests to green and commit**

```bash
.venv/bin/pytest tests/test_placebo_calibration.py -q
.venv/bin/ruff check helix/eval/placebo.py tests/test_placebo_calibration.py
git add helix/eval/placebo.py tests/test_placebo_calibration.py
git commit -m "feat: add vectorized placebo calibration core"
```

Expected: core, parity, quantile, isolation, and admission tests pass; script/config tests may remain red until their tasks.

### Task 3: Add typed admission configuration

**Files:**
- Modify: `helix/config.py`
- Modify later with generated values: `configs/default.yaml`
- Test: `tests/test_placebo_calibration.py`

- [ ] **Step 1: Implement strict models after their tests are red**

Add:

```python
class PlaceboThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ic_mean: float = Field(ge=0)
    icir: float = Field(ge=0)
    gini: float = Field(ge=0)
    quantile: float = Field(0.99, gt=0, lt=1)
    train_start: str
    train_end: str

    @field_validator("ic_mean", "icir", "gini")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("placebo thresholds must be finite")
        return value

    @model_validator(mode="after")
    def _ordered_training_dates(self):
        if not self.train_start or not self.train_end or self.train_start > self.train_end:
            raise ValueError("placebo training dates must be non-empty and ordered")
        return self


class FactorAdmissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    placebo_threshold: PlaceboThresholdConfig | None = None
```

Import NumPy and `model_validator`, and add
`factor_admission: FactorAdmissionConfig = FactorAdmissionConfig()` to `Config`. Keeping the nested threshold optional allows specialized configs such as `argus_neutral.yaml` to load before they are separately calibrated; the checked-in default config will contain the formal calibrated threshold.

- [ ] **Step 2: Run config-focused tests and commit**

```bash
.venv/bin/pytest tests/test_placebo_calibration.py -k config -q
.venv/bin/ruff check helix/config.py tests/test_placebo_calibration.py
git add helix/config.py tests/test_placebo_calibration.py
git commit -m "feat: add placebo admission configuration"
```

### Task 4: Implement the training-only CLI and artifact/report writers

**Files:**
- Create: `scripts/calibrate_placebo.py`
- Test: `tests/test_placebo_calibration.py`

- [ ] **Step 1: Implement the fixed-cutoff loader**

Define:

```python
FORMAL_TRAIN_END = "2024-09-04"
TARGET = "label_d2_hit_8pct"


def validate_train_end(train_end: str) -> str:
    if train_end != FORMAL_TRAIN_END:
        raise ValueError(f"train_end must equal formal cutoff {FORMAL_TRAIN_END}")
    return train_end


def load_training_frame(path: Path, columns: list[str], train_end: str) -> pd.DataFrame:
    validate_train_end(train_end)
    selected = ["trade_date", "stock_code", TARGET, *columns]
    frame = pd.read_parquet(
        path,
        columns=list(dict.fromkeys(selected)),
        filters=[("trade_date", "<=", train_end)],
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    if frame.empty or frame["trade_date"].max() > train_end:
        raise ValueError("training filter admitted an out-of-sample row or returned no rows")
    if train_end not in set(frame["trade_date"]):
        raise ValueError("train_end is absent from the filtered input")
    return frame
```

The test with extreme post-cutoff rows must pass here. Do not call `load_event_panel`, which loads the full file before filtering.

- [ ] **Step 2: Implement library replay and strict scope isolation**

Load the formal, n40, and multi libraries with `load_factors`, build the union of their `field_names`, then call `build_event_panel` on the already-filtered frame. Replay each library independently through `compute_factors`. Run `placebo_distribution` only on formal values. Run `factor_metrics` and `screen_factor_metrics` separately for all scopes, concatenate afterward, and assert the formal distribution metadata reports only the formal factor count.

- [ ] **Step 3: Implement artifact and Markdown output**

Write the distribution to `data/artifacts/placebo_ic_distribution.parquet` after adding `seed`, `train_start`, `train_end`, `n_train_dates`, and `formal_factor_count`. Write the combined screening table to `data/artifacts/placebo_factor_screening.parquet`. Implement `render_report(...) -> str` to produce `docs/risk/placebo_ic_calibration.md` with distribution summaries, the p95/p99/p99.9 table, formal factor rows, separate n40/multi tables and counts, and explicit training-only/G3 caveats.

Implement `write_threshold_config(path, quantiles, train_start, train_end)` as an explicit output operation. It renders an auto-generated block delimited by `# BEGIN PLACEBO CALIBRATION` and `# END PLACEBO CALIBRATION`, appends it when absent, and replaces only that block on recalibration. Populate all three numeric literals directly from the formal distribution's p99 row. This preserves the rest of the hand-written YAML and makes supplemental data structurally unable to affect configuration.

- [ ] **Step 4: Implement CLI arguments and fail-closed validation**

Support at least:

```text
--input
--formal-library
--n40-library
--multi-library
--train-end (required and exactly 2024-09-04)
--seed (default 20260813)
--permutations (default 1000)
--min-samples (default 50)
--distribution
--screening
--report
--write-config
```

Defaults point to the paths in the design. Reject missing/empty formal libraries, invalid labels, non-finite null output, and any cutoff mismatch. Print the threshold table and scope counts for audit.

- [ ] **Step 5: Run the whole new test file and lint to green, then commit**

```bash
.venv/bin/pytest tests/test_placebo_calibration.py -q
.venv/bin/ruff check helix/eval/placebo.py scripts/calibrate_placebo.py tests/test_placebo_calibration.py helix/config.py
git add scripts/calibrate_placebo.py tests/test_placebo_calibration.py
git commit -m "feat: add training-only placebo calibration CLI"
```

### Task 5: Run the real 1000-permutation calibration and fix the baseline

**Files:**
- Generate: `data/artifacts/placebo_ic_distribution.parquet`
- Generate: `data/artifacts/placebo_factor_screening.parquet`
- Generate: `docs/risk/placebo_ic_calibration.md`
- Modify: `configs/default.yaml`

- [ ] **Step 1: Run the real training-only command**

```bash
.venv/bin/python scripts/calibrate_placebo.py \
  --input data/raw/argus_quant_working.parquet \
  --formal-library data/artifacts/argus/event_factors.json \
  --n40-library data/artifacts/argus_n40/event_factors.json \
  --multi-library data/artifacts/argus_multi/event_factors.json \
  --train-end 2024-09-04 \
  --seed 20260813 \
  --permutations 1000 \
  --distribution data/artifacts/placebo_ic_distribution.parquet \
  --screening data/artifacts/placebo_factor_screening.parquet \
  --report docs/risk/placebo_ic_calibration.md \
  --write-config configs/default.yaml
```

Expected: exactly 1000 distribution rows, exactly 1 formal screening row, 12 n40 rows, and 17 multi rows; all metadata dates end at `2024-09-04`.

- [ ] **Step 2: Verify the command inserted the formal p99 values into default configuration**

The command must create one delimited `factor_admission.placebo_threshold` block. Its `ic_mean`,
`icir`, and `gini` literals come directly from
`quantiles.set_index("metric").loc[name, "p99"]`; `quantile` is `0.99`, `train_start` is
`2022-01-04`, and `train_end` is `2024-09-04`. Immediately load `Config.load()` and execute this
comparison:

```python
distribution = pd.read_parquet("data/artifacts/placebo_ic_distribution.parquet")
expected = metric_quantiles(distribution).set_index("metric")["p99"].to_dict()
configured = Config.load().factor_admission.placebo_threshold
assert configured is not None
assert configured.ic_mean == expected["ic_mean"]
assert configured.icir == expected["icir"]
assert configured.gini == expected["gini"]
```

Exact equality proves the checked-in literals came from the formal distribution rather than a
rounded report value or supplemental library.

- [ ] **Step 3: Validate generated artifact invariants and commit source/report/config**

```bash
.venv/bin/python -c 'import pandas as pd; d=pd.read_parquet("data/artifacts/placebo_ic_distribution.parquet"); s=pd.read_parquet("data/artifacts/placebo_factor_screening.parquet"); assert len(d)==1000; assert d.train_end.eq("2024-09-04").all(); assert s.scope.value_counts().to_dict()=={"argus_multi":17,"argus_n40":12,"formal":1}'
.venv/bin/pytest tests/test_placebo_calibration.py -q
git add configs/default.yaml docs/risk/placebo_ic_calibration.md
git commit -m "docs: record placebo calibration baseline"
```

The two Parquet files are intentionally ignored under `/data/`; keep them present in the workspace for delivery and report them in the final file list.

### Task 6: Close governance D12 without touching mining flow

**Files:**
- Modify: `docs/factor-governance.md`
- Reference: `docs/risk/placebo_ic_calibration.md`

- [ ] **Step 1: Update G1 with the exact generated table**

Replace provisional IC mean/ICIR/gini values with their p99 values from the generated report. State that all three must strictly exceed p99, thresholds are absolute direction-selected metrics, and the calibration uses 1000 within-date permutations on `2022-01-04` through `2024-09-04`. Link `risk/placebo_ic_calibration.md`.

- [ ] **Step 2: Mark D12 quantified and closed**

Change D12 to `已量化闭环`, record the formal library-only scope and report path, and retain the sentence that G3 remains the only admission veto. Do not edit GP, selection, deduplication, or ablation source files.

- [ ] **Step 3: Verify the documentation diff and commit**

```bash
git diff --check
rg -n "D12|p99|placebo_ic_calibration" docs/factor-governance.md docs/risk/placebo_ic_calibration.md
git add docs/factor-governance.md
git commit -m "docs: close placebo calibration governance gap"
```

### Task 7: Final requirements and regression verification

**Files:**
- Verify all modified and generated files.

- [ ] **Step 1: Prove the protected main flow was not changed**

```bash
git diff 4ceb0a4 -- helix/gp helix/pipeline.py helix/pipeline_events.py scripts/ablate_factors.py
```

Expected: no output.

- [ ] **Step 2: Run full tests and Ruff fresh**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Expected: all tests pass with exit code 0; Ruff prints `All checks passed!`.

- [ ] **Step 3: Audit outputs and configuration against the empirical distribution**

Run a read-only Python audit that recomputes p95/p99/p99.9 from the saved distribution with `method="linear"`, compares p99 to `Config.load().factor_admission.placebo_threshold`, counts screening levels by scope, verifies only the formal row can be eligible, and confirms all distribution rows have `train_end == "2024-09-04"`.

- [ ] **Step 4: Review the final diff and file inventory**

```bash
git status --short
git diff 4ceb0a4 --stat
git log --oneline 4ceb0a4..HEAD
```

Report the full verification evidence, core threshold table, formal/supplement scope counts, and every source/config/doc/generated-artifact path.
