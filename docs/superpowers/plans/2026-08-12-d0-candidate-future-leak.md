# D0 Candidate Future-Leak Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every D0 ranking pool uses only the point-in-time universe while future tradability is consulted only after selection during backtest execution validation.

**Architecture:** Preserve `LabelSet.valid` and label NaN semantics for supervised learning, expose the already-computed D+2 `touch_tradable` mask, and pass the D0 universe explicitly into selection code. Ranking happens before any label/future mask; selected trades are then validated without backfilling from deeper ranks.

OOS prediction generation must use a separate D0 `prediction_mask`; using label validity for the test index leaks the same future facts through prediction NaNs. Failed executions retain their fixed shortlist slots as cash, including zero-execution dates.

**Tech Stack:** Python 3, NumPy, pandas, pytest

---

### Task 1: Pin label and backtest mask separation

**Files:**
- Modify: `tests/test_labels.py`
- Modify: `tests/test_backtest.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Write failing label and selection tests**

Add assertions that `build_touch_label` returns `touch_tradable=False` for a D+2 suspension while retaining the existing `valid=False`/`y=NaN` behavior. Add a backtest regression where the highest-scoring D0 candidate is not D+2-tradable and assert that the lower-ranked stock is not substituted. Add a precision-at-k regression where the highest-scoring candidate has an unobservable label and assert that the day is unobservable instead of selecting a lower-ranked name.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_labels.py tests/test_backtest.py tests/test_metrics.py
```

Expected: failures because `LabelSet.touch_tradable` and the explicit D0 candidate-mask API do not yet exist, and because `precision_at_k` currently removes non-finite outcomes before ranking.

### Task 2: Separate D0 selection from future execution data

**Files:**
- Modify: `helix/labels/touch_label.py`
- Modify: `helix/eval/backtest.py`
- Modify: `helix/eval/metrics.py`

- [ ] **Step 1: Expose D+2 tradability without changing labels**

Add `touch_tradable: np.ndarray` to `LabelSet` and populate it from the existing local mask. Do not change `valid`, `y`, or price masking.

- [ ] **Step 2: Rank the D0 pool before execution validation**

Require a `candidate_mask` in `run_backtest`. Build scores from `candidate_mask & np.isfinite(predictions)`, select the fixed Top-K shortlist, then use `labels.valid`, `labels.touch_tradable`, and finite realized returns only on that shortlist. Do not replace rejected selections with deeper-ranked stocks.

- [ ] **Step 3: Make Top-K metrics follow the same order**

In `precision_at_k`, rank `mask & np.isfinite(score)` first. If a selected label is not finite, leave that date's precision undefined; compute the base rate only from observable labels after ranking.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest -q tests/test_labels.py tests/test_backtest.py tests/test_metrics.py
```

Expected: all selected tests pass.

### Task 3: Rewire the pipeline to D0 masks

**Files:**
- Modify: `helix/pipeline.py`
- Modify: `tests/test_pipeline_smoke.py` or add a focused pipeline test

- [ ] **Step 1: Write a failing pipeline wiring test**

Assert that GP liquidity-column sampling receives `Prepared.universe`, not `labels.valid`, and that backtest/Top-K calls receive the D0 universe.

- [ ] **Step 2: Verify RED**

Run the focused pipeline test and confirm it observes the current future-dependent masks.

- [ ] **Step 3: Apply the minimal wiring changes**

Use `prepared.universe[rows]` for liquidity-column sampling, pass `prepared.universe` into `run_backtest`, and pass `prepared.universe & tested` into `lift_at_k`. Keep `labels.valid` for label fitness, factor evaluation, and supervised training because those uses mean “target observable,” not “D0 selectable.”

- [ ] **Step 4: Verify GREEN**

Run the focused pipeline test and the pipeline smoke test.

### Task 4: Update contracts and verify the repository

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/factor-governance.md`

- [ ] **Step 1: Update the documented invariant**

Record that D0 candidate ranking uses `universe`, future tradability is applied only after Top-K selection, and the D2 leakage gap is closed without changing label generation.

- [ ] **Step 2: Audit forbidden data flow**

Run:

```bash
rg -n "touch_tradable|labels\.valid" helix tests docs
```

Expected: production reads of `touch_tradable` occur only in label construction and backtest execution validation; no candidate ranking or factor-population construction uses `labels.valid`.

- [ ] **Step 3: Run full verification**

Run:

```bash
pytest -q
```

Expected: the full suite passes with no failures.
