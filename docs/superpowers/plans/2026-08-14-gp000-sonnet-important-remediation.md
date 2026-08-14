# gp_000 Sonnet Important Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four Sonnet Important findings with executable adjustment-audit contracts, traceable logs, synchronized D10 governance, and unambiguous report wording without changing attribution calculations or rerunning numerical conclusions.

**Architecture:** Keep the existing evidence payload as the single source of truth. Strengthen `validate_evidence` so documented adjustment assumptions fail closed, derive structured audit-log records directly from the already-computed payload, and render/report the same frozen statistics. Restrict edits to the audit script, its tests, the design/plan, the risk report, and the governance ledger.

**Tech Stack:** Python 3.12, pandas, pytest, standard-library logging/JSON, Ruff, Markdown, Git/GitHub CLI.

---

### Task 1: Make adjustment validation promises executable

**Files:**
- Modify: `tests/test_gp000_loss_attribution.py`
- Modify: `scripts/gp000_loss_attribution.py:2310-2637`

- [ ] **Step 1: Write failing validation tests**

Add `validate_evidence` to the test imports, add `"是否核心": "否"` to the minimal root-cause contract, and add these tests:

```python
@pytest.mark.parametrize(
    "field",
    ("event_prices_match_raw", "event_returns_match_raw"),
)
def test_adjustment_consistency_must_pass_before_publish(field: str) -> None:
    evidence = _minimal_evidence()
    evidence["adjustment_audit"][field] = False

    with pytest.raises(ValueError, match="adjustment audit consistency"):
        validate_evidence(evidence)


def test_equal_factor_hit_flip_refuses_publish() -> None:
    evidence = _minimal_evidence()
    evidence["adjustment_audit"].update(
        hit_flip_count=2,
        adjustment_hit_flip_count=1,
        equal_factor_hit_flip_count=1,
    )

    with pytest.raises(ValueError, match="equal-factor hit flips"):
        validate_evidence(evidence)


@pytest.mark.parametrize("field", ("是否核心", "主导亏损"))
def test_each_root_cause_requires_both_loss_classifications(field: str) -> None:
    evidence = _minimal_evidence()
    evidence["root_causes"][0].pop(field)

    with pytest.raises(ValueError, match="full remediation contract"):
        validate_evidence(evidence)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_gp000_loss_attribution.py::test_adjustment_consistency_must_pass_before_publish \
  tests/test_gp000_loss_attribution.py::test_equal_factor_hit_flip_refuses_publish \
  tests/test_gp000_loss_attribution.py::test_each_root_cause_requires_both_loss_classifications -q
```

Expected: failures because false consistency flags and equal-factor flips are accepted, and `是否核心` is absent from `root_fields`.

- [ ] **Step 3: Add minimal fail-closed checks**

After adjustment numeric validation, add:

```python
if not (
    adjustment_audit["event_prices_match_raw"]
    and adjustment_audit["event_returns_match_raw"]
):
    raise ValueError("adjustment audit consistency checks must pass")
if adjustment_audit["event_return_rounding_error_max"] > 1e-6:
    raise ValueError("adjustment audit return rounding exceeds tolerance")
if adjustment_audit["equal_factor_hit_flip_count"] != 0:
    raise ValueError("adjustment audit equal-factor hit flips must be zero")
```

Add `"是否核心"` beside `"主导亏损"` in `root_fields`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command again. Expected: all selected tests pass.

### Task 2: Add report-to-log reverse reconciliation

**Files:**
- Modify: `tests/test_gp000_loss_attribution.py`
- Modify: `scripts/gp000_loss_attribution.py:1-25,2250-2310,2729-2772`

- [ ] **Step 1: Write failing trace-record and emission tests**

Import `logging`, `audit_trace_records`, and `emit_audit_trace`, then add:

```python
def test_audit_trace_records_cover_every_report_conclusion_input() -> None:
    evidence = _minimal_evidence()
    records = audit_trace_records(evidence)

    assert set(records) == {
        "adjustment_basis",
        "adjustment_statistics",
        "ex_right_counts",
        "ex_right_factor_diagnostics",
        "ex_right_return_errors",
        "ex_right_top4",
        "ex_right_portfolio",
        "quintiles",
        "quintile_monotonicity",
        "cost_split",
        "decay",
        "monthly",
        "style",
        "root_causes",
    }
    assert records["adjustment_statistics"] == evidence[
        "adjustment_stats"
    ].to_dict(orient="records")
    assert records["quintile_monotonicity"]["gross_spearman"] == -1.0
    assert records["root_causes"][1]["主导亏损"] == "否"


def test_emit_audit_trace_logs_each_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    evidence = _minimal_evidence()
    with caplog.at_level(logging.INFO, logger=audit_module.__name__):
        emit_audit_trace(evidence)

    for checkpoint in audit_trace_records(evidence):
        assert f"checkpoint={checkpoint}" in caplog.text
```

- [ ] **Step 2: Run trace tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_gp000_loss_attribution.py::test_audit_trace_records_cover_every_report_conclusion_input \
  tests/test_gp000_loss_attribution.py::test_emit_audit_trace_logs_each_checkpoint -q
```

Expected: collection failure because the trace functions do not exist.

- [ ] **Step 3: Implement trace extraction from the evidence payload**

Add standard logging plus project logging setup:

```python
import logging

from helix.logging_setup import setup_logging

log = logging.getLogger(__name__)
```

Add pure extraction and emission functions after `json_ready`:

```python
def audit_trace_records(evidence: dict[str, object]) -> dict[str, object]:
    def frame_records(name: str) -> list[dict[str, object]]:
        return evidence[name].to_dict(orient="records")

    return {
        "adjustment_basis": frame_records("adjustment_matrix"),
        "adjustment_statistics": frame_records("adjustment_stats"),
        "ex_right_counts": frame_records("ex_right_counts"),
        "ex_right_factor_diagnostics": frame_records(
            "ex_right_factor_diagnostics"
        ),
        "ex_right_return_errors": frame_records("ex_right_return_errors"),
        "ex_right_top4": dict(evidence["ex_right_top4_summary"]),
        "ex_right_portfolio": frame_records("ex_right_portfolio_comparison"),
        "quintiles": frame_records("quintiles"),
        "quintile_monotonicity": dict(evidence["quintile_monotonicity"]),
        "cost_split": frame_records("cost_split"),
        "decay": evidence["decay"]["summary"].to_dict(orient="records"),
        "monthly": frame_records("monthly"),
        "style": frame_records("style_table"),
        "root_causes": rank_root_causes(evidence),
    }


def emit_audit_trace(evidence: dict[str, object]) -> None:
    for checkpoint, payload in audit_trace_records(evidence).items():
        log.info(
            "audit checkpoint=%s data=%s",
            checkpoint,
            json.dumps(
                json_ready(payload),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
        )
```

In `main`, call `setup_logging()`, validate the evidence, emit the trace, and then publish outputs:

```python
setup_logging()
evidence = build_evidence(
    input_path=args.input,
    library_path=args.library,
    price_cache=args.price_cache,
    style_market_path=args.style_market,
    industries_path=args.industries,
    config_path=args.config,
)
validate_evidence(evidence)
emit_audit_trace(evidence)
write_outputs(evidence, paths)
```

- [ ] **Step 4: Run trace tests and the complete audit test module**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_gp000_loss_attribution.py -q
```

Expected: the complete module passes and logs are emitted only when requested or from CLI `main`.

### Task 3: Lock report wording and synchronize D10

**Files:**
- Modify: `tests/test_gp000_loss_attribution.py`
- Modify: `scripts/gp000_loss_attribution.py:1868-1876,2007-2123,2638-2728`
- Modify: `docs/risk/gp000_loss_attribution.md:3-7,59-80,918-927`
- Modify: `docs/factor-governance.md:549-570`

- [ ] **Step 1: Write failing report-contract tests**

Change the minimal evidence summary to contain the two explicit conclusions and add:

```python
def test_report_distinguishes_adjustment_issue_from_dominant_loss_cause() -> None:
    report = render_report(_minimal_evidence())

    assert "复权口径问题存在，但不是核心或主导亏损原因" in report
    assert "目标错配是主导亏损原因" in report
    assert "是否核心" in report
    assert "主导亏损" in report
```

Add a publication guard test that removes each conclusion phrase before calling
`write_outputs`:

```python
@pytest.mark.parametrize(
    "phrase",
    (
        "复权口径问题存在，但不是核心或主导亏损原因",
        "目标错配是主导亏损原因",
    ),
)
def test_report_conclusion_contract_refuses_publish(
    tmp_path: Path,
    phrase: str,
) -> None:
    evidence = _minimal_evidence()
    evidence["summary"] = str(evidence["summary"]).replace(phrase, "")
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    with pytest.raises(ValueError, match="required conclusions"):
        write_outputs(evidence, paths)
```

- [ ] **Step 2: Run report tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest \
  tests/test_gp000_loss_attribution.py::test_report_distinguishes_adjustment_issue_from_dominant_loss_cause \
  tests/test_gp000_loss_attribution.py::test_report_conclusion_contract_refuses_publish -q
```

Expected: report wording or publication-contract assertions fail until the renderer contract is updated.

- [ ] **Step 3: Update generated wording and publication guards**

Make the generated summary begin with two separate statements:

```python
summary = (
    "**复权口径问题存在，但不是核心或主导亏损原因。** "
    f"统一为点时后复权后，Top4 单笔净收益为 "
    f"{net_metrics['mean_trade_return']:.4%}、CAGR 为 "
    f"{net_metrics['cagr']:.2%}，仍为负；复权修正只改变单笔 "
    f"{adjustment_net_delta:.4%}。"
    "**目标错配是主导亏损原因。** "
    "正式库仍承载旧的 8% 触达优化目标，而生产验收是 D+2 收盘净收益；"
    f"对应 D+2 IC={d2_return_ic:.4f}，风格中性 CAGR="
    f"{style['style_neutral']['cagr']:.2%}，说明剥离风格后仍无可用纯 alpha。"
    "详见[治理台账 D10](../factor-governance.md)。"
)
```

Add these required report fragments alongside the heading checks in `write_outputs`:

```python
required_conclusions = (
    "复权口径问题存在，但不是核心或主导亏损原因",
    "目标错配是主导亏损原因",
    "是否核心",
    "主导亏损",
)
```

- [ ] **Step 4: Synchronize the frozen report and D10 ledger row**

Edit the checked-in report without running the numerical audit:

- Preserve every number and table row.
- Replace the execution-summary wording with the two explicit statements.
- Add a `治理台账 D10` link back to `../factor-governance.md`.

Replace D10's old estimate/status with the audited frozen values:

```markdown
| **D10** | 标签与历史 event 回测用原始价计算 D+1→D+2，和 canonical HFQ 路径错配 | 跨路径口径错配；对本窗 Top4 影响很小 | 1,425 个收益样本不同、48 个真实 hit 翻转；Top4 单笔净收益 +0.022226%，修正后仍为 -0.5233%、CAGR -53.86% | **专项审计完成（2026-08-14），工程缺陷确认、待统一入口，非核心/非主导亏损**；目标错配为主导原因。见[专项报告](risk/gp000_loss_attribution.md) |
```

- [ ] **Step 5: Run audit tests and diff checks**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest tests/test_gp000_loss_attribution.py -q
git diff --check
```

Expected: audit tests pass and there are no whitespace errors. Confirm with `git diff` that no file under `helix/` and no numerical table value changed.

### Task 4: Full verification, PR update, merge, and cleanup

**Files:**
- Verify only: entire repository

- [ ] **Step 1: Run full pytest**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest
```

Expected: zero failures.

- [ ] **Step 2: Run full Ruff**

Run:

```bash
PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/ruff check .
```

Expected: zero violations.

- [ ] **Step 3: Review scope and commit implementation**

Run `git status --short`, `git diff --stat HEAD~1`, and `git diff --check`. Confirm changed paths are limited to the five approved audit/document files plus this plan and the approved design document. Commit with:

```bash
git add scripts/gp000_loss_attribution.py tests/test_gp000_loss_attribution.py \
  docs/risk/gp000_loss_attribution.md docs/factor-governance.md \
  docs/superpowers/plans/2026-08-14-gp000-sonnet-important-remediation.md
git commit -m "fix: close gp000 audit traceability gaps"
```

- [ ] **Step 4: Push and verify PR state**

Run:

```bash
git push origin feature/gp000-loss-attribution
gh pr view 4 --json state,mergeable,reviewDecision,statusCheckRollup,url
```

Expected: PR #4 is open and mergeable with no failing checks.

- [ ] **Step 5: Merge PR #4 and update local main**

Run:

```bash
gh pr merge 4 --merge
git -C /Users/aochong/code/helix pull --ff-only origin main
```

Expected: PR #4 is merged and local `main` matches `origin/main`.

- [ ] **Step 6: Verify the merged result on main**

Run the full pytest and Ruff commands from `/Users/aochong/code/helix` again. Expected: zero failures and zero violations.

- [ ] **Step 7: Clean feature worktree and branches**

From `/Users/aochong/code/helix`, run:

```bash
git worktree remove /Users/aochong/code/helix/.worktrees/gp000-loss-attribution
git branch -d feature/gp000-loss-attribution
git push origin --delete feature/gp000-loss-attribution
git worktree prune
git status --short --branch
```

Expected: the feature worktree and feature branches are gone, while `main` is clean and aligned with `origin/main`.
