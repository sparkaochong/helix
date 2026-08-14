#!/usr/bin/env python3
"""Read-only fixed-score comparison for the legacy ``gp_000`` adjustment repair."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helix.config import BacktestConfig
from helix.eval.ic import daily_ic, summarize_ic
from helix.gp.library import load_factors
from scripts.gp000_loss_attribution import (
    FORMAL_EXPRESSION,
    FORMAL_FACTOR,
    TRAIN_END,
    TRAIN_START,
    _hash_file,
    _hash_files,
    _hash_frame,
    _top_k_selected_rows,
    audit_adjustment_chain,
    build_price_lookup,
    evaluate_top_k_book,
    event_grids,
    json_ready,
    load_audit_config,
    load_training_events,
    load_training_market,
    training_market_paths,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/argus_quant_working.parquet"
DEFAULT_LIBRARY = ROOT / "data/artifacts/argus/event_factors.json"
DEFAULT_PRICE_CACHE = ROOT / "data/raw/d2_exit_cache"
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
DEFAULT_REPORT = ROOT / "docs/risk/adjustment_unification_fix.md"

EXPECTED_RAW_IC = -0.0627748063907745
EXPECTED_HFQ_IC = -0.062899974234733
EXPECTED_RAW_NET_PER_TRADE = -0.005455654320765759
EXPECTED_HFQ_NET_PER_TRADE = -0.005233397934459387
EXPECTED_NET_DELTA = 0.0002222563863063718
EXPECTED_RAW_CAGR = -0.5517349330358576
EXPECTED_HFQ_CAGR = -0.5385714016648523
EXPECTED_RAW_SHARPE = -1.4420300457461805
EXPECTED_HFQ_SHARPE = -1.3882776746645582
EXPECTED_RAW_FINAL_EQUITY = 0.1274470164745505
EXPECTED_HFQ_FINAL_EQUITY = 0.1372782241838809
EXPECTED_D0_DATES = 647
EXPECTED_D0_END = "2024-09-02"
EXPECTED_EXIT_END = "2024-09-04"
ABS_TOLERANCE = 1e-12


def compare_fixed_scores(
    aligned: pd.DataFrame,
    config: BacktestConfig,
    *,
    min_ic_samples: int = 30,
) -> pd.DataFrame:
    """Compare two outcome bases while freezing scores and Top-K selections."""
    required = {
        "trade_date",
        "stock_code",
        "factor_score",
        "raw_return",
        "hfq_return",
    }
    missing = required - set(aligned.columns)
    if missing:
        raise KeyError(f"fixed-score comparison is missing: {sorted(missing)}")
    if aligned.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("fixed-score comparison contains duplicate event keys")

    selected = _top_k_selected_rows(aligned, config)
    selection_digest = _hash_frame(
        selected,
        ["trade_date", "stock_code", "factor_score"],
    )
    rows: list[dict[str, object]] = []
    for basis, return_column in (("raw", "raw_return"), ("hfq", "hfq_return")):
        _, score, target, mask = event_grids(
            aligned,
            "factor_score",
            return_column,
        )
        ic = summarize_ic(
            daily_ic(score, target, mask, min_samples=min_ic_samples)
        )
        metrics, _ = evaluate_top_k_book(
            aligned.assign(gross_return=aligned[return_column]),
            config,
            gross=False,
            overlap=2,
        )
        rows.append(
            {
                "price_basis": basis,
                "d2_close_ic": ic["ic_mean"],
                "net_per_trade": metrics["mean_trade_return"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "final_equity": metrics["final_equity"],
                "n_days": int(metrics["n_days"]),
                "selected_score_digest": selection_digest,
            }
        )
    return pd.DataFrame(rows)


def validate_frozen_boundary(
    aligned: pd.DataFrame,
    *,
    expected_dates: int = EXPECTED_D0_DATES,
    expected_d0_end: str = EXPECTED_D0_END,
    expected_exit_end: str = EXPECTED_EXIT_END,
) -> dict[str, object]:
    """Fail closed if the approved D+2-complete training boundary drifts."""
    required = {"trade_date", "exit_date"}
    missing = required - set(aligned.columns)
    if missing:
        raise KeyError(f"boundary frame is missing: {sorted(missing)}")
    d0_dates = aligned["trade_date"].astype(str)
    exit_dates = aligned["exit_date"].astype(str)
    actual_dates = int(d0_dates.nunique())
    if actual_dates != expected_dates:
        raise AssertionError(
            "D+2-complete date count drifted: "
            f"expected {expected_dates}, got {actual_dates}"
        )
    actual_d0_end = str(d0_dates.max())
    if actual_d0_end != expected_d0_end:
        raise AssertionError(
            f"D0 boundary drifted: expected {expected_d0_end}, got {actual_d0_end}"
        )
    actual_exit_end = str(exit_dates.max())
    if actual_exit_end != expected_exit_end or actual_exit_end > TRAIN_END:
        raise AssertionError(
            "D+2 exit boundary drifted: "
            f"expected {expected_exit_end}, got {actual_exit_end}"
        )
    return {
        "d2_complete_dates": actual_dates,
        "d0_end": actual_d0_end,
        "d2_exit_end": actual_exit_end,
    }


def validate_expected_impact(
    comparison: pd.DataFrame,
    *,
    absolute_tolerance: float = ABS_TOLERANCE,
) -> None:
    """Reject results that differ from the independently audited frozen values."""
    required = {
        "price_basis",
        "d2_close_ic",
        "net_per_trade",
        "cagr",
        "sharpe",
        "final_equity",
        "n_days",
        "selected_score_digest",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise AssertionError(
            f"historical audit tolerance cannot be checked; missing {sorted(missing)}"
        )
    if comparison["price_basis"].tolist() != ["raw", "hfq"]:
        raise AssertionError(
            "historical audit tolerance requires one raw row followed by one hfq row"
        )
    if comparison["selected_score_digest"].nunique() != 1:
        raise AssertionError(
            "historical audit tolerance failed: score selection digest changed by basis"
        )

    expected = {
        ("raw", "d2_close_ic"): EXPECTED_RAW_IC,
        ("hfq", "d2_close_ic"): EXPECTED_HFQ_IC,
        ("raw", "net_per_trade"): EXPECTED_RAW_NET_PER_TRADE,
        ("hfq", "net_per_trade"): EXPECTED_HFQ_NET_PER_TRADE,
        ("raw", "cagr"): EXPECTED_RAW_CAGR,
        ("hfq", "cagr"): EXPECTED_HFQ_CAGR,
        ("raw", "sharpe"): EXPECTED_RAW_SHARPE,
        ("hfq", "sharpe"): EXPECTED_HFQ_SHARPE,
        ("raw", "final_equity"): EXPECTED_RAW_FINAL_EQUITY,
        ("hfq", "final_equity"): EXPECTED_HFQ_FINAL_EQUITY,
    }
    indexed = comparison.set_index("price_basis")
    for (basis, metric), reference in expected.items():
        actual = float(indexed.loc[basis, metric])
        if not np.isclose(
            actual,
            reference,
            rtol=0.0,
            atol=absolute_tolerance,
        ):
            raise AssertionError(
                "historical audit tolerance failed for "
                f"{basis}.{metric}: expected {reference!r}, got {actual!r}"
            )
    if not (indexed["n_days"] == EXPECTED_D0_DATES).all():
        raise AssertionError("historical audit tolerance failed for D+2 date count")
    delta = float(
        indexed.loc["hfq", "net_per_trade"]
        - indexed.loc["raw", "net_per_trade"]
    )
    if not np.isclose(delta, EXPECTED_NET_DELTA, rtol=0.0, atol=absolute_tolerance):
        raise AssertionError(
            "historical audit tolerance failed for net-per-trade delta: "
            f"expected {EXPECTED_NET_DELTA!r}, got {delta!r}"
        )
    if not (indexed["net_per_trade"] < 0).all():
        raise AssertionError(
            "historical audit tolerance failed: adjustment unexpectedly reversed loss"
        )


def _fingerprinted_input(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _hash_file(resolved)}


def collect_input_fingerprints(
    *,
    input_path: Path,
    library_path: Path,
    price_cache: Path,
    config_path: Path,
) -> dict[str, dict[str, object]]:
    """Fingerprint every read-only input that determines the comparison."""
    market_paths, _ = training_market_paths(price_cache)
    resolved_cache = price_cache.resolve(strict=True)
    return {
        "event_table": _fingerprinted_input(input_path),
        "factor_library": _fingerprinted_input(library_path),
        "backtest_config": _fingerprinted_input(config_path),
        "price_cache": {
            "path": str(resolved_cache),
            "sha256": _hash_files(market_paths),
            "file_count": len(market_paths),
        },
        "comparison_script": _fingerprinted_input(Path(__file__)),
    }


def build_structured_output(
    comparison: pd.DataFrame,
    *,
    boundary: dict[str, object],
    inputs: dict[str, dict[str, object]],
) -> dict[str, Any]:
    """Build strict JSON evidence without writing any artifact."""
    indexed = comparison.set_index("price_basis")
    raw_net = float(indexed.loc["raw", "net_per_trade"])
    hfq_net = float(indexed.loc["hfq", "net_per_trade"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "fixed-score legacy-before versus governed-HFQ-outcome after",
        "formal_factor": {
            "name": FORMAL_FACTOR,
            "expression": FORMAL_EXPRESSION,
            "retrained": False,
            "score_basis_changed_between_arms": False,
        },
        "training_window": {
            "nominal_start": TRAIN_START,
            "nominal_end": TRAIN_END,
            **boundary,
        },
        "inputs": inputs,
        "legacy_unverified_lineage": True,
        "historical_reports_rewritten": False,
        "loss_conclusion_unchanged": bool(raw_net < 0 and hfq_net < 0),
        "target_mismatch_remains_dominant": True,
        "adjustment_mismatch_is_core_loss_cause": False,
        "net_per_trade_delta": hfq_net - raw_net,
        "comparison": json_ready(comparison),
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def build_baseline_evidence(
    *,
    input_path: Path,
    library_path: Path,
    price_cache: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Run the read-only legacy adapter and return validated structured evidence."""
    config = load_audit_config(config_path)
    library = load_factors(library_path)
    events = load_training_events(input_path, library)
    market, calendar = load_training_market(
        price_cache,
        set(events["stock_code"].astype(str)),
    )
    prices = build_price_lookup(market, calendar, events["stock_code"].unique())
    _, aligned = audit_adjustment_chain(events, prices)
    boundary = validate_frozen_boundary(aligned)
    comparison = compare_fixed_scores(aligned, config)
    validate_expected_impact(comparison)
    inputs = collect_input_fingerprints(
        input_path=input_path,
        library_path=library_path,
        price_cache=price_cache,
        config_path=config_path,
    )
    return build_structured_output(comparison, boundary=boundary, inputs=inputs)


def _metric_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparison = payload["comparison"]
    return {str(row["price_basis"]): row for row in comparison}


def render_report(payload: dict[str, Any]) -> str:
    """Render the validated fixed-score evidence as the new repair document."""
    rows = _metric_rows(payload)
    raw = rows["raw"]
    hfq = rows["hfq"]
    boundary = payload["training_window"]
    inputs = payload["inputs"]
    digest = str(raw["selected_score_digest"])

    def pct(value: float) -> str:
        return f"{value:.6%}"

    def input_row(name: str, label: str) -> str:
        item = inputs.get(name)
        if item is None:
            return f"| {label} | 测试内存输入 | 不适用 |"
        return f"| {label} | `{item['path']}` | `{item['sha256']}` |"

    report = f"""# 后复权基线统一修复说明

**生成日期：** 2026-08-14

**生成入口：** `scripts/adjustment_unification_baseline.py`

**证据性质：** 固定 `gp_000` 分数与选股的只读 before/after 对照；不重训、不重新挖掘、不覆盖历史产物。

## 执行摘要

**复权口径问题存在，但不是核心或主导亏损原因。** 同一组冻结的 `gp_000` 分数和 Top4 选择从 legacy raw outcome 切换为同日点时 HFQ outcome 后，单笔净收益仅从 {pct(float(raw['net_per_trade']))} 改善至 {pct(float(hfq['net_per_trade']))}，变化 {pct(float(payload['net_per_trade_delta']))}，收益仍为负。

**目标错配是主导亏损原因。** `gp_000` 的历史准入目标与 D+2 收盘净收益目标错配；本次工作只建立可审计的合规复权基线，不修复老因子的盈利能力。后续新一代 GP 因子必须从带四元血缘的 HFQ 新链路生成。

现有 `gp_000` 成品特征与 event 表仅作为“修复前 legacy 基线”保留。其上游价格口径和 `source_date/as_of_time/price_basis/adj_factor_version` 不可追溯，本报告不将其补写为已验证 raw 或 HFQ，也不改写任何既有实验结论。详见[治理台账 D10](../factor-governance.md)与[既有专项审计](gp000_loss_attribution.md)。

## 修复合同

- 新链路中，因子计算、标签生成、成交计价与收益核算只接受带四元血缘的点时 HFQ 价格，校验失败即终止。
- 原始 OHLC 只用于涨跌停判断和可成交性校验，不参与因子值、标签值或持仓收益核算。
- 本对照是 legacy baseline adapter：只读取既有 event 特征、正式因子库和行情缓存，并复用专项审计的纯计算函数；不调用历史报告写入器。
- 两个对照臂共享同一分数、同一 Top4 选择和同一成本/滑点配置，仅 outcome 价格口径不同。

## 固定窗口与 D+2 边界

| 项目 | 值 |
| --- | ---: |
| 名义训练窗 | {boundary['nominal_start']} 至 {boundary['nominal_end']} |
| D+2 完整 D0 数 | {boundary['d2_complete_dates']} |
| 最后 D0 | {boundary['d0_end']} |
| 最后 D+2 退出日 | {boundary['d2_exit_end']} |
| Top4 冻结选择摘要 | `{digest}` |

最后两个没有完整 D+2 outcome 的 D0 被严格排除；任何 D0 数量、最后 D0 或退出日变化都会触发 fail-closed，不会自动更新基线。

## gp_000 修复前后核心指标

| 指标 | 修复前：legacy raw outcome | 修复后：点时 HFQ outcome | 变化 |
| --- | ---: | ---: | ---: |
| D+2 close IC | {float(raw['d2_close_ic']):.10f} | {float(hfq['d2_close_ic']):.10f} | {float(hfq['d2_close_ic']) - float(raw['d2_close_ic']):+.10f} |
| Top4 单笔净收益 | {pct(float(raw['net_per_trade']))} | {pct(float(hfq['net_per_trade']))} | {pct(float(hfq['net_per_trade']) - float(raw['net_per_trade']))} |
| 年化 Sharpe | {float(raw['sharpe']):.6f} | {float(hfq['sharpe']):.6f} | {float(hfq['sharpe']) - float(raw['sharpe']):+.6f} |
| CAGR（补充） | {pct(float(raw['cagr']))} | {pct(float(hfq['cagr']))} | {pct(float(hfq['cagr']) - float(raw['cagr']))} |
| 期末净值（补充） | {float(raw['final_equity']):.6f} | {float(hfq['final_equity']):.6f} | {float(hfq['final_equity']) - float(raw['final_equity']):+.6f} |

该变化与既有专项审计预估一致：Top4 单笔净收益改善约 `0.022226` 个百分点，修复后约 `-0.5233%`，未逆转亏损结论。

## 输入追溯

| 输入 | 绝对路径 | SHA-256/集合摘要 |
| --- | --- | --- |
{input_row('event_table', 'legacy event 表')}
{input_row('factor_library', '正式 gp_000 因子库')}
{input_row('price_cache', 'D+2 行情缓存')}
{input_row('backtest_config', '成本与 Top4 配置')}
{input_row('comparison_script', '本报告生成脚本')}

CLI 标准输出同时提供严格 JSON，其中 `legacy_unverified_lineage=true`、`historical_reports_rewritten=false`、`loss_conclusion_unchanged=true`。这些标志防止把 outcome 修正误解为对 legacy 特征血缘或盈利能力的认证。

## 限制与后续基线

- “修复前 raw”只描述 event outcome 与历史行情缓存的重建关系，不代表 legacy 特征的上游价格口径已获认证。
- 本次没有重新训练、调参或改变正式因子方向；分数由冻结的正式表达式和既有成品特征确定，两个对照臂不重新选股。
- 本报告不替代、不覆盖 `docs/risk/gp000_loss_attribution.md`，也不修改历史 artifacts。
- 新一代 GP 因子不得通过 legacy adapter 进入正式训练；必须使用新的 HFQ 血缘强契约链路。
"""
    return report


def publish_report(report: str, path: Path) -> None:
    """Atomically publish only the new report and protect the historical report."""
    target = path.resolve()
    if target.name == "gp000_loss_attribution.md":
        raise ValueError("refusing to overwrite the historical gp_000 report")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(report)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--price-cache", type=Path, default=DEFAULT_PRICE_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_baseline_evidence(
        input_path=args.input,
        library_path=args.library,
        price_cache=args.price_cache,
        config_path=args.config,
    )
    publish_report(render_report(payload), args.report)
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
