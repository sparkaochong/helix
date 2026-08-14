#!/usr/bin/env python3
"""Measure formal training performance with a seeded moving-block bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helix.config import PROJECT_ROOT, BacktestConfig, Config
from helix.data.event_table import build_event_panel
from helix.eval.backtest import BacktestResult, run_backtest
from helix.eval.bootstrap import (
    PERFORMANCE_METRICS,
    bootstrap_performance_metrics,
    circular_block_bootstrap_indices,
    summarize_bootstrap_distribution,
    validate_bootstrap_seeds,
)
from helix.gp.library import FactorLibrary, FactorSpec, compute_factors, load_factors
from helix.labels.touch_label import build_touch_label
from helix.splits import complete_outcome_window
from scripts.d2_limit_down_bias import (
    build_exit_panel,
    load_open_dates,
    load_or_fetch_market,
)

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
DECISION_END = "2024-09-02"
FORMAL_FACTOR = "gp_000"
DEFAULT_SEEDS = (7, 13, 42, 101, 211, 307, 419, 523, 631, 743)
DEFAULT_BLOCK_LENGTH = 20
EXPECTED_DECISION_DATES = 647

DEFAULT_INPUT = PROJECT_ROOT / "data/raw/argus_quant_working.parquet"
DEFAULT_LIBRARY = PROJECT_ROOT / "data/artifacts/argus/event_factors.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_CALENDAR_CACHE = (
    PROJECT_ROOT / "data/raw/suspension_cache/calendar_20211006_20261022.parquet"
)
DEFAULT_MARKET_CACHE = PROJECT_ROOT / "data/raw/d2_exit_cache"
DEFAULT_RESULT = PROJECT_ROOT / "data/artifacts/performance_ci_bootstrap.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs/risk/performance_ci_bootstrap.md"

KEEP_DOWNGRADE = "KEEP_DOWNGRADE"
LIFT_DOWNGRADE = "LIFT_DOWNGRADE"


def _digits(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) != 8:
        raise ValueError(f"invalid trade date {value!r}")
    return digits


def _hyphenated(value: object) -> str:
    digits = _digits(value)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def complete_training_decision_dates(
    calendar: Sequence[object],
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    horizon: int = 2,
) -> np.ndarray:
    """Return D0 sessions whose full forward horizon stays inside training."""
    dates = np.unique(np.asarray([_digits(value) for value in calendar]))
    start = _digits(train_start)
    end = _digits(train_end)
    training = dates[(dates >= start) & (dates <= end)]
    if training.size == 0 or training[0] != start or training[-1] != end:
        raise ValueError("market calendar does not cover the exact training bounds")
    rows = complete_outcome_window(slice(0, len(training)), horizon)
    return training[rows]


def realistic_backtest_config(config: Config) -> BacktestConfig:
    """Enable observed deferred exits on an otherwise unchanged production config."""
    production = config.backtest
    if production.top_k != 4 or production.exit_rule != "close":
        raise ValueError("performance CI requires production Top4 with close exit")
    return production.model_copy(update={"enable_realistic_exit": True})


def validate_formal_library(library: FactorLibrary) -> FactorSpec:
    """Require the exact production-oriented formal event factor."""
    if library.kind != "event" or len(library.factors) != 1:
        raise ValueError("formal library must contain exactly one event factor gp_000")
    factor = library.factors[0]
    if factor.name != FORMAL_FACTOR or factor.sign != 1.0:
        raise ValueError("formal factor must be production-oriented gp_000 with sign +1")
    return factor


def align_event_scores(
    event_dates: np.ndarray,
    event_codes: np.ndarray,
    occupied: np.ndarray,
    scores: np.ndarray,
    market_dates: np.ndarray,
    market_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map packed event slots into a fixed market `(date, stock)` panel."""
    event_date_values = np.asarray([_digits(value) for value in event_dates])
    event_code_values = np.asarray(event_codes).astype(str)
    occupied_values = np.asarray(occupied, dtype=bool)
    score_values = np.asarray(scores, dtype=np.float64)
    if (
        event_code_values.shape != occupied_values.shape
        or score_values.shape != occupied_values.shape
        or occupied_values.ndim != 2
        or len(event_date_values) != occupied_values.shape[0]
    ):
        raise ValueError("event dates, codes, occupancy, and scores must align")
    fixed_dates = np.asarray([_digits(value) for value in market_dates])
    fixed_codes = np.asarray(market_codes).astype(str)
    date_positions = pd.Index(fixed_dates).get_indexer(event_date_values)
    event_rows, event_slots = np.nonzero(occupied_values)
    code_positions = pd.Index(fixed_codes).get_indexer(
        event_code_values[event_rows, event_slots]
    )
    if np.any(date_positions[event_rows] < 0) or np.any(code_positions < 0):
        raise ValueError("event dates or codes are absent from the market panel")
    aligned_scores = np.full(
        (len(fixed_dates), len(fixed_codes)), np.nan, dtype=np.float64
    )
    candidates = np.zeros_like(aligned_scores, dtype=bool)
    market_rows = date_positions[event_rows]
    aligned_scores[market_rows, code_positions] = score_values[event_rows, event_slots]
    candidates[market_rows, code_positions] = True
    return aligned_scores, candidates


def aggregate_performance_ci(
    result: BacktestResult, indices: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Combine canonical deterministic metrics with whole-date bootstrap inference."""
    required_daily = {"date", "portfolio_return"}
    required_trades = {"d0_date", "realistic_net_return", "unresolved_at_end"}
    if not required_daily <= set(result.daily):
        raise ValueError("backtest daily output is missing performance fields")
    if not required_trades <= set(result.trades):
        raise ValueError("backtest trade output is missing realistic-return fields")
    daily = result.daily.loc[:, ["date", "portfolio_return"]].copy()
    daily["date"] = daily["date"].map(_digits)
    if daily["date"].duplicated().any():
        raise ValueError("backtest daily output contains duplicate dates")
    returns = daily["portfolio_return"].to_numpy(dtype=np.float64)

    trades = result.trades.copy()
    trades["d0_date"] = trades["d0_date"].map(_digits)
    resolved = trades[
        ~trades["unresolved_at_end"].astype(bool)
        & np.isfinite(trades["realistic_net_return"])
    ]
    grouped = resolved.groupby("d0_date")["realistic_net_return"].agg(["sum", "count"])
    trade_sum = daily["date"].map(grouped["sum"]).fillna(0.0).to_numpy(dtype=np.float64)
    trade_count = (
        daily["date"].map(grouped["count"]).fillna(0.0).to_numpy(dtype=np.float64)
    )
    canonical_trade_mean = (
        float(trade_sum.sum() / trade_count.sum())
        if trade_count.sum() > 0
        else float("nan")
    )
    reported_trade_mean = float(result.summary["mean_trade_return_net"])
    if not np.isclose(canonical_trade_mean, reported_trade_mean, rtol=0.0, atol=1e-15):
        raise AssertionError("trade return aggregation drifted from canonical backtest")

    deterministic = {
        "cagr": float(result.summary["cagr"]),
        "sharpe": float(result.summary["sharpe"]),
        "day_win_rate": float(result.summary["day_win_rate"]),
        "mean_trade_return_net": reported_trade_mean,
    }
    bootstrap = summarize_bootstrap_distribution(
        bootstrap_performance_metrics(returns, trade_sum, trade_count, indices)
    )
    return {"deterministic": deterministic, "bootstrap": bootstrap}


def performance_ci_decision(sharpe_ci_low: float) -> str:
    """Lift the mandatory downgrade only for a finite, strictly positive lower bound."""
    return (
        LIFT_DOWNGRADE
        if np.isfinite(sharpe_ci_low) and sharpe_ci_low > 0
        else KEEP_DOWNGRADE
    )


def _format_metric(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if metric == "sharpe":
        return f"{value:.6f}"
    return f"{value:.4%}"


def render_report(payload: dict[str, object]) -> str:
    """Render the complete measured audit and its cache-only reproduction contract."""
    metadata = payload["metadata"]
    config = payload["config"]
    metrics = payload["metrics"]
    execution = payload["execution"]
    decision = str(payload["decision"])
    assert isinstance(metadata, dict)
    assert isinstance(config, dict)
    assert isinstance(metrics, dict)
    assert isinstance(execution, dict)
    labels = {
        "cagr": "CAGR",
        "sharpe": "年化夏普",
        "day_win_rate": "日胜率",
        "mean_trade_return_net": "单笔净收益均值",
    }
    rows = []
    for metric in PERFORMANCE_METRICS:
        summary = metrics[metric]
        assert isinstance(summary, dict)
        interval = (
            f"[{_format_metric(metric, float(summary['ci_low']))}, "
            f"{_format_metric(metric, float(summary['ci_high']))}]"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    labels[metric],
                    _format_metric(metric, float(summary["deterministic"])),
                    _format_metric(metric, float(summary["mean"])),
                    _format_metric(metric, float(summary["std"])),
                    interval,
                ]
            )
            + " |"
        )
    seed_values = ",".join(str(seed) for seed in metadata["seeds"])
    seed_rows = []
    for position, seed in enumerate(metadata["seeds"]):
        values = [str(seed)]
        for metric in PERFORMANCE_METRICS:
            summary = metrics[metric]
            assert isinstance(summary, dict)
            values.append(_format_metric(metric, float(summary["values"][position])))
        seed_rows.append("| " + " | ".join(values) + " |")
    sharpe = metrics["sharpe"]
    assert isinstance(sharpe, dict)
    lower = float(sharpe["ci_low"])
    conclusion = (
        "夏普 95% CI 下沿严格大于 0，解除强制降级。"
        if decision == LIFT_DOWNGRADE
        else "夏普 95% CI 下沿未严格大于 0，维持强制降级。"
    )
    return f"""# 正式策略训练集绩效置信区间实验

## 结论

夏普 95% CI 下沿为 **{lower:.6f}**；判定：**{decision}**。{conclusion}

该结论只覆盖训练集采样不确定性，不构成样本外盈利、冲击成本或容量验证。

## 实验口径

- 训练窗口：`{metadata['train_start']}` 至 `{metadata['train_end']}`；D+2 完整 D0
  截止 `{metadata['decision_end']}`，共 `{metadata['n_dates']}` 日。
- 正式因子：`gp_000`，生产方向；Top`{config['top_k']}` 等权，不补位。
- 退出：`{config['exit_rule']}`，真实跌停/停牌递延：
  `{config['enable_realistic_exit']}`。
- 成本（bp）：佣金 `{config['commission_bps']}`，过户费
  `{config['transfer_bps']}`，印花税 `{config['stamp_sell_bps_before_cut']}` →
  `{config['stamp_sell_bps']}`，单边滑点 `{config['slippage_bps']}`。
- circular moving block bootstrap：块长 `{metadata['block_length']}`，种子
  `{metadata['seeds']}`；完整交易日截面为最小重采样单位。

## 核心指标

| 指标 | 确定性全样本 | 10 种子均值 | 样本标准差 | 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 每种子结果

| seed | CAGR | 年化夏普 | 日胜率 | 单笔净收益均值 |
|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

## 成交审计

- 交易数：`{execution['n_trades']}`；成交率：
  `{float(execution['fill_rate']):.4%}`。
- D+2 跌停占比：`{float(execution['limit_down_exit_share']):.4%}`；训练边界未解析：
  `{execution['unresolved_at_end']}` 笔。

## 复现

```bash
.venv/bin/python -m scripts.performance_ci_bootstrap --cache-only \\
  --seeds {seed_values} \\
  --block-length {metadata['block_length']}
```

固定输入、缓存和种子时，报告与 JSON 结果必须逐字节一致。
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _load_formal_event_scores(
    input_path: Path,
    library_path: Path,
    decision_dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    library = load_factors(library_path)
    validate_formal_library(library)
    columns = ["trade_date", "stock_code", *library.field_names]
    frame = pd.read_parquet(
        input_path,
        columns=list(dict.fromkeys(columns)),
        filters=[
            ("trade_date", ">=", TRAIN_START),
            ("trade_date", "<=", DECISION_END),
        ],
    )
    frame["trade_date"] = frame["trade_date"].map(_hyphenated)
    wanted = set(decision_dates.tolist())
    frame = frame[frame["trade_date"].map(_digits).isin(wanted)].copy()
    actual_dates = np.unique(frame["trade_date"].map(_digits))
    if not np.array_equal(actual_dates, decision_dates):
        raise ValueError("event source does not match the complete D+2 decision calendar")
    panel = build_event_panel(frame, library.field_names, [])
    names, factor_values = compute_factors(library, panel.fields)
    if names != [FORMAL_FACTOR]:
        raise AssertionError("formal factor replay did not return exactly gp_000")
    return panel.dates, panel.codes, panel.occupied, factor_values[..., 0]


def _missing_market_cache_dates(cache_dir: Path, dates: Sequence[str]) -> list[str]:
    return [str(date) for date in dates if not (cache_dir / f"{date}.parquet").exists()]


def run_experiment(
    *,
    input_path: Path = DEFAULT_INPUT,
    library_path: Path = DEFAULT_LIBRARY,
    config_path: Path = DEFAULT_CONFIG,
    calendar_cache: Path = DEFAULT_CALENDAR_CACHE,
    market_cache: Path = DEFAULT_MARKET_CACHE,
    result_path: Path = DEFAULT_RESULT,
    report_path: Path = DEFAULT_REPORT,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    cache_only: bool = False,
    refresh_market: bool = False,
    download_workers: int = 8,
) -> dict[str, object]:
    """Run and persist the formal training-window performance CI experiment."""
    seed_values = validate_bootstrap_seeds(seeds, minimum=10)
    if block_length <= 0 or download_workers <= 0:
        raise ValueError("block length and download workers must be positive")
    if cache_only and refresh_market:
        raise ValueError("cache-only and refresh-market cannot be combined")
    config = Config.load(config_path)
    backtest_config = realistic_backtest_config(config)
    calendar = load_open_dates(calendar_cache, _digits(TRAIN_START), _digits(TRAIN_END))
    decision_dates = complete_training_decision_dates(calendar)
    if (
        len(decision_dates) != EXPECTED_DECISION_DATES
        or decision_dates[-1] != _digits(DECISION_END)
    ):
        raise ValueError("formal D+2 training decision calendar must contain 647 dates")
    missing_cache = _missing_market_cache_dates(market_cache, calendar)
    if cache_only and missing_cache:
        raise FileNotFoundError(
            f"cache-only run is missing {len(missing_cache)} market dates; "
            f"first missing date is {missing_cache[0]}"
        )
    market = load_or_fetch_market(
        calendar,
        market_cache,
        config,
        refresh=refresh_market,
        max_workers=download_workers,
    )
    event_dates, event_codes, occupied, event_scores = _load_formal_event_scores(
        input_path, library_path, decision_dates
    )
    selected_codes = sorted(set(event_codes[occupied].astype(str)))
    market = market[market["ts_code"].astype(str).isin(selected_codes)].copy()
    market_panel = build_exit_panel(market, calendar, selected_codes)
    scores, candidates = align_event_scores(
        event_dates,
        event_codes,
        occupied,
        event_scores,
        market_panel.dates,
        market_panel.codes,
    )
    labels = build_touch_label(market_panel, candidates, config.label)
    result = run_backtest(
        scores,
        labels,
        candidates,
        market_panel.dates,
        config.label,
        backtest_config,
        panel=market_panel,
    )
    actual_dates = result.daily["date"].map(_digits).to_numpy()
    if not np.array_equal(actual_dates, decision_dates):
        raise AssertionError("canonical backtest did not preserve all complete D0 dates")
    indices = circular_block_bootstrap_indices(
        len(decision_dates), block_length, seed_values
    )
    aggregated = aggregate_performance_ci(result, indices)
    metrics = {
        metric: {
            "deterministic": aggregated["deterministic"][metric],
            **aggregated["bootstrap"][metric],
        }
        for metric in PERFORMANCE_METRICS
    }
    sharpe_lower = float(metrics["sharpe"]["ci_low"])
    payload: dict[str, object] = {
        "metadata": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "decision_end": DECISION_END,
            "n_dates": len(decision_dates),
            "block_length": block_length,
            "seeds": list(seed_values),
            "input_sha256": _sha256(input_path),
            "library_sha256": _sha256(library_path),
            "config_sha256": _sha256(config_path),
        },
        "config": backtest_config.model_dump(),
        "metrics": metrics,
        "execution": {
            "n_trades": len(result.trades),
            "fill_rate": float(result.summary["fill_rate"]),
            "limit_down_exit_share": float(result.summary["limit_down_exit_share"]),
            "d2_suspension_count": int(result.summary["d2_suspension_count"]),
            "unresolved_at_end": int(result.summary["unresolved_at_end"]),
        },
        "decision": performance_ci_decision(sharpe_lower),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_text(result_path, serialized + "\n")
    _atomic_text(report_path, render_report(payload))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calendar-cache", type=Path, default=DEFAULT_CALENDAR_CACHE)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--refresh-market", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_experiment(
        input_path=args.input,
        library_path=args.library,
        config_path=args.config,
        calendar_cache=args.calendar_cache,
        market_cache=args.market_cache,
        result_path=args.result,
        report_path=args.report,
        seeds=tuple(int(seed) for seed in args.seeds.split(",") if seed.strip()),
        block_length=args.block_length,
        cache_only=args.cache_only,
        refresh_market=args.refresh_market,
        download_workers=args.download_workers,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "sharpe": payload["metrics"]["sharpe"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
