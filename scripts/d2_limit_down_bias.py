#!/usr/bin/env python3
"""Quantify the D+2 forced-close bias on one fixed, no-replacement trade ledger.

The model is fit exactly as in :mod:`backtest_argus`, but selection stops at Top-K before
D+1 fillability is inspected. Actual daily bars and exchange-provided limit prices are
cached by date, then every entered position is resolved by
``helix.eval.backtest.resolve_realistic_exit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from helix.config import Config, LabelConfig
from helix.data.panel import Panel
from helix.data.tushare_source import TushareSource
from helix.eval.backtest import resolve_realistic_exit, summarize_portfolio_returns
from helix.eval.shared_entry_check import entry_is_fillable

STAMP_CUT_DATE = "20230828"
PRICE_COLUMNS = ("open", "high", "close", "vol")
MARKET_COLUMNS = (
    "trade_date",
    "ts_code",
    "open",
    "high",
    "close",
    "vol",
    "adj_factor",
    "up_limit",
    "down_limit",
)
COMPARISON_METRICS = (
    "cagr",
    "max_drawdown",
    "day_win_rate",
    "profit_loss_ratio",
    "max_consecutive_losses",
    "sharpe",
)
SELECTION_ALGORITHM_VERSION = 2


def digits(value: str) -> str:
    return "".join(character for character in str(value) if character.isdigit())


def split_training_and_scoring_pool(
    frame: pd.DataFrame,
    *,
    split_date: str,
    embargo_days: int,
    required_training_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Filter outcomes only for training; preserve the full point-in-time test pool."""
    dates = np.asarray(sorted(frame["trade_date"].astype(str).unique()))
    if dates.size == 0:
        raise ValueError("candidate input has no trade dates")
    cut = int(np.searchsorted(dates, split_date, side="right"))
    train_end = str(dates[max(cut - 1, 0)])
    test_position = min(cut + embargo_days, len(dates) - 1)
    test_start = str(dates[test_position])
    train = frame[frame["trade_date"] <= train_end].dropna(
        subset=list(required_training_columns)
    )
    test = frame[frame["trade_date"] >= test_start].copy()
    return train, test, train_end, test_start


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_selection_signature(
    *,
    input_path: Path,
    seeds: Sequence[int],
    top_k: int,
    split_date: str,
    embargo_days: int,
    features: Sequence[str],
) -> dict:
    """Bind a shortlist cache to all inputs that can change the selected holdings."""
    from fill_impact import MODEL_PARAMS

    stat = input_path.stat()
    feature_payload = json.dumps(list(features), ensure_ascii=False, separators=(",", ":"))
    return {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "input_sha256": _sha256(input_path),
        "seeds": [int(seed) for seed in seeds],
        "top_k": int(top_k),
        "split_date": str(split_date),
        "embargo_days": int(embargo_days),
        "rank_by": "classify",
        "model_params": MODEL_PARAMS,
        "feature_count": len(features),
        "feature_sha256": hashlib.sha256(feature_payload.encode("utf-8")).hexdigest(),
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
    }


def select_fixed_top_k(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Fix each D0 Top-K before applying entry fillability; never promote rank K+1."""
    selected: list[pd.DataFrame] = []
    for _, group in scored.groupby("trade_date", sort=True):
        duplicated = group[group.duplicated("stock_code", keep=False)]
        price_columns = [
            column
            for column in ("label_px_d1_open", "label_px_d2_close")
            if column in duplicated
        ]
        if price_columns and not duplicated.empty:
            conflicts = (
                duplicated.groupby("stock_code", sort=False)[price_columns]
                .nunique(dropna=False)
                .gt(1)
                .any(axis=1)
            )
            if conflicts.any():
                codes = conflicts.index[conflicts].tolist()
                raise ValueError(f"duplicate stock rows have conflicting prices: {codes}")

        best_rows = group.groupby("stock_code", sort=False)["score"].idxmax()
        unique_stocks = group.loc[best_rows]
        if len(unique_stocks) < top_k:
            continue
        picked = unique_stocks.nlargest(top_k, "score", keep="first").copy()
        picked["selected_rank"] = range(1, top_k + 1)
        picked["entry_filled"] = (
            ~picked["unfillable"].astype(bool)
            if "unfillable" in picked
            else True
        )
        selected.append(picked)
    if not selected:
        return scored.iloc[0:0].assign(
            selected_rank=pd.Series(dtype=int), entry_filled=pd.Series(dtype=bool)
        )
    return pd.concat(selected, ignore_index=True)


def aggregate_aligned_daily(
    ledger: pd.DataFrame,
    dates: Sequence[str],
    top_k: int,
    overlap: int,
) -> pd.DataFrame:
    """Aggregate both exit assumptions from the same rows and fixed cash denominator."""
    if top_k <= 0 or overlap <= 0:
        raise ValueError("top_k and overlap must be positive")
    columns = ["matched_baseline_net_return", "matched_realistic_net_return"]
    missing = [column for column in ("d0_date", *columns) if column not in ledger]
    if missing:
        raise KeyError(f"trade ledger is missing columns {missing}")

    grouped = ledger.groupby("d0_date", sort=False)[columns].sum()
    daily = pd.DataFrame({"date": [str(date) for date in dates]})
    daily = daily.join(grouped, on="date")
    daily[columns] = daily[columns].fillna(0.0)
    denominator = top_k * overlap
    daily["baseline_portfolio_return"] = (
        daily["matched_baseline_net_return"] / denominator
    )
    daily["realistic_portfolio_return"] = (
        daily["matched_realistic_net_return"] / denominator
    )
    return daily.drop(columns=columns)


def build_exit_panel(
    market: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
) -> Panel:
    """Build an aligned sparse panel without filling absent actual limit prices."""
    required = {
        "trade_date",
        "ts_code",
        "open",
        "high",
        "close",
        "vol",
        "adj_factor",
        "up_limit",
        "down_limit",
    }
    missing = sorted(required.difference(market.columns))
    if missing:
        raise KeyError(f"market data is missing columns {missing}")
    frame = market.loc[:, sorted(required)].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market data has duplicate (trade_date, ts_code) rows")

    aligned_dates = np.asarray([str(date) for date in dates])
    aligned_codes = np.asarray([str(code) for code in codes])

    def pivot(column: str) -> np.ndarray:
        wide = frame.pivot(index="trade_date", columns="ts_code", values=column)
        return wide.reindex(index=aligned_dates, columns=aligned_codes).to_numpy(dtype=float)

    raw_open = pivot("open")
    raw_high = pivot("high")
    raw_close = pivot("close")
    adj_factor = pivot("adj_factor")
    up_limit = pivot("up_limit")
    down_limit = pivot("down_limit")
    volume = pivot("vol")
    fields = {
        "open": raw_open,
        "high": raw_high,
        "close": raw_close,
        "open_hfq": raw_open * adj_factor,
        "close_hfq": raw_close * adj_factor,
        "up_limit": up_limit,
        "down_limit": down_limit,
        "limit_price_observed": (np.isfinite(up_limit) & np.isfinite(down_limit)).astype(float),
        "is_trading": (np.isfinite(raw_close) & (np.nan_to_num(volume) > 0)).astype(float),
    }
    return Panel(dates=aligned_dates, codes=aligned_codes, fields=fields)


def load_open_dates(path: Path, start_date: str, end_date: str) -> list[str]:
    """Read an existing Tushare calendar cache and return real open dates."""
    calendar = pd.read_parquet(path)
    date_column = "cal_date" if "cal_date" in calendar else "trade_date"
    calendar[date_column] = calendar[date_column].astype(str).map(digits)
    if "is_open" in calendar:
        calendar = calendar[pd.to_numeric(calendar["is_open"], errors="coerce") == 1]
    dates = calendar[date_column]
    return sorted(dates[(dates >= start_date) & (dates <= end_date)].unique().tolist())


def _fetch_one_date(source: TushareSource, trade_date: str) -> pd.DataFrame:
    daily = source._call(
        "daily",
        trade_date=trade_date,
        fields="ts_code,trade_date,open,high,close,vol",
    )
    adjustment = source._call(
        "adj_factor",
        trade_date=trade_date,
        fields="ts_code,trade_date,adj_factor",
    )
    limits = source._call(
        "stk_limit",
        trade_date=trade_date,
        fields="ts_code,trade_date,up_limit,down_limit",
    )
    if daily.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    merged = daily.merge(
        adjustment,
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    ).merge(
        limits,
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    return merged.loc[:, MARKET_COLUMNS]


def load_or_fetch_market(
    dates: Sequence[str],
    cache_dir: Path,
    cfg: Config,
    *,
    refresh: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Load date caches, fetching only missing dates through the repository client."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = [digits(date) for date in dates]
    missing = [
        date for date in wanted if refresh or not (cache_dir / f"{date}.parquet").exists()
    ]
    source = TushareSource(cfg) if missing else None
    if source is not None:
        def fetch_and_cache(date: str) -> str:
            frame = _fetch_one_date(source, date)
            frame.to_parquet(
                cache_dir / f"{date}.parquet", index=False, compression="zstd"
            )
            return date

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_and_cache, date) for date in missing]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="D+2 exit market",
                unit="d",
            ):
                future.result()

    frames = [pd.read_parquet(cache_dir / f"{date}.parquet") for date in wanted]
    if not frames:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    market = pd.concat(frames, ignore_index=True)
    market["trade_date"] = market["trade_date"].astype(str).map(digits)
    if market.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("cached market data has duplicate (trade_date, ts_code) rows")
    return market


def fit_argus_top_k(
    input_path: Path,
    seeds: Sequence[int],
    top_k: int,
    split_date: str,
    embargo_days: int,
) -> tuple[dict[int, pd.DataFrame], list[str], dict[str, str | int]]:
    """Recreate current OOS scores, fixing each seed's Top-K before fill checks."""
    from backtest_argus import fit_and_score
    from fill_impact import feature_columns
    features = feature_columns(str(input_path))
    price_columns = ["label_px_d1_open", "label_px_d2_close"]
    needed = sorted(
        {
            "trade_date",
            "stock_code",
            "label_d2_hit_8pct",
            "label_d2_peak_return",
            "label_d2_return",
            *price_columns,
            *features,
        }
    )
    frame = pd.read_parquet(input_path, columns=needed)
    frame["trade_date"] = frame["trade_date"].astype(str)
    train, test, train_end, test_start = split_training_and_scoring_pool(
        frame,
        split_date=split_date,
        embargo_days=embargo_days,
        required_training_columns=[
            "label_d2_hit_8pct",
            "label_d2_peak_return",
            "label_d2_return",
            *price_columns,
        ],
    )

    selected_by_seed: dict[int, pd.DataFrame] = {}
    keep_columns = [
        "trade_date",
        "stock_code",
        "label_px_d1_open",
        "label_px_d2_close",
    ]
    for seed in seeds:
        scored = test.loc[:, keep_columns].copy()
        scored["score"] = fit_and_score(
            "classify",
            seed,
            train,
            test,
            features,
            "label_d2_hit_8pct",
            "label_d2_return",
        )
        selected = select_fixed_top_k(scored, top_k)
        selected["seed"] = seed
        selected_by_seed[seed] = selected

    metadata: dict[str, str | int] = {
        "features": len(features),
        "train_end": train_end,
        "test_start": test_start,
        "test_end": str(test["trade_date"].max()),
    }
    test_dates = [digits(date) for date in sorted(test["trade_date"].unique())]
    return selected_by_seed, test_dates, metadata


def _sell_rate(cfg: Config, trade_date: str) -> float:
    backtest = cfg.backtest
    stamp = (
        backtest.stamp_sell_bps_before_cut
        if digits(trade_date) < STAMP_CUT_DATE
        else backtest.stamp_sell_bps
    )
    return (
        backtest.commission_bps
        + backtest.transfer_bps
        + backtest.slippage_bps
        + stamp
    ) / 10_000.0


def _net_return(gross: float, buy_rate: float, sell_rate: float) -> float:
    return (1.0 + gross) * (1.0 - sell_rate) / (1.0 + buy_rate) - 1.0


def resolve_selected_ledger(
    selected: pd.DataFrame,
    panel: Panel,
    label_cfg: LabelConfig,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Resolve actual Top-K entries and produce both returns on the same rows."""
    date_index = {str(date): index for index, date in enumerate(panel.dates)}
    code_index = {str(code): index for index, code in enumerate(panel.codes)}
    buy_rate = (
        cfg.backtest.commission_bps
        + cfg.backtest.transfer_bps
        + cfg.backtest.slippage_bps
    ) / 10_000.0
    rows: list[dict] = []
    raw_entry_price_errors: list[float] = []
    raw_d2_price_errors: list[float] = []
    filled = 0

    for record in selected.itertuples(index=False):
        d0_date = digits(record.trade_date)
        if d0_date not in date_index or record.stock_code not in code_index:
            raise ValueError(f"selected holding is missing from market panel: {record}")
        d0_index = date_index[d0_date]
        stock_index = code_index[record.stock_code]
        if not entry_is_fillable(panel, d0_index, stock_index, label_cfg):
            continue
        filled += 1
        entry_index = d0_index + label_cfg.entry_offset
        entry_price = float(panel["open_hfq"][entry_index, stock_index])
        raw_entry = float(panel["open"][entry_index, stock_index])
        published_entry = float(record.label_px_d1_open)
        if np.isfinite(published_entry):
            raw_entry_price_errors.append(abs(raw_entry - published_entry))
        if np.isfinite(published_entry) and not np.isclose(
            raw_entry, published_entry, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"D+1 raw open mismatch for {record.stock_code} on {d0_date}: "
                f"market={raw_entry}, event={published_entry}"
            )

        resolution = resolve_realistic_exit(panel, d0_index, stock_index, label_cfg)
        touch_index = d0_index + label_cfg.touch_offset
        baseline_exit = float(panel["close_hfq"][touch_index, stock_index])
        raw_baseline_exit = float(panel["close"][touch_index, stock_index])
        published_baseline_exit = float(record.label_px_d2_close)
        if np.isfinite(published_baseline_exit) and np.isfinite(raw_baseline_exit):
            raw_d2_price_errors.append(abs(raw_baseline_exit - published_baseline_exit))
        if np.isfinite(published_baseline_exit) and not np.isclose(
            raw_baseline_exit, published_baseline_exit, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"D+2 raw close mismatch for {record.stock_code} on {d0_date}: "
                f"market={raw_baseline_exit}, event={published_baseline_exit}"
            )
        baseline_gross = baseline_exit / entry_price - 1.0
        realistic_gross = (
            resolution.exit_price / entry_price - 1.0
            if not resolution.unresolved_at_end
            else float("nan")
        )
        baseline_date = str(panel.dates[touch_index])
        baseline_net = _net_return(baseline_gross, buy_rate, _sell_rate(cfg, baseline_date))
        realistic_net = (
            _net_return(
                realistic_gross,
                buy_rate,
                _sell_rate(cfg, str(panel.dates[resolution.exit_index])),
            )
            if resolution.exit_index is not None
            else float("nan")
        )
        comparable = np.isfinite(baseline_net) and np.isfinite(realistic_net)
        rows.append(
            {
                "seed": int(record.seed),
                "d0_date": d0_date,
                "ts_code": record.stock_code,
                "selected_rank": int(record.selected_rank),
                "entry_price": entry_price,
                "baseline_exit_price": baseline_exit,
                "exit_date": (
                    str(panel.dates[resolution.exit_index])
                    if resolution.exit_index is not None
                    else ""
                ),
                "exit_price": resolution.exit_price,
                "exit_session": resolution.exit_session,
                "d2_limit_down": resolution.d2_limit_down,
                "d2_suspended": resolution.d2_suspended,
                "encountered_suspension": resolution.encountered_suspension,
                "missing_limit_data": resolution.missing_limit_data,
                "delay_days": resolution.delay_days,
                "holding_days": resolution.holding_days,
                "unresolved_at_end": resolution.unresolved_at_end,
                "baseline_gross_return": baseline_gross,
                "realistic_gross_return": realistic_gross,
                "matched_baseline_net_return": baseline_net if comparable else 0.0,
                "matched_realistic_net_return": realistic_net if comparable else 0.0,
            }
        )

    audit = {
        "selected": float(len(selected)),
        "filled": float(filled),
        "fill_rate": float(filled / len(selected)) if len(selected) else float("nan"),
        "max_raw_entry_price_error": max(raw_entry_price_errors, default=0.0),
        "max_raw_d2_price_error": max(raw_d2_price_errors, default=0.0),
    }
    return pd.DataFrame(rows), audit


def summarize_seed(
    ledger: pd.DataFrame,
    test_dates: Sequence[str],
    top_k: int,
    overlap: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    daily = aggregate_aligned_daily(ledger, test_dates, top_k, overlap)
    baseline = summarize_portfolio_returns(daily["baseline_portfolio_return"].to_numpy())
    realistic = summarize_portfolio_returns(daily["realistic_portfolio_return"].to_numpy())
    limit_down = ledger[ledger["d2_limit_down"]]
    resolved_limit_down = limit_down[~limit_down["unresolved_at_end"]]
    summary: dict[str, float] = {}
    for metric in COMPARISON_METRICS:
        summary[f"baseline_{metric}"] = baseline[metric]
        summary[f"realistic_{metric}"] = realistic[metric]
        summary[f"delta_{metric}"] = realistic[metric] - baseline[metric]
    summary.update(
        {
            "n_days": float(len(daily)),
            "n_trades": float(len(ledger)),
            "limit_down_count": float(len(limit_down)),
            "limit_down_exit_share": (
                float(len(limit_down) / len(ledger)) if len(ledger) else float("nan")
            ),
            "avg_delay_days": (
                float(resolved_limit_down["delay_days"].mean())
                if len(resolved_limit_down)
                else float("nan")
            ),
            "avg_limit_down_loss": (
                float(resolved_limit_down["realistic_gross_return"].mean())
                if len(resolved_limit_down)
                else float("nan")
            ),
            "avg_limit_down_return_impact": (
                float(
                    (
                        resolved_limit_down["realistic_gross_return"]
                        - resolved_limit_down["baseline_gross_return"]
                    ).mean()
                )
                if len(resolved_limit_down)
                else float("nan")
            ),
            "avg_suspension_delay_days": (
                float(
                    ledger.loc[
                        ledger["d2_suspended"] & ~ledger["unresolved_at_end"],
                        "delay_days",
                    ].mean()
                )
                if (ledger["d2_suspended"] & ~ledger["unresolved_at_end"]).any()
                else float("nan")
            ),
            "avg_suspension_return": (
                float(
                    ledger.loc[
                        ledger["d2_suspended"] & ~ledger["unresolved_at_end"],
                        "realistic_gross_return",
                    ].mean()
                )
                if (ledger["d2_suspended"] & ~ledger["unresolved_at_end"]).any()
                else float("nan")
            ),
            "missing_limit_count": float(ledger["missing_limit_data"].sum()),
            "unresolved_at_end": float(ledger["unresolved_at_end"].sum()),
            "d2_suspension_count": float(ledger["d2_suspended"].sum()),
        }
    )
    daily["baseline_equity"] = (1.0 + daily["baseline_portfolio_return"]).cumprod()
    daily["realistic_equity"] = (1.0 + daily["realistic_portfolio_return"]).cumprod()
    return summary, daily


def aggregate_seed_summaries(runs: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = runs[0].keys() if runs else []
    output: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        finite = values[np.isfinite(values)]
        output[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        }
    return output


def _format_metric(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if metric in {"cagr", "max_drawdown", "day_win_rate"}:
        return f"{value:.2%}"
    if metric == "max_consecutive_losses":
        return f"{value:.1f} 天"
    return f"{value:.3f}"


def _format_mean_std(summary: dict, key: str, metric: str) -> str:
    return (
        f"{_format_metric(metric, summary[key]['mean'])} ± "
        f"{_format_metric(metric, summary[key]['std'])}"
    )


def _append_exception_table(
    lines: list[str],
    title: str,
    records: Sequence[dict],
    columns: Sequence[tuple[str, str]],
) -> None:
    lines.extend(["", f"### {title}", ""])
    if not records:
        lines.append("无。")
        return
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for record in records:
        cells = []
        for key, _ in columns:
            value = record.get(key, "")
            cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")


def render_markdown_report(payload: dict) -> str:
    summary = payload["summary"]
    labels = {
        "cagr": "CAGR",
        "max_drawdown": "最大回撤",
        "day_win_rate": "日胜率",
        "profit_loss_ratio": "盈亏比",
        "max_consecutive_losses": "最大连续亏损",
        "sharpe": "夏普比率",
    }
    lines = [
        "# D+2 跌停递延平仓偏差量化",
        "",
        f"> 数据区间：{payload['metadata']['test_start']} 至 "
        f"{payload['metadata']['test_end']}；行情截止：{payload['market_end']}；"
        f"结果为 {len(payload['seeds'])} 个既有随机种子的均值 ± 样本标准差。",
        "",
    ]
    candidate_audit = payload["metadata"].get("candidate_pool_audit", {})
    if candidate_audit.get("future_conditioned"):
        lines.extend(
            [
                "> [!WARNING] 当前 ARGUS 事件源已在上游排除 D+1/D+2 停牌行。脚本本身"
                "不会再按未来结果过滤测试池，但物理缺失的候选无法从该文件恢复；因此"
                "下列数值是既有历史持仓的退出偏差量化，不是修复后完整 D0 候选池的最终验收值。",
                "",
            ]
        )
    lines.extend(
        [
        "## 核心指标",
        "",
        "| 指标 | 开关关闭：D+2 收盘 | 开关开启：真实递延 | 差异 |",
        "|---|---:|---:|---:|",
        ]
    )
    for metric, label in labels.items():
        lines.append(
            f"| {label} | {_format_mean_std(summary, f'baseline_{metric}', metric)} | "
            f"{_format_mean_std(summary, f'realistic_{metric}', metric)} | "
            f"{_format_mean_std(summary, f'delta_{metric}', metric)} |"
        )

    lines.extend(
        [
            "",
            "### 逐种子明细",
            "",
            "| seed | 原 CAGR | 修正 CAGR | 原最大回撤 | 修正最大回撤 | "
            "原夏普 | 修正夏普 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in payload["runs"]:
        lines.append(
            f"| {int(run['seed'])} | {run['baseline_cagr']:.2%} | "
            f"{run['realistic_cagr']:.2%} | {run['baseline_max_drawdown']:.2%} | "
            f"{run['realistic_max_drawdown']:.2%} | {run['baseline_sharpe']:.3f} | "
            f"{run['realistic_sharpe']:.3f} |"
        )

    share = summary["limit_down_exit_share"]["mean"]
    delay = summary["avg_delay_days"]["mean"]
    loss = summary["avg_limit_down_loss"]["mean"]
    impact = summary["avg_limit_down_return_impact"]["mean"]
    lines.extend(
        [
            "",
            "## 跌停与递延样本",
            "",
            f"- 每个种子平均固定 **{summary['n_days']['mean']:.0f}** 个 D0 交易日，"
            f"实际建仓 **{summary['n_trades']['mean']:.1f}** 笔。",
            f"- D+2 全天封死跌停平均 **{summary['limit_down_count']['mean']:.1f}** "
            "笔/种子。",
            f"- D+2 全天封死跌停占实际建仓的 **{share:.4%}**。",
            f"- 已完成平仓的跌停样本平均递延 **{delay:.3f} 个交易日**。",
            f"- 跌停样本按真实退出价计算的平均毛收益为 **{loss:.2%}**。",
            f"- 相对 D+2 强制收盘假设的平均毛收益变化为 **{impact:.2%}**。",
            f"- D+2 停牌递延：**{summary['d2_suspension_count']['mean']:.1f}** 笔/种子。",
            f"- 缺少实际涨跌停价并递延：**{summary['missing_limit_count']['mean']:.1f}** "
            "笔/种子。",
            f"- 行情末端未平仓：**{summary['unresolved_at_end']['mean']:.1f}** 笔/种子。",
            "",
            "### 逐种子风险统计",
            "",
            "| seed | 实际建仓 | D+2封死跌停 | 跌停占比 | 平均递延 | 平均真实毛收益 | "
            "平均收益变化 | D+2停牌 | 未平仓 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in payload["runs"]:
        lines.append(
            f"| {int(run['seed'])} | {int(run['n_trades'])} | "
            f"{int(run['limit_down_count'])} | {run['limit_down_exit_share']:.4%} | "
            f"{_format_metric('max_consecutive_losses', run['avg_delay_days'])} | "
            f"{_format_metric('cagr', run['avg_limit_down_loss'])} | "
            f"{_format_metric('cagr', run['avg_limit_down_return_impact'])} | "
            f"{int(run['d2_suspension_count'])} | {int(run['unresolved_at_end'])} |"
        )
    lines.extend(
        [
            "",
            "## 指标定义",
            "",
            "- CAGR：期末净值按 252 个交易日年化。",
            "- 最大回撤：累计净值相对历史峰值（含初始净值 1.0）的最小跌幅。",
            "- 日胜率：日组合收益严格大于 0 的天数占比；零收益日不是胜日。",
            "- 盈亏比：正收益日均值 / 负收益日均值绝对值。",
            "- 最大连续亏损：连续负收益交易日的最长长度，零收益日中断。",
            "- 夏普比率：日均收益 / 日收益样本标准差 × `sqrt(252)`。",
            "",
            "## 口径",
            "",
            "D0 每日先固定 Top4，再判断 D+1 是否按当日实际涨停价可买；未成交不补位。"
            "D+2 停牌或全天封死跌停时逐日递延：开盘高于实际跌停价则按开盘卖出，"
            "否则仅在收盘高于跌停价时按收盘卖出。盘中开板但收盘回封不假设成交。",
            "",
            "两组指标来自同一逐笔账本。未成交和末端未平仓槽位在两组日收益中均为零，"
            "每日分母始终为 `Top4 × overlap(2)`。收益仍归属于原 D0 篮子，以隔离退出"
            "价格偏差，不另建动态资金占用模型。",
            "",
            "不同股票和制度时期的涨跌停差异直接读取 Tushare `stk_limit` 的当日"
            "`up_limit/down_limit`，没有用代码前缀或固定百分比推断。成交计价使用后复权"
            "开盘/收盘价，交易状态判断只使用当日原始 OHLC、成交量和涨跌停价。",
            "",
            "## 数据限制",
            "",
            "当前标准面板没有可复用的历史 `predictions.npz`。本报告只能从仓库既有 "
            "ARGUS 事件表复现历史持仓；该源文件已在上游物理排除 D+1/D+2 停牌候选，"
            "无法在报告脚本内恢复缺失行。脚本已保证训练标签过滤不作用于测试打分池，"
            "但这不能修复输入文件本身的缺失。因此停牌统计只描述既有持仓，不代表全市场"
            "风险；完整验收仍需用修复后流水线重新生成 `predictions.npz`。",
            "",
            "事件表的 `label_px_d1_open/label_px_d2_close` 已与 Tushare 原始价逐笔校验，"
            "最大误差为 0；组合收益则统一改用同一行情面板的后复权价，以避免除权在"
            "跨日收益中制造跳变。",
            "",
            "## 可复现命令",
            "",
            "```bash",
            ".venv/bin/python scripts/d2_limit_down_bias.py",
            "```",
            "",
            "逐种子结果和机器可读汇总位于 "
            "`data/artifacts/d2_limit_down_bias_summary.json`，逐笔记录位于 "
            "`data/artifacts/d2_limit_down_bias_trades.parquet`。",
        ]
    )
    exceptions = payload.get("exceptions", {})
    _append_exception_table(
        lines,
        "D+2 停牌递延明细",
        exceptions.get("d2_suspensions", []),
        (
            ("seed", "seed"),
            ("d0_date", "D0"),
            ("ts_code", "代码"),
            ("exit_date", "退出日"),
            ("delay_days", "递延天数"),
            ("realistic_gross_return", "真实毛收益"),
        ),
    )
    _append_exception_table(
        lines,
        "行情末端未平仓明细",
        exceptions.get("unresolved", []),
        (
            ("seed", "seed"),
            ("d0_date", "D0"),
            ("ts_code", "代码"),
            ("entry_price", "入场价"),
            ("holding_days", "已持有交易日"),
        ),
    )
    _append_exception_table(
        lines,
        "实际涨跌停价缺失明细",
        exceptions.get("missing_limit_data", []),
        (
            ("seed", "seed"),
            ("d0_date", "D0"),
            ("ts_code", "代码"),
            ("exit_date", "退出日"),
            ("delay_days", "递延天数"),
        ),
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/argus_quant_working.parquet"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seeds", default="7,13,42")
    parser.add_argument("--split-date", default="2024-09-04")
    parser.add_argument("--embargo-days", type=int, default=3)
    parser.add_argument(
        "--calendar-cache",
        type=Path,
        default=Path("data/raw/suspension_cache/calendar_20211006_20261022.parquet"),
    )
    parser.add_argument("--market-cache", type=Path, default=Path("data/raw/d2_exit_cache"))
    parser.add_argument(
        "--selection-cache",
        type=Path,
        default=Path("data/artifacts/d2_limit_down_bias_selected.parquet"),
    )
    parser.add_argument(
        "--selection-meta",
        type=Path,
        default=Path("data/artifacts/d2_limit_down_bias_selected.json"),
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/artifacts/d2_limit_down_bias_summary.json"),
    )
    parser.add_argument(
        "--trades-out",
        type=Path,
        default=Path("data/artifacts/d2_limit_down_bias_trades.parquet"),
    )
    parser.add_argument(
        "--daily-out",
        type=Path,
        default=Path("data/artifacts/d2_limit_down_bias_daily.parquet"),
    )
    parser.add_argument(
        "--report-out", type=Path, default=Path("docs/risk/d2_limit_down_bias.md")
    )
    parser.add_argument("--refresh-market", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    from fill_impact import feature_columns

    features = feature_columns(str(args.input))
    selection_signature = build_selection_signature(
        input_path=args.input,
        seeds=seeds,
        top_k=args.top_k,
        split_date=args.split_date,
        embargo_days=args.embargo_days,
        features=features,
    )
    cached_metadata = (
        json.loads(args.selection_meta.read_text(encoding="utf-8"))
        if args.selection_cache.exists() and args.selection_meta.exists()
        else {}
    )
    cache_matches = cached_metadata.get("selection_signature") == selection_signature
    if cache_matches:
        selected_all = pd.read_parquet(args.selection_cache)
        selected_by_seed = {
            seed: selected_all[selected_all["seed"] == seed].copy() for seed in seeds
        }
        metadata = cached_metadata
        test_dates = list(metadata["test_dates"])
    else:
        selected_by_seed, test_dates, metadata = fit_argus_top_k(
            args.input, seeds, args.top_k, args.split_date, args.embargo_days
        )
        metadata["selection_signature"] = selection_signature
        metadata["test_dates"] = test_dates
        selected_all = pd.concat(selected_by_seed.values(), ignore_index=True)
        args.selection_cache.parent.mkdir(parents=True, exist_ok=True)
        selected_all.to_parquet(args.selection_cache, index=False, compression="zstd")
        args.selection_meta.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    suspension_audit_path = Path("data/raw/suspension_report.json")
    if args.input.name == "argus_quant_working.parquet" and suspension_audit_path.exists():
        suspension_audit = json.loads(suspension_audit_path.read_text(encoding="utf-8"))
        metadata["candidate_pool_audit"] = {
            "future_conditioned": suspension_audit.get("verdict") == "excluded_upstream",
            "source": str(suspension_audit_path),
            "verdict": suspension_audit.get("verdict", "unknown"),
            "d1_suspension_rows": suspension_audit.get("control", {})
            .get("D+1", {})
            .get("rows"),
            "d2_suspension_rows": suspension_audit.get("control", {})
            .get("D+2", {})
            .get("rows"),
        }
    today = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y%m%d")
    market_dates = load_open_dates(args.calendar_cache, min(test_dates), today)
    market = load_or_fetch_market(
        market_dates,
        args.market_cache,
        cfg,
        refresh=args.refresh_market,
        max_workers=args.download_workers,
    )
    codes = sorted(selected_all["stock_code"].astype(str).unique())
    market = market[market["ts_code"].astype(str).isin(codes)]
    panel = build_exit_panel(market, market_dates, codes)

    ledgers: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    runs: list[dict[str, float]] = []
    audits: dict[str, dict[str, float]] = {}
    overlap = max(cfg.label.touch_offset - cfg.label.entry_offset + 1, 1)
    for seed, selected in selected_by_seed.items():
        ledger, audit = resolve_selected_ledger(selected, panel, cfg.label, cfg)
        run, daily = summarize_seed(ledger, test_dates, args.top_k, overlap)
        run["seed"] = float(seed)
        ledger["seed"] = seed
        daily["seed"] = seed
        ledgers.append(ledger)
        daily_frames.append(daily)
        runs.append(run)
        audits[str(seed)] = audit

    all_trades = pd.concat(ledgers, ignore_index=True)
    all_daily = pd.concat(daily_frames, ignore_index=True)
    detail_columns = [
        "seed",
        "d0_date",
        "ts_code",
        "entry_price",
        "exit_date",
        "delay_days",
        "holding_days",
        "realistic_gross_return",
    ]
    exceptions = {
        "d2_suspensions": all_trades.loc[
            all_trades["d2_suspended"], detail_columns
        ].to_dict(orient="records"),
        "unresolved": all_trades.loc[
            all_trades["unresolved_at_end"], detail_columns
        ].to_dict(orient="records"),
        "missing_limit_data": all_trades.loc[
            all_trades["missing_limit_data"], detail_columns
        ].to_dict(orient="records"),
    }
    payload = {
        "metadata": metadata,
        "market_end": market_dates[-1],
        "top_k": args.top_k,
        "overlap": overlap,
        "seeds": seeds,
        "cost_bps": {
            "commission": cfg.backtest.commission_bps,
            "transfer": cfg.backtest.transfer_bps,
            "stamp_after_cut": cfg.backtest.stamp_sell_bps,
            "stamp_before_cut": cfg.backtest.stamp_sell_bps_before_cut,
            "slippage": cfg.backtest.slippage_bps,
        },
        "audits": audits,
        "runs": runs,
        "summary": aggregate_seed_summaries(runs),
        "exceptions": exceptions,
    }
    for path in (args.summary_out, args.trades_out, args.daily_out, args.report_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_trades.to_parquet(args.trades_out, index=False, compression="zstd")
    all_daily.to_parquet(args.daily_out, index=False, compression="zstd")
    args.report_out.write_text(render_markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
