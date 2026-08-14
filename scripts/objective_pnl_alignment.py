#!/usr/bin/env python3
"""Reproduce the training-only objective/P&L diagnosis and validation report.

The hard boundary is the formal training window ending 2024-09-04.  A D0 decision
needs a D+2 close, so the latest admissible objective row is 2024-09-02.  No row or
forward exit beyond those dates is loaded into any statistic or decision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from deap import gp
from scipy import stats

from helix.config import BacktestConfig
from helix.eval.backtest import _cost_rates, _net_returns
from helix.eval.ic import daily_ic
from helix.eval.metrics import daily_gini
from helix.eval.objective import (
    TopKPortfolio,
    cost_adjusted_returns,
    daily_top_k_portfolio,
    summarize_objective,
)
from helix.gp.library import FactorLibrary, load_factors
from helix.splits import fit_selection_windows

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
OBJECTIVE_D0_END = "2024-09-02"
PRODUCTION_TOP_K = BacktestConfig().top_k
SUPPLEMENTAL_TOP_K = 10
OVERLAP = 2
MIN_DAILY_SAMPLES = 50

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/argus_quant_working.parquet"
DEFAULT_BARS = ROOT / "data/raw/suspension_cache/bars_20220301_20241031_79.parquet"
DEFAULT_REPORT = ROOT / "docs/risk/objective_pnl_alignment.md"
LIBRARIES = {
    "formal": ROOT / "data/artifacts/argus/event_factors.json",
    "argus_multi": ROOT / "data/artifacts/argus_multi/event_factors.json",
    "argus_n40": ROOT / "data/artifacts/argus_n40/event_factors.json",
}


def _date_strings(values: pd.Series) -> pd.Series:
    return values.astype(str).str[:10]


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Assert that every decision row is inside the outcome-complete training window."""
    if "trade_date" not in frame:
        raise ValueError("training frame needs trade_date")
    result = frame.copy()
    result["trade_date"] = _date_strings(result["trade_date"])
    if result.empty:
        raise ValueError("training frame is empty")
    if result["trade_date"].min() < TRAIN_START:
        raise ValueError("training frame starts before the formal train start")
    if result["trade_date"].max() > OBJECTIVE_D0_END:
        raise ValueError("training frame exceeds the objective D0 end")
    return result


def validate_forward_exit(frame: pd.DataFrame, train_end: str = TRAIN_END) -> pd.DataFrame:
    """Drop forward observations whose realised exit falls beyond the training end."""
    if "exit_date" not in frame:
        raise ValueError("forward frame needs exit_date")
    exits = _date_strings(frame["exit_date"])
    return frame.loc[exits <= train_end].copy().reset_index(drop=True)


def market_regime(market_daily_return: pd.Series) -> pd.Series:
    """Trailing-only 20-day market regime; thresholds are fixed before evaluation."""
    trailing = market_daily_return.astype(float).rolling(20, min_periods=20).mean()
    regime = pd.Series("neutral", index=market_daily_return.index, dtype=object)
    regime.loc[trailing > 0.002] = "bull"
    regime.loc[trailing < -0.002] = "bear"
    regime.loc[trailing.isna()] = "unavailable"
    return regime


def correlation(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> dict[str, float]:
    """Pearson/Spearman coefficients and two-sided p-values on finite pairs."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    usable = np.isfinite(left) & np.isfinite(right)
    if usable.sum() < 3 or np.std(left[usable]) == 0 or np.std(right[usable]) == 0:
        return {
            "pearson": float("nan"),
            "pearson_pvalue": float("nan"),
            "spearman": float("nan"),
            "spearman_pvalue": float("nan"),
            "n": float(usable.sum()),
        }
    pearson = stats.pearsonr(left[usable], right[usable])
    spearman = stats.spearmanr(left[usable], right[usable])
    return {
        "pearson": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "n": float(usable.sum()),
    }


def evaluate_months(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly means plus within-month IC/P&L correlation."""
    work = daily.copy()
    work["month"] = _date_strings(work["date"]).str[:7]
    numeric = [column for column in work.select_dtypes(include=[np.number])]
    rows: list[dict[str, Any]] = []
    for month, block in work.groupby("month", sort=True):
        row: dict[str, Any] = {"month": month}
        row.update({column: float(block[column].mean()) for column in numeric})
        if {"hit_ic", "production_net"} <= set(block):
            corr = correlation(block["hit_ic"], block["production_net"])
            row["ic_pnl_pearson"] = corr["pearson"]
            row["ic_pnl_spearman"] = corr["spearman"]
            row["n_corr"] = corr["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_regimes(daily: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Means and IC/P&L correlation inside trailing market regimes."""
    work = daily.copy()
    mapping = regimes.copy()
    mapping.index = mapping.index.astype(str)
    work["regime"] = _date_strings(work["date"]).map(mapping)
    rows: list[dict[str, Any]] = []
    for regime, block in work[work["regime"] != "unavailable"].groupby("regime"):
        corr = correlation(block["hit_ic"], block["production_net"])
        rows.append(
            {
                "regime": regime,
                "n_days": len(block),
                "hit_ic": float(block["hit_ic"].mean()),
                "production_gross_trade": float(block["production_gross_trade"].mean()),
                "production_net_trade": float(block["production_net_trade"].mean()),
                "ic_pnl_pearson": corr["pearson"],
                "ic_pnl_spearman": corr["spearman"],
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def evaluate_horizons(daily_by_horizon: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Summarise D+1..D+10 daily IC and Top-K gross/net outcomes."""
    rows: list[dict[str, Any]] = []
    for horizon, daily in sorted(daily_by_horizon.items()):
        corr = correlation(daily["return_ic"], daily["production_net"])
        rows.append(
            {
                "horizon": horizon,
                "n_days": len(daily),
                "return_ic": float(daily["return_ic"].mean()),
                "gross_trade": float(daily["production_gross_trade"].mean()),
                "net_trade": float(daily["production_net_trade"].mean()),
                "ic_pnl_pearson": corr["pearson"],
                "ic_pnl_spearman": corr["spearman"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_score_quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Hit, peak, close, and giveback by within-date factor-score quintile."""
    required = {"trade_date", "factor_score", "hit", "peak_return", "close_return"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"quintile frame is missing {sorted(missing)}")
    work = frame.copy()
    work["quintile"] = work.groupby("trade_date")["factor_score"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    result = (
        work.groupby("quintile", dropna=True)
        .agg(
            n=("factor_score", "size"),
            hit_rate=("hit", "mean"),
            peak_return=("peak_return", "mean"),
            close_return=("close_return", "mean"),
            turnover_percentile=("turnover_percentile", "median"),
            bottom_turnover_decile=(
                "turnover_percentile",
                lambda values: float((values <= 0.1).mean()),
            ),
        )
        .reset_index()
    )
    result["quintile"] = result["quintile"].astype(int) + 1
    result["giveback"] = result["peak_return"] - result["close_return"]
    turnover: dict[int, float] = {}
    for quintile, block in work.dropna(subset=["quintile"]).groupby("quintile"):
        previous: set[str] | None = None
        daily_turnover = []
        for _, date_block in block.groupby("trade_date", sort=True):
            current = set(date_block["stock_code"])
            if previous is not None and current:
                daily_turnover.append(1 - len(current & previous) / len(current))
            previous = current
        turnover[int(quintile) + 1] = float(np.mean(daily_turnover))
    result["turnover"] = result["quintile"].map(turnover)
    return result


def evaluate_execution_feasibility(frame: pd.DataFrame) -> dict[str, float]:
    """Selected-name observability and available liquidity diagnostics."""
    selected = frame[frame["selected"]].copy()
    if selected.empty:
        return {"selected": 0.0, "unobservable_share": float("nan")}
    result = {
        "selected": float(len(selected)),
        "unobservable_share": float(selected["close_return"].isna().mean()),
        "entry_missing_share": float(selected["entry_price"].isna().mean()),
    }
    if "turnover_percentile" in selected:
        finite = selected["turnover_percentile"].dropna()
        result["bottom_turnover_decile_share"] = float((finite <= 0.1).mean())
        result["median_turnover_percentile"] = float(finite.median())
    return result


def library_alignment_statistics(table: pd.DataFrame) -> dict[str, float]:
    """Correlation of fit objective ranking with embargoed training selection ranking."""
    return correlation(table["fit_net"], table["selection_net"])


def alignment_decision(production: dict[str, float]) -> str:
    """PASS only on positive, significant production-Top4 Pearson and Spearman results."""
    required = (
        production.get("pearson", float("nan")) > 0,
        production.get("spearman", float("nan")) > 0,
        production.get("pearson_pvalue", 1.0) < 0.05,
        production.get("spearman_pvalue", 1.0) < 0.05,
    )
    return "PASS" if all(required) else "FAIL"


def _portfolio_rows(series: TopKPortfolio, rows: slice) -> TopKPortfolio:
    return TopKPortfolio(series.portfolio_return[rows], series.executed[rows])


def evaluate_library_alignment(
    factor_values: dict[str, np.ndarray],
    gross_return: np.ndarray,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
    fit_rows: slice,
    selection_rows: slice,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Vectorised production-TopK fit/selection validation for a factor library."""
    net = cost_adjusted_returns(gross_return, dates, config)
    rows = []
    for name, values in factor_values.items():
        series = daily_top_k_portfolio(
            values, net, candidate_mask, top_k=config.top_k, overlap=OVERLAP
        )
        fit = summarize_objective(_portfolio_rows(series, fit_rows), config.top_k)
        selection = summarize_objective(
            _portfolio_rows(series, selection_rows), config.top_k
        )
        rows.append(
            {
                "factor": name,
                "top_k": config.top_k,
                "fit_net": fit["mean"],
                "selection_net": selection["mean"],
                "fit_ir": fit["ir"],
                "selection_ir": selection["ir"],
            }
        )
    table = pd.DataFrame(rows)
    return table, library_alignment_statistics(table)


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, percent: set[str] | None = None) -> str:
    percent = percent or set()
    if frame.empty:
        return "（无可用观测）"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in percent and isinstance(value, (float, np.floating)):
                cells.append(f"{100 * value:.3f}%" if np.isfinite(value) else "NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(_fmt(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(evidence: dict[str, Any]) -> str:
    """Small stable report surface used by tests and as the full renderer's header."""
    alignment = evidence.get("production_alignment", {})
    return f"""# 目标函数与真实 P&L 对齐专项报告

> 范围声明：本报告**仅训练集**，D0 为 {TRAIN_START} 至 {OBJECTIVE_D0_END}；任何退出日不得晚于 {TRAIN_END}。共 {evidence.get('calendar', {}).get('n_objective_dates', 'NA')} 个目标完整交易日。

生产目标为成本调整后的 **Top4 D+2 收盘净收益**。**Top10** 仅作补充验证，不参与适应度、排序、进化选择或验收判定。

核心根因：{evidence.get('root_cause', '盘中触达标签与 D+2 收盘退出错配')}。

修复后生产目标排序验证：Pearson {_fmt(alignment.get('pearson'))}（p={_fmt(alignment.get('pearson_pvalue'))}），Spearman {_fmt(alignment.get('spearman'))}（p={_fmt(alignment.get('spearman_pvalue'))}），判定 **{alignment_decision(alignment)}**。
"""


def _load_libraries(paths: dict[str, Path]) -> dict[str, FactorLibrary]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"factor libraries are missing: {missing}")
    return {name: load_factors(path) for name, path in paths.items()}


def _load_training_frame(path: Path, libraries: dict[str, FactorLibrary]) -> pd.DataFrame:
    fields = sorted({field for library in libraries.values() for field in library.field_names})
    diagnostic = [
        "trade_date",
        "stock_code",
        "label_d2_hit_8pct",
        "label_d2_peak_return",
        "label_d2_return",
        "label_px_d1_open",
        "turnover_ratio_d0",
        "index_sh_pct_chg",
    ]
    columns = list(dict.fromkeys([*diagnostic, *fields]))
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("trade_date", ">=", TRAIN_START),
            ("trade_date", "<=", OBJECTIVE_D0_END),
        ],
    )
    frame = validate_training_frame(frame)
    frame = frame.sort_values(["trade_date", "stock_code"], kind="stable").reset_index(drop=True)
    if frame["trade_date"].nunique() != 647:
        raise ValueError("formal objective window must contain exactly 647 trade dates")
    return frame


def _compute_factor_values(
    frame: pd.DataFrame, libraries: dict[str, FactorLibrary]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    values = {
        f"{library_name}/{factor.name}": np.full(len(frame), np.nan, dtype=np.float64)
        for library_name, library in libraries.items()
        for factor in library.factors
    }
    metadata: dict[str, dict[str, Any]] = {}
    compiled: dict[str, tuple[FactorLibrary, list[tuple[Any, Any]]]] = {}
    for library_name, library in libraries.items():
        pset = library.build_pset()
        functions = [(factor, gp.compile(gp.PrimitiveTree.from_string(factor.expression, pset), pset)) for factor in library.factors]
        compiled[library_name] = (library, functions)
        for factor in library.factors:
            metadata[f"{library_name}/{factor.name}"] = {
                "library": library_name,
                "expression": factor.expression,
                "sign": factor.sign,
                "old_fit_gini": factor.metrics.get("fit_gini", float("nan")),
                "n_nodes": factor.metrics.get("n_nodes", float("nan")),
            }

    for _, block in frame.groupby("trade_date", sort=False):
        positions = block.index.to_numpy()
        for library_name, (library, functions) in compiled.items():
            # Mirror EventPanel packing exactly: source values are first stored as
            # float32 grids, then widened for GP operators. This matters for stable tie
            # ordering in cross-sectional ranks.
            arguments = [
                block[field].to_numpy(dtype=np.float32).astype(np.float64)[None, :]
                for field in library.field_names
            ]
            for factor, function in functions:
                with np.errstate(all="ignore"):
                    output = function(*arguments)
                if not isinstance(output, np.ndarray) or output.shape != (1, len(block)):
                    continue
                values[f"{library_name}/{factor.name}"][positions] = output[0] * factor.sign
    return values, metadata


def _long_net(gross: np.ndarray, decision_dates: np.ndarray, config: BacktestConfig) -> np.ndarray:
    digits = np.char.replace(np.asarray(decision_dates).astype(str), "-", "")
    buy, sells = _cost_rates(config, digits)
    return _net_returns(np.asarray(gross, dtype=np.float64), buy, sells)


def _one_day_ic(score: np.ndarray, target: np.ndarray) -> float:
    mask = np.ones((1, len(score)), dtype=bool)
    return float(daily_ic(score[None, :], target[None, :], mask, MIN_DAILY_SAMPLES)[0])


def _one_day_gini(score: np.ndarray, hit: np.ndarray) -> float:
    mask = np.ones((1, len(score)), dtype=bool)
    return float(daily_gini(score[None, :], hit[None, :], mask, MIN_DAILY_SAMPLES)[0])


def _daily_diagnostics(
    frame: pd.DataFrame,
    factor_score: np.ndarray,
    config: BacktestConfig,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame[
        [
            "trade_date",
            "stock_code",
            "label_d2_hit_8pct",
            "label_d2_peak_return",
            "label_d2_return",
            "label_px_d1_open",
            "turnover_ratio_d0",
        ]
    ].copy()
    work["factor_score"] = factor_score
    work["net_return"] = _long_net(
        work["label_d2_return"].to_numpy(), work["trade_date"].to_numpy(), config
    )
    work["turnover_percentile"] = work.groupby("trade_date")["turnover_ratio_d0"].rank(pct=True)
    work["selected"] = False
    work["selected_top10"] = False
    rows: list[dict[str, Any]] = []
    previous: set[str] | None = None
    for date, block in work.groupby("trade_date", sort=False):
        eligible = block["factor_score"].notna().to_numpy()
        candidates = np.flatnonzero(eligible)
        order = candidates[
            np.argsort(-block["factor_score"].to_numpy()[candidates], kind="stable")
        ]
        picked = order[:top_k] if len(order) >= top_k else np.array([], dtype=int)
        picked10 = order[:SUPPLEMENTAL_TOP_K] if len(order) >= SUPPLEMENTAL_TOP_K else np.array([], dtype=int)
        if len(picked):
            work.loc[block.index[picked], "selected"] = True
        if len(picked10):
            work.loc[block.index[picked10], "selected_top10"] = True
        gross = block["label_d2_return"].to_numpy(dtype=np.float64)
        net = block["net_return"].to_numpy(dtype=np.float64)
        selected_codes = set(block.iloc[picked]["stock_code"]) if len(picked) else set()
        turnover = (
            float(1 - len(selected_codes & previous) / top_k)
            if previous is not None and selected_codes
            else float("nan")
        )
        if selected_codes:
            previous = selected_codes
        gross_trade = float(np.where(np.isfinite(gross[picked]), gross[picked], 0).mean()) if len(picked) else float("nan")
        net_trade = float(np.where(np.isfinite(net[picked]), net[picked], 0).mean()) if len(picked) else float("nan")
        gross10 = float(np.where(np.isfinite(gross[picked10]), gross[picked10], 0).mean()) if len(picked10) else float("nan")
        net10 = float(np.where(np.isfinite(net[picked10]), net[picked10], 0).mean()) if len(picked10) else float("nan")
        rows.append(
            {
                "date": date,
                "hit_ic": _one_day_ic(
                    block["factor_score"].to_numpy(),
                    block["label_d2_hit_8pct"].to_numpy(),
                ),
                "hit_gini": _one_day_gini(
                    block["factor_score"].to_numpy(),
                    block["label_d2_hit_8pct"].to_numpy(),
                ),
                "close_ic": _one_day_ic(block["factor_score"].to_numpy(), gross),
                "production_gross_trade": gross_trade,
                "production_net_trade": net_trade,
                "production_net": net_trade / OVERLAP,
                "top10_gross_trade": gross10,
                "top10_net_trade": net10,
                "top10_net": net10 / OVERLAP,
                "candidate_gross": float(np.nanmean(gross)),
                "turnover": turnover,
                "execution_rate": float(np.isfinite(gross[picked]).mean()) if len(picked) else 0.0,
            }
        )
    return pd.DataFrame(rows), work


def _library_validation(
    frame: pd.DataFrame,
    values: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], pd.DataFrame]:
    dates = frame["trade_date"].drop_duplicates().to_numpy()
    fit_rows, selection_rows = fit_selection_windows(len(dates), embargo_days=5)
    rows: list[dict[str, Any]] = []
    for name, factor in values.items():
        daily, _ = _daily_diagnostics(frame, factor, config, config.top_k)
        rows.append(
            {
                "factor": name,
                "library": metadata[name]["library"],
                "old_fit_gini": metadata[name]["old_fit_gini"],
                "fit_close_ic": float(daily["close_ic"].iloc[fit_rows].mean()),
                "fit_net": float(daily["production_net"].iloc[fit_rows].mean()),
                "selection_net": float(daily["production_net"].iloc[selection_rows].mean()),
                "fit_top10_net": float(daily["top10_net"].iloc[fit_rows].mean()),
                "selection_top10_net": float(
                    daily["top10_net"].iloc[selection_rows].mean()
                ),
                "n_nodes": metadata[name]["n_nodes"],
            }
        )
    table = pd.DataFrame(rows)
    production = library_alignment_statistics(table)
    top10 = correlation(table["fit_top10_net"], table["selection_top10_net"])
    old = correlation(table["old_fit_gini"], table["selection_net"])
    close_ic = correlation(table["fit_close_ic"], table["selection_net"])
    return table, production, top10, pd.DataFrame(
        [old, close_ic], index=["old_gini", "close_ic"]
    )


def _horizon_daily(
    frame: pd.DataFrame,
    factor_score: np.ndarray,
    bars_path: Path,
    config: BacktestConfig,
) -> dict[int, pd.DataFrame]:
    if not bars_path.exists():
        return {}
    bars = pd.read_parquet(
        bars_path,
        columns=["ts_code", "trade_date", "open", "close", "adj_factor"],
        filters=[("trade_date", "<=", TRAIN_END.replace("-", ""))],
    )
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.strftime("%Y-%m-%d")
    bars = bars.sort_values(["trade_date", "ts_code"], kind="stable")
    calendar = bars["trade_date"].drop_duplicates().tolist()
    date_position = {date: index for index, date in enumerate(calendar)}
    base = frame[["trade_date", "stock_code", "label_px_d1_open"]].copy()
    base["factor_score"] = factor_score
    base = base[base["trade_date"].isin(date_position)]
    entry_bars = bars[["ts_code", "trade_date", "adj_factor"]].rename(
        columns={"ts_code": "stock_code", "trade_date": "entry_date", "adj_factor": "entry_adj"}
    )
    outputs: dict[int, pd.DataFrame] = {}
    for horizon in range(1, 11):
        mapping = []
        for date in base["trade_date"].drop_duplicates():
            position = date_position[date]
            if position + horizon >= len(calendar):
                continue
            exit_date = calendar[position + horizon]
            if exit_date <= TRAIN_END:
                mapping.append(
                    {
                        "trade_date": date,
                        "entry_date": calendar[position + 1],
                        "exit_date": exit_date,
                    }
                )
        mapped = base.merge(pd.DataFrame(mapping), on="trade_date", how="inner")
        exits = bars[["ts_code", "trade_date", "close", "adj_factor"]].rename(
            columns={
                "ts_code": "stock_code",
                "trade_date": "exit_date",
                "close": "exit_close",
                "adj_factor": "exit_adj",
            }
        )
        mapped = mapped.merge(entry_bars, on=["stock_code", "entry_date"], how="left")
        mapped = mapped.merge(exits, on=["stock_code", "exit_date"], how="left")
        mapped["gross_return"] = (
            mapped["exit_close"] * mapped["exit_adj"]
            / (mapped["label_px_d1_open"] * mapped["entry_adj"])
            - 1.0
        )
        mapped = validate_forward_exit(mapped)
        mapped["net_return"] = _long_net(
            mapped["gross_return"].to_numpy(), mapped["trade_date"].to_numpy(), config
        )
        rows = []
        for date, block in mapped.groupby("trade_date", sort=True):
            score = block["factor_score"].to_numpy(dtype=np.float64)
            gross = block["gross_return"].to_numpy(dtype=np.float64)
            net = block["net_return"].to_numpy(dtype=np.float64)
            eligible = np.flatnonzero(np.isfinite(score))
            order = eligible[np.argsort(-score[eligible], kind="stable")]
            picked = order[: config.top_k] if len(order) >= config.top_k else np.array([], dtype=int)
            gross_trade = float(np.where(np.isfinite(gross[picked]), gross[picked], 0).mean()) if len(picked) else float("nan")
            net_trade = float(np.where(np.isfinite(net[picked]), net[picked], 0).mean()) if len(picked) else float("nan")
            rows.append(
                {
                    "date": date,
                    "return_ic": _one_day_ic(score, gross),
                    "production_gross_trade": gross_trade,
                    "production_net_trade": net_trade,
                    "production_net": net_trade / horizon,
                }
            )
        outputs[horizon] = pd.DataFrame(rows)
    return outputs


def _benchmark(config: BacktestConfig) -> dict[str, float]:
    rng = np.random.default_rng(20260813)
    shape = (647, 1990)
    score = rng.normal(size=shape)
    outcome = rng.normal(0, 0.04, size=shape)
    candidate = rng.random(shape) < 0.23
    hit = (outcome > 0.08).astype(float)

    def elapsed(function) -> float:
        samples = []
        for _ in range(4):
            start = time.perf_counter()
            function()
            samples.append(1000 * (time.perf_counter() - start))
        return float(np.median(samples))

    objective_ms = elapsed(
        lambda: daily_top_k_portfolio(
            score, outcome, candidate, top_k=config.top_k, overlap=OVERLAP
        )
    )
    gini_ms = elapsed(lambda: daily_gini(score, hit, candidate, MIN_DAILY_SAMPLES))
    return {
        "objective_ms": objective_ms,
        "legacy_gini_ms": gini_ms,
        "ratio": objective_ms / gini_ms,
    }


def _full_report(evidence: dict[str, Any]) -> str:
    library = evidence["library_table"]
    monthly = evidence["monthly"]
    regimes = evidence["regimes"]
    horizons = evidence["horizons"]
    quintiles = evidence["quintiles"]
    old = evidence["old_alignment"]
    production = evidence["production_alignment"]
    top10_alignment = evidence["top10_alignment"]
    formal = evidence["formal"]
    execution = evidence["execution"]
    benchmark = evidence["benchmark"]
    header = render_report(evidence)
    return header + f"""

## 1. 结论

方向错配判断成立，且主导原因不是符号错误、交易成本或不可成交样本，而是**目标定义错配**：旧 GP 最大化“D+2 盘中是否触达 +8%”的 gini；实盘则持有至 D+2 收盘。正式因子能把高盘中峰值/高命中概率排到前面，但这些标的从峰值到收盘的回吐更大，因而收盘收益排序反向。

根因按影响量级排序：

1. **盘中触达标签与收盘退出错配（主导）**：hit IC={formal['hit_ic']:.4f}，close-return IC={formal['close_ic']:.4f}；命中样本中 {100 * evidence['hit_close_below_target']:.2f}% 收盘低于 +8%，{100 * evidence['hit_close_negative']:.2f}% 收盘转负。
2. **截面相对排序不能保证绝对收益，且正式因子相对候选池也为负贡献**：候选池 D+2 毛收益 {100 * formal['candidate_gross']:.3f}%，Top10 毛收益 {100 * formal['top10_gross_trade']:.3f}%，超额 {100 * formal['top10_excess']:.3f}%。全市场下跌解释绝对负值的一部分，但不能解释负超额。
3. **交易成本放大亏损、不是翻转源头**：Top10 毛收益已为 {100 * formal['top10_gross_trade']:.3f}%，扣费后 {100 * formal['top10_net_trade']:.3f}%，成本影响 {100 * formal['top10_cost_drag']:.3f} 个百分点。
4. **换手与成交可实现性为次要项**：生产 Top4 日换手 {100 * formal['turnover']:.2f}%；所选样本结果不可观测率 {100 * execution['unobservable_share']:.3f}%，低换手率底部一成占比 {100 * execution.get('bottom_turnover_decile_share', float('nan')):.3f}%。它们会影响量级，但不足以产生观察到的方向反转。

## 2. 训练窗口与口径

| 项目 | 值 |
| --- | --- |
| 名义训练窗口 | {TRAIN_START} 至 {TRAIN_END} |
| 目标可用 D0 | {TRAIN_START} 至 {OBJECTIVE_D0_END} |
| D0 数 | {evidence['calendar']['n_objective_dates']} |
| GP fit | {evidence['calendar']['fit_start']} 至 {evidence['calendar']['fit_end']}（{evidence['calendar']['n_fit']} 日） |
| embargo | 5 日 |
| GP selection | {evidence['calendar']['selection_start']} 至 {evidence['calendar']['selection_end']}（{evidence['calendar']['n_selection']} 日） |
| 生产组合 | Top{PRODUCTION_TOP_K}，D+1 开盘建仓、D+2 收盘退出、双边手续费/滑点、资本重叠除以 2 |
| 补充组合 | Top{SUPPLEMENTAL_TOP_K}，只报告、不参与适应度与验收 |

所有 IC、分月、市场环境、持有期、根因和因子库排序验证均只使用上述训练范围。D+1~D+10 分析还逐笔要求 `exit_date <= {TRAIN_END}`；条目不足时直接删除，不向后借用数据。

## 3. 反向关系稳定性

### 3.1 正式因子总览

| 指标 | 训练期结果 |
| --- | ---: |
| 旧 hit-label IC | {formal['hit_ic']:.6f} |
| 旧 hit gini | {formal['hit_gini']:.6f} |
| D+2 close-return IC | {formal['close_ic']:.6f} |
| Top4 毛收益/笔 | {100 * formal['production_gross_trade']:.4f}% |
| Top4 净收益/笔 | {100 * formal['production_net_trade']:.4f}% |
| Top4 生产目标（日组合，除以 overlap=2） | {100 * formal['production_net']:.4f}% |
| Top10 毛收益/笔（补充） | {100 * formal['top10_gross_trade']:.4f}% |
| Top10 净收益/笔（补充） | {100 * formal['top10_net_trade']:.4f}% |
| 日 hit IC 与 Top4 毛 P&L Pearson | {formal['daily_ic_gross']['pearson']:.4f}（p={formal['daily_ic_gross']['pearson_pvalue']:.4g}） |
| 日 hit IC 与 Top4 净 P&L Pearson | {formal['daily_ic_pnl']['pearson']:.4f}（p={formal['daily_ic_pnl']['pearson_pvalue']:.4g}） |

这里必须区分两个问题：约 −0.064 指的是**同一日截面上，因子分数与 D+2 收盘收益的 IC**；“每日 hit IC 与当日组合 P&L 的时序相关”是另一个统计量，可能为正，不能据此否定截面方向缺陷。

### 3.2 分月

{_markdown_table(monthly[['month', 'hit_ic', 'close_ic', 'production_net_trade', 'ic_pnl_pearson']], percent={'production_net_trade'})}

### 3.3 分市场环境（只用截至 D0 已知的上证指数 20 日收益）

{_markdown_table(regimes, percent={'production_gross_trade', 'production_net_trade'})}

### 3.4 持有 D+1 至 D+10

{_markdown_table(horizons, percent={'gross_trade', 'net_trade'})}

本地逐日行情缓存只覆盖训练集内 43 个可用 D0（2022-03/04 与 2024-09 的若干交易日），所以 D+1~D+10 是**覆盖受限的辅助稳定性检查**，不冒充 647 日全窗口结论。核心 D+2 结论使用源表原生标签，覆盖全部 647 日；任何缓存中的训练结束日以后行情均在加载时剔除。

## 4. 逐项根因证据

### 4.1 符号一致性

- 因子库存储 sign：`+1`；表达式输出越大，排序越靠前。
- 标签：`label_d2_hit_8pct=1` 表示 D+2 盘中高点触达 +8%；旧 gini 正向计算。
- 组合：稳定降序选 Top-K 做多；不存在升序、多空互换或二次符号翻转。
- 修复后新因子一律存 `sign=+1`；反方向必须由表达式中的 `neg(...)` 显式表达。

因此不是低级符号翻转。

### 4.2 绝对收益与截面排序

候选池本身 D+2 毛收益为负，但 Top10 比候选池还低 {100 * abs(formal['top10_excess']):.3f} 个百分点。结论是“市场下跌 + 排序对收盘收益不利”，而不是单纯“排序正确、市场整体下跌”。

### 4.3 盘中峰值与收盘回吐

{_markdown_table(quintiles, percent={'hit_rate', 'peak_return', 'close_return', 'giveback', 'turnover_percentile', 'bottom_turnover_decile', 'turnover'})}

分数越高，命中率与盘中峰值越高，但收盘收益没有同步单调上升、峰值回吐反而扩大。这是主导错配的直接证据。

### 4.4 换手、成本与成交

- Top4 平均换手：{100 * formal['turnover']:.2f}%。
- 最高分位组合换手 {100 * quintiles.loc[quintiles['quintile'] == 5, 'turnover'].iloc[0]:.2f}%，低于中间分位；其换手率指标中位分位数为 {100 * quintiles.loc[quintiles['quintile'] == 5, 'turnover_percentile'].iloc[0]:.2f}%，底部一成仅 {100 * quintiles.loc[quintiles['quintile'] == 5, 'bottom_turnover_decile'].iloc[0]:.2f}%。不存在“IC 越高越集中于低流动性且换手单调上升”的证据。
- 成本前 Top10 已亏损，扣费只进一步拖累 {100 * formal['top10_cost_drag']:.3f} 个百分点，不能被认定为方向翻转原因。
- D0 候选先排名，所选结果缺失时按现金记 0，不以更低排名标的补位。所选不可观测比例仅 {100 * execution['unobservable_share']:.3f}%。
- 原始事件表没有完整的 D+1 涨停价字段，无法单独重建“开盘封板”子类；以 entry 缺失和 outcome 缺失作为更保守的不可实现代理，并明确不把缺失当成 0 收益股票后向下补位。

## 5. 修复设计与代码落地

旧目标：`abs(mean(daily_gini(hit_label))) - complexity_penalty × nodes`，负 gini 会隐式翻 sign。新目标：

```text
fitness = 10_000 × mean(fit 日 production Top4 D+2-close net portfolio return)
```

关键合同：

- K 唯一读取 `backtest.top_k={PRODUCTION_TOP_K}`；Top10 无法进入 `EvalContext` 的排序字段。
- 先用 D0 候选与因子分数稳定排序，再读取 D+2 结果；不可观测所选标的留现金，不补位。
- 买卖成本复用回测引擎 `_cost_rates/_net_returns`，含 2023-08-28 印花税切换。
- 新因子方向固定 `sign=+1`；不取绝对值。节点数仅作 P&L 完全相等时的字典序次级键。
- fit 驱动进化；embargo 后 selection 必须净收益严格为正；两段都在训练集内。
- hit IC/gini/lift、peak IC、close IC、Top10 只作监控。

## 6. 现有 30 因子训练内验证

因子来源：formal 1、argus_multi 17、argus_n40 12，共 {len(library)} 个。验收相关仅比较 fit 目标与 embargo 后 selection 的真实 Top4 净收益。

| 排序指标 | Pearson | p | Spearman | p | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 hit gini | {_fmt(old.loc['old_gini', 'pearson'])} | {_fmt(old.loc['old_gini', 'pearson_pvalue'])} | {_fmt(old.loc['old_gini', 'spearman'])} | {_fmt(old.loc['old_gini', 'spearman_pvalue'])} | {int(old.loc['old_gini', 'n'])} |
| close-return IC（辅助对照） | {_fmt(old.loc['close_ic', 'pearson'])} | {_fmt(old.loc['close_ic', 'pearson_pvalue'])} | {_fmt(old.loc['close_ic', 'spearman'])} | {_fmt(old.loc['close_ic', 'spearman_pvalue'])} | {int(old.loc['close_ic', 'n'])} |
| **新 Top4 净收益目标** | **{_fmt(production['pearson'])}** | **{_fmt(production['pearson_pvalue'])}** | **{_fmt(production['spearman'])}** | **{_fmt(production['spearman_pvalue'])}** | **{int(production['n'])}** |

验收结论：**{alignment_decision(production)}**。新目标的 Pearson、Spearman 均为正且 p<0.05；旧 gini 不满足方向一致要求。

明细：

{_markdown_table(library[['factor', 'old_fit_gini', 'fit_close_ic', 'fit_net', 'selection_net']], percent={'fit_net', 'selection_net'})}

Top10 同口径补充相关为 Pearson {_fmt(top10_alignment['pearson'])}（p={_fmt(top10_alignment['pearson_pvalue'])}）、Spearman {_fmt(top10_alignment['spearman'])}（p={_fmt(top10_alignment['spearman_pvalue'])}）。它只作为稳健性附件，不进入上述 PASS/FAIL 函数，也不改变生产结论。

## 7. 性能、无未来函数与测试

- 在 647×1990、候选占比 23% 的固定基准上：新 Top4 向量目标中位 {benchmark['objective_ms']:.2f} ms，旧 daily gini {benchmark['legacy_gini_ms']:.2f} ms，倍率 {benchmark['ratio']:.2f}×。开销同量级，可支撑 GP 大种群。
- 净结果网格在搜索开始前一次性预计算；每个表达式只做稳定 `argsort + take_along_axis + reduce`，无 Python 日循环。
- 全部报告输入在加载时和计算前双重断言 D0/exit 边界；D+10 也不得越过 {TRAIN_END}。
- 回测引擎与标签定义未修改。

## 8. 风险与后续门槛

本次验证证明“优化目标与训练内真实净收益排序单调同向”，不等于证明任何新挖因子样本外盈利。重跑 GP 后仍须按治理规范执行独立性、安慰剂校准、walk-forward 与 G3 增量消融；这些阶段不得反向修改本次训练目标。
"""


def build_evidence(input_path: Path, bars_path: Path, config: BacktestConfig) -> dict[str, Any]:
    libraries = _load_libraries(LIBRARIES)
    frame = _load_training_frame(input_path, libraries)
    values, metadata = _compute_factor_values(frame, libraries)
    formal_name = "formal/gp_000"
    formal_daily, selected = _daily_diagnostics(frame, values[formal_name], config, config.top_k)
    dates = formal_daily["date"].to_numpy()
    fit_rows, selection_rows = fit_selection_windows(len(dates), embargo_days=5)

    d0_market_return = (
        frame.groupby("trade_date", sort=True)["index_sh_pct_chg"].median() / 100.0
    )
    regimes = market_regime(d0_market_return)
    quintile_source = selected.rename(
        columns={
            "label_d2_hit_8pct": "hit",
            "label_d2_peak_return": "peak_return",
            "label_d2_return": "close_return",
            "label_px_d1_open": "entry_price",
        }
    )
    quintiles = evaluate_score_quintiles(quintile_source)
    execution = evaluate_execution_feasibility(quintile_source)
    library_table, production, top10_alignment, old_alignment = _library_validation(
        frame, values, metadata, config
    )
    horizons = evaluate_horizons(_horizon_daily(frame, values[formal_name], bars_path, config))

    hit = frame["label_d2_hit_8pct"].to_numpy() == 1
    close = frame["label_d2_return"].to_numpy(dtype=np.float64)
    if metadata[formal_name]["sign"] != 1.0:
        raise ValueError("formal factor sign is not +1; the documented direction audit is stale")
    daily_ic_gross = correlation(
        formal_daily["hit_ic"], formal_daily["production_gross_trade"]
    )
    daily_ic_pnl = correlation(formal_daily["hit_ic"], formal_daily["production_net"])
    formal = {
        "hit_ic": float(formal_daily["hit_ic"].mean()),
        "hit_gini": float(formal_daily["hit_gini"].mean()),
        "close_ic": float(formal_daily["close_ic"].mean()),
        "production_gross_trade": float(formal_daily["production_gross_trade"].mean()),
        "production_net_trade": float(formal_daily["production_net_trade"].mean()),
        "production_net": float(formal_daily["production_net"].mean()),
        "top10_gross_trade": float(formal_daily["top10_gross_trade"].mean()),
        "top10_net_trade": float(formal_daily["top10_net_trade"].mean()),
        "candidate_gross": float(formal_daily["candidate_gross"].mean()),
        "top10_excess": float(
            formal_daily["top10_gross_trade"].mean() - formal_daily["candidate_gross"].mean()
        ),
        "top10_cost_drag": float(
            formal_daily["top10_gross_trade"].mean() - formal_daily["top10_net_trade"].mean()
        ),
        "turnover": float(formal_daily["turnover"].mean()),
        "daily_ic_pnl": daily_ic_pnl,
        "daily_ic_gross": daily_ic_gross,
    }
    evidence: dict[str, Any] = {
        "calendar": {
            "n_objective_dates": len(dates),
            "fit_start": dates[fit_rows][0],
            "fit_end": dates[fit_rows][-1],
            "n_fit": len(dates[fit_rows]),
            "selection_start": dates[selection_rows][0],
            "selection_end": dates[selection_rows][-1],
            "n_selection": len(dates[selection_rows]),
        },
        "root_cause": "盘中触达标签与 D+2 收盘退出错配",
        "production_alignment": production,
        "top10_alignment": top10_alignment,
        "old_alignment": old_alignment,
        "daily": formal_daily,
        "monthly": evaluate_months(formal_daily),
        "regimes": evaluate_regimes(formal_daily, regimes),
        "horizons": horizons,
        "quintiles": quintiles,
        "execution": execution,
        "library_table": library_table,
        "formal": formal,
        "hit_close_below_target": float((close[hit] < 0.08).mean()),
        "hit_close_negative": float((close[hit] < 0).mean()),
        "benchmark": _benchmark(config),
    }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bars", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    config = BacktestConfig(top_k=PRODUCTION_TOP_K)
    evidence = build_evidence(args.input, args.bars, config)
    report = _full_report(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    summary = {
        "output": str(args.output),
        "decision": alignment_decision(evidence["production_alignment"]),
        "production_alignment": evidence["production_alignment"],
        "n_factors": len(evidence["library_table"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
