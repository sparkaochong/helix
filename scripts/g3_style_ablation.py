#!/usr/bin/env python3
"""Run the formal gp_000 G3 ablation against explicit economic styles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from helix.config import PROJECT_ROOT, BacktestConfig, Config
from helix.data.event_table import build_event_panel
from helix.data.tushare_source import TushareSource
from helix.eval.backtest import _cost_rates, _net_returns, summarize_portfolio_returns
from helix.eval.ic import daily_ic, summarize_ic
from helix.eval.metrics import daily_gini, summarize_daily
from helix.eval.style_neutralize import build_style_design, style_residualize
from helix.gp.library import FactorLibrary, FactorSpec, compute_factors, load_factors

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TRAIN_DATES = 649
TRAIN_DATE_DIGEST = "df8186eafc50efa3e7ae9432e6e6327a333f7050b677f130838a17b03571e381"
TARGET = "label_d2_hit_8pct"
FORMAL_FACTOR = "gp_000"
DEFAULT_SEEDS = (7, 13, 42, 101, 211, 307, 419, 523, 631, 743)
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 20)
DEFAULT_BLOCK_LENGTH = 20
DEFAULT_TOP_K = 10
STYLE_COLUMNS = (
    "log_total_mv",
    "momentum_20d",
    "volatility_20d",
    "turnover_mean_20d",
)
LABEL_COLUMNS = (TARGET, "label_px_d1_open", "label_px_d2_close")
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

DEFAULT_INPUT = PROJECT_ROOT / "data/raw/argus_quant_working.parquet"
DEFAULT_LIBRARY = PROJECT_ROOT / "data/artifacts/argus/event_factors.json"
DEFAULT_PLACEBO = PROJECT_ROOT / "data/artifacts/placebo_ic_distribution.parquet"
DEFAULT_MARKET_CACHE = PROJECT_ROOT / "data/artifacts/g3_style_market.parquet"
DEFAULT_INDUSTRY_CACHE = PROJECT_ROOT / "data/artifacts/g3_sw2021_members.parquet"
DEFAULT_RESULT = PROJECT_ROOT / "data/artifacts/g3_style_ablation.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs/risk/g3_style_ablation.md"


def _date_text(values: pd.Series | np.ndarray | Sequence[object]) -> np.ndarray:
    return np.asarray(values).astype(str)


def _digits(value: str) -> str:
    return "".join(character for character in str(value) if character.isdigit())


def _hyphenated(value: str) -> str:
    digits = _digits(value)
    if len(digits) != 8:
        raise ValueError(f"invalid date {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def validate_seed_contract(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) for seed in values):
        raise ValueError("seeds must be integers")
    unique = tuple(dict.fromkeys(int(seed) for seed in values))
    if len(unique) < 3:
        raise ValueError("at least three unique seeds are required")
    return unique


def validate_training_calendar(
    dates: Sequence[object],
    *,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    expected_count: int = TRAIN_DATES,
    expected_digest: str = TRAIN_DATE_DIGEST,
) -> np.ndarray:
    unique = np.unique(_date_text(dates))
    digest = hashlib.sha256("\n".join(unique).encode()).hexdigest()
    valid = (
        unique.size == expected_count
        and unique.size > 0
        and unique[0] == train_start
        and unique[-1] == train_end
        and digest == expected_digest
    )
    if not valid:
        raise ValueError(
            "training calendar does not match the approved bounds, count, and digest"
        )
    return unique


def split_evaluation_windows(
    frame: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "trade_date" not in frame:
        raise KeyError("frame is missing trade_date")
    normalized = frame.copy()
    normalized["trade_date"] = normalized["trade_date"].astype(str)
    normalized = normalized.sort_values(
        [column for column in ("trade_date", "stock_code") if column in normalized],
        kind="stable",
    )
    train = normalized.loc[
        normalized["trade_date"].between(train_start, train_end)
    ].copy()
    oos = normalized.loc[normalized["trade_date"] > train_end].copy()
    return train.reset_index(drop=True), oos.reset_index(drop=True)


def circular_block_bootstrap_indices(
    n_dates: int, block_length: int, seed: int
) -> np.ndarray:
    if n_dates <= 0 or block_length <= 0:
        raise ValueError("n_dates and block_length must be positive")
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(n_dates / block_length))
    starts = rng.integers(0, n_dates, size=block_count)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_dates).reshape(-1)[:n_dates]


def bootstrap_metric_summary(
    n_dates: int,
    block_length: int,
    seeds: Sequence[int],
    metric: Callable[[np.ndarray], dict[str, float]],
) -> dict[str, dict[str, float | list[float]]]:
    seed_values = validate_seed_contract(seeds)
    runs = [
        metric(circular_block_bootstrap_indices(n_dates, block_length, seed))
        for seed in seed_values
    ]
    keys = tuple(dict.fromkeys(key for run in runs for key in run))
    output: dict[str, dict[str, float | list[float]]] = {}
    for key in keys:
        values = np.asarray([run.get(key, np.nan) for run in runs], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "values": values.tolist(),
        }
    return output


def compute_trailing_styles(
    market: pd.DataFrame, calendar: np.ndarray, window: int = 20
) -> pd.DataFrame:
    required = {
        "trade_date",
        "stock_code",
        "pct_chg",
        "total_mv",
        "turnover_rate_f",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise KeyError(f"market data is missing columns: {missing}")
    if window <= 1:
        raise ValueError("style window must be greater than one")
    frame = market.loc[:, sorted(required)].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("market data contains duplicate date/stock rows")

    dates = np.unique(_date_text(calendar))
    codes = np.array(sorted(frame["stock_code"].unique()), dtype=str)
    indexed = frame.set_index(["trade_date", "stock_code"])

    def pivot(column: str) -> pd.DataFrame:
        return indexed[column].unstack("stock_code").reindex(index=dates, columns=codes)

    returns = pivot("pct_chg") / 100.0
    gross = 1.0 + returns
    log_gross = np.log(gross.where(gross > 0))
    momentum = np.expm1(log_gross.rolling(window, min_periods=window).sum())
    volatility = returns.rolling(window, min_periods=window).std(ddof=1)
    turnover = pivot("turnover_rate_f").rolling(window, min_periods=window).mean()
    total_mv = pivot("total_mv")
    log_total_mv = np.log(total_mv.where(total_mv > 0))

    n_dates, n_codes = log_total_mv.shape
    return pd.DataFrame(
        {
            "trade_date": np.repeat(dates, n_codes),
            "stock_code": np.tile(codes, n_dates),
            "log_total_mv": log_total_mv.to_numpy().reshape(-1),
            "momentum_20d": momentum.to_numpy().reshape(-1),
            "volatility_20d": volatility.to_numpy().reshape(-1),
            "turnover_mean_20d": turnover.to_numpy().reshape(-1),
        }
    )


def forward_return_panel(
    entry_open: np.ndarray,
    exit_close: np.ndarray,
    entry_adjustment: np.ndarray,
    exit_adjustment: np.ndarray,
    exit_dates: np.ndarray,
    window_end: str,
) -> np.ndarray:
    arrays = tuple(
        np.asarray(array)
        for array in (entry_open, exit_close, entry_adjustment, exit_adjustment, exit_dates)
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("forward return inputs must have the same shape")
    entry, exit_, entry_adj, exit_adj, dates = arrays
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (
            exit_.astype(np.float64)
            * exit_adj.astype(np.float64)
            / (entry.astype(np.float64) * entry_adj.astype(np.float64))
            - 1.0
        )
    valid = (
        np.isfinite(entry.astype(np.float64))
        & np.isfinite(exit_.astype(np.float64))
        & np.isfinite(entry_adj.astype(np.float64))
        & np.isfinite(exit_adj.astype(np.float64))
        & (_date_text(dates) <= window_end)
    )
    return np.where(valid, result, np.nan)


def decide_go(
    neutral_icir: float,
    placebo_icir_p95: float,
    raw_net_return: float,
    neutral_net_return: float,
) -> dict[str, object]:
    values = np.asarray(
        [neutral_icir, placebo_icir_p95, raw_net_return, neutral_net_return],
        dtype=np.float64,
    )
    finite = bool(np.isfinite(values).all())
    icir_pass = finite and abs(neutral_icir) > placebo_icir_p95
    direction_pass = (
        finite
        and raw_net_return != 0.0
        and neutral_net_return != 0.0
        and np.sign(raw_net_return) == np.sign(neutral_net_return)
    )
    return {
        "decision": "GO" if icir_pass and direction_pass else "NO-GO",
        "neutral_icir": float(neutral_icir),
        "placebo_icir_p95": float(placebo_icir_p95),
        "icir_pass": bool(icir_pass),
        "raw_net_return": float(raw_net_return),
        "neutral_net_return": float(neutral_net_return),
        "direction_pass": bool(direction_pass),
    }


def validate_formal_library(library: FactorLibrary) -> FactorSpec:
    if library.kind != "event":
        raise ValueError("formal gp_000 library must be an event library")
    if len(library.factors) != 1:
        raise ValueError("formal library must contain exactly one factor")
    factor = library.factors[0]
    if factor.name != FORMAL_FACTOR:
        raise ValueError("formal library factor must be gp_000")
    return factor


def load_placebo_icir_p95(path: str | Path) -> float:
    distribution = pd.read_parquet(path)
    required = {"icir", "train_start", "train_end"}
    if not required <= set(distribution.columns):
        raise ValueError("placebo distribution is missing training metadata")
    if not distribution["train_start"].astype(str).eq(TRAIN_START).all() or not distribution[
        "train_end"
    ].astype(str).eq(TRAIN_END).all():
        raise ValueError("placebo distribution training window does not match the experiment")
    values = distribution["icir"].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("placebo ICIR distribution must be finite and nonempty")
    return float(np.quantile(values, 0.95, method="linear"))


def backtest_top_k(
    score: np.ndarray,
    label: np.ndarray,
    entry_price: np.ndarray,
    exit_price: np.ndarray,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
    *,
    overlap: int = 2,
) -> tuple[dict[str, float], pd.DataFrame]:
    arrays = tuple(
        np.asarray(array)
        for array in (score, label, entry_price, exit_price, candidate_mask)
    )
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("backtest arrays must share one two-dimensional shape")
    if len(dates) != arrays[0].shape[0]:
        raise ValueError("dates must align with the backtest rows")
    if overlap <= 0:
        raise ValueError("overlap must be positive")

    scores, labels, entries, exits, mask = arrays
    candidates = mask.astype(bool) & np.isfinite(scores)
    ranked = np.argsort(np.where(candidates, -scores, np.inf), axis=1, kind="stable")
    cost_dates = np.asarray([_digits(date) for date in dates])
    buy_rate, sell_rates = _cost_rates(config, cost_dates)
    rows: list[dict[str, float]] = []
    executed_returns: list[np.ndarray] = []
    for date_index in range(scores.shape[0]):
        if int(candidates[date_index].sum()) < config.top_k:
            continue
        selected = ranked[date_index, : config.top_k]
        valid_execution = (
            np.isfinite(labels[date_index, selected])
            & np.isfinite(entries[date_index, selected])
            & (entries[date_index, selected] > 0)
            & np.isfinite(exits[date_index, selected])
        )
        executed = selected[valid_execution]
        gross = exits[date_index, executed] / entries[date_index, executed] - 1.0
        net = _net_returns(gross, buy_rate, float(sell_rates[date_index]))
        if net.size:
            executed_returns.append(net)
        observable = candidates[date_index] & np.isfinite(labels[date_index])
        rows.append(
            {
                "date": str(dates[date_index]),
                "n_selected": float(config.top_k),
                "n_executed": float(executed.size),
                "hit_rate": (
                    float(labels[date_index, executed].mean())
                    if executed.size
                    else float("nan")
                ),
                "base_rate": (
                    float(labels[date_index, observable].mean())
                    if observable.any()
                    else float("nan")
                ),
                "portfolio_return": float(net.sum() / config.top_k / overlap),
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        empty = {key: float("nan") for key in REQUIRED_METRICS}
        return empty, daily
    portfolio = daily["portfolio_return"].to_numpy(dtype=np.float64)
    portfolio_summary = summarize_portfolio_returns(portfolio)
    trades = np.concatenate(executed_returns) if executed_returns else np.array([])
    hit_rate = float(daily["hit_rate"].mean())
    base_rate = float(daily["base_rate"].mean())
    metrics = {
        "top10_hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": hit_rate / base_rate if base_rate > 0 else float("nan"),
        "net_return": float(portfolio_summary["final_equity"] - 1.0),
        "net_per_trade": float(trades.mean()) if trades.size else float("nan"),
        "cagr": float(portfolio_summary["cagr"]),
        "sharpe": float(portfolio_summary["sharpe"]),
        "max_drawdown": float(portfolio_summary["max_drawdown"]),
        "n_days": float(len(daily)),
    }
    return metrics, daily


def _predictive_metrics(
    score: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    *,
    min_samples: int = 50,
) -> dict[str, float]:
    ic = summarize_ic(daily_ic(score, label, mask, min_samples=min_samples))
    gini = summarize_daily(daily_gini(score, label, mask, min_samples=min_samples))
    return {
        "ic_mean": float(ic["ic_mean"]),
        "icir": float(ic["icir"]),
        "gini": float(gini["mean"]),
        "coverage": float(ic["coverage"]),
    }


def _evaluate_arm(
    score: np.ndarray,
    label: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
    mask: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
) -> dict[str, float]:
    predictive = _predictive_metrics(score, label, mask)
    economic, _ = backtest_top_k(score, label, entry, exit_, mask, dates, config)
    result = predictive | economic
    missing = [key for key in REQUIRED_METRICS if key not in result]
    if missing:
        raise AssertionError(f"arm metrics are incomplete: {missing}")
    return result


def _decay_metrics(
    score: np.ndarray,
    forward_return: np.ndarray,
    mask: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
) -> dict[str, float]:
    usable = mask & np.isfinite(score) & np.isfinite(forward_return)
    ic = summarize_ic(daily_ic(score, forward_return, usable, min_samples=30))
    candidates = usable & np.isfinite(score)
    ranked = np.argsort(np.where(candidates, -score, np.inf), axis=1, kind="stable")
    buy, sells = _cost_rates(config, np.asarray([_digits(date) for date in dates]))
    daily_net: list[float] = []
    for date_index in range(score.shape[0]):
        if int(candidates[date_index].sum()) < config.top_k:
            continue
        selected = ranked[date_index, : config.top_k]
        gross = forward_return[date_index, selected]
        net = _net_returns(gross, buy, float(sells[date_index]))
        daily_net.extend(net.tolist())
    return {
        "ic_mean": float(ic["ic_mean"]),
        "icir": float(ic["icir"]),
        "net_per_trade": float(np.mean(daily_net)) if daily_net else float("nan"),
    }


@dataclass(frozen=True)
class PriceLookup:
    dates: np.ndarray
    codes: np.ndarray
    adjusted_open: np.ndarray
    adjusted_close: np.ndarray
    date_positions: dict[str, int]
    code_positions: dict[str, int]


def _price_lookup(market: pd.DataFrame, calendar: np.ndarray) -> PriceLookup:
    required = {"trade_date", "stock_code", "open", "close", "pct_chg"}
    if not required <= set(market):
        raise KeyError(f"market price data is missing: {sorted(required - set(market))}")
    frame = market.loc[:, sorted(required)].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    dates = np.unique(_date_text(calendar))
    codes = np.array(sorted(frame["stock_code"].unique()), dtype=str)
    indexed = frame.set_index(["trade_date", "stock_code"])

    def pivot(column: str) -> np.ndarray:
        return (
            indexed[column]
            .unstack("stock_code")
            .reindex(index=dates, columns=codes)
            .to_numpy(dtype=np.float64)
        )

    raw_open = pivot("open")
    raw_close = pivot("close")
    daily_gross = 1.0 + pivot("pct_chg") / 100.0
    adjusted_close = np.cumprod(np.where(np.isfinite(daily_gross), daily_gross, 1.0), axis=0)
    adjusted_open = adjusted_close * raw_open / raw_close
    adjusted_close = np.where(np.isfinite(raw_close), adjusted_close, np.nan)
    adjusted_open = np.where(np.isfinite(raw_open), adjusted_open, np.nan)
    return PriceLookup(
        dates=dates,
        codes=codes,
        adjusted_open=adjusted_open,
        adjusted_close=adjusted_close,
        date_positions={date: index for index, date in enumerate(dates)},
        code_positions={code: index for index, code in enumerate(codes)},
    )


def _forward_returns_for_events(
    keys: pd.DataFrame,
    prices: PriceLookup,
    horizon: int,
    window_end: str,
) -> np.ndarray:
    if horizon < 1:
        raise ValueError("holding horizon must be positive")
    d0_positions = np.asarray(
        [prices.date_positions.get(str(date), -1) for date in keys["trade_date"]],
        dtype=int,
    )
    code_positions = np.asarray(
        [prices.code_positions.get(str(code), -1) for code in keys["stock_code"]],
        dtype=int,
    )
    entry_positions = d0_positions + 1
    exit_positions = d0_positions + horizon
    in_bounds = (
        (d0_positions >= 0)
        & (code_positions >= 0)
        & (entry_positions < len(prices.dates))
        & (exit_positions < len(prices.dates))
    )
    safe_entry = np.clip(entry_positions, 0, len(prices.dates) - 1)
    safe_exit = np.clip(exit_positions, 0, len(prices.dates) - 1)
    safe_codes = np.clip(code_positions, 0, len(prices.codes) - 1)
    entry = prices.adjusted_open[safe_entry, safe_codes]
    exit_ = prices.adjusted_close[safe_exit, safe_codes]
    exit_dates = prices.dates[safe_exit]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = exit_ / entry - 1.0
    valid = in_bounds & np.isfinite(entry) & np.isfinite(exit_) & (exit_dates <= window_end)
    return np.where(valid, returns, np.nan)


def _align_industries(
    keys: pd.DataFrame, members: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, str]]:
    required = {"index_code", "industry_name", "stock_code", "in_date", "out_date"}
    if not required <= set(members):
        raise KeyError(f"industry cache is missing: {sorted(required - set(members))}")
    taxonomy = (
        members[["index_code", "industry_name"]]
        .drop_duplicates()
        .sort_values("index_code")
        .reset_index(drop=True)
    )
    taxonomy["industry_code"] = np.arange(len(taxonomy), dtype=int)
    intervals = members.merge(taxonomy, on=["index_code", "industry_name"], how="left")
    joined = keys.reset_index(names="_event_row").merge(
        intervals,
        on="stock_code",
        how="left",
        sort=False,
    )
    date = joined["trade_date"].astype(str)
    active = (joined["in_date"].astype(str) <= date) & (
        joined["out_date"].isna() | (date <= joined["out_date"].astype(str))
    )
    active_rows = joined.loc[active, ["_event_row", "industry_code"]]
    if active_rows.duplicated("_event_row").any():
        raise ValueError("overlapping SW2021 memberships found for an event row")
    aligned = keys.reset_index(drop=True).copy()
    aligned["industry_code"] = np.nan
    aligned.loc[
        active_rows["_event_row"].to_numpy(dtype=int), "industry_code"
    ] = active_rows["industry_code"].to_numpy(dtype=float)
    names = dict(
        zip(
            taxonomy["industry_code"].astype(int),
            taxonomy["industry_name"].astype(str),
            strict=True,
        )
    )
    return aligned, names


def _evaluate_window(
    frame: pd.DataFrame,
    library: FactorLibrary,
    styles: pd.DataFrame,
    members: pd.DataFrame,
    prices: PriceLookup,
    window_end: str,
    seeds: Sequence[int],
    block_length: int,
    horizons: Sequence[int],
    config: BacktestConfig,
    *,
    with_bootstrap: bool,
) -> dict[str, object]:
    source_panel = build_event_panel(
        frame,
        library.field_names,
        list(LABEL_COLUMNS),
    )
    names, values = compute_factors(library, source_panel.fields)
    if names != [FORMAL_FACTOR]:
        raise ValueError("formal factor replay did not return exactly gp_000")
    columns = {"factor": values[..., 0]}
    columns.update({name: source_panel.f64(name) for name in LABEL_COLUMNS})
    aligned = source_panel.to_long(columns)
    aligned = aligned.merge(styles, on=["trade_date", "stock_code"], how="left", sort=False)
    industries, industry_names = _align_industries(
        aligned[["trade_date", "stock_code"]], members
    )
    aligned["industry_code"] = industries["industry_code"].to_numpy()
    for horizon in horizons:
        aligned[f"forward_{horizon}"] = _forward_returns_for_events(
            aligned[["trade_date", "stock_code"]], prices, horizon, window_end
        )

    feature_columns = ["factor", *STYLE_COLUMNS, "industry_code"]
    outcome_columns = [*LABEL_COLUMNS, *[f"forward_{horizon}" for horizon in horizons]]
    panel = build_event_panel(aligned, feature_columns, outcome_columns)
    raw = panel.f64("factor")
    continuous = np.stack([panel.f64(name) for name in STYLE_COLUMNS], axis=2)
    industry = panel.f64("industry_code")
    common_mask = (
        panel.occupied
        & np.isfinite(raw)
        & np.isfinite(continuous).all(axis=2)
        & np.isfinite(industry)
    )
    levels = np.arange(len(industry_names), dtype=float)
    neutral = style_residualize(
        raw,
        continuous,
        industry,
        common_mask,
        industry_levels=levels,
    )
    label = panel.f64(TARGET)
    entry = panel.f64("label_px_d1_open")
    exit_ = panel.f64("label_px_d2_close")
    arms = {"raw": raw, "style_neutral": neutral}
    deterministic = {
        name: _evaluate_arm(score, label, entry, exit_, common_mask, panel.dates, config)
        for name, score in arms.items()
    }

    bootstrap: dict[str, object] = {}
    if with_bootstrap:
        for name, score in arms.items():
            bootstrap[name] = bootstrap_metric_summary(
                len(panel.dates),
                block_length,
                seeds,
                lambda index, score=score: _evaluate_arm(
                    score[index],
                    label[index],
                    entry[index],
                    exit_[index],
                    common_mask[index],
                    panel.dates[index],
                    config,
                ),
            )

    decay: list[dict[str, object]] = []
    for horizon in horizons:
        target = panel.f64(f"forward_{horizon}")
        for name, score in arms.items():
            point = _decay_metrics(score, target, common_mask, panel.dates, config)
            row: dict[str, object] = {"horizon": horizon, "arm": name, **point}
            if with_bootstrap:
                row["bootstrap"] = bootstrap_metric_summary(
                    len(panel.dates),
                    block_length,
                    seeds,
                    lambda index, score=score, target=target: _decay_metrics(
                        score[index],
                        target[index],
                        common_mask[index],
                        panel.dates[index],
                        config,
                    ),
                )
            decay.append(row)

    design, valid = build_style_design(
        continuous,
        industry,
        common_mask & np.isfinite(neutral),
        industry_levels=levels,
    )
    exposure = np.matmul(
        design.transpose(0, 2, 1), np.nan_to_num(neutral)[..., None]
    )
    scale = np.maximum(valid.sum(axis=1), 1)[:, None, None]
    orthogonality = {
        "max_abs_exposure": float(np.max(np.abs(exposure / scale))),
        "style_complete_rows": int(common_mask.sum()),
        "total_rows": int(panel.occupied.sum()),
        "style_coverage": float(common_mask.sum() / panel.occupied.sum()),
        "industry_count": len(industry_names),
    }
    return {
        "dates": panel.dates.tolist(),
        "deterministic": deterministic,
        "bootstrap": bootstrap,
        "decay": decay,
        "orthogonality": orthogonality,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无可用数据。_"
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.8g}" if np.isfinite(value) else "NaN")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator, *rows])


def render_report(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    decision = payload["decision"]
    deterministic = payload["deterministic"]
    bootstrap = payload["bootstrap"]
    assert isinstance(metadata, dict)
    assert isinstance(decision, dict)
    assert isinstance(deterministic, dict)
    assert isinstance(bootstrap, dict)

    metric_rows = []
    for arm in ("raw", "style_neutral"):
        values = deterministic[arm]
        metric_rows.append({"arm": arm, **{key: values[key] for key in REQUIRED_METRICS}})
    seed_rows = []
    for arm in ("raw", "style_neutral"):
        values = bootstrap.get(arm, {})
        row: dict[str, object] = {"arm": arm}
        for key in REQUIRED_METRICS:
            summary = values.get(key, {})
            row[key] = (
                f"{summary.get('mean', float('nan')):.6g} ± "
                f"{summary.get('std', float('nan')):.3g}"
            )
        seed_rows.append(row)
    decay_rows = []
    for item in payload.get("decay", []):
        row = {
            "horizon": item["horizon"],
            "arm": item["arm"],
            "ic_mean": item["ic_mean"],
            "icir": item["icir"],
            "net_per_trade": item["net_per_trade"],
        }
        if "bootstrap" in item:
            for key in ("ic_mean", "icir", "net_per_trade"):
                row[f"{key}_seed_std"] = item["bootstrap"][key]["std"]
        decay_rows.append(row)

    oos = payload.get("oos") or {}
    oos_rows = [
        {"arm": arm, **{key: values.get(key, np.nan) for key in REQUIRED_METRICS}}
        for arm, values in oos.items()
    ]
    return f"""# G3 Style Ablation

**{decision['decision']}**。本结论只读取训练窗口确定性统计量；样本外结果不参与 GO/NO-GO。

## 判定规则与输入

- 训练窗口：`{metadata['train_start']}` 至 `{metadata['train_end']}`
- 中性因子 `|ICIR| = {abs(float(decision['neutral_icir'])):.12g}`
- 安慰剂 ICIR p95：`{float(decision['placebo_icir_p95']):.12g}`
- ICIR 条件：`{decision['icir_pass']}`（严格大于，不接受相等）
- 原始/中性净收益每笔：`{float(decision['raw_net_return']):.8g}` / `{float(decision['neutral_net_return']):.8g}`
- 收益同向条件：`{decision['direction_pass']}`

## 数据与泄漏控制

正式因子为 `data/artifacts/argus/event_factors.json` 中唯一的 `gp_000`。五类风格为
对数总市值、申万 2021 一级行业哑变量、20 日动量、20 日波动率和 20 日平均自由
流通换手率。每个 D0 的风格只使用 D0 及此前 19 个市场交易日；每日 QR 回归相互
独立。训练窗末日之后的标签、退出价、衰减目标和样本外指标均不进入判定函数。

## 确定性核心指标

{_markdown_table(pd.DataFrame(metric_rows))}

## 种子稳健性（均值 ± 样本标准差）

循环移动块 bootstrap；种子：`{metadata['seeds']}`，块长：`{metadata.get('block_length', 20)}`。

{_markdown_table(pd.DataFrame(seed_rows))}

## 中性化正交审计

{_markdown_table(pd.DataFrame([payload.get('orthogonality', {})]))}

## 因子收益衰减

持仓期限从 D+1 开盘计至 D+h 收盘。训练窗内任何退出日在 `{metadata['train_end']}`
之后的观察均为 NaN，不借用样本外价格补齐。

{_markdown_table(pd.DataFrame(decay_rows))}

## 样本外附录

以下数据仅作对照参考，**不参与 GO/NO-GO**，也不能覆盖训练窗结论。

{_markdown_table(pd.DataFrame(oos_rows))}

## 局限

- 行业分类使用 SW2021 的历史 `in_date/out_date` 区间；无有效行业映射的行不插补。
- 移动块 bootstrap 衡量日期采样不确定性；确定性全训练窗估计才是门控输入。
- 本实验回答风格暴露问题，不关闭 D6 冲击成本或 D13 递延资金占用等独立风险。

## 复现命令

```bash
{metadata['command']}
```

结果参数与摘要：`{metadata.get('result_path', 'data/artifacts/g3_style_ablation.json')}`。
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _trade_calendar(source: TushareSource, start: str, end: str) -> np.ndarray:
    calendar = source._call(
        "trade_cal",
        exchange="SSE",
        start_date=_digits(start),
        end_date=_digits(end),
        fields="cal_date,is_open",
    )
    open_dates = calendar.loc[
        pd.to_numeric(calendar["is_open"], errors="coerce") == 1, "cal_date"
    ]
    return np.array(sorted(_hyphenated(date) for date in open_dates.astype(str)))


def _refresh_market_cache(
    path: Path,
    event_dates: np.ndarray,
    event_codes: set[str],
    max_horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    source = TushareSource(Config.load())
    start = (pd.Timestamp(event_dates.min()) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    requested_end = pd.Timestamp(event_dates.max()) + pd.Timedelta(days=max_horizon * 2 + 15)
    end = min(requested_end, pd.Timestamp.today()).strftime("%Y-%m-%d")
    calendar = _trade_calendar(source, start, end)
    first = int(np.searchsorted(calendar, event_dates.min()))
    last = min(int(np.searchsorted(calendar, event_dates.max(), side="right")) + max_horizon, len(calendar))
    required_dates = calendar[max(first - 19, 0) : last]
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(existing["trade_date"].astype(str)) if not existing.empty else set()
    pending: list[pd.DataFrame] = []
    missing_dates = [date for date in required_dates if date not in have]
    for number, date in enumerate(missing_dates, start=1):
        daily = source._call(
            "daily",
            trade_date=_digits(date),
            fields="ts_code,trade_date,open,close,pct_chg",
        ).rename(columns={"ts_code": "stock_code"})
        basic = source._call(
            "daily_basic",
            trade_date=_digits(date),
            fields="ts_code,trade_date,total_mv,turnover_rate_f",
        ).rename(columns={"ts_code": "stock_code"})
        merged = daily.merge(
            basic,
            on=["stock_code", "trade_date"],
            how="outer",
            validate="one_to_one",
        )
        merged = merged.loc[merged["stock_code"].astype(str).isin(event_codes)].copy()
        merged["trade_date"] = date
        pending.append(merged)
        if len(pending) >= 50 or number == len(missing_dates):
            market = pd.concat([existing, *pending], ignore_index=True)
            market = market.drop_duplicates(["trade_date", "stock_code"], keep="last")
            market = market.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
            _atomic_parquet(path, market)
            existing = market
            pending.clear()
            print(
                f"style market cache: {number}/{len(missing_dates)} missing dates fetched",
                flush=True,
            )
    if existing.empty:
        raise ValueError("style market cache refresh returned no rows")
    return existing, calendar


def _refresh_industry_cache(path: Path) -> pd.DataFrame:
    source = TushareSource(Config.load())
    taxonomy = source._call(
        "index_classify",
        level="L1",
        src="SW2021",
        fields="index_code,industry_name,level,src",
    )
    frames = []
    for row in taxonomy.itertuples(index=False):
        members = source._call(
            "index_member",
            index_code=row.index_code,
            fields="index_code,con_code,in_date,out_date,is_new",
        )
        members["industry_name"] = row.industry_name
        frames.append(members)
    result = pd.concat(frames, ignore_index=True).rename(columns={"con_code": "stock_code"})
    for column in ("in_date", "out_date"):
        present = result[column].notna()
        result.loc[present, column] = result.loc[present, column].astype(str).map(_hyphenated)
    result = result.sort_values(["index_code", "stock_code", "in_date"]).reset_index(drop=True)
    _atomic_parquet(path, result)
    return result


def _load_caches(
    market_cache: Path,
    industry_cache: Path,
    event_dates: np.ndarray,
    event_codes: set[str],
    max_horizon: int,
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if refresh:
        market, calendar = _refresh_market_cache(
            market_cache, event_dates, event_codes, max_horizon
        )
        members = _refresh_industry_cache(industry_cache)
        return market, members, calendar
    if not market_cache.is_file() or not industry_cache.is_file():
        raise FileNotFoundError("style caches are absent; rerun with --refresh-style-cache")
    market = pd.read_parquet(market_cache)
    members = pd.read_parquet(industry_cache)
    market["trade_date"] = market["trade_date"].astype(str)
    calendar = np.array(sorted(market["trade_date"].unique()), dtype=str)
    missing_event_dates = sorted(set(event_dates) - set(calendar))
    first = int(np.searchsorted(calendar, event_dates.min()))
    if missing_event_dates or first < 19:
        raise ValueError("style cache does not cover every event date plus 19 lookback sessions")
    return market, members, calendar


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_experiment(
    *,
    input_path: Path = DEFAULT_INPUT,
    library_path: Path = DEFAULT_LIBRARY,
    placebo_path: Path = DEFAULT_PLACEBO,
    market_cache: Path = DEFAULT_MARKET_CACHE,
    industry_cache: Path = DEFAULT_INDUSTRY_CACHE,
    result_path: Path = DEFAULT_RESULT,
    report_path: Path = DEFAULT_REPORT,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    top_k: int = DEFAULT_TOP_K,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    refresh_style_cache: bool = False,
) -> dict[str, object]:
    seed_values = validate_seed_contract(seeds)
    if block_length <= 0 or top_k <= 0:
        raise ValueError("block length and top_k must be positive")
    horizon_values = tuple(sorted(set(int(value) for value in horizons)))
    if not horizon_values or horizon_values[0] < 1:
        raise ValueError("horizons must be positive")
    library = load_factors(library_path)
    validate_formal_library(library)
    columns = ["trade_date", "stock_code", *library.field_names, *LABEL_COLUMNS]
    frame = pd.read_parquet(input_path, columns=list(dict.fromkeys(columns)))
    train, oos = split_evaluation_windows(frame)
    training_dates = validate_training_calendar(train["trade_date"])
    if train.empty:
        raise ValueError("training window is empty")

    event_dates = np.unique(frame["trade_date"].astype(str))
    event_codes = set(frame["stock_code"].astype(str))
    market, members, calendar = _load_caches(
        market_cache,
        industry_cache,
        event_dates,
        event_codes,
        max(horizon_values),
        refresh=refresh_style_cache,
    )
    styles = compute_trailing_styles(market, calendar)
    prices = _price_lookup(market, calendar)
    config = BacktestConfig(top_k=top_k)
    train_result = _evaluate_window(
        train,
        library,
        styles,
        members,
        prices,
        TRAIN_END,
        seed_values,
        block_length,
        horizon_values,
        config,
        with_bootstrap=True,
    )
    placebo_p95 = load_placebo_icir_p95(placebo_path)
    raw_metrics = train_result["deterministic"]["raw"]
    neutral_metrics = train_result["deterministic"]["style_neutral"]
    decision = decide_go(
        neutral_metrics["icir"],
        placebo_p95,
        raw_metrics["net_per_trade"],
        neutral_metrics["net_per_trade"],
    )

    oos_metrics: dict[str, object] = {}
    if not oos.empty:
        oos_result = _evaluate_window(
            oos,
            library,
            styles,
            members,
            prices,
            str(oos["trade_date"].max()),
            seed_values,
            block_length,
            horizon_values,
            config,
            with_bootstrap=False,
        )
        oos_metrics = oos_result["deterministic"]

    command = (
        ".venv/bin/python scripts/g3_style_ablation.py --cache-only "
        f"--seeds {','.join(map(str, seed_values))} --bootstrap-block-length {block_length} "
        f"--top-k {top_k} --horizons {','.join(map(str, horizon_values))}"
    )
    payload: dict[str, object] = {
        "metadata": {
            "train_start": str(training_dates[0]),
            "train_end": str(training_dates[-1]),
            "n_train_dates": int(training_dates.size),
            "seeds": list(seed_values),
            "block_length": block_length,
            "top_k": top_k,
            "horizons": list(horizon_values),
            "input_path": str(input_path),
            "library_path": str(library_path),
            "library_sha256": _sha256(library_path),
            "market_cache_sha256": _sha256(market_cache),
            "industry_cache_sha256": _sha256(industry_cache),
            "result_path": str(result_path),
            "command": command,
        },
        "decision": decision,
        "placebo_icir_p95": placebo_p95,
        "deterministic": train_result["deterministic"],
        "bootstrap": train_result["bootstrap"],
        "orthogonality": train_result["orthogonality"],
        "decay": train_result["decay"],
        "oos": oos_metrics,
    }
    report = render_report(payload)
    _atomic_text(result_path, json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    _atomic_text(report_path, report)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--placebo", type=Path, default=DEFAULT_PLACEBO)
    parser.add_argument("--market-cache", type=Path, default=DEFAULT_MARKET_CACHE)
    parser.add_argument("--industry-cache", type=Path, default=DEFAULT_INDUSTRY_CACHE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--refresh-style-cache", action="store_true")
    cache.add_argument("--cache-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_experiment(
        input_path=args.input,
        library_path=args.library,
        placebo_path=args.placebo,
        market_cache=args.market_cache,
        industry_cache=args.industry_cache,
        result_path=args.result,
        report_path=args.report,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value.strip()),
        block_length=args.bootstrap_block_length,
        top_k=args.top_k,
        horizons=tuple(int(value) for value in args.horizons.split(",") if value.strip()),
        refresh_style_cache=args.refresh_style_cache,
    )
    print(f"{payload['decision']['decision']}: wrote {args.report} and {args.result}")


if __name__ == "__main__":
    main()
