#!/usr/bin/env python3
"""Reproducible training-window audit for the formal ``gp_000`` factor."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from helix.config import BacktestConfig
from helix.data.event_table import build_event_panel
from helix.eval.backtest import _cost_rates, _net_returns, summarize_portfolio_returns
from helix.eval.ic import daily_ic, summarize_ic
from helix.eval.style_neutralize import build_style_design, style_residualize
from helix.gp.library import FactorLibrary, FactorSpec

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TRAIN_DATES = 649
FORMAL_FACTOR = "gp_000"
FORMAL_EXPRESSION = (
    "add(add(stock_intra_amp_d1d3_mean, "
    "div(stock_vwap_dev_d1, vol_burst_count_20d)), stock_intra_amp_d0)"
)
STYLE_COLUMNS = (
    "log_total_mv",
    "momentum_20d",
    "volatility_20d",
    "turnover_mean_20d",
)


@dataclass(frozen=True)
class PriceLookup:
    """Raw and point-in-time adjusted market arrays on one trading calendar."""

    dates: np.ndarray
    codes: np.ndarray
    raw_open: np.ndarray
    raw_high: np.ndarray
    raw_close: np.ndarray
    adj_factor: np.ndarray
    hfq_open: np.ndarray
    hfq_high: np.ndarray
    hfq_close: np.ndarray
    ex_right: np.ndarray
    date_positions: dict[str, int]
    code_positions: dict[str, int]


@dataclass(frozen=True)
class OutputPaths:
    """Tracked and machine-readable outputs emitted from one evidence payload."""

    report: Path
    json: Path
    daily: Path
    equity_svg: Path
    decay_svg: Path


def _hyphenated(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) != 8:
        raise ValueError(f"invalid date {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def outcome_complete_dates(
    d0_dates: Sequence[str] | np.ndarray,
    calendar: Sequence[str] | np.ndarray,
    horizon: int,
    train_end: str = TRAIN_END,
) -> np.ndarray:
    """Return D0 sessions whose D+h exit exists on or before ``train_end``."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    dates = np.asarray(d0_dates).astype(str)
    sessions = np.asarray(calendar).astype(str)
    if sessions.size == 0:
        return np.array([], dtype=str)
    if np.any(sessions[1:] <= sessions[:-1]):
        raise ValueError("calendar must be strictly increasing and unique")

    positions = np.searchsorted(sessions, dates)
    safe_positions = np.clip(positions, 0, len(sessions) - 1)
    exits = positions + horizon
    safe_exits = np.clip(exits, 0, len(sessions) - 1)
    valid = (
        (positions < len(sessions))
        & (sessions[safe_positions] == dates)
        & (exits < len(sessions))
        & (sessions[safe_exits] <= train_end)
    )
    return dates[valid]


def validate_formal_factor(library: FactorLibrary) -> FactorSpec:
    """Fail closed if the persisted formal factor contract has changed."""
    if library.kind != "event" or len(library.factors) != 1:
        raise ValueError("formal library must contain exactly one event factor")
    factor = library.factors[0]
    if factor.name != FORMAL_FACTOR:
        raise ValueError("formal factor identity changed")
    if factor.expression != FORMAL_EXPRESSION:
        raise ValueError("formal factor expression changed")
    if factor.sign != 1.0:
        raise ValueError("formal factor direction changed")
    return factor


def build_price_lookup(
    market: pd.DataFrame,
    calendar: Sequence[str] | np.ndarray,
    codes: Sequence[str] | np.ndarray,
) -> PriceLookup:
    """Pivot date-local raw prices and adjustment factors without future scaling."""
    required = {"trade_date", "ts_code", "open", "high", "close", "adj_factor"}
    missing = required - set(market.columns)
    if missing:
        raise KeyError(f"market cache is missing: {sorted(missing)}")
    frame = market.loc[:, sorted(required)].copy()
    frame["trade_date"] = frame["trade_date"].map(_hyphenated)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market cache contains duplicate date/stock rows")

    dates = np.asarray([_hyphenated(value) for value in calendar], dtype=str)
    names = np.asarray(sorted({str(code) for code in codes}), dtype=str)
    if dates.size == 0 or names.size == 0:
        raise ValueError("calendar and codes must be nonempty")
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("calendar must be strictly increasing and unique")

    def pivot(column: str) -> np.ndarray:
        return (
            frame.pivot(index="trade_date", columns="ts_code", values=column)
            .reindex(index=dates, columns=names)
            .to_numpy(dtype=np.float64)
        )

    raw_open = pivot("open")
    raw_high = pivot("high")
    raw_close = pivot("close")
    adj_factor = pivot("adj_factor")
    ex_right = np.zeros(adj_factor.shape, dtype=bool)
    ex_right[1:] = (
        np.isfinite(adj_factor[1:])
        & np.isfinite(adj_factor[:-1])
        & ~np.isclose(adj_factor[1:], adj_factor[:-1], rtol=0.0, atol=1e-12)
    )
    return PriceLookup(
        dates=dates,
        codes=names,
        raw_open=raw_open,
        raw_high=raw_high,
        raw_close=raw_close,
        adj_factor=adj_factor,
        hfq_open=raw_open * adj_factor,
        hfq_high=raw_high * adj_factor,
        hfq_close=raw_close * adj_factor,
        ex_right=ex_right,
        date_positions={date: index for index, date in enumerate(dates)},
        code_positions={code: index for index, code in enumerate(names)},
    )


def align_event_prices(
    events: pd.DataFrame,
    prices: PriceLookup,
    horizon: int,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """Attach raw/HFQ D+1-entry to D+h-exit prices to event rows."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = {"trade_date", "stock_code"}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"events are missing: {sorted(missing)}")

    work = events.copy()
    work["trade_date"] = work["trade_date"].map(_hyphenated)
    date_positions = work["trade_date"].map(prices.date_positions)
    code_positions = work["stock_code"].astype(str).map(prices.code_positions)
    if date_positions.isna().any() or code_positions.isna().any():
        raise ValueError("event keys are absent from the market cache")

    d0 = date_positions.to_numpy(dtype=int)
    code = code_positions.to_numpy(dtype=int)
    entry = d0 + 1
    exit_ = d0 + horizon
    in_bounds = (entry < len(prices.dates)) & (exit_ < len(prices.dates))
    safe_entry = np.clip(entry, 0, len(prices.dates) - 1)
    safe_exit = np.clip(exit_, 0, len(prices.dates) - 1)
    work["entry_date"] = prices.dates[safe_entry]
    work["exit_date"] = prices.dates[safe_exit]
    work["raw_entry"] = prices.raw_open[safe_entry, code]
    work["raw_exit_high"] = prices.raw_high[safe_exit, code]
    work["raw_exit"] = prices.raw_close[safe_exit, code]
    work["entry_adj_factor"] = prices.adj_factor[safe_entry, code]
    work["exit_adj_factor"] = prices.adj_factor[safe_exit, code]
    work["hfq_entry"] = prices.hfq_open[safe_entry, code]
    work["hfq_exit_high"] = prices.hfq_high[safe_exit, code]
    work["hfq_exit"] = prices.hfq_close[safe_exit, code]
    with np.errstate(invalid="ignore", divide="ignore"):
        work["raw_return"] = work["raw_exit"] / work["raw_entry"] - 1.0
        work["hfq_return"] = work["hfq_exit"] / work["hfq_entry"] - 1.0
    work["d0_ex_right"] = prices.ex_right[d0, code]
    work["entry_ex_right"] = prices.ex_right[safe_entry, code]
    work["exit_ex_right"] = prices.ex_right[safe_exit, code]
    work["holding_ex_right"] = [
        bool(prices.ex_right[start : stop + 1, column].any())
        for start, stop, column in zip(entry, exit_, code, strict=True)
    ]
    valid = in_bounds & (work["exit_date"].to_numpy(dtype=str) <= train_end)
    return work.loc[valid].reset_index(drop=True)


def audit_adjustment_chain(
    events: pd.DataFrame,
    prices: PriceLookup,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare persisted raw event labels with reconstructed raw and HFQ outcomes."""
    aligned = align_event_prices(events, prices, horizon=2)
    raw_price_checks = (
        ("label_px_d1_open", "raw_entry"),
        ("label_px_d2_high", "raw_exit_high"),
        ("label_px_d2_close", "raw_exit"),
    )
    price_matches = all(
        np.allclose(
            aligned[label].to_numpy(dtype=float),
            aligned[reconstructed].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        )
        for label, reconstructed in raw_price_checks
    )
    if not price_matches:
        raise ValueError("persisted event prices do not match the raw market cache")

    aligned["return_delta"] = aligned["hfq_return"] - aligned["raw_return"]
    aligned["raw_hit"] = aligned["label_d2_hit_8pct"].astype(bool)
    aligned["hfq_hit"] = aligned["hfq_exit_high"] >= aligned["hfq_entry"] * 1.08
    raw_return_matches = np.isclose(
        aligned["label_d2_return"],
        aligned["raw_return"],
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )
    return_mismatch = ~np.isclose(
        aligned["raw_return"],
        aligned["hfq_return"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    summary: dict[str, object] = {
        "event_prices_match_raw": price_matches,
        "event_returns_match_raw": bool(raw_return_matches.all()),
        "return_mismatch_count": int(return_mismatch.sum()),
        "hit_flip_count": int((aligned["raw_hit"] != aligned["hfq_hit"]).sum()),
        "holding_ex_right_count": int(aligned["holding_ex_right"].sum()),
        "mean_return_delta": float(aligned["return_delta"].mean()),
        "max_abs_return_delta": float(aligned["return_delta"].abs().max()),
    }
    return summary, aligned


def apply_cost_by_d0(
    returns: Sequence[float] | np.ndarray,
    d0_dates: Sequence[str] | np.ndarray,
    config: BacktestConfig,
) -> np.ndarray:
    """Apply the canonical date-sensitive, two-sided cost model per trade."""
    values = np.asarray(returns, dtype=np.float64)
    dates = np.asarray(d0_dates).astype(str)
    if values.shape != dates.shape:
        raise ValueError("returns and d0_dates must have the same shape")
    buy, sell = _cost_rates(config, dates)
    return _net_returns(values, buy, sell)


def evaluate_quintiles(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Calculate equal-count daily factor quintiles, Q1 low through Q5 high."""
    required = {"trade_date", "factor_score", "gross_return"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"quintile frame is missing: {sorted(missing)}")
    work = frame.loc[
        np.isfinite(frame["factor_score"]) & np.isfinite(frame["gross_return"])
    ].copy()
    work["quintile"] = work.groupby("trade_date")["factor_score"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            5,
            labels=False,
        )
        + 1
    )
    work["net_return"] = apply_cost_by_d0(
        work["gross_return"].to_numpy(dtype=float),
        work["trade_date"].to_numpy(dtype=str),
        config,
    )
    aggregations: dict[str, tuple[str, str]] = {
        "n": ("factor_score", "size"),
        "n_dates": ("trade_date", "nunique"),
        "gross_return": ("gross_return", "mean"),
        "net_return": ("net_return", "mean"),
    }
    if "hit_hfq" in work:
        aggregations["hit_rate"] = ("hit_hfq", "mean")
    return (
        work.groupby("quintile", observed=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values("quintile")
        .reset_index(drop=True)
    )


def evaluate_top_k_book(
    frame: pd.DataFrame,
    config: BacktestConfig,
    *,
    gross: bool,
    overlap: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a fixed-selection Top-K book with missing outcomes held as cash."""
    if overlap < 1:
        raise ValueError("overlap must be positive")
    required = {"trade_date", "factor_score", "gross_return"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Top-K frame is missing: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    executed_trades: list[float] = []
    for date, block in frame.groupby("trade_date", sort=True):
        scores = block["factor_score"].to_numpy(dtype=float)
        eligible = np.flatnonzero(np.isfinite(scores))
        if eligible.size < config.top_k:
            continue
        order = eligible[np.argsort(-scores[eligible], kind="stable")]
        picked = block.iloc[order[: config.top_k]]
        values = picked["gross_return"].to_numpy(dtype=float)
        if not gross:
            values = apply_cost_by_d0(
                values,
                np.repeat(str(date), len(values)),
                config,
            )
        finite = values[np.isfinite(values)]
        executed_trades.extend(finite.tolist())
        rows.append(
            {
                "date": str(date),
                "n_selected": config.top_k,
                "n_executed": int(finite.size),
                "portfolio_return": float(finite.sum() / config.top_k / overlap),
            }
        )

    daily = pd.DataFrame(
        rows,
        columns=["date", "n_selected", "n_executed", "portfolio_return"],
    )
    performance = summarize_portfolio_returns(
        daily["portfolio_return"].to_numpy(dtype=float)
    )
    performance["mean_trade_return"] = (
        float(np.mean(executed_trades)) if executed_trades else float("nan")
    )
    performance["n_days"] = float(len(daily))
    performance["n_trades"] = float(len(executed_trades))
    total_selected = float(daily["n_selected"].sum())
    performance["execution_rate"] = (
        float(daily["n_executed"].sum() / total_selected)
        if total_selected
        else float("nan")
    )
    return performance, daily


def evaluate_monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound paired gross/net daily series into a calendar-month table."""
    required = {"date", "gross_portfolio_return", "net_portfolio_return"}
    missing = required - set(daily.columns)
    if missing:
        raise KeyError(f"daily frame is missing: {sorted(missing)}")
    if daily["date"].duplicated().any():
        raise ValueError("daily frame contains duplicate dates")
    work = daily.copy()
    work["month"] = work["date"].astype(str).str[:7]
    rows = []
    for month, block in work.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "n_days": len(block),
                "gross_return": float(
                    np.prod(1.0 + block["gross_portfolio_return"]) - 1.0
                ),
                "net_return": float(
                    np.prod(1.0 + block["net_portfolio_return"]) - 1.0
                ),
                "day_win_rate": float((block["net_portfolio_return"] > 0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result.assign(gross_equity=[], net_equity=[])
    result["gross_equity"] = (1.0 + result["gross_return"]).cumprod()
    result["net_equity"] = (1.0 + result["net_return"]).cumprod()
    return result


def event_grids(
    frame: pd.DataFrame,
    score_column: str,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a long event table into aligned date-by-stock IC grids."""
    required = {"trade_date", "stock_code", score_column, target_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"event grid frame is missing: {sorted(missing)}")
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("event grid contains duplicate date/stock rows")
    dates = np.asarray(sorted(frame["trade_date"].astype(str).unique()), dtype=str)
    codes = np.asarray(sorted(frame["stock_code"].astype(str).unique()), dtype=str)

    def pivot(column: str) -> np.ndarray:
        return (
            frame.pivot(index="trade_date", columns="stock_code", values=column)
            .reindex(index=dates, columns=codes)
            .to_numpy(dtype=float)
        )

    score = pivot(score_column)
    target = pivot(target_column)
    mask = np.isfinite(score) & np.isfinite(target)
    return dates, score, target, mask


def evaluate_horizon_decay(
    events: pd.DataFrame,
    prices: PriceLookup,
    config: BacktestConfig,
    horizons: Sequence[int] = tuple(range(1, 11)),
    *,
    min_ic_samples: int = 30,
) -> dict[str, pd.DataFrame]:
    """Evaluate IC and Top-K close-return decay for D+1 through D+h."""
    summaries: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    for horizon in horizons:
        aligned = align_event_prices(events, prices, int(horizon))
        if aligned.empty:
            raise ValueError(f"no outcome-complete events for D+{horizon}")
        aligned["gross_return"] = aligned["hfq_return"]
        _, score, target, mask = event_grids(
            aligned,
            "factor_score",
            "gross_return",
        )
        ic_summary = summarize_ic(
            daily_ic(score, target, mask, min_samples=min_ic_samples)
        )
        gross_summary, gross_daily = evaluate_top_k_book(
            aligned,
            config,
            gross=True,
            overlap=int(horizon),
        )
        net_summary, net_daily = evaluate_top_k_book(
            aligned,
            config,
            gross=False,
            overlap=int(horizon),
        )
        daily = gross_daily.rename(
            columns={
                "n_executed": "n_executed_gross",
                "portfolio_return": "gross_portfolio_return",
            }
        ).drop(columns="n_selected")
        net = net_daily.rename(
            columns={
                "n_executed": "n_executed_net",
                "portfolio_return": "net_portfolio_return",
            }
        )
        daily = daily.merge(net, on="date", validate="one_to_one")
        exit_by_d0 = aligned.groupby("trade_date")["exit_date"].first()
        daily["exit_date"] = daily["date"].map(exit_by_d0)
        daily["horizon"] = int(horizon)
        daily["overlap"] = int(horizon)
        daily_frames.append(daily)
        summaries.append(
            {
                "horizon": int(horizon),
                "n_days": len(daily),
                "d0_start": str(aligned["trade_date"].min()),
                "d0_end": str(aligned["trade_date"].max()),
                "exit_end": str(aligned["exit_date"].max()),
                "ic_mean": ic_summary["ic_mean"],
                "icir": ic_summary["icir"],
                "ic_days": int(ic_summary["n_days"]),
                "gross_per_trade": gross_summary["mean_trade_return"],
                "net_per_trade": net_summary["mean_trade_return"],
                "net_cagr": net_summary["cagr"],
                "net_sharpe": net_summary["sharpe"],
                "net_final_equity": net_summary["final_equity"],
            }
        )
    return {
        "summary": pd.DataFrame(summaries),
        "daily": pd.concat(daily_frames, ignore_index=True),
    }


def align_industries(
    keys: pd.DataFrame,
    members: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Point-in-time align SW2021 membership intervals to event rows."""
    required = {"index_code", "industry_name", "stock_code", "in_date", "out_date"}
    missing = required - set(members.columns)
    if missing:
        raise KeyError(f"industry cache is missing: {sorted(missing)}")
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
        raise ValueError("overlapping industry memberships found for an event row")
    aligned = keys.reset_index(drop=True).copy()
    aligned["industry_code"] = np.nan
    aligned.loc[
        active_rows["_event_row"].to_numpy(dtype=int),
        "industry_code",
    ] = active_rows["industry_code"].to_numpy(dtype=float)
    names = dict(
        zip(
            taxonomy["industry_code"].astype(int),
            taxonomy["industry_name"].astype(str),
            strict=True,
        )
    )
    return aligned, names


def _style_orthogonality(
    neutral: np.ndarray,
    continuous: np.ndarray,
    industry: np.ndarray,
    mask: np.ndarray,
    levels: np.ndarray,
) -> dict[str, float]:
    design, valid = build_style_design(
        continuous,
        industry,
        mask & np.isfinite(neutral),
        industry_levels=levels,
    )
    residual = np.nan_to_num(neutral)
    exposure = np.matmul(design.transpose(0, 2, 1), residual[..., None])[:, :, 0]
    design_norm = np.sqrt(np.einsum("tnk,tnk->tk", design, design))
    residual_norm = np.sqrt(np.einsum("tn,tn->t", residual, residual))
    denominator = design_norm * residual_norm[:, None]
    normalized = np.divide(
        exposure,
        denominator,
        out=np.zeros_like(exposure),
        where=denominator > 0,
    )
    counts = np.maximum(valid.sum(axis=1), 1)[:, None]
    covariance = exposure / counts
    return {
        "max_abs_normalized_exposure": float(np.max(np.abs(normalized))),
        "max_abs_covariance": float(np.max(np.abs(covariance))),
    }


def evaluate_style_neutral_book(
    events: pd.DataFrame,
    styles: pd.DataFrame,
    members: pd.DataFrame,
    config: BacktestConfig,
    *,
    min_ic_samples: int = 30,
) -> dict[str, object]:
    """Compare raw and date-local style-neutral scores on one common universe."""
    style_keys = ["trade_date", "stock_code"]
    if styles.duplicated(style_keys).any():
        raise ValueError("style cache contains duplicate date/stock rows")
    aligned = events.merge(styles, on=style_keys, how="left", validate="many_to_one")
    industries, industry_names = align_industries(aligned[style_keys], members)
    aligned["industry_code"] = industries["industry_code"].to_numpy()
    panel = build_event_panel(
        aligned,
        ["factor_score", *STYLE_COLUMNS, "industry_code"],
        ["gross_return"],
    )
    raw = panel.f64("factor_score")
    target = panel.f64("gross_return")
    continuous = np.stack([panel.f64(name) for name in STYLE_COLUMNS], axis=2)
    industry = panel.f64("industry_code")
    common = (
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
        common,
        industry_levels=levels,
    )
    arms: dict[str, dict[str, float]] = {}
    daily_frames: dict[str, pd.DataFrame] = {}
    for name, score in {"raw": raw, "style_neutral": neutral}.items():
        long = panel.to_long(
            {
                "factor_score": score,
                "gross_return": target,
                "common": common.astype(float),
            }
        )
        long = long[long["common"] == 1.0]
        metrics, daily = evaluate_top_k_book(
            long,
            config,
            gross=False,
            overlap=2,
        )
        ic = summarize_ic(
            daily_ic(
                score,
                target,
                common,
                min_samples=min_ic_samples,
            )
        )
        metrics.update(
            {
                "ic_mean": ic["ic_mean"],
                "icir": ic["icir"],
                "ic_days": ic["n_days"],
            }
        )
        arms[name] = metrics
        daily_frames[name] = daily
    orthogonality = _style_orthogonality(
        neutral,
        continuous,
        industry,
        common,
        levels,
    )
    raw_rows = panel.occupied & np.isfinite(raw)
    orthogonality.update(
        {
            "style_complete_rows": int(common.sum()),
            "raw_factor_rows": int(raw_rows.sum()),
            "analysis_coverage_of_raw_factor": float(common.sum() / raw_rows.sum()),
            "industry_count": len(industry_names),
        }
    )
    return {
        **arms,
        "raw_daily": daily_frames["raw"],
        "style_neutral_daily": daily_frames["style_neutral"],
        "orthogonality": orthogonality,
    }


CATEGORY_ORDER = {"工程 bug": 0, "参数配置": 1, "因子 alpha": 2}


def rank_root_causes(evidence: dict[str, object]) -> list[dict[str, object]]:
    """Enforce the requested engineering/configuration/alpha presentation order."""
    causes = [dict(row) for row in evidence["root_causes"]]
    unknown = {str(row.get("category")) for row in causes} - set(CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"unknown root-cause categories: {sorted(unknown)}")
    return sorted(
        causes,
        key=lambda row: (
            CATEGORY_ORDER[str(row["category"])],
            int(row.get("priority", 0)),
        ),
    )


def _display_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    if frame.empty:
        return "_无样本。_"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_display_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_report(evidence: dict[str, object]) -> str:
    """Render the complete Chinese audit report from calculated evidence."""
    metadata = evidence["metadata"]
    sections = [
        "# gp_000 亏损根因排查与复权全链路审计",
        "## 执行摘要",
        str(evidence["summary"]),
        (
            f"审计窗口严格限定为 `{metadata['train_start']}` 至 "
            f"`{metadata['train_end']}`，所有 D+h 结果均按各自出场日做边界截断。"
        ),
        "## 第一部分：复权全链路审计",
        "### 四节点口径与点时性",
        markdown_table(evidence["adjustment_matrix"]),
        "### 口径一致性校验",
        markdown_table(evidence["adjustment_stats"]),
        "### 除权日专项校验",
        markdown_table(evidence["ex_right_samples"]),
        "## 第二部分：gp_000 亏损归因",
        "### 五分位单调性",
        markdown_table(evidence["quintiles"]),
        "### 成本拆分",
        markdown_table(evidence["cost_split"]),
        "### 收益衰减",
        markdown_table(evidence["decay"]["summary"]),
        "![D+1 至 D+10 衰减](assets/gp000_loss_attribution_decay.svg)",
        "### 时间分布",
        markdown_table(evidence["monthly"]),
        "![累计收益](assets/gp000_loss_attribution_equity.svg)",
        "### 风格中性收益",
        markdown_table(evidence["style_table"]),
        "## 根因优先级",
        markdown_table(pd.DataFrame(rank_root_causes(evidence))),
        "## 修复路径与预期效果",
        markdown_table(pd.DataFrame(evidence["repairs"])),
        "## 复现方式",
        f"```bash\n{metadata['command']}\n```",
    ]
    return "\n\n".join(sections) + "\n"


def line_chart_svg(
    series: dict[str, Sequence[float] | np.ndarray],
    *,
    title: str,
    y_label: str,
) -> str:
    """Render deterministic shared-axis line series as standalone SVG."""
    prepared = {
        str(name): np.asarray(values, dtype=float)
        for name, values in series.items()
        if np.asarray(values).size
    }
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in prepared.values()]
    )
    if finite_values.size == 0:
        raise ValueError("chart has no finite values")
    width, height = 960.0, 480.0
    left, right, top, bottom = 75.0, 30.0, 55.0, 55.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = float(min(finite_values.min(), 1.0))
    y_max = float(max(finite_values.max(), 1.0))
    if np.isclose(y_min, y_max):
        y_min -= 0.01
        y_max += 0.01
    margin = (y_max - y_min) * 0.05
    y_min -= margin
    y_max += margin

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 480">',
        '<rect width="960" height="480" fill="#ffffff"/>',
        f'<text x="480" y="30" text-anchor="middle" font-size="20">{escape(title)}</text>',
        f'<text x="18" y="240" text-anchor="middle" font-size="13" transform="rotate(-90 18 240)">{escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{y_position(1.0):.2f}" x2="{width-right}" y2="{y_position(1.0):.2f}" stroke="#94a3b8" stroke-dasharray="4 4"/>',
    ]
    for index, (name, values) in enumerate(prepared.items()):
        denominator = max(len(values) - 1, 1)
        points = [
            f"{left + position / denominator * plot_width:.2f},{y_position(value):.2f}"
            for position, value in enumerate(values)
            if np.isfinite(value)
        ]
        color = colors[index % len(colors)]
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
            + " ".join(points)
            + '"/>'
        )
        legend_x = left + index * 155
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="462" x2="{legend_x + 24}" y2="462" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="466" font-size="12">{escape(name)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="11">{y_max:.3f}</text>',
            f'<text x="{left - 8}" y="{height-bottom}" text-anchor="end" font-size="11">{y_min:.3f}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def render_equity_svg(daily: pd.DataFrame) -> str:
    """Render D+2 gross, net, and style-neutral cumulative equity."""
    required = {
        "gross_portfolio_return",
        "net_portfolio_return",
        "style_neutral_return",
    }
    missing = required - set(daily.columns)
    if missing:
        raise KeyError(f"equity daily frame is missing: {sorted(missing)}")
    series = {
        "毛收益": np.r_[
            1.0,
            np.cumprod(1.0 + daily["gross_portfolio_return"].to_numpy(dtype=float)),
        ],
        "净收益": np.r_[
            1.0,
            np.cumprod(1.0 + daily["net_portfolio_return"].to_numpy(dtype=float)),
        ],
        "风格中性净收益": np.r_[
            1.0,
            np.cumprod(1.0 + daily["style_neutral_return"].to_numpy(dtype=float)),
        ],
    }
    return line_chart_svg(series, title="gp_000 Top4 累计收益", y_label="净值")


def render_decay_svg(decay: dict[str, pd.DataFrame]) -> str:
    """Render one Top-K net-equity path per tested holding horizon."""
    daily = decay["daily"]
    series = {
        f"D+{int(horizon)}": np.r_[
            1.0,
            np.cumprod(1.0 + block["net_portfolio_return"].to_numpy(dtype=float)),
        ]
        for horizon, block in daily.groupby("horizon", sort=True)
    }
    return line_chart_svg(series, title="gp_000 Top4 净收益衰减", y_label="净值")


def json_ready(value: object) -> object:
    """Recursively convert analytical objects to strict JSON-compatible values."""
    if isinstance(value, pd.DataFrame):
        return json_ready(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_ready(value.tolist())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def atomic_text(path: Path, content: str) -> None:
    """Write UTF-8 text atomically through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a parquet DataFrame atomically through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_outputs(evidence: dict[str, object], paths: OutputPaths) -> None:
    """Emit every artifact from the same immutable-in-practice evidence mapping."""
    payload = json.dumps(
        json_ready(evidence),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    atomic_text(paths.report, render_report(evidence))
    atomic_text(paths.json, payload + "\n")
    atomic_parquet(paths.daily, evidence["daily"])
    atomic_text(paths.equity_svg, render_equity_svg(evidence["daily"]))
    atomic_text(paths.decay_svg, render_decay_svg(evidence["decay"]))
