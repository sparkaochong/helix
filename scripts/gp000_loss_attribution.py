#!/usr/bin/env python3
"""Reproducible training-window audit for the formal ``gp_000`` factor."""

from __future__ import annotations

import argparse
import hashlib
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
from helix.gp.library import FactorLibrary, FactorSpec, compute_factors, load_factors
from scripts.g3_style_ablation import compute_trailing_styles

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TRAIN_DATES = 649
TRAIN_DATE_DIGEST = "df8186eafc50efa3e7ae9432e6e6327a333f7050b677f130838a17b03571e381"
FORMAL_FACTOR = "gp_000"
FORMAL_EXPRESSION = (
    "add(add(stock_intra_amp_d1d3_mean, "
    "div(stock_vwap_dev_d1, vol_burst_count_20d)), stock_intra_amp_d0)"
)
FORMAL_FIELDS = (
    "stock_intra_amp_d1d3_mean",
    "stock_vwap_dev_d1",
    "vol_burst_count_20d",
    "stock_intra_amp_d0",
)
STYLE_COLUMNS = (
    "log_total_mv",
    "momentum_20d",
    "volatility_20d",
    "turnover_mean_20d",
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/argus_quant_working.parquet"
DEFAULT_LIBRARY = ROOT / "data/artifacts/argus/event_factors.json"
DEFAULT_PRICE_CACHE = ROOT / "data/raw/d2_exit_cache"
DEFAULT_STYLE_MARKET = ROOT / "data/artifacts/g3_style_market.parquet"
DEFAULT_INDUSTRIES = ROOT / "data/artifacts/g3_sw2021_members.parquet"
DEFAULT_REPORT = ROOT / "docs/risk/gp000_loss_attribution.md"
DEFAULT_JSON = ROOT / "data/artifacts/gp000_loss_attribution.json"
DEFAULT_DAILY = ROOT / "data/artifacts/gp000_loss_attribution_daily.parquet"
DEFAULT_EQUITY_SVG = ROOT / "docs/risk/assets/gp000_loss_attribution_equity.svg"
DEFAULT_DECAY_SVG = ROOT / "docs/risk/assets/gp000_loss_attribution_decay.svg"


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


def validate_training_calendar(
    dates: Sequence[object] | np.ndarray,
    *,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    expected_count: int = TRAIN_DATES,
    expected_digest: str = TRAIN_DATE_DIGEST,
) -> np.ndarray:
    """Validate the approved formal training calendar by bounds, count, and hash."""
    unique = np.unique(np.asarray(dates).astype(str))
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


def replay_formal_factor(
    events: pd.DataFrame,
    library: FactorLibrary,
) -> pd.DataFrame:
    """Replay only expression-referenced inputs through the canonical GP evaluator."""
    factor = validate_formal_factor(library)
    missing = set(FORMAL_FIELDS) - set(events.columns)
    if missing:
        raise KeyError(f"formal factor inputs are missing: {sorted(missing)}")
    replay_library = FactorLibrary(
        factors=[factor],
        field_names=list(FORMAL_FIELDS),
        windows=[],
        kind="event",
    )
    panel = build_event_panel(events, list(FORMAL_FIELDS), [])
    names, values = compute_factors(replay_library, panel.fields)
    if names != [FORMAL_FACTOR] or values.shape[-1] != 1:
        raise ValueError("formal factor replay did not return exactly gp_000")
    return panel.to_long({"factor_score": values[..., 0]})


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
    aligned["reconstructed_raw_hit"] = (
        aligned["raw_exit_high"] >= aligned["raw_entry"] * 1.08
    )
    aligned["hfq_hit"] = aligned["hfq_exit_high"] >= aligned["hfq_entry"] * 1.08
    return_rounding_error = (
        aligned["label_d2_return"] - aligned["raw_return"]
    ).abs()
    raw_return_matches = np.isclose(
        aligned["label_d2_return"],
        aligned["raw_return"],
        rtol=0.0,
        atol=1e-6,
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
        "event_return_rounding_error_max": float(return_rounding_error.max()),
        "event_raw_hit_mismatch_count": int(
            (aligned["raw_hit"] != aligned["reconstructed_raw_hit"]).sum()
        ),
        "return_mismatch_count": int(return_mismatch.sum()),
        "hit_flip_count": int(
            (aligned["reconstructed_raw_hit"] != aligned["hfq_hit"]).sum()
        ),
        "event_label_to_hfq_hit_difference_count": int(
            (aligned["raw_hit"] != aligned["hfq_hit"]).sum()
        ),
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
    compact_dates = np.asarray(
        ["".join(character for character in date if character.isdigit()) for date in dates]
    )
    if np.any(np.char.str_len(compact_dates) != 8):
        raise ValueError("d0_dates must contain valid YYYY-MM-DD or YYYYMMDD dates")
    buy, sell = _cost_rates(config, compact_dates)
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


def load_training_events(path: Path, library: FactorLibrary) -> pd.DataFrame:
    """Load only formal inputs and labels inside the nominal training D0 window."""
    validate_formal_factor(library)
    labels = [
        "label_d2_hit_8pct",
        "label_d2_return",
        "label_px_d1_open",
        "label_px_d2_high",
        "label_px_d2_close",
    ]
    columns = ["trade_date", "stock_code", *FORMAL_FIELDS, *labels]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("trade_date", ">=", TRAIN_START),
            ("trade_date", "<=", TRAIN_END),
        ],
    )
    if frame.empty:
        raise ValueError("formal training event table is empty")
    frame["trade_date"] = frame["trade_date"].map(_hyphenated)
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame = frame.drop_duplicates(["trade_date", "stock_code"], keep="last")
    frame = frame.sort_values(["trade_date", "stock_code"], kind="stable")
    frame = frame.reset_index(drop=True)
    validate_training_calendar(frame["trade_date"])
    replayed = replay_formal_factor(frame, library)
    return frame.merge(
        replayed,
        on=["trade_date", "stock_code"],
        how="left",
        validate="one_to_one",
    )


def load_training_market(
    directory: Path,
    event_codes: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load raw OHLC and date-local adjustment factors through train end only."""
    paths = [
        path
        for path in sorted(directory.glob("*.parquet"))
        if TRAIN_START.replace("-", "") <= path.stem <= TRAIN_END.replace("-", "")
    ]
    calendar = np.asarray([_hyphenated(path.stem) for path in paths], dtype=str)
    validate_training_calendar(calendar)
    columns = ["trade_date", "ts_code", "open", "high", "close", "adj_factor"]
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame.loc[frame["ts_code"].astype(str).isin(event_codes)])
    market = pd.concat(frames, ignore_index=True)
    if market.empty:
        raise ValueError("training market cache has no matching event stocks")
    return market, calendar


def load_training_styles(
    market_path: Path,
    training_dates: np.ndarray,
) -> pd.DataFrame:
    """Build trailing-only styles with 19 pre-window sessions and no future rows."""
    market = pd.read_parquet(
        market_path,
        columns=[
            "trade_date",
            "stock_code",
            "pct_chg",
            "total_mv",
            "turnover_rate_f",
        ],
        filters=[("trade_date", "<=", TRAIN_END)],
    )
    market["trade_date"] = market["trade_date"].map(_hyphenated)
    calendar = np.asarray(sorted(market["trade_date"].unique()), dtype=str)
    first_training = int(np.searchsorted(calendar, TRAIN_START))
    if first_training < 19 or not set(training_dates) <= set(calendar):
        raise ValueError("style cache lacks the training calendar or 19 lookback sessions")
    styles = compute_trailing_styles(market, calendar)
    return styles.loc[styles["trade_date"].isin(training_dates)].reset_index(drop=True)


def summarize_ex_right_samples(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every event/ex-date mapping and an all-date anomaly summary."""
    work = aligned.copy()
    daily_mean = work.groupby("trade_date")["factor_score"].transform("mean")
    daily_std = work.groupby("trade_date")["factor_score"].transform("std")
    work["factor_daily_z"] = (work["factor_score"] - daily_mean) / daily_std
    work["hit_flip"] = work["reconstructed_raw_hit"] != work["hfq_hit"]
    work["raw_gap_removed"] = (
        work["raw_return"].abs() > 0.10
    ) & (work["hfq_return"].abs() <= 0.10)
    stages = (
        ("D0", "trade_date", "d0_ex_right"),
        ("D+1", "entry_date", "entry_ex_right"),
        ("D+2", "exit_date", "exit_ex_right"),
    )
    details = []
    for stage, date_column, flag_column in stages:
        block = work.loc[work[flag_column]].copy()
        block["ex_right_date"] = block[date_column]
        block["stage"] = stage
        details.append(block)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    if detail.empty:
        columns = [
            "ex_right_date",
            "stage",
            "n",
            "factor_mean",
            "factor_abs_z_max",
            "raw_return_mean",
            "hfq_return_mean",
            "return_delta_mean",
            "return_delta_abs_max",
            "hit_flip_count",
            "raw_gap_removed_count",
        ]
        return detail, pd.DataFrame(columns=columns)
    summary = (
        detail.groupby(["ex_right_date", "stage"], sort=True)
        .agg(
            n=("factor_score", "size"),
            factor_mean=("factor_score", "mean"),
            factor_abs_z_max=("factor_daily_z", lambda values: values.abs().max()),
            raw_return_mean=("raw_return", "mean"),
            hfq_return_mean=("hfq_return", "mean"),
            return_delta_mean=("return_delta", "mean"),
            return_delta_abs_max=("return_delta", lambda values: values.abs().max()),
            hit_flip_count=("hit_flip", "sum"),
            raw_gap_removed_count=("raw_gap_removed", "sum"),
        )
        .reset_index()
    )
    return detail, summary


def _hash_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_daily_books(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    gross_metrics, gross_daily = evaluate_top_k_book(
        frame,
        config,
        gross=True,
        overlap=2,
    )
    net_metrics, net_daily = evaluate_top_k_book(
        frame,
        config,
        gross=False,
        overlap=2,
    )
    daily = gross_daily[["date", "n_selected", "n_executed", "portfolio_return"]].rename(
        columns={"portfolio_return": "gross_portfolio_return"}
    )
    daily = daily.merge(
        net_daily[["date", "portfolio_return"]].rename(
            columns={"portfolio_return": "net_portfolio_return"}
        ),
        on="date",
        validate="one_to_one",
    )
    return gross_metrics, net_metrics, daily


def _style_metric_row(name: str, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "组合": name,
        "D+2 IC": metrics["ic_mean"],
        "ICIR": metrics["icir"],
        "CAGR": metrics["cagr"],
        "夏普": metrics["sharpe"],
        "单笔净收益": metrics["mean_trade_return"],
        "累计净值": metrics["final_equity"],
        "最大回撤": metrics["max_drawdown"],
        "交易日": int(metrics["n_days"]),
    }


def build_evidence(
    *,
    input_path: Path = DEFAULT_INPUT,
    library_path: Path = DEFAULT_LIBRARY,
    price_cache: Path = DEFAULT_PRICE_CACHE,
    style_market_path: Path = DEFAULT_STYLE_MARKET,
    industries_path: Path = DEFAULT_INDUSTRIES,
) -> dict[str, object]:
    """Run the complete training-only adjustment and loss-attribution audit."""
    library = load_factors(library_path)
    factor = validate_formal_factor(library)
    events = load_training_events(input_path, library)
    training_dates = validate_training_calendar(events["trade_date"])
    market, calendar = load_training_market(
        price_cache,
        set(events["stock_code"].astype(str)),
    )
    prices = build_price_lookup(market, calendar, events["stock_code"].unique())
    adjustment, aligned = audit_adjustment_chain(events, prices)
    if aligned["trade_date"].nunique() != 647:
        raise ValueError("D+2-complete training window must contain exactly 647 D0 dates")
    if aligned["exit_date"].max() > TRAIN_END:
        raise AssertionError("D+2 exit crossed the formal training end")
    aligned["gross_return"] = aligned["hfq_return"]
    aligned["hit_hfq"] = aligned["hfq_hit"].astype(float)

    config = BacktestConfig(top_k=4)
    quintiles = evaluate_quintiles(aligned, config)
    gross_metrics, net_metrics, daily = _paired_daily_books(aligned, config)

    raw_path = aligned.copy()
    raw_path["gross_return"] = raw_path["raw_return"]
    raw_adjusted_metrics, raw_adjusted_daily = evaluate_top_k_book(
        raw_path,
        config,
        gross=False,
        overlap=2,
    )
    daily = daily.merge(
        raw_adjusted_daily[["date", "portfolio_return"]].rename(
            columns={"portfolio_return": "raw_unadjusted_net_return"}
        ),
        on="date",
        validate="one_to_one",
    )

    decay = evaluate_horizon_decay(events, prices, config)
    if decay["daily"]["exit_date"].max() > TRAIN_END:
        raise AssertionError("decay exit crossed the formal training end")

    styles = load_training_styles(style_market_path, training_dates)
    members = pd.read_parquet(industries_path)
    style = evaluate_style_neutral_book(aligned, styles, members, config)
    raw_style_daily = style["raw_daily"]
    neutral_style_daily = style["style_neutral_daily"]
    if not raw_style_daily["date"].equals(neutral_style_daily["date"]):
        raise ValueError("raw and style-neutral books do not share the same dates")
    daily = daily.merge(
        neutral_style_daily[["date", "portfolio_return"]].rename(
            columns={"portfolio_return": "style_neutral_return"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if daily["style_neutral_return"].isna().any():
        raise ValueError("style-neutral book does not cover every D+2 training date")
    daily["gross_equity"] = (1.0 + daily["gross_portfolio_return"]).cumprod()
    daily["net_equity"] = (1.0 + daily["net_portfolio_return"]).cumprod()
    daily["style_neutral_equity"] = (1.0 + daily["style_neutral_return"]).cumprod()
    monthly = evaluate_monthly_returns(daily)

    ex_right_detail, ex_right_summary = summarize_ex_right_samples(aligned)
    _, hit_score, hit_target, hit_mask = event_grids(
        aligned,
        "factor_score",
        "hit_hfq",
    )
    hit_ic = summarize_ic(daily_ic(hit_score, hit_target, hit_mask, min_samples=30))
    d2_return_ic = decay["summary"].loc[
        decay["summary"]["horizon"] == 2,
        "ic_mean",
    ].iloc[0]
    quintile_slope = float(
        np.corrcoef(quintiles["quintile"], quintiles["gross_return"])[0, 1]
    )
    adjustment_net_delta = (
        net_metrics["mean_trade_return"] - raw_adjusted_metrics["mean_trade_return"]
    )
    cost_drag = gross_metrics["mean_trade_return"] - net_metrics["mean_trade_return"]
    adjusted_still_loses = net_metrics["cagr"] < 0
    d0_ex_right = ex_right_detail.loc[ex_right_detail["stage"] == "D0"]
    d2_ex_right = ex_right_detail.loc[ex_right_detail["stage"] == "D+2"]
    d0_factor_outliers = int((d0_ex_right["factor_daily_z"].abs() > 5.0).sum())
    raw_gaps_removed = int(d2_ex_right["raw_gap_removed"].sum())
    worst_months = monthly.nsmallest(5, "net_return")
    worst_month_text = "、".join(
        f"{row.month} {row.net_return:.2%}" for row in worst_months.itertuples()
    )
    decay_d1 = decay["summary"].loc[decay["summary"]["horizon"] == 1].iloc[0]
    decay_d2 = decay["summary"].loc[decay["summary"]["horizon"] == 2].iloc[0]
    decay_d10 = decay["summary"].loc[decay["summary"]["horizon"] == 10].iloc[0]

    adjustment_matrix = pd.DataFrame(
        [
            {
                "节点": "数据源层",
                "实现": "原始 OHLC + 当日 adj_factor；审计重建 raw×adj_factor",
                "口径": "原始价与后复权价并存",
                "点时性": "每个交易日仅使用该日因子",
                "未来函数风险": "未发现训练末统一缩放",
                "结论": "可重建、可审计",
            },
            {
                "节点": "因子计算层",
                "实现": "正式表达式只读取 4 个上游截面字段；Helix 内不再复权",
                "口径": "无直接价格输入",
                "点时性": "表达式无 label/future/lead；上游生成血缘未入库",
                "未来函数风险": "代码内未发现；上游契约风险未闭环",
                "结论": "需补 as-of/复权血缘元数据",
            },
            {
                "节点": "标签计算层",
                "实现": "event 表持久化 raw D+1 open/D+2 high/close；panel 标签用 raw×adj_factor",
                "口径": "event 不复权，panel 后复权",
                "点时性": "D+2 仅作为训练结果，边界已截断",
                "未来函数风险": "无越界；存在跨路径口径错配",
                "结论": "工程 bug",
            },
            {
                "节点": "回测引擎层",
                "实现": "canonical panel 回测消费后复权 LabelSet；历史 event 脚本直接消费 raw 标签",
                "口径": "两条回测路径不统一；本专项统一使用后复权",
                "点时性": "费用按 D0，出场逐 horizon 截断",
                "未来函数风险": "专项未发现",
                "结论": "工程 bug，需统一入口",
            },
        ]
    )
    adjustment_stats = pd.DataFrame(
        [
            {"指标": "D+2 完整事件数", "值": len(aligned), "结论": "仅 647 个 D0 日"},
            {
                "指标": "原始价标签与行情缓存一致",
                "值": adjustment["event_prices_match_raw"],
                "结论": "一致",
            },
            {
                "指标": "event 收益与 raw 价格公式一致",
                "值": adjustment["event_returns_match_raw"],
                "结论": (
                    "最大舍入误差 "
                    f"{adjustment['event_return_rounding_error_max']:.2e}"
                ),
            },
            {
                "指标": "event hit 与 raw 高价重算不一致",
                "值": adjustment["event_raw_hit_mismatch_count"],
                "结论": "阈值/舍入口径差异，需同时修复",
            },
            {
                "指标": "raw/HFQ 收益不同样本",
                "值": adjustment["return_mismatch_count"],
                "结论": "复权错配影响范围",
            },
            {
                "指标": "8% hit 翻转样本",
                "值": adjustment["hit_flip_count"],
                "结论": "仅计 raw 重算与 HFQ 的真实复权翻转",
            },
            {
                "指标": "持有期含除权事件",
                "值": adjustment["holding_ex_right_count"],
                "结论": "需按 HFQ 计算",
            },
            {
                "指标": "Top4 单笔净收益修正量",
                "值": adjustment_net_delta,
                "结论": "HFQ 净收益减 raw 净收益",
            },
            {
                "指标": "修正后是否仍亏损",
                "值": adjusted_still_loses,
                "结论": "决定复权 bug 是否为核心原因",
            },
            {
                "指标": "D0 除权因子 |z|>5",
                "值": d0_factor_outliers,
                "结论": "未见除权日因子系统性跳空",
            },
            {
                "指标": "D+2 raw >10% 跳空被 HFQ 消除",
                "值": raw_gaps_removed,
                "结论": "raw 回测错误计入除权缺口",
            },
        ]
    )
    cost_split = pd.DataFrame(
        [
            {
                "口径": "毛收益（零手续费/滑点）",
                "CAGR": gross_metrics["cagr"],
                "夏普": gross_metrics["sharpe"],
                "单笔收益": gross_metrics["mean_trade_return"],
                "累计净值": gross_metrics["final_equity"],
                "最大回撤": gross_metrics["max_drawdown"],
            },
            {
                "口径": "净收益（生产成本）",
                "CAGR": net_metrics["cagr"],
                "夏普": net_metrics["sharpe"],
                "单笔收益": net_metrics["mean_trade_return"],
                "累计净值": net_metrics["final_equity"],
                "最大回撤": net_metrics["max_drawdown"],
            },
            {
                "口径": "成本拖累（毛－净）",
                "CAGR": gross_metrics["cagr"] - net_metrics["cagr"],
                "夏普": gross_metrics["sharpe"] - net_metrics["sharpe"],
                "单笔收益": cost_drag,
                "累计净值": gross_metrics["final_equity"] - net_metrics["final_equity"],
                "最大回撤": gross_metrics["max_drawdown"] - net_metrics["max_drawdown"],
            },
        ]
    )
    style_table = pd.DataFrame(
        [
            _style_metric_row("原始因子（共同样本）", style["raw"]),
            _style_metric_row("风格中性因子（共同样本）", style["style_neutral"]),
        ]
    )

    adjustment_conclusion = (
        f"**审计结论：口径不统一，确认是工程 bug。** event 标签保留 raw 价格，"
        f"而 panel 标签与 canonical 回测使用 `raw×adj_factor`。共有 "
        f"{adjustment['return_mismatch_count']} 个 D+2 收益样本因复权而不同，"
        f"{adjustment['hit_flip_count']} 个 8% hit 发生真实翻转，另有 "
        f"{adjustment['event_raw_hit_mismatch_count']} 个 event hit 与 raw 高价公式的"
        f"阈值/舍入差异。全部 D0 除权样本中 |因子日内 z|>5 的数量为 "
        f"{d0_factor_outliers}，未发现因子在除权日系统性跳空；D+2 有 "
        f"{raw_gaps_removed} 个超过 10% 的 raw 跳空被 HFQ 消除。修正后策略仍亏损，"
        "因此该 bug 必须修，但不是负收益的核心解释。"
    )
    quintile_conclusion = (
        f"**方向不匹配。** hit rate 从 Q1 的 {quintiles.iloc[0]['hit_rate']:.2%} "
        f"单调升至 Q5 的 {quintiles.iloc[-1]['hit_rate']:.2%}，说明因子确实预测“盘中触达”；"
        f"但 D+2 毛收益从 Q1 的 {quintiles.iloc[0]['gross_return']:.4%} 降至 "
        f"Q5 的 {quintiles.iloc[-1]['gross_return']:.4%}，分位－收益相关为 "
        f"{quintile_slope:.4f}。高分组对收盘收益反而最差。"
    )
    cost_conclusion = (
        f"**成本不是由盈转亏的首因。** 零成本 Top4 单笔已为 "
        f"{gross_metrics['mean_trade_return']:.4%}、CAGR {gross_metrics['cagr']:.2%}；"
        f"生产成本再拖累单笔 {cost_drag:.4%}，使净 CAGR 降至 "
        f"{net_metrics['cagr']:.2%}。成本放大亏损，但毛 alpha 本身已经为负。"
    )
    decay_conclusion = (
        f"**预测在 D+1 尾部短暂有效，D+2 起反转并持续恶化。** D+1 Top4 单笔净收益 "
        f"{decay_d1['net_per_trade']:.4%}，到 D+2 变为 {decay_d2['net_per_trade']:.4%}；"
        f"D+2 IC={decay_d2['ic_mean']:.4f}，D+10 IC={decay_d10['ic_mean']:.4f}、"
        f"单笔净收益={decay_d10['net_per_trade']:.4%}。这不是缓慢衰减，而是隔夜后"
        "方向反转并随持有期累积。"
    )
    time_conclusion = (
        f"**亏损集中月份：** {worst_month_text}。2023 年毛收益阶段性修复，但成本后"
        "全年净收益仍略负；2022 年与 2024 年再次出现大幅毛亏，说明问题不是单一"
        "费率阶段或单个除权季造成。"
    )
    style_conclusion = (
        f"共同样本覆盖原始因子行的 {style['orthogonality']['analysis_coverage_of_raw_factor']:.2%}；"
        f"中性残差最大归一化暴露为 "
        f"{style['orthogonality']['max_abs_normalized_exposure']:.2e}。"
        f"剥离规模、20 日动量、波动、换手和 SW2021 行业后，D+2 IC 仍为 "
        f"{style['style_neutral']['ic_mean']:.4f}，Top4 CAGR 进一步降至 "
        f"{style['style_neutral']['cagr']:.2%}。**纯 alpha 仍为负。**"
    )

    root_causes = [
        {
            "category": "工程 bug",
            "priority": 1,
            "cause": "event 标签/回测用 raw，panel 链用 HFQ，跨路径复权口径错配",
            "evidence": (
                f"{adjustment['return_mismatch_count']} 个收益样本不同、"
                f"{adjustment['hit_flip_count']} 个 hit 翻转；Top4 单笔净收益修正 "
                f"{adjustment_net_delta:.6%}"
            ),
            "是否核心": "否" if adjusted_still_loses else "是",
        },
        {
            "category": "工程 bug",
            "priority": 2,
            "cause": "4 个上游因子字段缺少 as-of 与复权血缘契约",
            "evidence": "表达式本身无未来算子，但仓库无法证明上游特征生成时的复权版本和可见时间",
            "是否核心": "未证实",
        },
        {
            "category": "参数配置",
            "priority": 1,
            "cause": "正式因子按 8% 盘中触达目标筛选，却以 D+2 收盘净收益验收",
            "evidence": (
                f"正式库 fit_gini={factor.metrics.get('fit_gini', np.nan):.6f}，"
                f"HFQ hit IC={hit_ic['ic_mean']:.6f}，D+2 close IC={d2_return_ic:.6f}"
            ),
            "是否核心": "是",
        },
        {
            "category": "参数配置",
            "priority": 2,
            "cause": "成本门槛未阻止毛收益不足的因子进入正式库",
            "evidence": (
                f"Top4 毛收益单笔={gross_metrics['mean_trade_return']:.6%}，"
                f"成本拖累={cost_drag:.6%}"
            ),
            "是否核心": "次要" if gross_metrics["mean_trade_return"] < 0 else "是",
        },
        {
            "category": "因子 alpha",
            "priority": 1,
            "cause": "gp_000 对 D+2 收盘收益缺乏同方向纯 alpha",
            "evidence": (
                f"D+2 IC={d2_return_ic:.6f}，五分位线性相关={quintile_slope:.6f}，"
                f"风格中性 Top4 CAGR={style['style_neutral']['cagr']:.6%}"
            ),
            "是否核心": "是",
        },
    ]
    repairs = [
        {
            "类别": "工程 bug",
            "修复路径": "删除 raw event 收益的独立消费路径；标签和回测统一调用点时 raw×adj_factor 价格服务，并把除权日翻转测试纳入 CI",
            "预期效果": f"消除 {adjustment['return_mismatch_count']} 个收益错配和 {adjustment['hit_flip_count']} 个 hit 翻转；本样本 Top4 单笔净收益变化 {adjustment_net_delta:.6%}",
        },
        {
            "类别": "工程 bug",
            "修复路径": "为每个上游字段固化 source_date/as_of_time/price_basis/adj_factor_version，训练加载时 fail-closed",
            "预期效果": "把当前无法证伪的上游未来函数风险转为可自动审计契约；不承诺直接增厚收益",
        },
        {
            "类别": "参数配置",
            "修复路径": "废止当前正式 gp_000；以 D+2 HFQ Top4 净收益为 GP 主目标，并增加毛收益>0、净收益>0、方向与五分位单调门槛",
            "预期效果": "阻止 hit 指标好但 close PnL 为负的候选再次晋级；需重新挖掘后用独立 OOS 确认收益",
        },
        {
            "类别": "参数配置",
            "修复路径": "把实际双边费率和 2023-08-28 印花税切换前置到适应度，报告同时保留零成本上限",
            "预期效果": f"显式覆盖本窗口单笔 {cost_drag:.6%} 的成本拖累，避免把成本后置成解释项",
        },
        {
            "类别": "因子 alpha",
            "修复路径": "不对 gp_000 机械反向；重新挖掘 close-return alpha，并通过 walk-forward、风格中性和安慰剂门槛后再发布",
            "预期效果": "当前负 alpha 从正式库下线；新因子收益不做训练内外推承诺",
        },
    ]
    summary = (
        "**结论：存在真实的复权工程 bug，但它不是 gp_000 亏损的核心原因。** "
        f"统一为点时后复权后，Top4 单笔净收益为 {net_metrics['mean_trade_return']:.4%}、"
        f"CAGR 为 {net_metrics['cagr']:.2%}，仍为负；复权修正只改变单笔 "
        f"{adjustment_net_delta:.4%}。核心原因是正式库仍承载旧的 8% 触达优化目标，"
        f"而生产验收是 D+2 收盘净收益；对应 D+2 IC={d2_return_ic:.4f}，"
        f"风格中性 CAGR={style['style_neutral']['cagr']:.2%}，说明剥离风格后仍无可用纯 alpha。"
    )
    metadata = {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "nominal_dates": int(len(training_dates)),
        "d2_complete_dates": int(aligned["trade_date"].nunique()),
        "d2_d0_end": str(aligned["trade_date"].max()),
        "d2_exit_end": str(aligned["exit_date"].max()),
        "event_rows_nominal": len(events),
        "event_rows_d2_complete": len(aligned),
        "formal_factor": FORMAL_FACTOR,
        "formal_expression": FORMAL_EXPRESSION,
        "formal_sign": factor.sign,
        "calendar_digest": TRAIN_DATE_DIGEST,
        "event_content_digest": _hash_frame(
            events,
            ["trade_date", "stock_code", *FORMAL_FIELDS],
        ),
        "library_sha256": _hash_file(library_path),
        "command": "PYTHONPATH=. .venv/bin/python scripts/gp000_loss_attribution.py",
    }
    return {
        "metadata": metadata,
        "summary": summary,
        "adjustment_matrix": adjustment_matrix,
        "adjustment_stats": adjustment_stats,
        "adjustment_audit": adjustment,
        "adjustment_conclusion": adjustment_conclusion,
        "ex_right_samples": ex_right_summary,
        "ex_right_event_sample_count": len(ex_right_detail),
        "quintiles": quintiles,
        "quintile_direction_correlation": quintile_slope,
        "quintile_conclusion": quintile_conclusion,
        "cost_split": cost_split,
        "cost_conclusion": cost_conclusion,
        "decay": decay,
        "decay_conclusion": decay_conclusion,
        "monthly": monthly,
        "time_conclusion": time_conclusion,
        "daily": daily,
        "style_table": style_table,
        "style_orthogonality": style["orthogonality"],
        "style_conclusion": style_conclusion,
        "hit_ic": hit_ic,
        "root_causes": root_causes,
        "repairs": repairs,
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


def _display_cell(value: object, column: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        percent_columns = (
            "return",
            "收益",
            "CAGR",
            "回撤",
            "hit_rate",
            "win_rate",
            "coverage",
        )
        if any(token in column for token in percent_columns):
            return f"{float(value):.4%}"
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
        "| "
        + " | ".join(
            _display_cell(value, column)
            for column, value in zip(headers, row, strict=True)
        )
        + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_report(evidence: dict[str, object]) -> str:
    """Render the complete Chinese audit report from calculated evidence."""
    metadata = evidence["metadata"]
    metadata_table = pd.DataFrame(
        [
            {"校验项": "名义训练日", "值": metadata.get("nominal_dates", "—")},
            {"校验项": "D+2 完整日", "值": metadata.get("d2_complete_dates", "—")},
            {"校验项": "D+2 最晚 D0", "值": metadata.get("d2_d0_end", "—")},
            {"校验项": "D+2 最晚出场", "值": metadata.get("d2_exit_end", "—")},
            {"校验项": "正式因子", "值": metadata.get("formal_factor", "—")},
            {"校验项": "正式方向", "值": metadata.get("formal_sign", "—")},
            {"校验项": "训练日历 SHA256", "值": metadata.get("calendar_digest", "—")},
            {"校验项": "事件内容 SHA256", "值": metadata.get("event_content_digest", "—")},
            {"校验项": "因子库 SHA256", "值": metadata.get("library_sha256", "—")},
        ]
    )
    sections = [
        "# gp_000 亏损根因排查与复权全链路审计",
        "## 执行摘要",
        str(evidence["summary"]),
        (
            f"审计窗口严格限定为 `{metadata['train_start']}` 至 "
            f"`{metadata['train_end']}`，所有 D+h 结果均按各自出场日做边界截断。"
        ),
        "### 边界与复现契约",
        markdown_table(metadata_table),
        "## 第一部分：复权全链路审计",
        "### 四节点口径与点时性",
        markdown_table(evidence["adjustment_matrix"]),
        "### 口径一致性校验",
        markdown_table(evidence["adjustment_stats"]),
        str(evidence.get("adjustment_conclusion", "")),
        "### 除权日专项校验",
        "下表覆盖全部除权事件，并按除权发生在 D0、D+1 或 D+2 分层聚合；不是抽样展示。",
        markdown_table(evidence["ex_right_samples"]),
        "## 第二部分：gp_000 亏损归因",
        "### 五分位单调性",
        markdown_table(evidence["quintiles"]),
        str(evidence.get("quintile_conclusion", "")),
        "### 成本拆分",
        markdown_table(evidence["cost_split"]),
        str(evidence.get("cost_conclusion", "")),
        "### 收益衰减",
        markdown_table(evidence["decay"]["summary"]),
        str(evidence.get("decay_conclusion", "")),
        "![D+1 至 D+10 衰减](assets/gp000_loss_attribution_decay.svg)",
        "### 时间分布",
        markdown_table(evidence["monthly"]),
        str(evidence.get("time_conclusion", "")),
        "![累计收益](assets/gp000_loss_attribution_equity.svg)",
        "### 风格中性收益",
        markdown_table(evidence["style_table"]),
        str(evidence.get("style_conclusion", "")),
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
    reference: float = 1.0,
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
    legend_columns = min(5, len(prepared))
    legend_rows = int(np.ceil(len(prepared) / legend_columns))
    left, right, top, bottom = 75.0, 30.0, 55.0, 45.0 + 20.0 * legend_rows
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = float(min(finite_values.min(), reference))
    y_max = float(max(finite_values.max(), reference))
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
        f'<line x1="{left}" y1="{y_position(reference):.2f}" x2="{width-right}" y2="{y_position(reference):.2f}" stroke="#94a3b8" stroke-dasharray="4 4"/>',
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
        legend_column = index % legend_columns
        legend_row = index // legend_columns
        legend_x = left + legend_column * (plot_width / legend_columns)
        legend_y = height - 12.0 - (legend_rows - 1 - legend_row) * 20.0
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-size="12">{escape(name)}</text>',
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
    series = {}
    for horizon, block in daily.groupby("horizon", sort=True):
        equity = np.r_[
            1.0,
            np.cumprod(1.0 + block["net_portfolio_return"].to_numpy(dtype=float)),
        ]
        if (equity <= 0).any():
            raise ValueError("decay equity must stay positive for log-scale rendering")
        series[f"D+{int(horizon)}"] = np.log(equity)
    return line_chart_svg(
        series,
        title="gp_000 Top4 净收益衰减（对数净值）",
        y_label="log 净值",
        reference=0.0,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--price-cache", type=Path, default=DEFAULT_PRICE_CACHE)
    parser.add_argument("--style-market", type=Path, default=DEFAULT_STYLE_MARKET)
    parser.add_argument("--industries", type=Path, default=DEFAULT_INDUSTRIES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--equity-svg", type=Path, default=DEFAULT_EQUITY_SVG)
    parser.add_argument("--decay-svg", type=Path, default=DEFAULT_DECAY_SVG)
    args = parser.parse_args()

    evidence = build_evidence(
        input_path=args.input,
        library_path=args.library,
        price_cache=args.price_cache,
        style_market_path=args.style_market,
        industries_path=args.industries,
    )
    paths = OutputPaths(
        report=args.report,
        json=args.json,
        daily=args.daily,
        equity_svg=args.equity_svg,
        decay_svg=args.decay_svg,
    )
    write_outputs(evidence, paths)
    summary = {
        "report": str(paths.report),
        "formal_factor": evidence["metadata"]["formal_factor"],
        "nominal_dates": evidence["metadata"]["nominal_dates"],
        "d2_complete_dates": evidence["metadata"]["d2_complete_dates"],
        "d2_exit_end": evidence["metadata"]["d2_exit_end"],
        "root_causes": rank_root_causes(evidence),
    }
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
