#!/usr/bin/env python3
"""Reproducible training-window audit for the formal ``gp_000`` factor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from helix.gp.library import FactorLibrary, FactorSpec

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TRAIN_DATES = 649
FORMAL_FACTOR = "gp_000"
FORMAL_EXPRESSION = (
    "add(add(stock_intra_amp_d1d3_mean, "
    "div(stock_vwap_dev_d1, vol_burst_count_20d)), stock_intra_amp_d0)"
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
