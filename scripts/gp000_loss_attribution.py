#!/usr/bin/env python3
"""Reproducible training-window audit for the formal ``gp_000`` factor."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from helix.gp.library import FactorLibrary, FactorSpec

TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TRAIN_DATES = 649
FORMAL_FACTOR = "gp_000"
FORMAL_EXPRESSION = (
    "add(add(stock_intra_amp_d1d3_mean, "
    "div(stock_vwap_dev_d1, vol_burst_count_20d)), stock_intra_amp_d0)"
)


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
