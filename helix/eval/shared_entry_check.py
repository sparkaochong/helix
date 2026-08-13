"""Shared D+1 entry fillability checks for labels, backtests, and reports."""

from __future__ import annotations

import numpy as np

from ..config import LabelConfig
from ..data.panel import Panel

_ENTRY_FIELDS = ("open", "open_hfq", "up_limit", "limit_price_observed", "is_trading")


def entry_is_fillable(
    panel: Panel,
    d0_index: int | np.ndarray,
    stock_index: int | np.ndarray,
    label_cfg: LabelConfig,
) -> bool | np.ndarray:
    """Check D+``entry_offset`` against its observed market state and actual limit.

    Scalar indices return one ``bool``. Broadcastable index arrays return a boolean
    array, which lets label construction apply the identical rule across the panel.
    """
    d0, stock = np.broadcast_arrays(
        np.asarray(d0_index, dtype=np.intp),
        np.asarray(stock_index, dtype=np.intp),
    )
    result = np.zeros(d0.size, dtype=bool)
    if any(field not in panel for field in _ENTRY_FIELDS):
        return bool(result.item()) if d0.ndim == 0 else result.reshape(d0.shape)

    entry = (d0 + label_cfg.entry_offset).ravel()
    stocks = stock.ravel()
    in_bounds = (
        (entry >= 0)
        & (entry < panel.shape[0])
        & (stocks >= 0)
        & (stocks < panel.shape[1])
    )
    positions = np.flatnonzero(in_bounds)
    if positions.size:
        entry_rows = entry[positions]
        stock_columns = stocks[positions]
        raw_open = np.asarray(
            panel["open"][entry_rows, stock_columns], dtype=np.float64
        )
        adjusted_open = np.asarray(
            panel["open_hfq"][entry_rows, stock_columns], dtype=np.float64
        )
        up_limit = np.asarray(
            panel["up_limit"][entry_rows, stock_columns], dtype=np.float64
        )
        fillable = (
            (panel["is_trading"][entry_rows, stock_columns] > 0)
            & (panel["limit_price_observed"][entry_rows, stock_columns] > 0)
            & np.isfinite(raw_open)
            & np.isfinite(adjusted_open)
            & (adjusted_open > 0)
            & np.isfinite(up_limit)
        )
        if label_cfg.exclude_entry_limit_up:
            fillable &= raw_open < up_limit - label_cfg.limit_price_eps
        result[positions] = fillable

    reshaped = result.reshape(d0.shape)
    return bool(reshaped.item()) if d0.ndim == 0 else reshaped
