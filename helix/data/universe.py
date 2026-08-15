"""Point-in-time tradable universe mask.

The ST filter is the part that is easy to get wrong: using the *current* stock name
to exclude ST names leaks the future (a stock that becomes ST in 2024 would be
dropped from 2019 samples too). Helix reconstructs the name in effect on each date
from Tushare's ``namechange`` history instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import UniverseConfig
from ..logging_setup import get_logger
from . import schema
from .panel import Panel
from .st_status import point_in_time_st_mask
from .store import ParquetStore

log = get_logger(__name__)


def st_mask(panel: Panel, store: ParquetStore) -> np.ndarray:
    """``(T, N)`` bool: True where the stock was ST / delisting-flagged on that date."""
    return point_in_time_st_mask(panel.dates, panel.codes, store)


def listing_mask(panel: Panel, store: ParquetStore, min_list_days: int) -> np.ndarray:
    """``(T, N)`` bool: True where the stock is listed, seasoned, and not yet delisted.

    ``min_list_days`` is counted in *trading* days present in the panel calendar.
    """
    dates, codes = panel.dates, panel.codes
    mask = np.zeros((len(dates), len(codes)), dtype=bool)
    basic = store.read_static(schema.STOCK_BASIC)
    if basic.empty:
        log.warning("stock_basic is empty; skipping listing/seasoning filter")
        return np.ones_like(mask)

    code_index = {code: j for j, code in enumerate(codes)}
    for row in basic.itertuples(index=False):
        j = code_index.get(str(row.ts_code))
        if j is None or pd.isna(row.list_date):
            continue
        start = int(np.searchsorted(dates, str(row.list_date), "left")) + min_list_days
        end = len(dates)
        if pd.notna(row.delist_date):
            end = int(np.searchsorted(dates, str(row.delist_date), "left"))
        if end > start:
            mask[start:end, j] = True
    return mask


def build_universe(panel: Panel, store: ParquetStore, cfg: UniverseConfig) -> np.ndarray:
    """``(T, N)`` bool mask of stocks eligible to be *selected* on each date D0."""
    is_trading = panel["is_trading"] > 0
    mask = is_trading & listing_mask(panel, store, cfg.min_list_days)

    if cfg.exclude_st:
        mask &= ~st_mask(panel, store)

    if cfg.exclude_exchanges:
        suffixes = tuple(f".{ex.upper()}" for ex in cfg.exclude_exchanges)
        excluded = np.array([code.upper().endswith(suffixes) for code in panel.codes])
        mask &= ~excluded[None, :]

    close = panel["close"]
    mask &= np.nan_to_num(close, nan=-1.0) >= cfg.min_close
    mask &= np.nan_to_num(close, nan=np.inf) <= cfg.max_close
    mask &= np.nan_to_num(panel["amount"], nan=-1.0) >= cfg.min_amount_kcny

    coverage = mask.sum(axis=1)
    log.info(
        "universe: median %d stocks/day (min %d, max %d)",
        int(np.median(coverage)), int(coverage.min()), int(coverage.max()),
    )
    return mask
