"""Point-in-time ST / *ST / delisting-name detection.

Shared by the tradable-universe mask (``universe.py``) and the ST-aware limit-price
fallback (``panel.py``). Lives below both so neither has to import the other:
``universe.py`` already depends on ``panel.py`` for ``Panel``, so a reverse import
from ``panel.py`` back into ``universe.py`` would be circular. This module only
depends on ``store``/``schema``, taking plain ``dates``/``codes`` arrays instead of
a ``Panel``.

Using the *current* stock name to flag ST leaks the future (a stock that becomes ST
in 2024 would be dropped from 2019 samples too); reconstructing the name in effect on
each date from Tushare's ``namechange`` history avoids that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .store import ParquetStore

#: Every marker treated as "not a normal tradable name" for universe exclusion.
ST_MARKERS = ("ST", "*ST", "退")

#: Risk-warning markers only, excluding "退" (delisting consolidation). Real-data
#: validation against the 2026-08-15 audit's local store confirmed "退"-named stocks
#: trade under their own near-unbounded/sentinel Tushare convention (observed
#: up_limit ~999,999, down_limit 0.01), not the 5% ST band -- folding them into the
#: 5% rule was flatly wrong (2.6% agreement with official stk_limit values, vs.
#: 95.6% once "退" is excluded and the check is restricted to main-board names).
RISK_WARNING_MARKERS = ("ST", "*ST")


def looks_st(name: str) -> bool:
    upper = str(name).upper().replace(" ", "")
    return any(marker in upper for marker in ST_MARKERS)


def _point_in_time_name_mask(
    dates: np.ndarray, codes: np.ndarray, store: ParquetStore, markers: tuple[str, ...]
) -> np.ndarray:
    mask = np.zeros((len(dates), len(codes)), dtype=bool)
    code_index = {code: j for j, code in enumerate(codes)}

    def hit(name: object) -> bool:
        upper = str(name).upper().replace(" ", "")
        return any(marker in upper for marker in markers)

    changes = store.read_static(schema.NAMECHANGE)
    covered: set[str] = set()
    if not changes.empty:
        changes = changes.dropna(subset=["ts_code", "name", "start_date"]).copy()
        changes["ts_code"] = changes["ts_code"].astype(str)
        changes["start_date"] = changes["start_date"].astype(str)
        covered = set(changes["ts_code"])
        for row in changes.itertuples(index=False):
            j = code_index.get(row.ts_code)
            if j is None or not hit(row.name):
                continue
            lo = int(np.searchsorted(dates, row.start_date, "left"))
            end = str(row.end_date) if pd.notna(row.end_date) else ""
            hi = int(np.searchsorted(dates, end, "right")) if end else len(dates)
            if hi > lo:
                mask[lo:hi, j] = True

    # Stocks with no namechange history: fall back to the current name for all dates.
    basic = store.read_static(schema.STOCK_BASIC)
    if not basic.empty:
        for row in basic.itertuples(index=False):
            code = str(row.ts_code)
            if code in covered:
                continue
            j = code_index.get(code)
            if j is not None and hit(row.name):
                mask[:, j] = True
    return mask


def point_in_time_st_mask(dates: np.ndarray, codes: np.ndarray, store: ParquetStore) -> np.ndarray:
    """``(T, N)`` bool: True where the stock's point-in-time name looked ST/delisting-flagged."""
    return _point_in_time_name_mask(dates, codes, store, ST_MARKERS)


def point_in_time_risk_warning_mask(dates: np.ndarray, codes: np.ndarray, store: ParquetStore) -> np.ndarray:
    """``(T, N)`` bool: True where the point-in-time name was ST/*ST (risk warning only,
    not "退" delisting consolidation -- see :data:`RISK_WARNING_MARKERS`)."""
    return _point_in_time_name_mask(dates, codes, store, RISK_WARNING_MARKERS)
