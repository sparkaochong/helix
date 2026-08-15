"""Cross-source unit conversions for Tushare fields that share a name but not a unit.

``stock_bak_daily`` is not ingested by Helix (see docs/data-sources.md) -- nothing in
this codebase reads ``bak_daily.amount`` today. This module exists so that *if* it
ever is ingested, the conversion is a single reviewed call site instead of tribal
knowledge someone has to rediscover the hard way: ``daily.amount`` (the table Helix
does ingest, ``schema.DAILY``) is in thousand CNY (千元); ``bak_daily.amount`` is in
ten-thousand CNY (万元). Mixing the two without conversion silently produces a 10x
error.

Confirmed via a live paired lookup (2026-08-15 data baseline audit, 000001.SZ /
20260731): ``stk_factor_pro.amount`` (千元-denominated, matches ``daily.amount``
exactly) was 2,318,839.88; ``bak_daily.amount`` for the same code/date was
231,883.98 -- exactly a factor of 10.
"""

from __future__ import annotations

import numpy as np

#: Multiply bak_daily.amount (万元) by this to get daily.amount's convention (千元).
BAK_DAILY_AMOUNT_TO_KCNY = 10.0


def bak_daily_amount_to_kcny(amount_wan_yuan: np.ndarray | float) -> np.ndarray:
    """Convert ``bak_daily.amount`` (万元) to ``daily.amount``'s unit (千元/kcny)."""
    return np.asarray(amount_wan_yuan, dtype=np.float64) * BAK_DAILY_AMOUNT_TO_KCNY
