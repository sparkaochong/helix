"""Reconstruct whether a D+1 entry could actually have been filled.

The argus_quant label asserts `high[D+2] >= open[D+1] * 1.08`. That is only a tradeable
claim if you can buy at `open[D+1]`, and you cannot when D+1 opens at its up-limit -- the
queue at the limit price never clears for a buyer. Such a row is not a win and not a
loss; it is **undefined**. `helix/labels/touch_label.py` drops these on the panel path
and `tests/test_labels.py` pins the behaviour; the event path consumes argus_quant's own
precomputed label instead, so the same property has to be reconstructed and checked.

The table carries no D0 close column, so the limit price is derived:

    close[D0] = open[D+1] / (1 + label_open_gap)
    up_limit  = round(close[D0] * (1 + board_rate), 2)

This lives in its own module because two scripts need the identical definition. Joining
a precomputed flag back on (stock_code, trade_date) is not an option: 514 rows share
those keys even including strategy_name, so the join is not one-to-one.

Standalone: numpy / pandas only, so it runs on the training host.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Exchange price limits by code prefix. ST names are capped at 5% instead, which this
#: table cannot know -- it carries no listing-name history. `st_suspect_count` bounds how
#: many rows that blind spot can touch.
BOARD_RATES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("300", "301", "302"), 0.20),   # 创业板
    (("688", "689"), 0.20),          # 科创板
    (("8", "4", "9"), 0.30),         # 北交所
)
MAIN_BOARD_RATE = 0.10

#: Limit prices are quoted to the fen. Half a fen of slack absorbs the rounding without
#: admitting a genuinely sub-limit open.
PRICE_EPS = 0.005

#: Columns `unfillable_mask` needs. `label_px_d1_high` is used only by the self-check.
REQUIRED_COLUMNS = ("stock_code", "label_px_d1_open", "label_open_gap")


def board_rate(codes: pd.Series) -> np.ndarray:
    """Per-row price-limit fraction, from the exchange code prefix."""
    rates = np.full(len(codes), MAIN_BOARD_RATE)
    numeric = codes.str.split(".").str[0]
    for prefixes, rate in BOARD_RATES:
        rates[numeric.str.startswith(prefixes).to_numpy()] = rate
    return rates


def close_d0(frame: pd.DataFrame) -> np.ndarray:
    """D0 close, backed out of the D+1 open and its gap."""
    open_d1 = frame["label_px_d1_open"].to_numpy(dtype=float)
    gap = frame["label_open_gap"].to_numpy(dtype=float)
    return open_d1 / (1.0 + gap)


def up_limit_d1(frame: pd.DataFrame) -> np.ndarray:
    return np.round(close_d0(frame) * (1.0 + board_rate(frame["stock_code"])), 2)


def unfillable_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where D+1 opens at or above its up-limit, so the entry never fills."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"fillability needs columns not present: {missing}")
    open_d1 = frame["label_px_d1_open"].to_numpy(dtype=float)
    return np.isfinite(open_d1) & (open_d1 >= up_limit_d1(frame) - PRICE_EPS)


def st_suspect_count(frame: pd.DataFrame) -> int:
    """Rows gapping ~+5%, which would be at the limit if the name were ST.

    Not a flag, a bound: without listing-name history these cannot be told apart from an
    ordinary 5% gap, so this counts how much the 10% assumption could be under-flagging.
    """
    gap = frame["label_open_gap"].to_numpy(dtype=float)
    return int((np.abs(gap - 0.05) < 0.002).sum())
