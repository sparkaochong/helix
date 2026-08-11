"""The prediction target.

Decision is made after the **D0 close** using D0-and-earlier information only.
The trade is entered at the **D+1 open** and the label asks whether the **D+2 high**
reaches ``entry * target_ratio`` (default 1.08).

Three details decide whether this label is usable or garbage:

1. **Back-adjusted prices for the ratio.** ``high_hfq[D+2] / open_hfq[D+1]`` is
   immune to an ex-dividend between the two days; the raw-price ratio is not.
2. **The D+1 entry must be fillable.** If D+1 opens at or above its up-limit there
   is no realistic fill, so the sample is dropped rather than labelled. Leaving
   these in inflates the hit rate precisely on the days the strategy looks best.
3. **D+1 and D+2 must actually trade.** A suspension makes the outcome unobservable,
   so the label is undefined (NaN) rather than 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import LabelConfig
from ..features.operators import lead
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class LabelSet:
    """Aligned to D0 rows: row ``t`` describes the trade decided at the close of ``dates[t]``."""

    y: np.ndarray          # (T, N) float, 1.0 touched / 0.0 not / NaN undefined
    valid: np.ndarray      # (T, N) bool, sample is usable for train / eval / trading
    entry_price: np.ndarray  # (T, N) back-adjusted D+1 open
    target_price: np.ndarray  # (T, N) entry * target_ratio
    exit_price: np.ndarray   # (T, N) back-adjusted D+2 close, for the non-touch exit

    @property
    def base_rate(self) -> float:
        return float(np.nanmean(np.where(self.valid, self.y, np.nan)))


def build_touch_label(panel, universe: np.ndarray, cfg: LabelConfig) -> LabelSet:
    """Build the touch label for every ``(date, stock)`` cell of ``panel``."""
    entry_off, touch_off = cfg.entry_offset, cfg.touch_offset

    open_hfq_entry = lead(panel.f64("open_hfq"), entry_off)
    high_hfq_touch = lead(panel.f64("high_hfq"), touch_off)
    close_hfq_exit = lead(panel.f64("close_hfq"), touch_off)

    target = open_hfq_entry * cfg.target_ratio
    touched = high_hfq_touch >= target

    trading = panel["is_trading"] > 0
    entry_tradable = lead(trading.astype(np.float64), entry_off) > 0
    touch_tradable = lead(trading.astype(np.float64), touch_off) > 0

    valid = (
        universe
        & entry_tradable
        & touch_tradable
        & np.isfinite(open_hfq_entry)
        & np.isfinite(high_hfq_touch)
        & (open_hfq_entry > 0)
    )

    if cfg.exclude_entry_limit_up:
        # Raw prices here: up_limit is quoted raw, so it must be compared raw.
        open_raw_entry = lead(panel.f64("open"), entry_off)
        up_limit_entry = lead(panel.f64("up_limit"), entry_off)
        unfillable = open_raw_entry >= (up_limit_entry - cfg.limit_price_eps)
        blocked = int(np.sum(valid & np.nan_to_num(unfillable).astype(bool)))
        valid &= ~np.nan_to_num(unfillable, nan=0.0).astype(bool)
        log.info("dropped %d samples that would open at the D+%d up-limit", blocked, entry_off)

    y = np.where(valid, touched.astype(np.float64), np.nan)
    labels = LabelSet(
        y=y,
        valid=valid,
        entry_price=np.where(valid, open_hfq_entry, np.nan),
        target_price=np.where(valid, target, np.nan),
        exit_price=np.where(valid, close_hfq_exit, np.nan),
    )
    log.info(
        "label: %d usable samples, base rate %.3f%% (target ratio %.2f)",
        int(valid.sum()), 100 * labels.base_rate, cfg.target_ratio,
    )
    return labels
