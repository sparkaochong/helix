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
from ..eval.shared_entry_check import entry_is_fillable
from ..features.operators import lead
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class LabelSet:
    """Aligned to D0 rows: row ``t`` describes the trade decided at the close of ``dates[t]``."""

    y: np.ndarray          # (T, N) float, 1.0 touched / 0.0 not / NaN undefined
    valid: np.ndarray      # (T, N) bool, sample has an observable train / eval target
    touch_tradable: np.ndarray  # (T, N) bool, D+touch_offset traded; backtest validation only
    entry_price: np.ndarray  # (T, N) back-adjusted D+1 open
    target_price: np.ndarray  # (T, N) entry * target_ratio
    exit_price: np.ndarray   # (T, N) back-adjusted D+2 close, for the non-touch exit
    entry_valid: np.ndarray | None = None  # (T, N) bool, D+1 entry actually fillable

    @property
    def executable_entry(self) -> np.ndarray:
        """D+1 entry observability, with compatibility for older constructed labels."""
        return self.valid if self.entry_valid is None else self.entry_valid

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
    touch_tradable = lead(trading.astype(np.float64), touch_off) > 0

    entry_fillable = np.asarray(
        entry_is_fillable(
            panel,
            np.arange(panel.shape[0])[:, None],
            np.arange(panel.shape[1])[None, :],
            cfg,
        ),
        dtype=bool,
    )
    entry_valid = universe & entry_fillable
    blocked = int(np.sum(universe & ~entry_fillable))
    log.info("dropped %d samples not fillable at D+%d entry", blocked, entry_off)

    valid = entry_valid & touch_tradable & np.isfinite(high_hfq_touch)
    y = np.where(valid, touched.astype(np.float64), np.nan)
    labels = LabelSet(
        y=y,
        valid=valid,
        touch_tradable=touch_tradable,
        entry_price=np.where(entry_valid, open_hfq_entry, np.nan),
        target_price=np.where(entry_valid, target, np.nan),
        exit_price=np.where(valid, close_hfq_exit, np.nan),
        entry_valid=entry_valid,
    )
    log.info(
        "label: %d usable samples, base rate %.3f%% (target ratio %.2f)",
        int(valid.sum()), 100 * labels.base_rate, cfg.target_ratio,
    )
    return labels
