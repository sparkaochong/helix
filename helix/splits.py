"""Purged walk-forward splits.

A sample decided at D0 is only resolved at D+``touch_offset``. Slicing train/valid/test
at adjacent indices therefore lets the tail of the training set overlap the head of the
validation window -- a small leak that reliably manufactures fake alpha. Every seam here
is separated by ``embargo_days``, which :class:`~helix.config.Config` forces to exceed
the label horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SplitConfig


@dataclass(frozen=True)
class Fold:
    index: int
    train: slice
    valid: slice
    test: slice

    def describe(self, dates: np.ndarray) -> str:
        def span(s: slice) -> str:
            return f"{dates[s][0]}~{dates[s][-1]}" if len(dates[s]) else "empty"

        return (
            f"fold {self.index}: train {span(self.train)} | "
            f"valid {span(self.valid)} | test {span(self.test)}"
        )


def walk_forward(n_dates: int, cfg: SplitConfig) -> list[Fold]:
    """Rolling train -> embargo -> valid -> embargo -> test windows, oldest first."""
    folds: list[Fold] = []
    gap = cfg.embargo_days
    block = cfg.train_days + gap + cfg.valid_days + gap + cfg.test_days
    if block > n_dates:
        raise ValueError(
            f"need at least {block} trade dates for one fold, panel has {n_dates}; "
            "shorten split.train_days or extend data.start_date"
        )

    start = 0
    while start + block <= n_dates:
        train_end = start + cfg.train_days
        valid_start = train_end + gap
        valid_end = valid_start + cfg.valid_days
        test_start = valid_end + gap
        test_end = test_start + cfg.test_days
        folds.append(
            Fold(
                index=len(folds),
                train=slice(start, train_end),
                valid=slice(valid_start, valid_end),
                test=slice(test_start, test_end),
            )
        )
        start += cfg.step_days
    return folds


def search_window(n_dates: int, cfg: SplitConfig) -> slice:
    """The oldest train block -- what factor *discovery* is allowed to look at.

    Mining factors on the full history and then "validating" on part of it is the
    single most common way these pipelines fool themselves. GP search is confined here.
    """
    return slice(0, min(cfg.train_days, n_dates))
