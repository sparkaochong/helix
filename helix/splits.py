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


def complete_outcome_window(rows: slice, horizon: int) -> slice:
    """Remove D0 rows whose forward outcome lands beyond ``rows``.

    This is for objective construction at a hard training boundary.  It is deliberately
    independent of whether a particular outcome cell happens to be present: every
    decision date follows the same calendar rule.
    """
    start = 0 if rows.start is None else rows.start
    if rows.stop is None or horizon < 1 or rows.stop - start <= horizon:
        raise ValueError("window is too short to build an outcome-complete slice")
    return slice(start, rows.stop - horizon)


def fit_selection_windows(
    n_rows: int,
    embargo_days: int,
    fit_fraction: float = 0.8,
    min_selection_rows: int = 20,
) -> tuple[slice, slice]:
    """Return the two training-only blocks used by GP and factor monitoring."""
    if n_rows <= 0 or embargo_days < 0 or not 0 < fit_fraction < 1:
        raise ValueError("invalid fit/selection window geometry")
    cut = int(n_rows * fit_fraction)
    selection_start = min(cut + embargo_days, n_rows)
    if n_rows - selection_start < min_selection_rows:
        raise ValueError(
            f"search window of {n_rows} dates is too short to hold out a selection block; "
            "increase split.train_days or lengthen the data range"
        )
    return slice(0, cut), slice(selection_start, n_rows)
