"""Pin the window arithmetic, because the whole comparison rests on it.

The claim these numbers support is that a 28-day drawdown and a 431-day drawdown are not
the same quantity. That argument is only worth making if the 28-day one is computed the
way a 28-day track would actually report it -- measured from the capital you started the
window with, not from wherever the curve happened to peak inside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from window_stats import window_metrics  # noqa: E402


def test_every_window_of_the_length_is_emitted_and_no_more():
    """n - w + 1 windows, one per possible start."""
    assert len(window_metrics(np.zeros(10), 3)) == 8
    assert len(window_metrics(np.zeros(10), 10)) == 1
    assert window_metrics(np.zeros(10), 11).empty


def test_cumulative_return_compounds_rather_than_sums():
    """+10% then +10% is +21%, not +20%. At 28 days the difference is not a rounding one."""
    got = window_metrics(np.array([0.10, 0.10]), 2)
    assert float(got["cum_return"].iloc[0]) == pytest.approx(0.21)


def test_a_window_that_opens_by_falling_has_drawn_down_from_its_own_start():
    """The baseline bug: a running peak seeded at day 1 instead of at the opening capital
    would call this window flat, because the curve never exceeds where it began."""
    got = window_metrics(np.array([-0.05, 0.02]), 2)
    assert float(got["max_drawdown"].iloc[0]) == pytest.approx(-0.05)


def test_drawdown_is_measured_from_the_peak_inside_the_window():
    up_then_down = window_metrics(np.array([0.20, -0.10]), 2)
    assert float(up_then_down["max_drawdown"].iloc[0]) == pytest.approx(-0.10)


def test_drawdown_only_deepens_as_the_window_grows():
    """The reason the reference comparison is confounded, stated as an invariant: a longer
    window contains every shorter one it spans, so its minimum cannot be shallower."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.02, 200)
    worst = [float(window_metrics(returns, w)["max_drawdown"].min()) for w in (10, 40, 160)]
    assert worst[0] >= worst[1] >= worst[2]


def test_positive_day_rate_counts_strictly_positive_days():
    """A flat day is not a positive day. Under a conviction gate flat days are common, and
    counting them as wins would inflate exactly the metric being compared."""
    got = window_metrics(np.array([0.01, 0.0, -0.01, 0.01]), 4)
    assert float(got["pos_day_rate"].iloc[0]) == pytest.approx(0.5)
