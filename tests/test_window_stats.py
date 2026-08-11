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
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from window_stats import select_curve, window_metrics  # noqa: E402


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


def _long_csv() -> pd.DataFrame:
    """Two gates × two book sizes × two seeds × two days, the shape backtest_argus.py
    writes. The ungated rows carry an empty min_score, which pandas reads back as NaN."""
    rows = []
    for gate in (np.nan, 0.6):
        for hold in (2, 5):
            for seed in (7, 13):
                for date in ("20250101", "20250102"):
                    rows.append({"date": date, "portfolio_return": 0.01, "equity": 1.0,
                                 "seed": seed, "hold_k": hold, "exit": "close",
                                 "slippage_bps": 10.0, "min_score": gate})
    return pd.DataFrame(rows)


def test_selecting_the_ungated_book_does_not_also_pull_in_the_gated_one():
    """The bug: min_score == None never matches, so an unfiltered gate column would
    concatenate every variant into one series -- a 431-day track read as 3,017 days, with a
    fabricated jump wherever one variant ends and the next begins."""
    sel = select_curve(_long_csv(), "close", 10.0, "", hold=2)
    assert len(sel) == 4                              # 2 seeds × 2 days, ungated only
    assert sel["min_score"].isna().all()


def test_a_named_gate_selects_only_that_gate():
    sel = select_curve(_long_csv(), "close", 10.0, "0.6", hold=2)
    assert len(sel) == 4
    assert (sel["min_score"] == 0.6).all()


def test_selecting_a_book_size_does_not_also_pull_in_the_other_ones():
    """Book size is the axis being compared, so mixing two of them into one series would
    average away the exact difference the run exists to measure."""
    sel = select_curve(_long_csv(), "close", 10.0, "", hold=5)
    assert len(sel) == 4
    assert (sel["hold_k"] == 5).all()


def test_a_filter_that_leaves_two_rows_on_one_day_is_refused():
    """Better to stop than to silently cut windows out of two interleaved curves. Leaving
    the book size unpinned is the easy way to trip this."""
    with pytest.raises(SystemExit):
        select_curve(_long_csv(), "close", 10.0, "")


def test_an_older_csv_without_the_column_says_so_instead_of_filtering_nothing():
    df = _long_csv().drop(columns=["hold_k"])
    with pytest.raises(SystemExit):
        select_curve(df, "close", 10.0, "", hold=2)


def test_an_empty_selection_is_refused_rather_than_silently_producing_no_windows():
    with pytest.raises(SystemExit):
        select_curve(_long_csv(), "target", 10.0, "", hold=2)


def test_positive_day_rate_counts_strictly_positive_days():
    """A flat day is not a positive day. Under a conviction gate flat days are common, and
    counting them as wins would inflate exactly the metric being compared."""
    got = window_metrics(np.array([0.01, 0.0, -0.01, 0.01]), 4)
    assert float(got["pos_day_rate"].iloc[0]) == pytest.approx(0.5)
