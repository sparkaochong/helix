"""Pin the accounting, because a costing bug reads as a strategy result.

The model needs xgboost and a GPU box; none of what is tested here does. These are the
parts that decide the *sign* of the answer while looking like arithmetic nobody needs to
check: how costs compose, what each exit rule pays out, whether an unfillable pick is
dropped or quietly replaced, and whether the regression target has had the market's daily
move taken out of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_argus import (  # noqa: E402
    COMMISSION_BPS,
    STAMP_SELL_BPS,
    TRANSFER_BPS,
    cost_rates,
    cross_sectional_z,
    gross_returns,
    net_return,
    run_book,
)

LABEL = "label_d2_hit_8pct"


def test_stamp_duty_is_charged_on_the_sell_side_only():
    buy, sell = cost_rates(0.0)
    assert buy == pytest.approx((COMMISSION_BPS + TRANSFER_BPS) / 1e4)
    assert sell - buy == pytest.approx(STAMP_SELL_BPS / 1e4)


def test_slippage_is_charged_on_both_sides():
    buy, sell = cost_rates(10.0)
    base_buy, base_sell = cost_rates(0.0)
    assert buy - base_buy == pytest.approx(10.0 / 1e4)
    assert sell - base_sell == pytest.approx(10.0 / 1e4)


def test_a_flat_trade_still_loses_the_round_trip():
    """Round-tripping at the entry price costs 2.6bp in, 7.6bp out -- 10.2bp all in."""
    assert float(net_return(np.array([0.0]), 0.0)[0]) == pytest.approx(-1.0197e-3, rel=1e-3)


def test_costs_scale_with_notional_so_this_is_not_a_subtraction():
    """The invariant a `gross - c` implementation would violate.

    Sell-side cost is charged on the exit value, so a winner pays more of it than a
    loser. Subtracting a constant would make the drag identical at every gross return and
    would flatter exactly the trades that matter most.
    """
    drag = lambda g: float(net_return(np.array([g]), 0.0)[0]) - g  # noqa: E731
    assert drag(0.20) < drag(0.0) < drag(-0.20)


def _picks(hit: tuple[int, ...], to_close: tuple[float, ...]) -> pd.DataFrame:
    open_d1 = 10.0
    return pd.DataFrame({
        LABEL: [float(h) for h in hit],
        "label_px_d1_open": [open_d1] * len(hit),
        "label_px_d2_close": [open_d1 * (1 + r) for r in to_close],
    })


def test_close_exit_ignores_the_label_entirely():
    frame = _picks(hit=(1, 0), to_close=(0.10, -0.05))
    assert gross_returns(frame, LABEL, 1.08, "close") == pytest.approx([0.10, -0.05])


def test_target_exit_caps_the_winner_and_lets_the_loser_run():
    """The whole finding in one assertion: +10% becomes +8%, -5% stays -5%."""
    frame = _picks(hit=(1, 0), to_close=(0.10, -0.05))
    assert gross_returns(frame, LABEL, 1.08, "target") == pytest.approx([0.08, -0.05])


def _book(scores, unfillable, hit, to_close) -> pd.DataFrame:
    frame = _picks(hit, to_close)
    frame["trade_date"] = "20250101"
    frame["score"] = scores
    frame["unfillable"] = unfillable
    return frame


def test_an_unfillable_pick_is_dropped_not_replaced_by_the_next_name():
    """Fill-aware, not queue-deeper.

    You submit k orders at D0 close. Substituting the next name down assumes you knew
    which of them would gap to the limit, which is the one thing D0 cannot tell you.
    """
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, False, False],
                 hit=(1, 0, 1), to_close=(0.09, -0.04, 0.09))
    res = run_book(book, LABEL, k=2, target_ratio=1.08, exit_rule="close", slippage_bps=0.0)
    assert res["avg_positions"] == 1.0                      # not 2: the third is not pulled in
    assert res["gross_per_trade"] == pytest.approx(-0.04)   # only the score-2.0 row
    assert res["hit_rate"] == 0.0


def test_base_rate_is_measured_over_fillable_rows_only():
    """Otherwise lift compares a fill-aware numerator to an as-is denominator."""
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, False, False],
                 hit=(1, 1, 0), to_close=(0.09, 0.09, -0.04))
    res = run_book(book, LABEL, k=2, target_ratio=1.08, exit_rule="close", slippage_bps=0.0)
    assert res["base_rate"] == pytest.approx(0.5)   # 1 of the 2 fillable, not 2 of 3


def _two_days() -> pd.DataFrame:
    # Same within-day ordering, shifted by a +10% market move on the second day.
    return pd.DataFrame({
        "trade_date": ["d1"] * 3 + ["d2"] * 3,
        "r": [-0.01, 0.00, 0.01, 0.09, 0.10, 0.11],
    })


def test_cross_sectional_z_removes_the_shared_daily_move():
    """A day that was broadly up must not outrank a flat day; only the within-day rank is
    information the book can act on."""
    z = cross_sectional_z(_two_days(), "r")
    assert z[:3] == pytest.approx(z[3:])
    assert z.mean() == pytest.approx(0.0, abs=1e-12)


def test_cross_sectional_z_is_zero_not_nan_when_a_day_has_no_dispersion():
    """Every name moving identically carries no cross-sectional signal. Dividing by a zero
    std would emit NaN/inf and poison the whole training target."""
    frame = pd.DataFrame({"trade_date": ["d1"] * 3, "r": [0.05, 0.05, 0.05]})
    z = cross_sectional_z(frame, "r")
    assert np.isfinite(z).all()
    assert z == pytest.approx([0.0, 0.0, 0.0])
