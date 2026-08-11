"""The board-rate taxonomy decides which rows count as unbuyable, so pin it.

A wrong rate fails silently and in the dangerous direction: assume 10% for a 创业板 name
and its 20% limit-up open is never flagged, so an entry you could not have filled stays
in the sample as a win. The audit's own empirical check (a flagged row must satisfy
``open == high``) catches a rate that is too *low*; nothing catches one that is too high
except a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fillability import (  # noqa: E402
    MAIN_BOARD_RATE,
    board_rate,
    st_suspect_count,
    unfillable_mask,
    up_limit_d1,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("600519.SH", 0.10),   # 沪主板
        ("000001.SZ", 0.10),   # 深主板
        ("002415.SZ", 0.10),   # 原中小板，与主板同为 10%
        ("300750.SZ", 0.20),   # 创业板
        ("301029.SZ", 0.20),
        ("688981.SH", 0.20),   # 科创板
        ("830799.BJ", 0.30),   # 北交所
        ("430139.BJ", 0.30),
    ],
)
def test_board_rate_matches_the_exchange_taxonomy(code, expected):
    assert board_rate(pd.Series([code]))[0] == pytest.approx(expected)


def test_unknown_prefix_falls_back_to_the_main_board_rate():
    """Unrecognised is the conservative choice: 10% flags more rows, never fewer."""
    assert board_rate(pd.Series(["123456.XX"]))[0] == pytest.approx(MAIN_BOARD_RATE)


def _frame(code: str, close_d0: float, gap: float) -> pd.DataFrame:
    return pd.DataFrame({
        "stock_code": [code],
        "label_px_d1_open": [round(close_d0 * (1 + gap), 2)],
        "label_open_gap": [gap],
    })


def test_limit_price_is_rounded_to_the_fen():
    # 10.005 * 1.1 = 11.0055 -> the exchange quotes 11.01, not 11.0055.
    assert up_limit_d1(_frame("600000.SH", 10.005, 0.0))[0] == pytest.approx(11.01)


def test_a_main_board_limit_up_open_is_unfillable_and_a_hair_below_is_not():
    assert unfillable_mask(_frame("600000.SH", 10.00, 0.10))[0]
    assert not unfillable_mask(_frame("600000.SH", 10.00, 0.09))[0]


def test_a_chinext_name_needs_twenty_percent_not_ten():
    """The failure this test exists for: 300xxx gapping +10% is NOT at its limit."""
    assert not unfillable_mask(_frame("300750.SZ", 10.00, 0.10))[0]
    assert unfillable_mask(_frame("300750.SZ", 10.00, 0.20))[0]


def test_a_beijing_name_needs_thirty_percent():
    assert not unfillable_mask(_frame("830799.BJ", 10.00, 0.20))[0]
    assert unfillable_mask(_frame("830799.BJ", 10.00, 0.30))[0]


def test_a_missing_open_is_not_flagged():
    """No open price means no entry to judge; NaN must not read as 'unfillable'."""
    frame = pd.DataFrame({
        "stock_code": ["600000.SH"],
        "label_px_d1_open": [np.nan],
        "label_open_gap": [0.10],
    })
    assert not unfillable_mask(frame)[0]


def test_missing_columns_are_a_clear_error_not_a_wrong_answer():
    with pytest.raises(KeyError, match="fillability needs columns"):
        unfillable_mask(pd.DataFrame({"stock_code": ["600000.SH"]}))


def test_st_suspects_bound_the_five_percent_blind_spot():
    """ST names cap at 5%; with no name history they can only be counted, not flagged."""
    frame = pd.DataFrame({"label_open_gap": [0.05, 0.0501, 0.048, 0.10, -0.05]})
    assert st_suspect_count(frame) == 2
