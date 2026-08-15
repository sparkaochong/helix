"""bak_daily.amount (万元) -> daily.amount's convention (千元). Real-world regression
fixture from the 2026-08-15 data baseline audit: 000001.SZ / 20260731, where
stk_factor_pro.amount (千元, matches daily.amount) and bak_daily.amount (万元) were
both fetched live for the same code/date."""

from __future__ import annotations

import numpy as np

from helix.data.unit_registry import BAK_DAILY_AMOUNT_TO_KCNY, bak_daily_amount_to_kcny

BAK_DAILY_AMOUNT_WAN_YUAN = 231883.98
STK_FACTOR_PRO_AMOUNT_KCNY = 2318839.88  # same code/date, independently confirmed 千元


def test_conversion_factor_is_ten():
    assert BAK_DAILY_AMOUNT_TO_KCNY == 10.0


def test_matches_the_real_paired_observation_within_rounding():
    converted = bak_daily_amount_to_kcny(BAK_DAILY_AMOUNT_WAN_YUAN)
    assert abs(converted - STK_FACTOR_PRO_AMOUNT_KCNY) < 0.1


def test_accepts_arrays():
    result = bak_daily_amount_to_kcny(np.array([1.0, 2.5, 0.0]))
    assert np.allclose(result, [10.0, 25.0, 0.0])


def test_accepts_a_plain_float():
    assert bak_daily_amount_to_kcny(3.0) == 30.0
