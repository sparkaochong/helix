"""Operator contracts. The no-look-ahead test is the important one here."""

from __future__ import annotations

import numpy as np
import pytest

from helix.features import operators as ops


@pytest.fixture
def panel() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, 8))


def test_delay_shifts_backward(panel):
    delayed = ops.delay(panel, 3)
    assert np.isnan(delayed[:3]).all()
    np.testing.assert_allclose(delayed[3:], panel[:-3])


def test_delay_rejects_zero_and_negative(panel):
    with pytest.raises(ValueError):
        ops.delay(panel, 0)
    with pytest.raises(ValueError):
        ops.delay(panel, -1)


@pytest.mark.parametrize(
    "op",
    [ops.ts_mean, ops.ts_std, ops.ts_max, ops.ts_min, ops.ts_sum,
     ops.ts_rank, ops.ts_delta, ops.ts_zscore, ops.ts_argmax, ops.ts_decay_linear],
)
def test_time_series_operators_never_see_the_future(op, panel):
    """Changing the last row must not change any earlier row of the output."""
    window = 5
    before = op(panel, window)
    mutated = panel.copy()
    mutated[-1] += 100.0
    after = op(mutated, window)

    head_before, head_after = before[:-1], after[:-1]
    both_nan = np.isnan(head_before) & np.isnan(head_after)
    assert np.allclose(head_before[~both_nan], head_after[~both_nan], equal_nan=True)


def test_lead_is_the_only_forward_looking_helper(panel):
    led = ops.lead(panel, 2)
    np.testing.assert_allclose(led[:-2], panel[2:])
    assert np.isnan(led[-2:]).all()


def test_protected_division_yields_nan_not_inf():
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[0.0, 1e-15, 2.0]])
    out = ops.div(x, y)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[0, 1])
    assert out[0, 2] == pytest.approx(1.5)
    assert np.isfinite(out[np.isfinite(out)]).all()


def test_cs_rank_ignores_nan_and_scales_to_unit_interval():
    x = np.array([[3.0, 1.0, np.nan, 2.0]])
    ranks = ops.cs_rank(x)
    assert np.isnan(ranks[0, 2])
    # three valid values -> ranks 1/3, 2/3, 3/3 assigned by ascending value
    np.testing.assert_allclose(ranks[0, [1, 3, 0]], [1 / 3, 2 / 3, 1.0])


def test_cs_rank_ordinal_reports_valid_mask():
    x = np.array([[np.nan, 5.0, 4.0]])
    ranks, valid = ops.cs_rank_ordinal(x)
    assert valid.tolist() == [[False, True, True]]
    assert ranks[0, 0] == 0.0
    assert ranks[0, 2] == 1.0 and ranks[0, 1] == 2.0


def test_cs_zscore_is_nan_for_constant_rows():
    x = np.array([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
    z = ops.cs_zscore(x)
    assert np.isnan(z[0]).all()
    assert z[1, 0] < 0 < z[1, 2]


def test_rolling_requires_a_full_window():
    x = np.arange(10, dtype=float).reshape(10, 1)
    out = ops.ts_mean(x, 4)
    assert np.isnan(out[:3, 0]).all()
    assert out[3, 0] == pytest.approx(1.5)
