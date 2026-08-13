"""Behavioral contract for fully vectorized, daily style neutralization."""

from __future__ import annotations

import numpy as np
import pytest
from helix.eval.style_neutralize import build_style_design, style_residualize


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260813)
    continuous = rng.normal(size=(2, 12, 4))
    industry = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        ],
        dtype=float,
    )
    nonlinear = continuous[..., 0] ** 2 + 0.4 * continuous[..., 1] ** 3
    factor = (
        nonlinear
        + 2.0 * continuous[..., 0]
        - 0.7 * continuous[..., 2]
        + np.where(industry == 1, 1.5, 0.0)
        - np.where(industry == 2, 0.8, 0.0)
    )
    mask = np.ones(factor.shape, dtype=bool)
    levels = np.array([0, 1, 2], dtype=float)
    return factor, continuous, industry, mask, levels


def test_residual_is_orthogonal_to_each_same_day_style_column():
    factor, continuous, industry, mask, levels = _sample()
    residual = style_residualize(
        factor, continuous, industry, mask, industry_levels=levels
    )
    design, valid = build_style_design(
        continuous, industry, mask, industry_levels=levels
    )

    assert np.isfinite(residual[valid]).all()
    products = np.matmul(design.transpose(0, 2, 1), np.nan_to_num(residual)[..., None])
    np.testing.assert_allclose(products, 0.0, atol=2e-9)


def test_changing_a_later_date_cannot_change_an_earlier_residual():
    factor, continuous, industry, mask, levels = _sample()
    before = style_residualize(
        factor, continuous, industry, mask, industry_levels=levels
    )
    changed_factor = factor.copy()
    changed_continuous = continuous.copy()
    changed_industry = industry.copy()
    changed_factor[1] = np.linspace(-1000.0, 1000.0, factor.shape[1])
    changed_continuous[1] = changed_continuous[1, ::-1] * 1e6
    changed_industry[1] = changed_industry[1, ::-1]

    after = style_residualize(
        changed_factor,
        changed_continuous,
        changed_industry,
        mask,
        industry_levels=levels,
    )

    np.testing.assert_array_equal(before[0], after[0])


def test_missing_rows_stay_nan_and_do_not_enter_the_regression():
    factor, continuous, industry, mask, levels = _sample()
    continuous[0, 3, 2] = np.nan
    with_missing = style_residualize(
        factor, continuous, industry, mask, industry_levels=levels
    )
    explicit_mask = mask.copy()
    explicit_mask[0, 3] = False
    explicitly_removed = style_residualize(
        factor, continuous, industry, explicit_mask, industry_levels=levels
    )

    assert np.isnan(with_missing[0, 3])
    np.testing.assert_allclose(with_missing, explicitly_removed, equal_nan=True, atol=1e-12)


def test_absent_and_collinear_industries_are_rank_safe():
    factor, continuous, industry, mask, _ = _sample()
    continuous[..., 3] = continuous[..., 0]
    industry[0] = 0
    levels = np.array([0, 1, 2, 3], dtype=float)

    residual = style_residualize(
        factor, continuous, industry, mask, industry_levels=levels
    )

    assert np.isfinite(residual[0]).all()
    assert np.isfinite(residual[1]).all()


def test_fully_explained_factor_is_nan_not_rankable_float_noise():
    _, continuous, industry, mask, levels = _sample()
    factor = (
        4.0
        + 2.5 * continuous[..., 0]
        - 1.2 * continuous[..., 1]
        + np.where(industry == 1, 0.7, 0.0)
        - np.where(industry == 2, 1.1, 0.0)
    )

    residual = style_residualize(
        factor, continuous, industry, mask, industry_levels=levels
    )

    assert np.isnan(residual).all()


def test_inputs_are_not_mutated():
    factor, continuous, industry, mask, levels = _sample()
    snapshots = tuple(array.copy() for array in (factor, continuous, industry, mask, levels))

    style_residualize(factor, continuous, industry, mask, industry_levels=levels)

    for actual, expected in zip(
        (factor, continuous, industry, mask, levels), snapshots, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_batched_result_matches_daily_lstsq_reference():
    factor, continuous, industry, mask, levels = _sample()
    factor[0, 0] = np.nan
    residual = style_residualize(
        factor, continuous, industry, mask, industry_levels=levels
    )
    design, design_valid = build_style_design(
        continuous,
        industry,
        mask & np.isfinite(factor),
        industry_levels=levels,
    )
    expected = np.full_like(factor, np.nan)
    for date_index in range(factor.shape[0]):
        valid = design_valid[date_index]
        x = design[date_index, valid]
        y = factor[date_index, valid]
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        expected[date_index, valid] = y - x @ coefficients

    np.testing.assert_allclose(residual, expected, equal_nan=True, atol=2e-9)


@pytest.mark.parametrize(
    ("factor_shape", "continuous_shape", "industry_shape", "mask_shape"),
    [
        ((2, 3), (2, 3, 4), (2, 3), (2, 2)),
        ((2, 3), (2, 2, 4), (2, 3), (2, 3)),
        ((2, 3), (2, 3, 4), (3, 3), (2, 3)),
    ],
)
def test_shapes_are_validated(
    factor_shape, continuous_shape, industry_shape, mask_shape
):
    with pytest.raises(ValueError, match="shape"):
        style_residualize(
            np.ones(factor_shape),
            np.ones(continuous_shape),
            np.zeros(industry_shape),
            np.ones(mask_shape, dtype=bool),
            industry_levels=np.array([0.0, 1.0]),
        )


@pytest.mark.parametrize(
    "levels",
    [np.array([]), np.array([0.0, 0.0]), np.array([0.0, np.nan])],
)
def test_industry_levels_are_validated(levels):
    factor, continuous, industry, mask, _ = _sample()
    with pytest.raises(ValueError, match="industry_levels"):
        style_residualize(
            factor, continuous, industry, mask, industry_levels=levels
        )


@pytest.mark.parametrize("fraction", [0.0, -1.0, np.nan, np.inf])
def test_residual_fraction_is_positive_and_finite(fraction):
    factor, continuous, industry, mask, levels = _sample()
    with pytest.raises(ValueError, match="min_residual_fraction"):
        style_residualize(
            factor,
            continuous,
            industry,
            mask,
            industry_levels=levels,
            min_residual_fraction=fraction,
        )
