"""Daily cross-sectional neutralization against explicit economic styles.

Unlike :mod:`helix.gp.neutralize`, this module works in value space against a fixed
economic design: continuous styles plus industry indicators. Every linear algebra
operation is batched by date; no observation from one date can enter another date's
fit.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _validate_shapes(
    factor: np.ndarray | None,
    continuous: np.ndarray,
    industry: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int]:
    if continuous.ndim != 3:
        raise ValueError("continuous must have shape (dates, names, styles)")
    shape = continuous.shape[:2]
    if industry.shape != shape:
        raise ValueError(f"industry shape {industry.shape} does not match {shape}")
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    if factor is not None and factor.shape != shape:
        raise ValueError(f"factor shape {factor.shape} does not match {shape}")
    return shape


def _validated_levels(
    industry: np.ndarray, industry_levels: np.ndarray | None
) -> np.ndarray:
    if industry_levels is None:
        levels = np.unique(industry[np.isfinite(industry)])
    else:
        levels = np.asarray(industry_levels, dtype=np.float64).reshape(-1).copy()
    if levels.size == 0 or not np.isfinite(levels).all():
        raise ValueError("industry_levels must be nonempty and finite")
    if np.unique(levels).size != levels.size:
        raise ValueError("industry_levels must contain unique values")
    return levels


def build_style_design(
    continuous: np.ndarray,
    industry: np.ndarray,
    mask: np.ndarray,
    *,
    industry_levels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an intercept, standardized styles, and ``L-1`` industry dummies.

    Standardization is cross-sectional and date-local. Rows lacking any required style
    are excluded from every column. The final industry is the reference category, so
    the intercept and indicators are not deterministically collinear.
    """
    continuous_array = np.asarray(continuous, dtype=np.float64)
    industry_array = np.asarray(industry, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    _validate_shapes(None, continuous_array, industry_array, mask_array)
    levels = _validated_levels(industry_array, industry_levels)

    valid = (
        mask_array
        & np.isfinite(continuous_array).all(axis=2)
        & np.isfinite(industry_array)
    )
    known_industry = np.any(industry_array[..., None] == levels, axis=2)
    valid &= known_industry

    weights = valid[..., None].astype(np.float64)
    counts = valid.sum(axis=1, keepdims=True)[..., None]
    safe_counts = np.maximum(counts, 1)
    values = np.where(valid[..., None], continuous_array, 0.0)
    means = values.sum(axis=1, keepdims=True) / safe_counts
    centered = np.where(valid[..., None], continuous_array - means, 0.0)
    squared = np.sum(centered * centered, axis=1, keepdims=True)
    denominator = np.maximum(counts - 1, 1)
    std = np.sqrt(squared / denominator)
    standardized = np.divide(
        centered,
        std,
        out=np.zeros_like(centered),
        where=std > EPS,
    )
    standardized *= weights

    dummies = (
        (industry_array[..., None] == levels[:-1]) & valid[..., None]
    ).astype(np.float64)
    intercept = valid[..., None].astype(np.float64)
    design = np.concatenate((intercept, standardized, dummies), axis=2)
    return np.ascontiguousarray(design), valid


def style_residualize(
    factor: np.ndarray,
    continuous: np.ndarray,
    industry: np.ndarray,
    mask: np.ndarray,
    *,
    industry_levels: np.ndarray | None = None,
    min_residual_fraction: float = 1e-6,
) -> np.ndarray:
    """Return the batched per-date residual after projecting onto style columns."""
    if not np.isfinite(min_residual_fraction) or min_residual_fraction <= 0:
        raise ValueError("min_residual_fraction must be positive and finite")

    factor_array = np.asarray(factor, dtype=np.float64)
    continuous_array = np.asarray(continuous, dtype=np.float64)
    industry_array = np.asarray(industry, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    _validate_shapes(factor_array, continuous_array, industry_array, mask_array)

    design, valid = build_style_design(
        continuous_array,
        industry_array,
        mask_array & np.isfinite(factor_array),
        industry_levels=industry_levels,
    )
    values = np.where(valid, factor_array, 0.0)
    transpose = design.transpose(0, 2, 1)
    gram = np.matmul(transpose, design)
    right_hand_side = np.matmul(transpose, values[..., None])
    inverse = np.linalg.pinv(gram, rcond=EPS, hermitian=True)
    coefficients = np.matmul(inverse, right_hand_side)
    projection = np.matmul(design, coefficients)[:, :, 0]
    residual = values - projection

    counts = valid.sum(axis=1)
    means = values.sum(axis=1) / np.maximum(counts, 1)
    centered = np.where(valid, values - means[:, None], 0.0)
    scale = np.einsum("tn,tn->t", centered, centered)
    magnitude = np.einsum("tn,tn->t", residual, residual)
    explained = magnitude <= (min_residual_fraction**2) * np.maximum(scale, EPS)
    return np.where(valid & ~explained[:, None], residual, np.nan)
