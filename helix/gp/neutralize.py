"""Cross-sectional neutralisation against a basis of existing columns.

Mining on raw IC finds factors that are good linear blends of columns the downstream
model already has. Measured honestly, such a factor adds nothing: orthogonalise it
against its own inputs and the residual IC collapses to zero. That is not a
hypothetical -- it is what the first mined factor here did (raw IC +0.087, residual IC
-0.006 against its three inputs).

So fitness is measured on the **residual** after projecting the factor onto the base
columns, per date. A factor only scores if it explains something the base set cannot.

The per-date basis is orthonormalised once up front, which turns each factor's
residualisation into two einsums instead of a least-squares solve per date -- the
difference between a usable inner loop and an unusable one.

**Known limitation.** The projection is linear in rank space, so it removes linear
combinations of the base columns but not arbitrary *monotone nonlinear* functions of
them. A factor that is, say, a steep sigmoid of a base column keeps part of its signal
through neutralisation. Treat a surviving factor as a candidate to verify, not as proof
of independence -- confirm incremental value against the real feature set before
trusting it.
"""

from __future__ import annotations

import numpy as np

from ..features.operators import cs_rank

EPS = 1e-9


def _ranked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks in (0, 1], zero-filled outside the mask.

    Zero-filling is safe because every vector in this module is masked identically, so
    the excluded cells contribute nothing to any inner product.
    """
    ranks = cs_rank(np.where(mask, values, np.nan))
    return np.where(mask & np.isfinite(ranks), ranks, 0.0)


def build_basis(base_grids: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """``(T, N, K+1)`` per-date orthonormal basis: an intercept plus each base column.

    Modified Gram-Schmidt, run across all dates at once. Degenerate directions (a
    constant column on some date, or a date with too few names) collapse to zero, which
    makes the projection a no-op there rather than producing garbage.
    """
    n_dates, n_slots = mask.shape
    columns = [mask.astype(np.float64)]  # intercept, restricted to occupied slots
    columns.extend(_ranked(np.asarray(g, dtype=np.float64), mask) for g in base_grids)

    basis = np.zeros((n_dates, n_slots, len(columns)), dtype=np.float64)
    for k, column in enumerate(columns):
        v = column.copy()
        for j in range(k):
            projection = np.einsum("tn,tn->t", basis[:, :, j], v)
            v -= basis[:, :, j] * projection[:, None]
        norm = np.sqrt(np.einsum("tn,tn->t", v, v))
        basis[:, :, k] = np.where(norm[:, None] > EPS, v / np.maximum(norm, EPS)[:, None], 0.0)
    return basis


def residualize(
    values: np.ndarray,
    basis: np.ndarray,
    mask: np.ndarray,
    min_residual_fraction: float = 1e-6,
) -> np.ndarray:
    """Component of ``values`` that the basis cannot explain, per date.

    Dates where the residual is numerically negligible are returned as NaN rather than
    as float noise. That guard is essential, not defensive: the metrics downstream are
    **rank based and scale free**, so a residual of magnitude 1e-16 still carries the
    original ordering and would score a fully-explained factor as if it were predictive.
    A factor the basis explains completely must contribute no dates at all.

    Returns NaN outside the mask so downstream metrics skip those cells exactly as they
    would for a raw factor.
    """
    ranks = _ranked(np.asarray(values, dtype=np.float64), mask)
    coefficients = np.einsum("tnk,tn->tk", basis, ranks)
    residual = ranks - np.einsum("tnk,tk->tn", basis, coefficients)

    scale = np.sqrt(np.einsum("tn,tn->t", ranks, ranks))
    magnitude = np.sqrt(np.einsum("tn,tn->t", residual, residual))
    explained = magnitude <= min_residual_fraction * np.maximum(scale, EPS)
    return np.where(mask & ~explained[:, None], residual, np.nan)


def basis_from_fields(
    fields: dict[str, np.ndarray], names: list[str], mask: np.ndarray
) -> np.ndarray:
    missing = [n for n in names if n not in fields]
    if missing:
        raise KeyError(f"neutralisation base refers to unknown columns: {missing}")
    return build_basis([fields[n] for n in names], mask)
