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


def build_basis(
    base_grids: list[np.ndarray], mask: np.ndarray, dtype=np.float32
) -> np.ndarray:
    """``(T, N, K+1)`` per-date orthonormal basis: an intercept plus each base column.

    Modified Gram-Schmidt, run across all dates at once. Degenerate directions (a
    constant column on some date, or a date with too few names) collapse to zero, which
    makes the projection a no-op there rather than producing garbage.

    Orthonormalisation runs in float64 for stability; the result is stored in ``dtype``
    (float32 by default) because :func:`residualize` runs once per candidate inside the
    GP loop and halving the bandwidth roughly halves that cost. Only ranks of the
    residual are ever used, so float32 is ample.
    """
    columns = [mask.astype(np.float64)]  # intercept, restricted to occupied slots
    columns.extend(_ranked(np.asarray(g, dtype=np.float64), mask) for g in base_grids)
    stacked = np.stack(columns, axis=-1)  # (T, N, K)

    # Batched QR rather than a Gram-Schmidt loop: one LAPACK call instead of K^2/2
    # passes over the panel. With 70 base columns that is seconds instead of minutes.
    # Rows outside the mask are zero in every column, so they stay zero in Q.
    q, r = np.linalg.qr(stacked)

    # Rank-deficient directions (a column constant on some date, or a date with fewer
    # names than columns) leave a negligible diagonal in R and an arbitrary direction in
    # Q. Zero those out so the projection is a no-op there rather than removing noise.
    diagonal = np.abs(np.diagonal(r, axis1=1, axis2=2))          # (T, K)
    tolerance = EPS * np.maximum(diagonal.max(axis=1, keepdims=True), EPS)
    q = np.where((diagonal > tolerance)[:, None, :], q, 0.0)
    return np.ascontiguousarray(q, dtype=dtype)


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
    ranks = _ranked(np.asarray(values, dtype=np.float64), mask).astype(basis.dtype, copy=False)

    # Two batched matmuls rather than einsum: both keep the contiguous basis as an
    # operand so they dispatch to BLAS. With a 70-column base this is the difference
    # between minutes and seconds per generation.
    coefficients = np.matmul(ranks[:, None, :], basis)                 # (T, 1, K)
    projection = np.matmul(basis, coefficients.transpose(0, 2, 1))     # (T, N, 1)
    residual = ranks - projection[:, :, 0]

    scale = np.einsum("tn,tn->t", ranks, ranks)
    magnitude = np.einsum("tn,tn->t", residual, residual)
    explained = magnitude <= (min_residual_fraction**2) * np.maximum(scale, EPS)
    return np.where(mask & ~explained[:, None], residual.astype(np.float64), np.nan)


def basis_from_fields(
    fields: dict[str, np.ndarray], names: list[str], mask: np.ndarray
) -> np.ndarray:
    missing = [n for n in names if n not in fields]
    if missing:
        raise KeyError(f"neutralisation base refers to unknown columns: {missing}")
    return build_basis([fields[n] for n in names], mask)
