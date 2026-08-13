"""Cross-sectional metrics for a binary, heavily imbalanced label.

Everything here is computed **per trade date and then averaged**, never pooled across
dates. Pooling lets a factor score well merely by knowing which days were strong
market-wide, which is useless for a strategy that must pick names within a day.

The AUC is derived from ranks in closed form rather than by sorting per row, so a
full ``(T, N)`` evaluation costs one argsort -- fast enough to sit inside the GP loop.
"""

from __future__ import annotations

import numpy as np

from ..features.operators import cs_rank_ordinal


def daily_auc(
    factor: np.ndarray, y: np.ndarray, mask: np.ndarray, min_samples: int = 50
) -> np.ndarray:
    """Per-date AUC of ``factor`` against binary ``y``. NaN on unusable dates."""
    usable = mask & np.isfinite(factor) & np.isfinite(y)
    scores = np.where(usable, factor, np.nan)
    ranks, valid = cs_rank_ordinal(scores)

    pos = valid & (y == 1.0)
    neg = valid & (y == 0.0)
    n_pos = pos.sum(axis=1).astype(np.float64)
    n_neg = neg.sum(axis=1).astype(np.float64)
    rank_sum_pos = np.where(pos, ranks, 0.0).sum(axis=1)

    denom = n_pos * n_neg
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / np.where(denom > 0, denom, 1.0)
    ok = (denom > 0) & ((n_pos + n_neg) >= min_samples)
    return np.where(ok, auc, np.nan)


def daily_gini(factor, y, mask, min_samples: int = 50) -> np.ndarray:
    """``2 * AUC - 1`` -- zero for a useless factor, symmetric around zero."""
    return 2.0 * daily_auc(factor, y, mask, min_samples) - 1.0


def summarize_daily(values: np.ndarray) -> dict[str, float]:
    """Mean / std / information ratio / coverage of a per-date metric series."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ir": float("nan"), "coverage": 0.0}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "ir": mean / std if std and std > 0 else float("nan"),
        "coverage": float(finite.size / values.size),
    }


def precision_at_k(
    score: np.ndarray, y: np.ndarray, mask: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-date hit rate of the D0 top-``k`` names, plus the observable base rate.

    ``mask`` is the point-in-time candidate pool. Ranking never consults ``y``; if a
    selected outcome is unobservable, precision remains NaN rather than substituting a
    deeper-ranked name. Base rate uses the observable labels inside the D0 pool.
    """
    candidates = mask & np.isfinite(score)
    scores = np.where(candidates, score, -np.inf)
    n_dates = score.shape[0]
    precision = np.full(n_dates, np.nan)
    base = np.full(n_dates, np.nan)

    counts = candidates.sum(axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    for t in range(n_dates):
        observable = candidates[t] & np.isfinite(y[t])
        if observable.any():
            base[t] = float(np.mean(y[t, observable]))
        if counts[t] < k:
            continue
        picked = order[t, :k]
        if not np.isfinite(y[t, picked]).all():
            continue
        precision[t] = float(np.mean(y[t, picked]))
    return precision, base


def lift_at_k(score, y, mask, k: int) -> float:
    """How many times better than random the top-``k`` selection is, averaged over dates."""
    precision, base = precision_at_k(score, y, mask, k)
    ok = np.isfinite(precision) & np.isfinite(base) & (base > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(precision[ok] / base[ok]))


def pairwise_max_abs_corr(candidate: np.ndarray, kept: list[np.ndarray]) -> float:
    """Largest ``|corr|`` between a candidate factor and any already-kept factor.

    Correlation is measured on cross-sectional ranks over the cells where both are
    defined, so it reflects "would these two pick the same names" rather than raw scale.
    """
    if not kept:
        return 0.0
    best = 0.0
    flat_c = candidate.ravel()
    for other in kept:
        flat_o = other.ravel()
        both = np.isfinite(flat_c) & np.isfinite(flat_o)
        if both.sum() < 100:
            continue
        a, b = flat_c[both], flat_o[both]
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        corr = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        best = max(best, abs(corr))
    return best
