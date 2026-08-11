"""Narrowing 459 raw columns down to a workable GP terminal set.

Two independent reasons to prune, both decisive:

* **Search space.** GP terminal choice is uniform; with 459 terminals most trees are
  built from columns that carry nothing, and the population spends its budget rediscovering
  that. Fifty to a hundred informative terminals converge far faster.
* **Redundancy.** The source table has whole families of near-identical columns
  (``stock_intra_amp_d0/_d1/_d2/_d1d3_mean``). Keeping all of them biases the search
  toward whichever idea happens to have the most aliases.

Selection runs **only on the search window**, exactly like the factors themselves --
screening features on data the walk-forward will later score against is the same leak
one level up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.ic import daily_ic, summarize_ic
from ..features.operators import cs_rank
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class FeatureScore:
    name: str
    ic_mean: float
    icir: float
    positive_rate: float
    coverage: float


def score_features(
    fields: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    min_samples: int = 30,
) -> list[FeatureScore]:
    """Univariate per-date IC of every column against the target, best first."""
    scores: list[FeatureScore] = []
    for name, values in fields.items():
        arr = np.asarray(values, dtype=np.float64)
        defined = np.isfinite(arr) & mask
        if defined.sum() < 0.2 * max(mask.sum(), 1):
            continue
        stats = summarize_ic(daily_ic(arr, target, mask, min_samples))
        if not np.isfinite(stats["ic_mean"]):
            continue
        scores.append(
            FeatureScore(
                name=name,
                ic_mean=stats["ic_mean"],
                icir=stats["icir"],
                positive_rate=stats["positive_rate"],
                coverage=stats["coverage"],
            )
        )
    scores.sort(key=lambda s: abs(s.ic_mean), reverse=True)
    return scores


def select_features(
    fields: dict[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    n_keep: int = 80,
    max_abs_corr: float = 0.85,
    min_abs_ic: float = 0.005,
    min_samples: int = 30,
) -> tuple[list[str], list[FeatureScore]]:
    """Greedy: take the strongest column, drop anything too correlated with it, repeat.

    Correlation is measured on cross-sectional ranks, so it reflects "do these two order
    the names the same way" rather than raw scale agreement.
    """
    scored = score_features(fields, target, mask, min_samples)
    log.info("scored %d/%d columns with a usable IC", len(scored), len(fields))

    kept: list[str] = []
    kept_ranks: list[np.ndarray] = []
    kept_scores: list[FeatureScore] = []

    for score in scored:
        if len(kept) >= n_keep:
            break
        if abs(score.ic_mean) < min_abs_ic:
            break  # sorted by |ic|, so nothing later can qualify either
        ranks = cs_rank(np.where(mask, np.asarray(fields[score.name], dtype=np.float64), np.nan))
        flat = ranks.ravel()
        if _max_abs_corr(flat, kept_ranks) > max_abs_corr:
            continue
        kept.append(score.name)
        kept_ranks.append(flat)
        kept_scores.append(score)

    log.info(
        "kept %d features (|IC| %.4f ~ %.4f, dedup at |corr| <= %.2f)",
        len(kept),
        abs(kept_scores[-1].ic_mean) if kept_scores else float("nan"),
        abs(kept_scores[0].ic_mean) if kept_scores else float("nan"),
        max_abs_corr,
    )
    return sorted(kept), kept_scores


def _max_abs_corr(candidate: np.ndarray, kept: list[np.ndarray]) -> float:
    best = 0.0
    for other in kept:
        both = np.isfinite(candidate) & np.isfinite(other)
        if both.sum() < 1000:
            continue
        a, b = candidate[both], other[both]
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        corr = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        best = max(best, abs(corr))
        if best > 0.999:
            break
    return best
