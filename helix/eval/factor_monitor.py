"""Training-only monitors for factors selected by the economic objective.

These diagnostics deliberately sit outside GP fitness.  They explain what a retained
factor is doing without giving hit-label IC, gini, or supplemental Top10 results any
vote in evolution or survivor ordering.
"""

from __future__ import annotations

import numpy as np

from ..config import BacktestConfig
from ..splits import fit_selection_windows
from .ic import daily_ic, summarize_ic
from .metrics import daily_gini, summarize_daily
from .objective import (
    cost_adjusted_returns,
    daily_top_k_portfolio,
    summarize_objective,
)

SUPPLEMENTAL_TOP_K = 10


def _top_k_hit(
    score: np.ndarray,
    hit_label: np.ndarray,
    candidate_mask: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    eligible = candidate_mask & np.isfinite(score)
    enough = eligible.sum(axis=1) >= top_k
    order = np.argsort(np.where(eligible, -score, np.inf), axis=1, kind="stable")
    picked = order[:, :top_k]
    selected = np.take_along_axis(hit_label, picked, axis=1)
    top_hit = float(np.where(np.isfinite(selected), selected, 0.0)[enough].mean()) if enough.any() else float("nan")
    observed = candidate_mask & np.isfinite(hit_label)
    base = float(hit_label[observed].mean()) if observed.any() else float("nan")
    lift = top_hit / base if np.isfinite(top_hit) and base > 0 else float("nan")
    return {
        "hit_rate": top_hit,
        "base_rate": base,
        "lift": float(lift),
        "coverage": float(enough.mean()),
    }


def _objective_role(
    score: np.ndarray,
    gross_return: np.ndarray,
    net_return: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    top_k: int,
    overlap: int,
    role: str,
) -> dict[str, object]:
    gross = daily_top_k_portfolio(
        score, gross_return, candidate_mask, top_k=top_k, overlap=overlap
    )
    net = daily_top_k_portfolio(
        score, net_return, candidate_mask, top_k=top_k, overlap=overlap
    )
    return {
        "role": role,
        "top_k": top_k,
        "gross": summarize_objective(gross, top_k),
        "net": summarize_objective(net, top_k),
    }


def _block(
    score: np.ndarray,
    hit_label: np.ndarray,
    peak_return: np.ndarray,
    gross_return: np.ndarray,
    net_return: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    production_top_k: int,
    overlap: int,
    min_samples: int,
) -> dict[str, object]:
    return {
        "production_objective": _objective_role(
            score,
            gross_return,
            net_return,
            candidate_mask,
            top_k=production_top_k,
            overlap=overlap,
            role="production_objective",
        ),
        "supplemental_top10": _objective_role(
            score,
            gross_return,
            net_return,
            candidate_mask,
            top_k=SUPPLEMENTAL_TOP_K,
            overlap=overlap,
            role="supplemental_only",
        ),
        "hit_ic": summarize_ic(daily_ic(score, hit_label, candidate_mask, min_samples)),
        "hit_gini": summarize_daily(
            daily_gini(score, hit_label, candidate_mask, min_samples)
        ),
        "top_k_hit": _top_k_hit(score, hit_label, candidate_mask, production_top_k),
        "peak_return_ic": summarize_ic(
            daily_ic(score, peak_return, candidate_mask, min_samples)
        ),
        "close_return_ic": summarize_ic(
            daily_ic(score, gross_return, candidate_mask, min_samples)
        ),
    }


def evaluate_training_monitors(
    *,
    score: np.ndarray,
    hit_label: np.ndarray,
    peak_return: np.ndarray,
    gross_return: np.ndarray,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
    entry_offset: int,
    touch_offset: int,
    embargo_days: int,
    min_samples: int,
) -> dict[str, dict[str, object]]:
    """Evaluate fit, selection, and full blocks of one outcome-complete training grid."""
    scores = np.asarray(score, dtype=np.float64)
    hit = np.asarray(hit_label, dtype=np.float64)
    peak = np.asarray(peak_return, dtype=np.float64)
    gross = np.asarray(gross_return, dtype=np.float64)
    candidates = np.asarray(candidate_mask, dtype=bool)
    shapes = {array.shape for array in (scores, hit, peak, gross, candidates)}
    if len(shapes) != 1 or scores.ndim != 2:
        raise ValueError("factor monitor arrays must share one two-dimensional shape")
    overlap = touch_offset - entry_offset + 1
    if overlap <= 0:
        raise ValueError("touch_offset must be greater than or equal to entry_offset")

    net = cost_adjusted_returns(gross, dates, config)
    fit, selection = fit_selection_windows(len(scores), embargo_days)
    blocks = {
        "fit": fit,
        "selection": selection,
        "training_full": slice(0, len(scores)),
    }
    return {
        name: _block(
            scores[rows],
            hit[rows],
            peak[rows],
            gross[rows],
            net[rows],
            candidates[rows],
            production_top_k=config.top_k,
            overlap=overlap,
            min_samples=min_samples,
        )
        for name, rows in blocks.items()
    }
