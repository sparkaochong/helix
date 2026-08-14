"""Vectorised production Top-K net-P&L objective.

Ranking uses only the point-in-time candidate mask and factor score.  Outcomes are
consulted after the selected names are fixed; an unobservable selection remains cash
instead of being replaced by a deeper-ranked name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BacktestConfig
from .backtest import _cost_rates, _net_returns


@dataclass(frozen=True)
class TopKPortfolio:
    portfolio_return: np.ndarray
    executed: np.ndarray


def _validate_aligned(*arrays: np.ndarray) -> tuple[int, int]:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("objective arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("objective arrays must be two-dimensional")
    return shape


def cost_adjusted_returns(
    gross_return: np.ndarray,
    dates: np.ndarray,
    config: BacktestConfig,
) -> np.ndarray:
    """Apply the backtest's historical buy/sell costs to an aligned return grid."""
    gross = np.asarray(gross_return, dtype=np.float64)
    date_values = np.asarray(dates)
    if gross.ndim != 2 or date_values.ndim != 1 or len(date_values) != gross.shape[0]:
        raise ValueError("dates must align with two-dimensional gross returns")
    date_digits = np.asarray([str(value).replace("-", "") for value in date_values])
    if np.any(date_digits[1:] <= date_digits[:-1]):
        raise ValueError("objective dates must be strictly increasing")
    buy, sells = _cost_rates(config, date_digits)
    return _net_returns(gross, buy, sells[:, None])


def daily_top_k_portfolio(
    score: np.ndarray,
    net_return: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    top_k: int,
    overlap: int,
) -> TopKPortfolio:
    """Daily fixed-Top-K net returns with stable ties and cash for failed fills."""
    scores = np.asarray(score, dtype=np.float64)
    net = np.asarray(net_return, dtype=np.float64)
    candidates = np.asarray(candidate_mask, dtype=bool)
    _validate_aligned(scores, net, candidates)
    if top_k <= 0 or overlap <= 0:
        raise ValueError("top_k and overlap must be positive")

    eligible = candidates & np.isfinite(scores)
    enough = eligible.sum(axis=1) >= top_k
    order = np.argsort(np.where(eligible, -scores, np.inf), axis=1, kind="stable")
    picked = order[:, :top_k]
    selected = np.take_along_axis(net, picked, axis=1)
    finite = np.isfinite(selected)
    portfolio = np.where(finite, selected, 0.0).sum(axis=1) / top_k / overlap
    return TopKPortfolio(
        portfolio_return=np.where(enough, portfolio, np.nan),
        executed=np.where(enough, finite.sum(axis=1), 0),
    )


def summarize_objective(series: TopKPortfolio, top_k: int) -> dict[str, float]:
    """Mean/IR and coverage diagnostics for an economic-objective series."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    usable = np.isfinite(series.portfolio_return)
    values = series.portfolio_return[usable]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "positive_rate": float("nan"),
            "execution_rate": float("nan"),
            "coverage": 0.0,
            "n_days": 0.0,
        }
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    mean = float(values.mean())
    return {
        "mean": mean,
        "std": std,
        "ir": mean / std if std > 0 else float("nan"),
        "positive_rate": float((values > 0).mean()),
        "execution_rate": float(series.executed[usable].sum() / (values.size * top_k)),
        "coverage": float(values.size / len(series.portfolio_return)),
        "n_days": float(values.size),
    }
