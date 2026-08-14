"""Production-economic fitness for a candidate factor.

Evolution is driven by cost-adjusted D+2-close Top-K P&L on the fit window.  The
embargoed selection window is reported separately and gates factor retention in the
search engine.  Direction belongs to the GP expression itself: fitness never takes an
absolute value or silently flips a factor's sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eval.objective import TopKPortfolio, daily_top_k_portfolio, summarize_objective
from .neutralize import residualize

INVALID = -1e9


@dataclass
class EvalContext:
    """Everything a factor is scored against. Built once and reused for the whole run."""

    field_arrays: list[np.ndarray]
    net_returns: np.ndarray
    candidate_mask: np.ndarray
    fit_rows: slice
    sel_rows: slice
    top_k: int
    overlap: int
    min_coverage: float = 0.4
    min_defined_fraction: float = 0.2
    #: Optional ``(T, N, K)`` orthonormal basis. When set, fitness scores the factor's
    #: residual after projecting it out, so a good linear blend of columns the model
    #: already has scores zero instead of scoring well.
    basis: np.ndarray | None = None
    _cache: dict[str, FactorScore] = field(default_factory=dict, repr=False)


@dataclass
class FactorScore:
    fitness: float
    sign: float
    fit_net_return: float
    fit_net_ir: float
    sel_net_return: float
    sel_net_ir: float
    coverage: float
    n_nodes: int

    def as_dict(self) -> dict[str, float]:
        return {
            "fitness": self.fitness,
            "sign": self.sign,
            "fit_net_return": self.fit_net_return,
            "fit_net_ir": self.fit_net_ir,
            "sel_net_return": self.sel_net_return,
            "sel_net_ir": self.sel_net_ir,
            "coverage": self.coverage,
            "n_nodes": float(self.n_nodes),
        }


def _invalid(n_nodes: int) -> FactorScore:
    return FactorScore(
        INVALID,
        1.0,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        0.0,
        n_nodes,
    )


def _rows(series: TopKPortfolio, rows: slice) -> TopKPortfolio:
    return TopKPortfolio(series.portfolio_return[rows], series.executed[rows])


def score_values(values: np.ndarray, ctx: EvalContext, n_nodes: int) -> FactorScore:
    """Score already-computed factor values. Separated out so tests can call it directly."""
    if (
        not isinstance(values, np.ndarray)
        or values.shape != ctx.net_returns.shape
        or ctx.candidate_mask.shape != ctx.net_returns.shape
    ):
        return _invalid(n_nodes)

    defined = np.isfinite(values) & ctx.candidate_mask
    if defined.sum() < ctx.min_defined_fraction * max(ctx.candidate_mask.sum(), 1):
        return _invalid(n_nodes)

    if ctx.basis is not None:
        values = residualize(values, ctx.basis, ctx.candidate_mask)

    portfolio = daily_top_k_portfolio(
        values,
        ctx.net_returns,
        ctx.candidate_mask,
        top_k=ctx.top_k,
        overlap=ctx.overlap,
    )
    fit_stats = summarize_objective(_rows(portfolio, ctx.fit_rows), ctx.top_k)
    if fit_stats["coverage"] < ctx.min_coverage or not np.isfinite(fit_stats["mean"]):
        return _invalid(n_nodes)

    sel_stats = summarize_objective(_rows(portfolio, ctx.sel_rows), ctx.top_k)
    fitness = 10_000.0 * fit_stats["mean"]
    return FactorScore(
        fitness=float(fitness),
        sign=1.0,
        fit_net_return=float(fit_stats["mean"]),
        fit_net_ir=float(fit_stats["ir"]),
        sel_net_return=float(sel_stats["mean"]),
        sel_net_ir=float(sel_stats["ir"]),
        coverage=float(fit_stats["coverage"]),
        n_nodes=n_nodes,
    )


def evaluate(individual, toolbox, ctx: EvalContext) -> FactorScore:
    """Compile, run and score one individual, memoised on its printed expression."""
    key = str(individual)
    cached = ctx._cache.get(key)
    if cached is not None:
        return cached

    n_nodes = len(individual)
    try:
        func = toolbox.compile(expr=individual)
        with np.errstate(all="ignore"):
            values = func(*ctx.field_arrays)
        score = score_values(values, ctx, n_nodes)
    except (ValueError, TypeError, FloatingPointError, MemoryError, ZeroDivisionError):
        score = _invalid(n_nodes)

    ctx._cache[key] = score
    return score
