"""Fitness for a candidate factor.

Two windows, not one. Evolution is driven by the *fit* window; a factor is only kept
if it still points the same way on a held-out *selection* window that sits after an
embargo. Optimising and selecting on the same rows is how GP pipelines end up with
beautiful in-sample factors and nothing else.

Sign is a free parameter: a factor with gini ``-0.2`` is exactly as useful as one with
``+0.2`` once negated, so fitness uses ``|gini|`` and the sign is recorded separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eval.metrics import daily_gini, summarize_daily

INVALID = -1e9


@dataclass
class EvalContext:
    """Everything a factor is scored against. Built once and reused for the whole run."""

    field_arrays: list[np.ndarray]
    y: np.ndarray
    mask: np.ndarray
    fit_rows: slice
    sel_rows: slice
    min_daily_samples: int = 50
    min_coverage: float = 0.4
    min_defined_fraction: float = 0.2
    complexity_penalty: float = 0.0015
    _cache: dict[str, FactorScore] = field(default_factory=dict, repr=False)


@dataclass
class FactorScore:
    fitness: float
    sign: float
    fit_gini: float
    fit_ir: float
    sel_gini: float
    coverage: float
    n_nodes: int

    def as_dict(self) -> dict[str, float]:
        return {
            "fitness": self.fitness,
            "sign": self.sign,
            "fit_gini": self.fit_gini,
            "fit_ir": self.fit_ir,
            "sel_gini": self.sel_gini,
            "coverage": self.coverage,
            "n_nodes": float(self.n_nodes),
        }


def _invalid(n_nodes: int) -> FactorScore:
    return FactorScore(INVALID, 1.0, float("nan"), float("nan"), float("nan"), 0.0, n_nodes)


def score_values(values: np.ndarray, ctx: EvalContext, n_nodes: int) -> FactorScore:
    """Score already-computed factor values. Separated out so tests can call it directly."""
    if not isinstance(values, np.ndarray) or values.shape != ctx.y.shape:
        return _invalid(n_nodes)

    defined = np.isfinite(values) & ctx.mask
    if defined.sum() < ctx.min_defined_fraction * max(ctx.mask.sum(), 1):
        return _invalid(n_nodes)

    fit = daily_gini(
        values[ctx.fit_rows], ctx.y[ctx.fit_rows], ctx.mask[ctx.fit_rows], ctx.min_daily_samples
    )
    fit_stats = summarize_daily(fit)
    if fit_stats["coverage"] < ctx.min_coverage or not np.isfinite(fit_stats["mean"]):
        return _invalid(n_nodes)

    sign = 1.0 if fit_stats["mean"] >= 0 else -1.0
    sel = daily_gini(
        values[ctx.sel_rows], ctx.y[ctx.sel_rows], ctx.mask[ctx.sel_rows], ctx.min_daily_samples
    )
    sel_stats = summarize_daily(sel)
    sel_gini = sign * sel_stats["mean"] if np.isfinite(sel_stats["mean"]) else float("nan")

    fitness = abs(fit_stats["mean"]) - ctx.complexity_penalty * n_nodes
    return FactorScore(
        fitness=float(fitness),
        sign=sign,
        fit_gini=float(sign * fit_stats["mean"]),
        fit_ir=float(fit_stats["ir"]),
        sel_gini=float(sel_gini),
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
