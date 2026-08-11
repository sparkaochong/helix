"""The evolutionary search loop.

Design choices that matter more than the GP hyper-parameters:

* **Search sees only the oldest training block.** Everything after it is reserved for
  walk-forward evaluation, so the reported out-of-sample numbers mean something.
* **Fit and select on different rows.** Evolution maximises |gini| on the fit rows;
  survival requires the same sign to hold on embargoed selection rows.
* **Deduplicate before handing factors to the network.** GP converges on families of
  near-identical expressions; feeding 24 copies of one idea to a GRU is worse than
  feeding 8 distinct ones.
* **Memoise on the expression string.** GP regenerates the same subtree constantly and
  panel evaluation is the entire cost of the run.
"""

from __future__ import annotations

import operator
import random
from dataclasses import dataclass

import numpy as np
from deap import algorithms, base, creator, gp, tools

from ..config import GPConfig
from ..eval.metrics import pairwise_max_abs_corr
from ..features.operators import cs_rank
from ..logging_setup import get_logger
from .fitness import INVALID, EvalContext, evaluate
from .generate import gen_grow, gen_half_and_half
from .library import FactorLibrary, FactorSpec
from .primitives import build_pset

log = get_logger(__name__)

_CREATOR_READY = False


def _ensure_creator() -> None:
    """DEAP's creator registers classes globally; do it once per process."""
    global _CREATOR_READY
    if _CREATOR_READY:
        return
    creator.create("HelixFitness", base.Fitness, weights=(1.0,))
    creator.create("HelixIndividual", gp.PrimitiveTree, fitness=creator.HelixFitness)
    _CREATOR_READY = True


def build_toolbox(pset: gp.PrimitiveSetTyped, cfg: GPConfig) -> base.Toolbox:
    _ensure_creator()
    toolbox = base.Toolbox()
    toolbox.register(
        "expr", gen_half_and_half, pset=pset, min_=cfg.init_depth[0], max_=cfg.init_depth[1]
    )
    toolbox.register("individual", tools.initIterate, creator.HelixIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gen_grow, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    for op in ("mate", "mutate"):
        toolbox.decorate(op, gp.staticLimit(operator.attrgetter("height"), cfg.max_depth))
        toolbox.decorate(op, gp.staticLimit(len, cfg.max_nodes))
    return toolbox


def make_context(
    fields: dict[str, np.ndarray],
    field_names: list[str],
    y: np.ndarray,
    mask: np.ndarray,
    cfg: GPConfig,
    embargo_days: int,
    fit_fraction: float = 0.8,
) -> EvalContext:
    """Split the search block into fit rows and embargoed selection rows."""
    n_rows = y.shape[0]
    cut = int(n_rows * fit_fraction)
    sel_start = min(cut + embargo_days, n_rows)
    if n_rows - sel_start < 20:
        raise ValueError(
            f"search window of {n_rows} dates is too short to hold out a selection block; "
            "increase split.train_days or lengthen the data range"
        )
    log.info(
        "GP search: fit rows [0:%d], selection rows [%d:%d] (embargo %d)",
        cut, sel_start, n_rows, embargo_days,
    )
    return EvalContext(
        field_arrays=[np.asarray(fields[n], dtype=np.float64) for n in field_names],
        y=y,
        mask=mask,
        fit_rows=slice(0, cut),
        sel_rows=slice(sel_start, n_rows),
        min_daily_samples=cfg.min_daily_samples,
        min_coverage=cfg.min_coverage,
        complexity_penalty=cfg.complexity_penalty,
    )


def liquidity_top_columns(amount: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the ``k`` most consistently liquid names in the window.

    Evolution runs on this subsample purely for speed; the surviving expressions are
    re-evaluated on the full universe afterwards.
    """
    n_cols = amount.shape[1]
    if k >= n_cols:
        return np.arange(n_cols)
    scored = np.where(mask, amount, np.nan)
    with np.errstate(all="ignore"):
        median_amount = np.nanmedian(scored, axis=0)
    median_amount = np.where(np.isfinite(median_amount), median_amount, -np.inf)
    return np.sort(np.argsort(-median_amount)[:k])


@dataclass
class SearchResult:
    library: FactorLibrary
    hall_of_fame: list[tuple[str, float, float]]  # expression, fit_gini, sel_gini


def run_search(
    fields: dict[str, np.ndarray],
    field_names: list[str],
    y: np.ndarray,
    mask: np.ndarray,
    cfg: GPConfig,
    embargo_days: int,
    pset: gp.PrimitiveSetTyped | None = None,
    kind: str = "panel",
) -> SearchResult:
    """Evolve factors. Pass ``pset``/``kind="event"`` for slot panels, where the default
    windowed operators would silently mix unrelated stocks."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    pset = pset if pset is not None else build_pset(field_names, cfg.windows)
    toolbox = build_toolbox(pset, cfg)
    ctx = make_context(fields, field_names, y, mask, cfg, embargo_days)

    population = toolbox.population(n=cfg.population)
    hof = tools.HallOfFame(cfg.hall_of_fame)

    for generation in range(cfg.generations + 1):
        pending = [ind for ind in population if not ind.fitness.valid]
        for ind in pending:
            ind.fitness.values = (evaluate(ind, toolbox, ctx).fitness,)
        hof.update(population)

        valid_fitness = [
            ind.fitness.values[0] for ind in population if ind.fitness.values[0] > INVALID / 2
        ]
        log.info(
            "gen %02d/%d | evaluated %4d | alive %3d/%d | best %.4f | mean %.4f | cache %d",
            generation, cfg.generations, len(pending), len(valid_fitness), len(population),
            max(valid_fitness) if valid_fitness else float("nan"),
            float(np.mean(valid_fitness)) if valid_fitness else float("nan"),
            len(ctx._cache),
        )
        if generation == cfg.generations:
            break

        offspring = toolbox.select(population, len(population) - 1)
        offspring = algorithms.varAnd(offspring, toolbox, cfg.crossover_prob, cfg.mutation_prob)
        population = [toolbox.clone(hof[0]), *offspring]  # elitism

    return _select_factors(hof, toolbox, ctx, cfg, field_names, kind)


def _select_factors(
    hof, toolbox, ctx: EvalContext, cfg: GPConfig, field_names: list[str], kind: str = "panel"
) -> SearchResult:
    """Rank the hall of fame by held-out performance, then greedily deduplicate."""
    ranked = []
    for ind in hof:
        score = ctx._cache.get(str(ind))
        if score is None or score.fitness <= INVALID / 2:
            continue
        if not np.isfinite(score.sel_gini) or score.sel_gini <= 0:
            continue  # sign did not survive the embargoed selection block
        ranked.append((str(ind), score))
    ranked.sort(key=lambda item: item[1].sel_gini, reverse=True)
    log.info("%d/%d hall-of-fame factors survived out-of-sample sign check", len(ranked), len(hof))

    kept_specs: list[FactorSpec] = []
    kept_ranks: list[np.ndarray] = []
    for expression, score in ranked:
        if len(kept_specs) >= cfg.n_keep:
            break
        tree = gp.PrimitiveTree.from_string(expression, toolbox.compile.keywords["pset"])
        func = toolbox.compile(expr=tree)
        with np.errstate(all="ignore"):
            values = func(*ctx.field_arrays)
        if not isinstance(values, np.ndarray) or values.shape != ctx.y.shape:
            continue
        ranks = cs_rank(np.where(ctx.mask, values * score.sign, np.nan))
        if pairwise_max_abs_corr(ranks, kept_ranks) > cfg.max_abs_corr:
            continue
        kept_specs.append(
            FactorSpec(
                name=f"gp_{len(kept_specs):03d}",
                expression=expression,
                sign=score.sign,
                metrics=score.as_dict(),
            )
        )
        kept_ranks.append(ranks)

    log.info("kept %d factors after correlation dedup (threshold %.2f)", len(kept_specs), cfg.max_abs_corr)
    for spec in kept_specs:
        log.info(
            "  %s  fit_gini=%.4f  sel_gini=%.4f  nodes=%d\n    %s",
            spec.name, spec.metrics["fit_gini"], spec.metrics["sel_gini"],
            int(spec.metrics["n_nodes"]), spec.expression,
        )

    return SearchResult(
        library=FactorLibrary(
            factors=kept_specs, field_names=field_names, windows=cfg.windows, kind=kind
        ),
        hall_of_fame=[(e, s.fit_gini, s.sel_gini) for e, s in ranked],
    )
