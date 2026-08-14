"""Contracts for the economic GP fitness function."""

from __future__ import annotations

import numpy as np
import pytest

from helix.gp.fitness import INVALID, EvalContext, evaluate, score_values
from helix.gp.neutralize import build_basis


def context(net_returns: np.ndarray, mask: np.ndarray | None = None) -> EvalContext:
    candidate = np.ones_like(net_returns, dtype=bool) if mask is None else mask
    return EvalContext(
        field_arrays=[],
        net_returns=net_returns,
        candidate_mask=candidate,
        fit_rows=slice(0, 3),
        sel_rows=slice(3, 5),
        top_k=2,
        overlap=1,
        min_coverage=0.5,
    )


def test_fitness_rewards_the_production_long_direction():
    returns = np.tile(np.array([-0.04, -0.02, 0.02, 0.04]), (5, 1))
    values = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))

    good = score_values(values, context(returns), n_nodes=3)
    bad = score_values(-values, context(returns), n_nodes=3)

    assert good.fitness > bad.fitness
    assert good.fit_net_return > 0
    assert bad.fit_net_return < 0
    assert good.sign == 1.0 == bad.sign


def test_node_count_cannot_reverse_pnl_ordering():
    returns = np.tile(np.array([-0.02, 0.00, 0.01, 0.03]), (5, 1))
    high = np.tile(np.array([1.0, 2.0, 3.0, 5.0]), (5, 1))
    low = np.tile(np.array([1.0, 3.0, 4.0, 2.0]), (5, 1))

    assert score_values(high, context(returns), 40).fitness > score_values(
        low, context(returns), 1
    ).fitness


def test_fitness_is_exactly_basis_points_of_fit_net_return():
    returns = np.tile(np.array([-0.04, -0.02, 0.02, 0.04]), (5, 1))
    values = np.tile(np.arange(4.0), (5, 1))

    score = score_values(values, context(returns), n_nodes=100)

    assert score.fitness == pytest.approx(10_000 * score.fit_net_return)
    assert score.as_dict()["sel_net_return"] == pytest.approx(0.03)


def test_wrong_shape_is_invalid():
    score = score_values(np.ones((5, 3)), context(np.ones((5, 4))), n_nodes=7)

    assert score.fitness == INVALID
    assert score.n_nodes == 7


def test_too_few_defined_candidate_values_are_invalid():
    values = np.full((5, 4), np.nan)
    values[0, :2] = [1.0, 2.0]
    ctx = context(np.ones((5, 4)))
    ctx.min_defined_fraction = 0.5

    assert score_values(values, ctx, n_nodes=2).fitness == INVALID


def test_fit_coverage_gate_does_not_look_at_selection_outcomes():
    returns = np.tile(np.array([-0.04, -0.02, 0.02, 0.04]), (5, 1))
    values = np.tile(np.arange(4.0), (5, 1))
    mask = np.ones_like(values, dtype=bool)
    mask[1:3, 1:] = False
    ctx = context(returns, mask)
    ctx.min_coverage = 0.3

    valid = score_values(values, ctx, n_nodes=1)
    ctx.min_coverage = 0.8
    invalid = score_values(values, ctx, n_nodes=1)

    assert valid.fitness != INVALID
    assert valid.coverage == pytest.approx(1 / 3)
    assert invalid.fitness == INVALID


def test_factor_fully_explained_by_basis_is_invalid():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(5, 20))
    ctx = EvalContext(
        field_arrays=[],
        net_returns=rng.normal(size=values.shape),
        candidate_mask=np.ones_like(values, dtype=bool),
        fit_rows=slice(0, 3),
        sel_rows=slice(3, 5),
        top_k=4,
        overlap=1,
        min_coverage=0.5,
        basis=build_basis([values], np.ones_like(values, dtype=bool)),
    )

    assert score_values(values, ctx, n_nodes=2).fitness == INVALID


def test_evaluate_memoizes_by_expression():
    class Individual:
        def __str__(self):
            return "factor"

        def __len__(self):
            return 3

    class Toolbox:
        calls = 0

        def compile(self, *, expr):
            self.calls += 1
            return lambda: np.tile(np.arange(4.0), (5, 1))

    toolbox = Toolbox()
    ctx = context(np.tile(np.array([-0.04, -0.02, 0.02, 0.04]), (5, 1)))

    first = evaluate(Individual(), toolbox, ctx)
    second = evaluate(Individual(), toolbox, ctx)

    assert first is second
    assert toolbox.calls == 1


def test_hit_label_is_not_part_of_the_fitness_context():
    assert "y" not in EvalContext.__dataclass_fields__
    assert "mask" not in EvalContext.__dataclass_fields__
