"""Neutralisation must kill exactly the factors that add nothing.

The motivating case: the first factor mined here scored raw IC +0.087 but was a linear
blend of three columns already in the training table. Residual IC was -0.006. Fitness
measured on the residual has to score that at zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from helix.eval.metrics import daily_gini, summarize_daily
from helix.gp.neutralize import build_basis, residualize


@pytest.fixture
def mask() -> np.ndarray:
    m = np.ones((30, 50), dtype=bool)
    m[:, 45:] = False           # some slots never occupied
    m[7, 30:] = False           # one thin date
    return m


@pytest.fixture
def base(mask) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=mask.shape)


def test_a_base_column_residualises_to_nothing(mask, base):
    """Fully explained -> every date is NaN, not float noise that still ranks correctly."""
    basis = build_basis([base], mask)
    residual = residualize(base, basis, mask)
    assert np.isnan(residual).all()


def test_a_monotone_transform_of_a_base_column_also_dies(mask, base):
    """Residualisation works on ranks, so any monotone rescaling is equally explained."""
    basis = build_basis([base], mask)
    assert np.isnan(residualize(np.exp(base), basis, mask)).all()
    assert np.isnan(residualize(base * 2.0 + 1.0, basis, mask)).all()


def test_a_linear_blend_of_base_columns_loses_its_predictive_power(mask):
    """The gp_000 failure mode: strong raw signal, no incremental signal.

    Re-ranking before projecting is not linear, so ~15% of the blend's *magnitude*
    survives. Magnitude is not the point -- what must collapse is the residual's ability
    to predict, because that is what fitness scores and what a downstream model would
    gain. gp_000 itself had raw IC +0.087 and residual IC -0.006.
    """
    rng = np.random.default_rng(1)
    from helix.features.operators import cs_rank

    shape = (120, 90)
    live = np.ones(shape, dtype=bool)
    a, b = rng.normal(size=shape), rng.normal(size=shape)
    blend = 0.6 * cs_rank(a) + 0.4 * cs_rank(b)
    y = (rng.uniform(size=shape) < 1 / (1 + np.exp(-6 * (blend - 0.5)))).astype(float)

    raw = summarize_daily(daily_gini(blend, y, live, min_samples=20))["mean"]
    basis = build_basis([a, b], live)
    residual = residualize(blend, basis, live)
    neutral = summarize_daily(daily_gini(residual, y, live, min_samples=20))["mean"]

    assert raw > 0.3, f"blend should look strong on its own, got {raw:.3f}"
    # Not zero: projection is linear in rank space, so a monotone nonlinear function of
    # the base columns leaves a residue. This synthetic case is adversarial (y is a clean
    # deterministic function of the blend); the real gp_000 residual IC was -0.006.
    assert abs(neutral) < 0.25 * raw, f"raw {raw:.3f} -> residual {neutral:.3f}"


def test_an_orthogonal_factor_survives(mask, base):
    rng = np.random.default_rng(2)
    independent = rng.normal(size=mask.shape)
    basis = build_basis([base], mask)
    residual = residualize(independent, basis, mask)
    assert np.nanstd(residual[mask]) > 0.05


def test_residual_is_nan_outside_the_mask(mask, base):
    rng = np.random.default_rng(9)
    basis = build_basis([base], mask)
    residual = residualize(rng.normal(size=mask.shape), basis, mask)
    assert np.isnan(residual[~mask]).all()
    assert not np.isnan(residual[mask]).any()


def test_basis_is_orthonormal_per_date(mask, base):
    rng = np.random.default_rng(3)
    basis = build_basis([base, rng.normal(size=mask.shape)], mask)
    gram = np.einsum("tnj,tnk->tjk", basis, basis)
    for t in range(mask.shape[0]):
        active = np.diag(gram[t]) > 0.5
        block = gram[t][np.ix_(active, active)]
        np.testing.assert_allclose(block, np.eye(active.sum()), atol=1e-9)


def test_a_constant_column_neither_explodes_nor_corrupts_the_basis(mask):
    """A constant carries no cross-sectional information: fully explained by the intercept."""
    rng = np.random.default_rng(11)
    constant = np.ones(mask.shape)

    assert np.isnan(residualize(constant, build_basis([constant], mask), mask)).all()

    # A degenerate column alongside a real one must not damage the real direction.
    informative = rng.normal(size=mask.shape)
    basis = build_basis([constant, informative], mask)
    assert np.isfinite(basis).all()
    assert np.isnan(residualize(informative, basis, mask)).all()
    assert np.isfinite(residualize(rng.normal(size=mask.shape), basis, mask)[mask]).all()


def test_neutralised_gini_collapses_for_a_redundant_factor():
    """End to end: a factor built only from base columns must score ~0 after residualising."""
    rng = np.random.default_rng(4)
    mask = np.ones((60, 80), dtype=bool)
    driver = rng.normal(size=mask.shape)
    y = (rng.uniform(size=mask.shape) < 1 / (1 + np.exp(-driver))).astype(float)

    raw_gini = summarize_daily(daily_gini(driver, y, mask, min_samples=10))["mean"]
    assert raw_gini > 0.2, "the driver should look strong before neutralisation"

    basis = build_basis([driver], mask)
    residual = residualize(driver * 2.0 + 1.0, basis, mask)
    stats = summarize_daily(daily_gini(residual, y, mask, min_samples=10))

    # Every date is fully explained, so no date contributes and there is no score at all.
    assert stats["coverage"] == 0.0
    assert np.isnan(stats["mean"])


def test_float_noise_residual_cannot_masquerade_as_signal():
    """Without the magnitude guard, a 1e-16 residual still ranks correctly and scores."""
    rng = np.random.default_rng(5)
    mask = np.ones((40, 60), dtype=bool)
    driver = rng.normal(size=mask.shape)
    y = (rng.uniform(size=mask.shape) < 1 / (1 + np.exp(-driver))).astype(float)
    basis = build_basis([driver], mask)

    unguarded = residualize(driver, basis, mask, min_residual_fraction=0.0)
    guarded = residualize(driver, basis, mask)

    noise_gini = summarize_daily(daily_gini(unguarded, y, mask, min_samples=10))["mean"]
    assert abs(noise_gini) > 0.02, "the unguarded path really does score float noise"
    assert np.isnan(guarded).all()


def test_unknown_base_column_is_a_clear_error(mask, base):
    from helix.gp.neutralize import basis_from_fields

    with pytest.raises(KeyError, match="unknown columns"):
        basis_from_fields({"a": base}, ["a", "missing"], mask)


def test_neutralised_rounds_find_a_second_independent_driver():
    """The point of multi-round mining: round 2 must not rediscover round 1's idea.

    ``y`` depends on two independent drivers, ``alpha`` more strongly than ``beta``.
    Unneutralised, the search locks onto ``alpha``. Neutralised against ``alpha``, it
    has to reach for ``beta`` -- which is exactly the behaviour that produces a set of
    factors worth adding to a model rather than one idea restated five times.
    """
    from helix.config import GPConfig
    from helix.gp.engine import run_search
    from helix.gp.event_primitives import build_event_pset

    rng = np.random.default_rng(17)
    shape = (300, 120)
    alpha = rng.normal(size=shape)
    beta = rng.normal(size=shape)
    noise = rng.normal(size=shape)
    logit = 1.6 * alpha + 0.9 * beta
    y = (rng.uniform(size=shape) < 1 / (1 + np.exp(-logit))).astype(float)
    mask = np.ones(shape, dtype=bool)

    fields = {"alpha": alpha, "beta": beta, "noise": noise}
    names = ["alpha", "beta", "noise"]
    cfg = GPConfig(
        population=60, generations=4, hall_of_fame=20, n_keep=3,
        windows=[], max_nodes=8, max_depth=3, min_daily_samples=20, seed=5,
    )
    pset = build_event_pset(names)

    plain = run_search(fields, names, y, mask, cfg, embargo_days=5, pset=pset, kind="event")
    assert plain.library.factors
    assert "alpha" in plain.library.factors[0].expression

    basis = build_basis([alpha], mask)
    neutralised = run_search(
        fields, names, y, mask, cfg, embargo_days=5, pset=pset, kind="event", basis=basis
    )
    assert neutralised.library.factors, "nothing survived once alpha was projected out"
    top = neutralised.library.factors[0].expression
    assert "beta" in top, f"expected the second driver, got {top}"
