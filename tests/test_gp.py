"""End-to-end sanity for the GP stage on synthetic data with a planted signal."""

from __future__ import annotations

import numpy as np
import pytest
from deap import gp

from helix.config import GPConfig
from helix.gp.engine import liquidity_top_columns, run_search
from helix.gp.library import (
    FactorLibrary,
    FactorSpec,
    compute_factors,
    load_factors,
    save_factors,
)
from helix.gp.primitives import build_pset, parse_expression


@pytest.fixture
def synthetic():
    """``alpha`` genuinely predicts the label; ``noise`` does not."""
    rng = np.random.default_rng(7)
    n_dates, n_codes = 260, 120
    alpha = rng.normal(size=(n_dates, n_codes))
    noise = rng.normal(size=(n_dates, n_codes))
    prob = 1.0 / (1.0 + np.exp(-1.5 * alpha))
    y = (rng.uniform(size=(n_dates, n_codes)) < prob).astype(np.float64)
    mask = np.ones((n_dates, n_codes), dtype=bool)
    return {"alpha": alpha, "noise": noise}, ["alpha", "noise"], y, mask


def test_pset_round_trips_an_expression():
    pset = build_pset(["a", "b"], [5, 10])
    tree = parse_expression("add(cs_rank(a), ts_mean(b, w5))", pset)
    func = gp.compile(tree, pset)
    out = func(np.arange(20.0).reshape(10, 2), np.ones((10, 2)))
    assert out.shape == (10, 2)


def test_windows_are_type_checked():
    """A price array must not be usable where a window is required."""
    pset = build_pset(["a"], [5])
    with pytest.raises(TypeError, match="does not match the expected one"):
        parse_expression("ts_mean(a, a)", pset)


def test_search_finds_the_planted_factor(synthetic):
    fields, names, y, mask = synthetic
    cfg = GPConfig(
        population=60, generations=4, hall_of_fame=20, n_keep=5,
        windows=[5], max_nodes=12, max_depth=4, min_daily_samples=20, seed=3,
    )
    result = run_search(fields, names, y, mask, cfg, embargo_days=5)

    assert result.library.factors, "expected at least one factor to survive selection"
    best = result.library.factors[0]
    assert best.metrics["sel_gini"] > 0.05
    assert "alpha" in best.expression


def test_kept_factors_are_not_near_duplicates(synthetic):
    fields, names, y, mask = synthetic
    cfg = GPConfig(
        population=60, generations=3, hall_of_fame=30, n_keep=6,
        windows=[5], max_nodes=12, max_depth=4, min_daily_samples=20,
        max_abs_corr=0.7, seed=11,
    )
    result = run_search(fields, names, y, mask, cfg, embargo_days=5)
    _, values = compute_factors(result.library, fields)

    flat = values.reshape(-1, values.shape[2])
    for i in range(flat.shape[1]):
        for j in range(i + 1, flat.shape[1]):
            both = np.isfinite(flat[:, i]) & np.isfinite(flat[:, j])
            if both.sum() < 100:
                continue
            corr = np.corrcoef(flat[both, i], flat[both, j])[0, 1]
            assert abs(corr) <= 0.95


def test_library_survives_a_save_load_round_trip(tmp_path, synthetic):
    fields, names, _, _ = synthetic
    library = FactorLibrary(
        factors=[FactorSpec(name="f0", expression="cs_rank(alpha)", sign=-1.0, metrics={})],
        field_names=names,
        windows=[5],
    )
    path = tmp_path / "factors.json"
    save_factors(path, library)

    reloaded = load_factors(path)
    assert reloaded.factors[0].expression == "cs_rank(alpha)"
    names_out, values = compute_factors(reloaded, fields)
    assert names_out == ["f0"]
    # the stored sign is applied, so the saved factor is the negation of the raw rank
    assert values.shape == (fields["alpha"].shape[0], fields["alpha"].shape[1], 1)
    assert np.nanmax(values) <= 0.0


def test_liquidity_subsample_picks_the_busiest_names():
    amount = np.tile(np.arange(10.0), (30, 1))
    mask = np.ones((30, 10), dtype=bool)
    cols = liquidity_top_columns(amount, mask, 3)
    assert cols.tolist() == [7, 8, 9]
