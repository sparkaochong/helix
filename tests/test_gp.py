"""End-to-end sanity for the GP stage on synthetic data with a planted signal."""

from __future__ import annotations

import numpy as np
import pytest
from deap import creator, gp

from helix.config import BacktestConfig, GPConfig
from helix.gp.engine import (
    _select_factors,
    build_toolbox,
    liquidity_top_columns,
    run_search,
)
from helix.gp.event_primitives import build_event_pset
from helix.gp.fitness import EvalContext, FactorScore
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
    """``alpha`` genuinely predicts realised D+2 return; ``noise`` does not."""
    rng = np.random.default_rng(7)
    n_dates, n_codes = 260, 120
    alpha = rng.normal(size=(n_dates, n_codes))
    noise = rng.normal(size=(n_dates, n_codes))
    gross = 0.005 + 0.025 * np.tanh(alpha) + rng.normal(0.0, 0.002, size=alpha.shape)
    mask = np.ones((n_dates, n_codes), dtype=bool)
    dates = np.array([f"{20200101 + i:08d}" for i in range(n_dates)])
    costs = BacktestConfig(
        top_k=4,
        commission_bps=0,
        transfer_bps=0,
        stamp_sell_bps=0,
        stamp_sell_bps_before_cut=0,
        slippage_bps=0,
    )
    return {"alpha": alpha, "noise": noise}, ["alpha", "noise"], gross, mask, dates, costs


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
    fields, names, gross, mask, dates, costs = synthetic
    cfg = GPConfig(
        population=60, generations=4, hall_of_fame=20, n_keep=5,
        windows=[5], max_nodes=12, max_depth=4, min_daily_samples=20, seed=3,
    )
    result = run_search(
        fields=fields,
        field_names=names,
        gross_returns=gross,
        candidate_mask=mask,
        dates=dates,
        cfg=cfg,
        backtest_cfg=costs,
        entry_offset=1,
        touch_offset=2,
        embargo_days=5,
    )

    assert result.library.factors, "expected at least one factor to survive selection"
    best = result.library.factors[0]
    assert best.metrics["sel_net_return"] > 0
    assert best.sign == 1.0
    assert "alpha" in best.expression


def test_kept_factors_are_not_near_duplicates(synthetic):
    fields, names, gross, mask, dates, costs = synthetic
    cfg = GPConfig(
        population=60, generations=3, hall_of_fame=30, n_keep=6,
        windows=[5], max_nodes=12, max_depth=4, min_daily_samples=20,
        max_abs_corr=0.7, seed=11,
    )
    result = run_search(
        fields=fields,
        field_names=names,
        gross_returns=gross,
        candidate_mask=mask,
        dates=dates,
        cfg=cfg,
        backtest_cfg=costs,
        entry_offset=1,
        touch_offset=2,
        embargo_days=5,
    )
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
    fields, names, *_ = synthetic
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


def test_deap_orders_by_pnl_then_smaller_tree():
    build_toolbox(build_event_pset(["alpha"]), GPConfig(windows=[]))
    assert creator.HelixFitness.weights == (1.0, -1.0)

    large = creator.HelixFitness((100.0, 12.0))
    small = creator.HelixFitness((100.0, 3.0))
    assert small > large


def test_selection_requires_positive_selection_pnl_and_uses_nodes_for_ties():
    rng = np.random.default_rng(19)
    shape = (30, 20)
    fields = [rng.normal(size=shape) for _ in range(3)]
    names = ["large", "small", "negative"]
    pset = build_event_pset(names)
    cfg = GPConfig(windows=[], n_keep=3, max_abs_corr=0.99)
    toolbox = build_toolbox(pset, cfg)
    individuals = [gp.PrimitiveTree.from_string(name, pset) for name in names]
    ctx = EvalContext(
        field_arrays=fields,
        net_returns=np.zeros(shape),
        candidate_mask=np.ones(shape, dtype=bool),
        fit_rows=slice(0, 20),
        sel_rows=slice(25, 30),
        top_k=4,
        overlap=2,
    )
    ctx._cache = {
        "large": FactorScore(100.0, 1.0, 0.01, 1.0, 0.02, 1.0, 1.0, 9),
        "small": FactorScore(100.0, 1.0, 0.01, 1.0, 0.02, 1.0, 1.0, 2),
        "negative": FactorScore(200.0, 1.0, 0.02, 1.0, -0.01, -1.0, 1.0, 1),
    }

    result = _select_factors(individuals, toolbox, ctx, cfg, names, kind="event")

    assert [factor.expression for factor in result.library.factors] == ["small", "large"]
    assert all(factor.sign == 1.0 for factor in result.library.factors)
    assert result.hall_of_fame == [
        ("small", 0.01, 0.02),
        ("large", 0.01, 0.02),
    ]
    assert "sel_gini" not in result.library.factors[0].metrics
