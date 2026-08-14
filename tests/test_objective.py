"""Contracts for the production Top-K economic objective."""

from __future__ import annotations

import numpy as np
import pytest

from helix.config import BacktestConfig
from helix.eval.backtest import _cost_rates, _net_returns
from helix.eval.objective import (
    cost_adjusted_returns,
    daily_top_k_portfolio,
    summarize_objective,
)


def test_vectorized_top_k_matches_looped_reference():
    rng = np.random.default_rng(7)
    score = rng.normal(size=(9, 13))
    gross = rng.normal(0.001, 0.04, size=score.shape)
    mask = rng.random(score.shape) > 0.2
    gross[2, np.argsort(-score[2])[:1]] = np.nan
    dates = np.array([f"202401{day:02d}" for day in range(2, 11)])
    cfg = BacktestConfig(top_k=4, slippage_bps=10.0)
    net = cost_adjusted_returns(gross, dates, cfg)

    actual = daily_top_k_portfolio(score, net, mask, top_k=cfg.top_k, overlap=2)

    expected_returns = []
    expected_executed = []
    for row in range(score.shape[0]):
        candidates = np.flatnonzero(mask[row] & np.isfinite(score[row]))
        if len(candidates) < cfg.top_k:
            expected_returns.append(np.nan)
            expected_executed.append(0)
            continue
        picked = candidates[np.argsort(-score[row, candidates], kind="stable")[: cfg.top_k]]
        selected = net[row, picked]
        finite = np.isfinite(selected)
        expected_returns.append(np.where(finite, selected, 0.0).sum() / cfg.top_k / 2)
        expected_executed.append(int(finite.sum()))

    np.testing.assert_allclose(actual.portfolio_return, expected_returns, equal_nan=True)
    np.testing.assert_array_equal(actual.executed, expected_executed)


def test_selected_nan_stays_cash_and_is_not_replaced():
    score = np.array([[4.0, 3.0, 2.0, 1.0]])
    net = np.array([[0.04, np.nan, 0.50, 0.60]])

    result = daily_top_k_portfolio(
        score, net, np.ones_like(score, dtype=bool), top_k=2, overlap=2
    )

    assert result.portfolio_return[0] == pytest.approx(0.04 / 2 / 2)
    assert result.executed[0] == 1


def test_stable_tie_break_matches_first_columns():
    score = np.ones((1, 4))
    net = np.array([[0.01, 0.02, 0.90, 1.00]])

    result = daily_top_k_portfolio(
        score, net, np.ones_like(score, dtype=bool), top_k=2, overlap=1
    )

    assert result.portfolio_return[0] == pytest.approx(0.015)


@pytest.mark.parametrize("top_k", [1, 4, 10])
def test_top_k_is_an_explicit_production_input(top_k):
    score = np.arange(12.0)[None, :]
    net = np.arange(12.0)[None, :] / 100.0

    result = daily_top_k_portfolio(
        score, net, np.ones_like(score, dtype=bool), top_k=top_k, overlap=1
    )

    expected = np.arange(12 - top_k, 12).mean() / 100.0
    assert result.portfolio_return[0] == pytest.approx(expected)


def test_cost_target_matches_canonical_backtest_helpers():
    gross = np.array([[0.10, -0.05], [0.10, -0.05]])
    dates = np.array(["2023-08-25", "2023-08-28"])
    cfg = BacktestConfig(slippage_bps=7.0)
    buy, sells = _cost_rates(cfg, np.array(["20230825", "20230828"]))

    result = cost_adjusted_returns(gross, dates, cfg)

    np.testing.assert_allclose(result, _net_returns(gross, buy, sells[:, None]))


def test_insufficient_candidates_make_the_date_unusable():
    score = np.array([[3.0, 2.0, 1.0]])
    net = np.full_like(score, 0.01)
    mask = np.array([[True, False, True]])

    result = daily_top_k_portfolio(score, net, mask, top_k=3, overlap=1)

    assert np.isnan(result.portfolio_return[0])
    assert result.executed[0] == 0


def test_summary_reports_coverage_and_execution_rate():
    score = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    net = np.array([[0.02, 0.01], [np.nan, -0.01], [0.03, 0.02]])
    mask = np.array([[True, True], [True, True], [True, False]])
    series = daily_top_k_portfolio(score, net, mask, top_k=2, overlap=1)

    result = summarize_objective(series, top_k=2)

    assert result["n_days"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["execution_rate"] == pytest.approx(3 / 4)


def test_inputs_are_not_mutated():
    score = np.array([[2.0, 1.0]])
    net = np.array([[0.02, 0.01]])
    mask = np.ones_like(score, dtype=bool)
    originals = (score.copy(), net.copy(), mask.copy())

    daily_top_k_portfolio(score, net, mask, top_k=1, overlap=1)

    for actual, expected in zip((score, net, mask), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("score", "net", "mask", "message"),
    [
        (np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)), "share one shape"),
        (np.ones(2), np.ones(2), np.ones(2), "two-dimensional"),
    ],
)
def test_objective_rejects_malformed_shapes(score, net, mask, message):
    with pytest.raises(ValueError, match=message):
        daily_top_k_portfolio(score, net, mask, top_k=1, overlap=1)


def test_cost_target_rejects_non_increasing_dates():
    with pytest.raises(ValueError, match="strictly increasing"):
        cost_adjusted_returns(
            np.ones((2, 1)), np.array(["20240103", "20240103"]), BacktestConfig()
        )
