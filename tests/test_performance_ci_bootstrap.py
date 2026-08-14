"""Contracts for the training performance confidence-interval experiment."""

from __future__ import annotations

import numpy as np
import pytest

from helix.eval.backtest import summarize_portfolio_returns
from helix.eval.bootstrap import (
    bootstrap_performance_metrics,
    circular_block_bootstrap_indices,
    summarize_bootstrap_distribution,
    summarize_metric_runs,
)


def test_circular_indices_are_seeded_and_preserve_complete_blocks():
    seeds = (7, 13, 42)

    first = circular_block_bootstrap_indices(11, block_length=4, seeds=seeds)
    second = circular_block_bootstrap_indices(11, block_length=4, seeds=seeds)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 11)
    assert not np.array_equal(first[0], first[1])
    for row in first:
        for start in range(0, len(row), 4):
            block = row[start : start + 4]
            assert np.all(np.diff(block) % 11 == 1)


@pytest.mark.parametrize(
    ("n_dates", "block_length", "seeds", "message"),
    [
        (0, 4, (1, 2), "positive"),
        (10, 0, (1, 2), "positive"),
        (10, 4, (1,), "at least 2 unique"),
        (10, 4, (1, 1), "at least 2 unique"),
    ],
)
def test_circular_indices_reject_invalid_contracts(
    n_dates, block_length, seeds, message
):
    with pytest.raises(ValueError, match=message):
        circular_block_bootstrap_indices(n_dates, block_length, seeds)


def test_vectorized_performance_matches_canonical_scalar_metrics():
    daily = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])
    trade_sum = np.array([0.025, -0.03, 0.08, 0.0, -0.015, 0.03])
    trade_count = np.array([2, 1, 3, 0, 1, 2])
    indices = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 0, 1],
            [5, 5, 4, 3, 2, 1],
        ]
    )

    result = bootstrap_performance_metrics(daily, trade_sum, trade_count, indices)

    for replicate, index in enumerate(indices):
        expected = summarize_portfolio_returns(daily[index])
        assert result["cagr"][replicate] == pytest.approx(expected["cagr"])
        assert result["sharpe"][replicate] == pytest.approx(expected["sharpe"])
        assert result["day_win_rate"][replicate] == pytest.approx(
            expected["day_win_rate"]
        )
        assert result["mean_trade_return_net"][replicate] == pytest.approx(
            trade_sum[index].sum() / trade_count[index].sum()
        )


def test_bootstrap_summary_uses_sample_std_and_linear_percentiles():
    values = {
        "sharpe": np.array([-0.4, -0.1, 0.2, 0.5]),
        "cagr": np.array([-0.2, 0.0, 0.1, 0.4]),
    }

    result = summarize_bootstrap_distribution(values)

    for metric, samples in values.items():
        assert result[metric]["mean"] == pytest.approx(samples.mean())
        assert result[metric]["std"] == pytest.approx(samples.std(ddof=1))
        assert result[metric]["ci_low"] == pytest.approx(
            np.quantile(samples, 0.025, method="linear")
        )
        assert result[metric]["ci_high"] == pytest.approx(
            np.quantile(samples, 0.975, method="linear")
        )
        assert result[metric]["values"] == samples.tolist()


@pytest.mark.parametrize(
    ("daily", "trade_sum", "trade_count", "indices", "message"),
    [
        (np.ones(3), np.ones(2), np.ones(3), np.array([[0, 1, 2]]), "aligned"),
        (np.ones((1, 3)), np.ones(3), np.ones(3), np.array([[0, 1, 2]]), "one-dimensional"),
        (np.ones(3), np.ones(3), np.ones(3), np.array([0, 1, 2]), "two-dimensional"),
        (np.ones(3), np.ones(3), np.ones(3), np.array([[0, 1, 3]]), "out of bounds"),
        (np.ones(3), np.ones(3), np.zeros(3), np.array([[0, 1, 2]]), "resolved trade"),
    ],
)
def test_vectorized_performance_rejects_malformed_inputs(
    daily, trade_sum, trade_count, indices, message
):
    with pytest.raises(ValueError, match=message):
        bootstrap_performance_metrics(daily, trade_sum, trade_count, indices)


def test_bootstrap_summary_rejects_nonfinite_or_singleton_distributions():
    with pytest.raises(ValueError, match="finite"):
        summarize_bootstrap_distribution({"sharpe": np.array([0.1, np.nan])})
    with pytest.raises(ValueError, match="at least two"):
        summarize_bootstrap_distribution({"sharpe": np.array([0.1])})


def test_generic_metric_run_summary_preserves_g3_nonfinite_compatibility():
    runs = [
        {"score": 1.0, "optional": np.nan},
        {"score": 2.0, "optional": 4.0},
        {"score": 3.0, "optional": np.nan},
    ]

    result = summarize_metric_runs(runs)

    assert result["score"]["mean"] == pytest.approx(2.0)
    assert result["score"]["std"] == pytest.approx(1.0)
    assert result["score"]["values"] == [1.0, 2.0, 3.0]
    assert result["optional"]["mean"] == pytest.approx(4.0)
    assert result["optional"]["std"] == 0.0
    assert np.isnan(result["optional"]["values"][0])
