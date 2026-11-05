"""Contracts for the training performance confidence-interval experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.config import BacktestConfig, Config
from helix.eval.backtest import BacktestResult, summarize_portfolio_returns
from helix.eval.bootstrap import (
    bootstrap_performance_metrics,
    circular_block_bootstrap_indices,
    summarize_bootstrap_distribution,
    summarize_metric_runs,
)
from helix.gp.library import FactorLibrary, FactorSpec
from scripts.performance_ci_bootstrap import (
    KEEP_DOWNGRADE,
    LIFT_DOWNGRADE,
    aggregate_performance_ci,
    align_event_scores,
    complete_training_decision_dates,
    performance_ci_decision,
    realistic_backtest_config,
    render_report,
    validate_formal_library,
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


def test_complete_training_dates_drop_the_d2_boundary_tail():
    calendar = np.array(
        ["20220104", "20220105", "20220106", "20220107", "20220110"]
    )

    result = complete_training_decision_dates(
        calendar,
        train_start="2022-01-04",
        train_end="2022-01-10",
        horizon=2,
    )

    assert result.tolist() == ["20220104", "20220105", "20220106"]


def test_realistic_config_preserves_production_costs_and_top4():
    production = BacktestConfig(
        top_k=4,
        exit_rule="close",
        enable_realistic_exit=False,
        commission_bps=3.1,
        transfer_bps=0.2,
        stamp_sell_bps=5.3,
        stamp_sell_bps_before_cut=10.4,
        slippage_bps=8.7,
    )
    config = Config(backtest=production)

    result = realistic_backtest_config(config)

    assert result.enable_realistic_exit is True
    assert result.model_dump(exclude={"enable_realistic_exit"}) == production.model_dump(
        exclude={"enable_realistic_exit"}
    )


def test_performance_ci_aggregates_complete_dates_and_resolved_trades():
    daily_returns = np.array([0.01, -0.02, 0.03, 0.0])
    daily = pd.DataFrame(
        {
            "date": ["20220104", "20220105", "20220106", "20220107"],
            "portfolio_return": daily_returns,
        }
    )
    trades = pd.DataFrame(
        {
            "d0_date": ["20220104", "20220104", "20220105", "20220106", "20220107"],
            "realistic_net_return": [0.02, 0.01, -0.03, 0.04, np.nan],
            "unresolved_at_end": [False, False, False, False, True],
        }
    )
    deterministic = summarize_portfolio_returns(daily_returns)
    result = BacktestResult(
        daily=daily,
        trades=trades,
        summary={
            **deterministic,
            "mean_trade_return_net": 0.01,
        },
    )
    indices = np.array(
        [
            [0, 1, 2, 3],
            [2, 2, 0, 1],
            [1, 0, 3, 2],
        ]
    )

    aggregated = aggregate_performance_ci(result, indices)

    assert aggregated["deterministic"]["cagr"] == pytest.approx(
        deterministic["cagr"]
    )
    assert aggregated["deterministic"]["mean_trade_return_net"] == pytest.approx(0.01)
    assert aggregated["bootstrap"]["mean_trade_return_net"]["values"][1] == pytest.approx(
        (0.04 + 0.04 + 0.03 - 0.03) / (1 + 1 + 2 + 1)
    )


@pytest.mark.parametrize(
    ("lower", "expected"),
    [
        (0.0001, LIFT_DOWNGRADE),
        (0.0, KEEP_DOWNGRADE),
        (-0.0001, KEEP_DOWNGRADE),
        (float("nan"), KEEP_DOWNGRADE),
    ],
)
def test_downgrade_lifts_only_for_a_strictly_positive_finite_bound(lower, expected):
    assert performance_ci_decision(lower) == expected


def test_report_renders_ci_decision_and_reproduction_contract():
    payload = {
        "metadata": {
            "train_start": "2022-01-04",
            "train_end": "2024-09-04",
            "decision_end": "2024-09-02",
            "n_dates": 647,
            "block_length": 20,
            "seeds": [7, 13, 42, 101, 211, 307, 419, 523, 631, 743],
        },
        "config": {
            "top_k": 4,
            "exit_rule": "close",
            "enable_realistic_exit": True,
            "commission_bps": 2.5,
            "transfer_bps": 0.1,
            "stamp_sell_bps": 5.0,
            "stamp_sell_bps_before_cut": 10.0,
            "slippage_bps": 10.0,
        },
        "metrics": {
            metric: {
                "deterministic": 0.1,
                "mean": 0.2,
                "std": 0.3,
                "ci_low": 0.012345,
                "ci_high": 0.5,
                "values": [float(index) for index in range(10)],
            }
            for metric in (
                "cagr",
                "sharpe",
                "day_win_rate",
                "mean_trade_return_net",
            )
        },
        "execution": {
            "n_trades": 100,
            "fill_rate": 0.9,
            "limit_down_exit_share": 0.01,
            "unresolved_at_end": 2,
        },
        "decision": LIFT_DOWNGRADE,
    }

    report = render_report(payload)

    assert "0.012345" in report
    assert LIFT_DOWNGRADE in report
    assert "2024-09-02" in report
    assert "百分位法" in report
    assert "--cache-only" in report
    assert "7,13,42,101,211,307,419,523,631,743" in report


def test_event_scores_align_to_fixed_market_coordinates_without_promoting_slots():
    event_dates = np.array(["2022-01-04", "2022-01-05"])
    event_codes = np.array([["A", "B"], ["B", ""]])
    occupied = np.array([[True, True], [True, False]])
    scores = np.array([[3.0, 2.0], [1.0, np.nan]])
    market_dates = np.array(["20220104", "20220105", "20220106"])
    market_codes = np.array(["A", "B", "C"])

    aligned_scores, candidates = align_event_scores(
        event_dates,
        event_codes,
        occupied,
        scores,
        market_dates,
        market_codes,
    )

    np.testing.assert_allclose(
        aligned_scores,
        [[3.0, 2.0, np.nan], [np.nan, 1.0, np.nan], [np.nan, np.nan, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        candidates,
        [[True, True, False], [False, True, False], [False, False, False]],
    )


def test_event_score_alignment_rejects_unknown_dates_or_codes():
    with pytest.raises(ValueError, match="absent from the market panel"):
        align_event_scores(
            np.array(["2022-01-04"]),
            np.array([["MISSING"]]),
            np.array([[True]]),
            np.array([[1.0]]),
            np.array(["20220104"]),
            np.array(["A"]),
        )


def test_formal_library_requires_one_oriented_gp_000_event_factor():
    valid = FactorLibrary(
        factors=[FactorSpec(name="gp_000", expression="x", sign=1.0)],
        field_names=["x"],
        windows=[],
        kind="event",
    )
    assert validate_formal_library(valid).name == "gp_000"

    wrong = FactorLibrary(
        factors=[FactorSpec(name="gp_001", expression="x", sign=1.0)],
        field_names=["x"],
        windows=[],
        kind="event",
    )
    with pytest.raises(ValueError, match="gp_000"):
        validate_formal_library(wrong)
