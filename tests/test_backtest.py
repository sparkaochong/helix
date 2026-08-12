"""The backtest must reproduce the label's economics exactly."""

from __future__ import annotations

import numpy as np
import pytest

from helix.config import BacktestConfig, LabelConfig
from helix.eval.backtest import run_backtest
from helix.labels.touch_label import LabelSet


def make_labels(y, entry, exit_price, valid=None) -> LabelSet:
    y = np.asarray(y, dtype=float)
    entry = np.asarray(entry, dtype=float)
    exit_price = np.asarray(exit_price, dtype=float)
    valid = np.ones_like(y, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return LabelSet(
        y=y,
        valid=valid,
        entry_price=entry,
        target_price=entry * 1.08,
        exit_price=exit_price,
    )


def free(**kwargs) -> BacktestConfig:
    """Costs switched off, so a test can assert on the exit rule alone."""
    return BacktestConfig(
        commission_bps=0.0,
        transfer_bps=0.0,
        stamp_sell_bps=0.0,
        stamp_sell_bps_before_cut=0.0,
        slippage_bps=0.0,
        **kwargs,
    )


@pytest.fixture
def cfg() -> LabelConfig:
    return LabelConfig(entry_offset=1, touch_offset=2, target_ratio=1.08)


def test_the_default_exit_holds_a_hit_all_the_way_to_the_d2_close(cfg):
    """Touching +8% is not being filled at +8%; the default books the close, up or down."""
    labels = make_labels(y=[[1.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[11.0, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(predictions, labels, np.array(["20240101"]), cfg, free(top_k=1))
    assert result.daily["gross_return"].iloc[0] == pytest.approx(0.10)
    assert result.summary["exit_rule"] == "close"


def test_a_hit_earns_exactly_the_target_under_the_label_mirroring_exit(cfg):
    labels = make_labels(y=[[1.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[3.0, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions, labels, np.array(["20240101"]), cfg, free(top_k=1, exit_rule="target")
    )
    assert result.daily["gross_return"].iloc[0] == pytest.approx(0.08)
    assert result.summary["hit_rate"] == pytest.approx(1.0)


def test_a_miss_exits_at_the_d2_close(cfg):
    labels = make_labels(y=[[0.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[9.0, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions, labels, np.array(["20240101"]), cfg, free(top_k=1, exit_rule="target")
    )
    assert result.daily["gross_return"].iloc[0] == pytest.approx(-0.10)


def test_costs_are_charged_per_side_against_notional(cfg):
    """Not ``gross - c``: the sell-side charge applies to the grown position."""
    labels = make_labels(y=[[1.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[10.8, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions,
        labels,
        np.array(["20240101"]),
        cfg,
        BacktestConfig(top_k=1, slippage_bps=10.0),
    )
    buy = (2.5 + 0.1 + 10.0) / 10_000.0
    sell = (2.5 + 0.1 + 5.0 + 10.0) / 10_000.0
    expected = 1.08 * (1.0 - sell) / (1.0 + buy) - 1.0
    assert result.daily["net_return"].iloc[0] == pytest.approx(expected)
    # Subtracting a flat round trip would be off by the notional growth on the sell leg.
    assert result.daily["net_return"].iloc[0] != pytest.approx(0.08 - buy - sell)


def test_stamp_duty_doubles_before_the_2023_cut(cfg):
    """A panel starting in 2018 straddles the halving; one rate for both is wrong."""
    labels = make_labels(y=[[0.0, 0.0]] * 2, entry=[[10.0, 10.0]] * 2, exit_price=[[10.0, 10.0]] * 2)
    predictions = np.tile([[1.0, 0.0]], (2, 1))
    result = run_backtest(
        predictions,
        labels,
        np.array(["20230825", "20230828"]),
        cfg,
        BacktestConfig(top_k=1, slippage_bps=0.0),
    )
    before, after = result.daily["net_return"].to_numpy()
    assert before < after
    assert after - before == pytest.approx(5.0 / 10_000.0, rel=1e-3)


def test_an_unknown_cost_key_is_rejected_rather_than_ignored(cfg):
    """The retired ``cost_bps`` must fail loudly; a silently ignored rate is invisible."""
    with pytest.raises(ValueError):
        BacktestConfig(top_k=1, cost_bps=15.0)


def test_selection_follows_the_prediction_ranking(cfg):
    labels = make_labels(
        y=[[0.0, 1.0, 0.0]], entry=[[10.0] * 3], exit_price=[[10.0] * 3]
    )
    predictions = np.array([[0.1, 0.9, 0.5]])
    result = run_backtest(predictions, labels, np.array(["20240101"]), cfg, free(top_k=1))
    assert result.daily["hit_rate"].iloc[0] == pytest.approx(1.0)
    assert result.daily["base_rate"].iloc[0] == pytest.approx(1 / 3)


def test_dates_with_too_few_candidates_are_skipped(cfg):
    labels = make_labels(y=[[1.0, 1.0]], entry=[[10.0, 10.0]], exit_price=[[10.0, 10.0]])
    predictions = np.array([[1.0, 0.5]])
    result = run_backtest(predictions, labels, np.array(["20240101"]), cfg, free(top_k=5))
    assert result.daily.empty
    assert result.summary == {}


def test_invalid_samples_are_never_traded(cfg):
    labels = make_labels(
        y=[[1.0, 0.0]],
        entry=[[10.0, 10.0]],
        exit_price=[[10.0, 9.0]],
        valid=[[False, True]],
    )
    predictions = np.array([[0.99, 0.01]])
    result = run_backtest(predictions, labels, np.array(["20240101"]), cfg, free(top_k=1))
    # The high-scoring name is untradable, so the loser is what actually gets bought.
    assert result.daily["hit_rate"].iloc[0] == pytest.approx(0.0)
    assert result.daily["gross_return"].iloc[0] == pytest.approx(-0.10)


def test_equity_divides_capital_across_overlapping_tranches(cfg):
    n = 4
    # Exits at +8% under either rule, so this pins the tranche maths and nothing else.
    labels = make_labels(
        y=np.ones((n, 2)), entry=np.full((n, 2), 10.0), exit_price=np.full((n, 2), 10.8)
    )
    predictions = np.tile([[1.0, 0.0]], (n, 1))
    dates = np.array([f"2024010{i}" for i in range(1, n + 1)])
    result = run_backtest(predictions, labels, dates, cfg, free(top_k=1))

    # Holding spans D+1 and D+2, so a new book overlaps the previous one: 8% / 2 per day.
    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.04)
    assert result.summary["final_equity"] == pytest.approx(1.04**n)
