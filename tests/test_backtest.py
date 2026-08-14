"""The backtest must reproduce the label's economics exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.config import BacktestConfig, LabelConfig
from helix.data.panel import Panel
from helix.data.price_lineage import AdjustmentStamp
from helix.eval import backtest as backtest_module
from helix.eval.backtest import run_backtest
from helix.labels.touch_label import LabelSet

TEST_ADJ_VERSION = "raw-times-same-day-adj-v1:" + "0" * 64


def make_labels(
    y, entry, exit_price, valid=None, touch_tradable=None, entry_valid=None
) -> LabelSet:
    y = np.asarray(y, dtype=float)
    entry = np.asarray(entry, dtype=float)
    exit_price = np.asarray(exit_price, dtype=float)
    valid = np.ones_like(y, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    touch_tradable = (
        valid.copy()
        if touch_tradable is None
        else np.asarray(touch_tradable, dtype=bool)
    )
    return LabelSet(
        y=y,
        valid=valid,
        touch_tradable=touch_tradable,
        entry_price=entry,
        target_price=entry * 1.08,
        exit_price=exit_price,
        adjustment=AdjustmentStamp("hfq", TEST_ADJ_VERSION),
        entry_valid=(
            valid.copy()
            if entry_valid is None
            else np.asarray(entry_valid, dtype=bool)
        ),
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


def make_exit_panel(n_dates=8, n_codes=1, **overrides) -> Panel:
    """Daily bars aligned to D0 rows, with exact per-symbol limit prices."""
    shape = (n_dates, n_codes)
    fields = {
        "open": np.full(shape, 10.0),
        "high": np.full(shape, 10.0),
        "close": np.full(shape, 10.0),
        "open_hfq": np.full(shape, 10.0),
        "close_hfq": np.full(shape, 10.0),
        "up_limit": np.full(shape, 11.0),
        "down_limit": np.full(shape, 9.0),
        "limit_price_observed": np.ones(shape),
        "is_trading": np.ones(shape),
    }
    fields.update(overrides)
    return Panel(
        dates=np.array([f"2024010{i}" for i in range(1, n_dates + 1)]),
        codes=np.array([f"00000{j}.SZ" for j in range(n_codes)]),
        fields={k: np.asarray(v, dtype=float) for k, v in fields.items()},
    )


@pytest.fixture
def cfg() -> LabelConfig:
    return LabelConfig(entry_offset=1, touch_offset=2, target_ratio=1.08)


def test_three_sealed_limit_down_days_exit_at_fourth_day_open(cfg):
    panel = make_exit_panel()
    for row in (2, 3, 4):
        panel.fields["open"][row, 0] = 9.0
        panel.fields["high"][row, 0] = 9.0
        panel.fields["close"][row, 0] = 9.0
    panel.fields["open"][5, 0] = 9.5
    panel.fields["open_hfq"][5, 0] = 9.5

    resolved = backtest_module.resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.exit_index == 5
    assert resolved.exit_price == pytest.approx(9.5)
    assert resolved.exit_session == "open"
    assert resolved.d2_limit_down
    assert resolved.delay_days == 3
    assert resolved.holding_days == 5
    assert resolved.exit_price / 10.0 - 1.0 == pytest.approx(-0.05)


def test_intraday_opening_that_reseals_at_close_keeps_deferring(cfg):
    panel = make_exit_panel()
    panel.fields["open"][2:4, 0] = 9.0
    panel.fields["close"][2:4, 0] = 9.0
    panel.fields["high"][2, 0] = 9.0
    panel.fields["high"][3, 0] = 9.6  # intraday opening does not provide a sampled fill
    panel.fields["open"][4, 0] = 9.4
    panel.fields["open_hfq"][4, 0] = 9.4

    resolved = backtest_module.resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.exit_index == 4
    assert resolved.exit_session == "open"
    assert resolved.delay_days == 2


def test_d2_suspension_defers_to_next_tradable_open(cfg):
    panel = make_exit_panel()
    panel.fields["is_trading"][2, 0] = 0.0
    panel.fields["open"][3, 0] = 9.4
    panel.fields["open_hfq"][3, 0] = 9.4

    resolved = backtest_module.resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.exit_index == 3
    assert resolved.exit_price == pytest.approx(9.4)
    assert resolved.exit_session == "open"
    assert resolved.encountered_suspension
    assert resolved.delay_days == 1


def test_unresolved_position_at_panel_end_has_no_fabricated_price(cfg):
    panel = make_exit_panel(n_dates=4)
    panel.fields["open"][2:, 0] = 9.0
    panel.fields["high"][2:, 0] = 9.0
    panel.fields["close"][2:, 0] = 9.0

    resolved = backtest_module.resolve_realistic_exit(panel, 0, 0, cfg)

    assert resolved.unresolved_at_end
    assert resolved.exit_index is None
    assert np.isnan(resolved.exit_price)


def test_each_symbol_uses_its_actual_daily_limit(cfg):
    down = np.column_stack([np.full(6, 9.0), np.full(6, 8.0)])
    panel = make_exit_panel(n_dates=6, n_codes=2, down_limit=down)
    panel.fields["open"][2] = [9.0, 9.0]
    panel.fields["high"][2] = [9.0, 9.0]
    panel.fields["close"][2] = [9.0, 9.0]
    panel.fields["open"][3, 0] = 9.3
    panel.fields["open_hfq"][3, 0] = 9.3

    first = backtest_module.resolve_realistic_exit(panel, 0, 0, cfg)
    second = backtest_module.resolve_realistic_exit(panel, 0, 1, cfg)

    assert first.d2_limit_down
    assert first.exit_index == 3
    assert not second.d2_limit_down
    assert second.exit_index == 2
    assert second.exit_session == "d2_close"


def test_realistic_exit_switch_defaults_off():
    assert BacktestConfig().enable_realistic_exit is False


def test_realistic_mode_uses_deferred_price_and_records_trade(cfg):
    panel = make_exit_panel(n_dates=5)
    panel.fields["open"][2, 0] = 9.0
    panel.fields["high"][2, 0] = 9.0
    panel.fields["close"][2, 0] = 9.0
    panel.fields["open"][3, 0] = 9.2
    panel.fields["open_hfq"][3, 0] = 9.2
    labels = make_labels(
        y=np.zeros((5, 1)),
        entry=np.full((5, 1), 10.0),
        exit_price=np.full((5, 1), 9.0),
    )
    predictions = np.full((5, 1), np.nan)
    predictions[0, 0] = 1.0
    candidates = np.zeros((5, 1), dtype=bool)
    candidates[0, 0] = True

    result = run_backtest(
        predictions,
        labels,
        candidates,
        panel.dates,
        cfg,
        free(top_k=1, enable_realistic_exit=True),
        panel=panel,
    )

    assert result.daily["gross_return"].iloc[0] == pytest.approx(-0.08)
    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(-0.08 / 2)
    assert result.trades["baseline_gross_return"].iloc[0] == pytest.approx(-0.10)
    assert result.trades["realistic_gross_return"].iloc[0] == pytest.approx(-0.08)
    assert result.trades["d2_limit_down"].iloc[0]
    assert result.trades["delay_days"].iloc[0] == 1
    assert result.summary["limit_down_exit_share"] == pytest.approx(1.0)
    assert result.summary["avg_delay_days"] == pytest.approx(1.0)


def test_realistic_mode_enters_before_a_d2_suspension(cfg):
    panel = make_exit_panel(n_dates=5)
    panel.fields["is_trading"][2, 0] = 0.0
    panel.fields["open"][3, 0] = 9.5
    panel.fields["open_hfq"][3, 0] = 9.5
    labels = make_labels(
        y=np.full((5, 1), np.nan),
        entry=np.full((5, 1), 10.0),
        exit_price=np.full((5, 1), np.nan),
        valid=np.zeros((5, 1), dtype=bool),
        touch_tradable=np.zeros((5, 1), dtype=bool),
        entry_valid=np.ones((5, 1), dtype=bool),
    )
    predictions = np.full((5, 1), np.nan)
    predictions[0, 0] = 1.0
    candidates = np.zeros((5, 1), dtype=bool)
    candidates[0, 0] = True

    result = run_backtest(
        predictions,
        labels,
        candidates,
        panel.dates,
        cfg,
        free(top_k=1, enable_realistic_exit=True),
        panel=panel,
    )

    assert result.daily["n_executed"].iloc[0] == 1
    assert result.trades["encountered_suspension"].iloc[0]
    assert result.trades["exit_date"].iloc[0] == panel.dates[3]
    assert result.summary["d2_suspension_count"] == pytest.approx(1.0)


def test_realistic_mode_requires_an_observed_d1_up_limit(cfg):
    panel = make_exit_panel(n_dates=5)
    panel.fields["limit_price_observed"][1, 0] = 0.0
    labels = make_labels(
        y=np.zeros((5, 1)),
        entry=np.full((5, 1), 10.0),
        exit_price=np.full((5, 1), 10.0),
        entry_valid=np.ones((5, 1), dtype=bool),
    )
    predictions = np.full((5, 1), np.nan)
    predictions[0, 0] = 1.0
    candidates = np.zeros((5, 1), dtype=bool)
    candidates[0, 0] = True

    result = run_backtest(
        predictions,
        labels,
        candidates,
        panel.dates,
        cfg,
        free(top_k=1, enable_realistic_exit=True),
        panel=panel,
    )

    assert result.daily["n_selected"].iloc[0] == 1
    assert result.daily["n_executed"].iloc[0] == 0
    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.0)
    assert result.trades.empty


def test_unresolved_realistic_exit_keeps_the_slot_in_cash(cfg):
    panel = make_exit_panel(n_dates=4)
    panel.fields["open"][2:, 0] = 9.0
    panel.fields["high"][2:, 0] = 9.0
    panel.fields["close"][2:, 0] = 9.0
    labels = make_labels(
        y=np.zeros((4, 1)),
        entry=np.full((4, 1), 10.0),
        exit_price=np.full((4, 1), 9.0),
    )
    predictions = np.full((4, 1), np.nan)
    predictions[0, 0] = 1.0
    candidates = np.zeros((4, 1), dtype=bool)
    candidates[0, 0] = True

    result = run_backtest(
        predictions,
        labels,
        candidates,
        panel.dates,
        cfg,
        free(top_k=1, enable_realistic_exit=True),
        panel=panel,
    )

    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.0)
    assert result.summary["unresolved_at_end"] == pytest.approx(1.0)
    assert result.trades["unresolved_at_end"].iloc[0]
    assert np.isnan(result.trades["realistic_gross_return"].iloc[0])


def test_switch_off_ignores_realistic_market_data(cfg):
    labels = make_labels([[1.0]], [[10.0]], [[11.0]])
    predictions = np.array([[1.0]])
    candidates = np.ones((1, 1), dtype=bool)
    dates = np.array(["20240101"])
    config = free(top_k=1)

    without_panel = run_backtest(predictions, labels, candidates, dates, cfg, config)
    with_panel = run_backtest(
        predictions, labels, candidates, dates, cfg, config, panel=make_exit_panel()
    )

    pd.testing.assert_frame_equal(without_panel.daily, with_panel.daily)
    assert without_panel.summary.keys() == with_panel.summary.keys()
    for key, expected in without_panel.summary.items():
        actual = with_panel.summary[key]
        if isinstance(expected, float) and np.isnan(expected):
            assert isinstance(actual, float) and np.isnan(actual)
        else:
            assert actual == expected


def test_requested_performance_metrics_have_explicit_daily_semantics():
    returns = np.array([0.10, -0.05, -0.02, 0.0, 0.03])

    metrics = backtest_module.summarize_portfolio_returns(returns)

    equity = np.cumprod(1.0 + returns)
    assert metrics["cagr"] == pytest.approx(equity[-1] ** (252 / len(returns)) - 1.0)
    assert metrics["day_win_rate"] == pytest.approx(2 / 5)
    assert metrics["profit_loss_ratio"] == pytest.approx(0.065 / 0.035)
    assert metrics["max_consecutive_losses"] == pytest.approx(2.0)
    assert metrics["max_drawdown"] == pytest.approx(backtest_module._max_drawdown(equity))


def test_max_drawdown_includes_the_initial_equity_peak():
    metrics = backtest_module.summarize_portfolio_returns(np.array([-0.10, 0.20]))

    assert metrics["max_drawdown"] == pytest.approx(-0.10)


def test_switch_off_preserves_the_legacy_drawdown_baseline(cfg):
    labels = make_labels(
        y=[[0.0], [0.0]],
        entry=[[10.0], [10.0]],
        exit_price=[[8.0], [12.0]],
    )

    result = run_backtest(
        np.ones((2, 1)),
        labels,
        np.ones((2, 1), dtype=bool),
        np.array(["20240101", "20240102"]),
        cfg,
        free(top_k=1),
    )

    assert result.daily["portfolio_return"].tolist() == pytest.approx([-0.10, 0.10])
    assert result.summary["max_drawdown"] == pytest.approx(0.0)


def test_the_default_exit_holds_a_hit_all_the_way_to_the_d2_close(cfg):
    """Touching +8% is not being filled at +8%; the default books the close, up or down."""
    labels = make_labels(y=[[1.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[11.0, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=1),
    )
    assert result.daily["gross_return"].iloc[0] == pytest.approx(0.10)
    assert result.summary["exit_rule"] == "close"


def test_a_hit_earns_exactly_the_target_under_the_label_mirroring_exit(cfg):
    labels = make_labels(y=[[1.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[3.0, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=1, exit_rule="target"),
    )
    assert result.daily["gross_return"].iloc[0] == pytest.approx(0.08)
    assert result.summary["hit_rate"] == pytest.approx(1.0)


def test_a_miss_exits_at_the_d2_close(cfg):
    labels = make_labels(y=[[0.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[9.0, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=1, exit_rule="target"),
    )
    assert result.daily["gross_return"].iloc[0] == pytest.approx(-0.10)


def test_costs_are_charged_per_side_against_notional(cfg):
    """Not ``gross - c``: the sell-side charge applies to the grown position."""
    labels = make_labels(y=[[1.0, 0.0]], entry=[[10.0, 10.0]], exit_price=[[10.8, 10.0]])
    predictions = np.array([[1.0, 0.0]])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
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
        np.ones_like(predictions, dtype=bool),
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
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=1),
    )
    assert result.daily["hit_rate"].iloc[0] == pytest.approx(1.0)
    assert result.daily["base_rate"].iloc[0] == pytest.approx(1 / 3)


def test_dates_with_too_few_candidates_are_skipped(cfg):
    labels = make_labels(y=[[1.0, 1.0]], entry=[[10.0, 10.0]], exit_price=[[10.0, 10.0]])
    predictions = np.array([[1.0, 0.5]])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=5),
    )
    assert result.daily.empty
    assert result.summary == {}


def test_future_untradability_does_not_replace_the_d0_top_pick(cfg):
    labels = make_labels(
        y=[[np.nan, 0.0]],
        entry=[[10.0, 10.0]],
        exit_price=[[10.0, 9.0]],
        valid=[[False, True]],
        touch_tradable=[[False, True]],
    )
    predictions = np.array([[0.99, 0.01]])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=1),
    )

    # D0 selects the high-scoring name before D+2 tradability is known. Execution
    # validation may reject it, but must not reach deeper into the ranking for a loser.
    assert len(result.daily) == 1
    assert result.daily["n_selected"].iloc[0] == 1
    assert result.daily["n_executed"].iloc[0] == 0
    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.0)


def test_partial_execution_keeps_unfilled_slots_in_cash(cfg):
    labels = make_labels(
        y=[[1.0, np.nan]],
        entry=[[10.0, 10.0]],
        exit_price=[[11.0, 10.0]],
        valid=[[True, False]],
        touch_tradable=[[True, False]],
    )
    predictions = np.array([[0.9, 0.8]])

    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101"]),
        cfg,
        free(top_k=2),
    )

    assert result.daily["n_selected"].iloc[0] == 2
    assert result.daily["n_executed"].iloc[0] == 1
    # One +10% fill occupies one of two slots, then the tranche is split over D+1/D+2.
    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.10 / 2 / 2)


def test_trade_summary_weights_individual_fills_not_daily_baskets(cfg):
    labels = make_labels(
        y=[[1.0, np.nan], [0.0, 0.0]],
        entry=[[10.0, 10.0], [10.0, 10.0]],
        exit_price=[[11.0, 10.0], [9.0, 9.0]],
        valid=[[True, False], [True, True]],
        touch_tradable=[[True, False], [True, True]],
    )
    predictions = np.array([[0.9, 0.8], [0.9, 0.8]])

    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        np.array(["20240101", "20240102"]),
        cfg,
        free(top_k=2),
    )

    # Three fills: +10%, -10%, -10%. Equal-weighting the two daily basket means
    # would incorrectly report 0% and a 50% win rate.
    assert result.summary["mean_trade_return_net"] == pytest.approx(-0.10 / 3)
    assert result.summary["trade_win_rate"] == pytest.approx(1 / 3)


def test_equity_divides_capital_across_overlapping_tranches(cfg):
    n = 4
    # Exits at +8% under either rule, so this pins the tranche maths and nothing else.
    labels = make_labels(
        y=np.ones((n, 2)), entry=np.full((n, 2), 10.0), exit_price=np.full((n, 2), 10.8)
    )
    predictions = np.tile([[1.0, 0.0]], (n, 1))
    dates = np.array([f"2024010{i}" for i in range(1, n + 1)])
    result = run_backtest(
        predictions,
        labels,
        np.ones_like(predictions, dtype=bool),
        dates,
        cfg,
        free(top_k=1),
    )

    # Holding spans D+1 and D+2, so a new book overlaps the previous one: 8% / 2 per day.
    assert result.daily["portfolio_return"].iloc[0] == pytest.approx(0.04)
    assert result.summary["final_equity"] == pytest.approx(1.04**n)
