"""The risk report must reuse a fixed shortlist and one aligned trade ledger."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helix.config import BacktestConfig, Config, LabelConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import d2_limit_down_bias as report  # noqa: E402


def test_test_scoring_pool_keeps_rows_without_future_outcomes():
    frame = pd.DataFrame(
        {
            "trade_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-04",
                "2024-01-04",
            ],
            "stock_code": ["A", "A", "A", "B"],
            "label": [0.0, 1.0, 0.0, np.nan],
            "return": [0.0, 0.1, -0.1, np.nan],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    train, test, train_end, test_start = report.split_training_and_scoring_pool(
        frame,
        split_date="2024-01-02",
        embargo_days=1,
        required_training_columns=["label", "return"],
    )

    assert train_end == "2024-01-02"
    assert test_start == "2024-01-04"
    assert not train[["label", "return"]].isna().any().any()
    assert test["stock_code"].tolist() == ["A", "B"]
    assert np.isnan(test.loc[test["stock_code"] == "B", "label"]).all()


def test_selection_signature_binds_parameters_and_input(tmp_path):
    input_path = tmp_path / "events.parquet"
    pd.DataFrame({"x": [1.0]}).to_parquet(input_path, index=False)

    first = report.build_selection_signature(
        input_path=input_path,
        seeds=[7, 13, 42],
        top_k=4,
        split_date="2024-09-04",
        embargo_days=3,
        features=["x"],
    )
    different_top_k = report.build_selection_signature(
        input_path=input_path,
        seeds=[7, 13, 42],
        top_k=5,
        split_date="2024-09-04",
        embargo_days=3,
        features=["x"],
    )
    pd.DataFrame({"x": [2.0]}).to_parquet(input_path, index=False)
    different_input = report.build_selection_signature(
        input_path=input_path,
        seeds=[7, 13, 42],
        top_k=4,
        split_date="2024-09-04",
        embargo_days=3,
        features=["x"],
    )

    assert first != different_top_k
    assert first != different_input
    assert first["selection_algorithm_version"] >= 2


def test_fixed_top_k_never_replaces_an_unfillable_name():
    scored = pd.DataFrame(
        {
            "trade_date": ["20250102"] * 5,
            "stock_code": ["A", "B", "C", "D", "E"],
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
            "unfillable": [True, False, False, False, False],
        }
    )

    selected = report.select_fixed_top_k(scored, top_k=4)

    assert selected["stock_code"].tolist() == ["A", "B", "C", "D"]
    assert selected["selected_rank"].tolist() == [1, 2, 3, 4]
    assert selected.loc[selected["entry_filled"], "stock_code"].tolist() == ["B", "C", "D"]
    assert "E" not in selected["stock_code"].tolist()


def test_fixed_top_k_counts_each_stock_only_once():
    scored = pd.DataFrame(
        {
            "trade_date": ["20250102"] * 6,
            "stock_code": ["A", "A", "B", "C", "D", "E"],
            "score": [6.0, 5.5, 5.0, 4.0, 3.0, 2.0],
            "label_px_d1_open": [10.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "label_px_d2_close": [9.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        }
    )

    selected = report.select_fixed_top_k(scored, top_k=4)

    assert selected["stock_code"].tolist() == ["A", "B", "C", "D"]
    assert not selected.duplicated(["trade_date", "stock_code"]).any()


def test_duplicate_stock_rows_must_agree_on_published_prices():
    scored = pd.DataFrame(
        {
            "trade_date": ["20250102"] * 5,
            "stock_code": ["A", "A", "B", "C", "D"],
            "score": [6.0, 5.5, 5.0, 4.0, 3.0],
            "label_px_d1_open": [10.0, 10.1, 11.0, 12.0, 13.0],
            "label_px_d2_close": [9.0, 9.0, 10.0, 11.0, 12.0],
        }
    )

    with pytest.raises(ValueError, match="conflicting prices"):
        report.select_fixed_top_k(scored, top_k=4)


def test_fixed_selection_can_defer_all_fill_checks_to_actual_market_data():
    scored = pd.DataFrame(
        {
            "trade_date": ["20250102"] * 5,
            "stock_code": ["A", "B", "C", "D", "E"],
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    selected = report.select_fixed_top_k(scored, top_k=4)

    assert selected["stock_code"].tolist() == ["A", "B", "C", "D"]
    assert selected["entry_filled"].all()


def test_aligned_daily_comparison_keeps_dates_and_top_k_denominator():
    ledger = pd.DataFrame(
        {
            "d0_date": ["20250102", "20250102", "20250103"],
            "matched_baseline_net_return": [0.08, 0.0, -0.04],
            "matched_realistic_net_return": [0.04, 0.0, -0.06],
        }
    )

    daily = report.aggregate_aligned_daily(
        ledger,
        dates=["20250102", "20250103", "20250106"],
        top_k=4,
        overlap=2,
    )

    assert daily["date"].tolist() == ["20250102", "20250103", "20250106"]
    assert daily["baseline_portfolio_return"].tolist() == pytest.approx(
        [0.08 / 8, -0.04 / 8, 0.0]
    )
    assert daily["realistic_portfolio_return"].tolist() == pytest.approx(
        [0.04 / 8, -0.06 / 8, 0.0]
    )


def test_unfilled_and_unresolved_slots_do_not_change_the_denominator():
    ledger = pd.DataFrame(
        {
            "d0_date": ["20250102"],
            "matched_baseline_net_return": [0.0],
            "matched_realistic_net_return": [0.0],
        }
    )

    daily = report.aggregate_aligned_daily(
        ledger, dates=["20250102"], top_k=4, overlap=2
    )

    assert daily["baseline_portfolio_return"].iloc[0] == 0.0
    assert daily["realistic_portfolio_return"].iloc[0] == 0.0


def test_market_panel_keeps_suspensions_and_actual_limits():
    market = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250102", "20250103"],
            "ts_code": ["A", "B", "A"],
            "open": [9.0, 19.0, 9.2],
            "high": [9.0, 19.5, 9.4],
            "close": [9.0, 19.2, 9.3],
            "vol": [100.0, 100.0, 100.0],
            "adj_factor": [2.0, 3.0, 2.0],
            "up_limit": [11.0, 22.0, 9.9],
            "down_limit": [9.0, 18.0, 8.1],
        }
    )

    panel = report.build_exit_panel(
        market, dates=["20250102", "20250103"], codes=["A", "B"]
    )

    assert panel["down_limit"][0].tolist() == pytest.approx([9.0, 18.0])
    assert panel["open_hfq"][0].tolist() == pytest.approx([18.0, 57.0])
    assert panel["is_trading"][1].tolist() == pytest.approx([1.0, 0.0])
    assert panel["limit_price_observed"][1].tolist() == pytest.approx([1.0, 0.0])


def test_entry_fill_uses_actual_up_limit_not_a_board_guess():
    market = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250102"],
            "ts_code": ["ST", "NORMAL"],
            "open": [10.5, 10.5],
            "high": [10.5, 10.5],
            "close": [10.5, 10.5],
            "vol": [100.0, 100.0],
            "adj_factor": [1.0, 1.0],
            "up_limit": [10.5, 11.0],
            "down_limit": [9.5, 9.0],
        }
    )
    panel = report.build_exit_panel(
        market, dates=["20250101", "20250102"], codes=["ST", "NORMAL"]
    )

    label_cfg = LabelConfig()
    assert not report.entry_is_fillable(panel, 0, 0, label_cfg)
    assert report.entry_is_fillable(panel, 0, 1, label_cfg)


def test_ledger_uses_adjusted_prices_but_audits_event_raw_prices():
    market = pd.DataFrame(
        {
            "trade_date": ["20250101", "20250102", "20250103"],
            "ts_code": ["A", "A", "A"],
            "open": [10.0, 10.0, 9.0],
            "high": [10.0, 10.0, 9.5],
            "close": [10.0, 10.0, 9.0],
            "vol": [100.0, 100.0, 100.0],
            "adj_factor": [2.0, 2.0, 3.0],
            "up_limit": [11.0, 11.0, 9.9],
            "down_limit": [9.0, 9.0, 8.1],
        }
    )
    panel = report.build_exit_panel(
        market, dates=["20250101", "20250102", "20250103"], codes=["A"]
    )
    selected = pd.DataFrame(
        {
            "seed": [7],
            "trade_date": ["20250101"],
            "stock_code": ["A"],
            "selected_rank": [1],
            "label_px_d1_open": [10.0],
            "label_px_d2_close": [9.0],
        }
    )
    cfg = Config(
        backtest=BacktestConfig(
            top_k=1,
            commission_bps=0.0,
            transfer_bps=0.0,
            stamp_sell_bps=0.0,
            stamp_sell_bps_before_cut=0.0,
            slippage_bps=0.0,
        )
    )

    ledger, audit = report.resolve_selected_ledger(
        selected, panel, LabelConfig(), cfg
    )

    assert audit["max_raw_entry_price_error"] == pytest.approx(0.0)
    assert ledger["entry_price"].iloc[0] == pytest.approx(20.0)
    assert ledger["baseline_exit_price"].iloc[0] == pytest.approx(27.0)
    assert ledger["baseline_gross_return"].iloc[0] == pytest.approx(0.35)


def test_ledger_accepts_a_missing_d2_baseline_and_records_suspension():
    market = pd.DataFrame(
        {
            "trade_date": ["20250101", "20250102", "20250104"],
            "ts_code": ["A", "A", "A"],
            "open": [10.0, 10.0, 9.5],
            "high": [10.0, 10.0, 9.8],
            "close": [10.0, 10.0, 9.6],
            "vol": [100.0, 100.0, 100.0],
            "adj_factor": [1.0, 1.0, 1.0],
            "up_limit": [11.0, 11.0, 9.9],
            "down_limit": [9.0, 9.0, 8.1],
        }
    )
    panel = report.build_exit_panel(
        market,
        dates=["20250101", "20250102", "20250103", "20250104"],
        codes=["A"],
    )
    selected = pd.DataFrame(
        {
            "seed": [7],
            "trade_date": ["20250101"],
            "stock_code": ["A"],
            "selected_rank": [1],
            "label_px_d1_open": [10.0],
            "label_px_d2_close": [np.nan],
        }
    )
    cfg = Config(
        backtest=BacktestConfig(
            top_k=1,
            commission_bps=0.0,
            transfer_bps=0.0,
            stamp_sell_bps=0.0,
            stamp_sell_bps_before_cut=0.0,
            slippage_bps=0.0,
        )
    )

    ledger, _ = report.resolve_selected_ledger(selected, panel, LabelConfig(), cfg)

    assert ledger["d2_suspended"].iloc[0]
    assert ledger["exit_date"].iloc[0] == "20250104"
    assert np.isnan(ledger["baseline_gross_return"].iloc[0])
    assert ledger["realistic_gross_return"].iloc[0] == pytest.approx(-0.05)
    assert ledger["matched_baseline_net_return"].iloc[0] == 0.0
    assert ledger["matched_realistic_net_return"].iloc[0] == 0.0
