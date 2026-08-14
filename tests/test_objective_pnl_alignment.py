"""Contracts for the reproducible training-only objective/P&L audit."""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import pytest

from helix.config import BacktestConfig
from scripts import objective_pnl_alignment as alignment_audit
from scripts.objective_pnl_alignment import (
    OBJECTIVE_D0_END,
    PRODUCTION_TOP_K,
    SUPPLEMENTAL_TOP_K,
    TRAIN_END,
    TRAIN_START,
    alignment_decision,
    correlation,
    evaluate_library_alignment,
    evaluate_months,
    library_alignment_statistics,
    market_regime,
    render_report,
    validate_forward_exit,
    validate_training_frame,
)


def test_formal_calendar_and_metric_roles_are_fixed():
    assert TRAIN_START == "2022-01-04"
    assert TRAIN_END == "2024-09-04"
    assert OBJECTIVE_D0_END == "2024-09-02"
    assert PRODUCTION_TOP_K == BacktestConfig().top_k == 4
    assert SUPPLEMENTAL_TOP_K == 10


def test_training_frame_rejects_a_decision_after_objective_end():
    frame = pd.DataFrame({"trade_date": [OBJECTIVE_D0_END, "2024-09-03"]})
    with pytest.raises(ValueError, match="objective D0 end"):
        validate_training_frame(frame)


def test_market_regime_is_trailing_and_uses_fixed_thresholds():
    returns = pd.Series(
        [0.01] * 20 + [-0.01] * 20,
        index=pd.date_range("2024-01-02", periods=40, freq="B").strftime("%Y-%m-%d"),
    )
    result = market_regime(returns)
    assert result.iloc[18] == "unavailable"
    assert result.iloc[19] == "bull"
    assert result.iloc[-1] == "bear"


def test_correlation_reports_pearson_and_spearman_pvalues():
    result = correlation(np.arange(10.0), np.arange(10.0))
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson_pvalue"] < 0.05
    assert result["spearman_pvalue"] < 0.05


def test_monthly_table_keeps_each_month_separate():
    daily = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"],
            "hit_ic": [0.1, 0.2, 0.3, 0.4],
            "production_net": [0.01, 0.02, -0.01, -0.02],
        }
    )
    result = evaluate_months(daily)
    assert result["month"].tolist() == ["2024-01", "2024-02"]
    assert result["production_net"].tolist() == pytest.approx([0.015, -0.015])


def test_horizon_filter_drops_only_exits_after_train_end():
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-09-02", "2024-09-03", "2024-09-04"],
            "exit_date": ["2024-09-04", "2024-09-05", "2024-09-06"],
            "gross_return": [0.01, 9.0, 9.0],
        }
    )
    result = validate_forward_exit(frame, train_end=TRAIN_END)
    assert result["gross_return"].tolist() == [0.01]


def test_top10_cannot_change_production_acceptance():
    production = {
        "pearson": 0.4,
        "pearson_pvalue": 0.01,
        "spearman": 0.3,
        "spearman_pvalue": 0.02,
    }
    assert list(inspect.signature(alignment_decision).parameters) == ["production"]
    assert alignment_decision(production) == "PASS"


def test_library_acceptance_uses_fit_against_selection():
    table = pd.DataFrame(
        {"fit_net": [1.0, 2.0, 3.0, 4.0], "selection_net": [2.0, 4.0, 6.0, 8.0]}
    )
    result = library_alignment_statistics(table)
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)


def test_library_evaluation_uses_production_k_from_config():
    rows, names = 30, 8
    base = np.tile(np.arange(names, dtype=float), (rows, 1))
    returns = np.tile(np.linspace(-0.02, 0.03, names), (rows, 1))
    table, _ = evaluate_library_alignment(
        factor_values={"good": base, "bad": -base},
        gross_return=returns,
        candidate_mask=np.ones_like(base, dtype=bool),
        dates=np.array([f"{20200101 + index:08d}" for index in range(rows)]),
        config=BacktestConfig(
            top_k=4,
            commission_bps=0,
            transfer_bps=0,
            stamp_sell_bps=0,
            stamp_sell_bps_before_cut=0,
            slippage_bps=0,
        ),
        fit_rows=slice(0, 20),
        selection_rows=slice(20, 30),
    )
    assert table.loc[table["factor"] == "good", "fit_net"].item() > table.loc[
        table["factor"] == "bad", "fit_net"
    ].item()
    assert set(table["top_k"]) == {4}


def test_audit_library_validation_reuses_vectorized_alignment(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=130).strftime("%Y-%m-%d")
    codes = [f"{index:06d}.SZ" for index in range(12)]
    frame = pd.DataFrame(
        [(date, code) for date in dates for code in codes],
        columns=["trade_date", "stock_code"],
    )
    stock_rank = np.tile(np.arange(len(codes), dtype=float), len(dates))
    day_wave = np.repeat(np.sin(np.arange(len(dates)) / 7), len(codes))
    frame["label_d2_hit_8pct"] = (stock_rank >= 8).astype(float)
    frame["label_d2_peak_return"] = stock_rank / 100
    frame["label_d2_return"] = stock_rank / 1000 + day_wave / 100
    frame["label_px_d1_open"] = 10.0
    frame["turnover_ratio_d0"] = stock_rank + 1
    values = {
        "ascending": stock_rank,
        "descending": -stock_rank,
        "alternating": stock_rank * np.repeat(
            np.where(np.arange(len(dates)) % 2, 1.0, -1.0), len(codes)
        ),
    }
    metadata = {
        name: {"library": "test", "old_fit_gini": 0.0, "n_nodes": 1.0}
        for name in values
    }
    calls = []
    original = alignment_audit.evaluate_library_alignment

    def tracked_evaluation(**kwargs):
        table, statistics = original(**kwargs)
        calls.append(table.copy())
        return table, statistics

    monkeypatch.setattr(alignment_audit, "evaluate_library_alignment", tracked_evaluation)

    table, _, _, _ = alignment_audit._library_validation(
        frame,
        values,
        metadata,
        BacktestConfig(
            top_k=4,
            commission_bps=0,
            transfer_bps=0,
            stamp_sell_bps=0,
            stamp_sell_bps_before_cut=0,
            slippage_bps=0,
        ),
    )

    assert {call["factor"].item() for call in calls} == set(values)
    tested = pd.concat(calls).set_index("factor").sort_index()
    audited = table.set_index("factor").sort_index()
    pd.testing.assert_frame_equal(
        audited[["top_k", "fit_net", "selection_net", "fit_ir", "selection_ir"]],
        tested[["top_k", "fit_net", "selection_net", "fit_ir", "selection_ir"]],
    )


def test_missing_horizon_bars_logs_explicit_degradation(tmp_path, caplog):
    bars_path = tmp_path / "missing-bars.parquet"

    with caplog.at_level(logging.WARNING, logger=alignment_audit.__name__):
        result = alignment_audit._horizon_daily(
            pd.DataFrame(), np.array([]), bars_path, BacktestConfig()
        )

    assert result == {}
    assert str(bars_path) in caplog.text
    assert "exists=False" in caplog.text
    assert "empty horizon results" in caplog.text


def test_report_renderer_states_scope_root_cause_and_roles():
    report = render_report(
        {
            "calendar": {"n_objective_dates": 647},
            "root_cause": "盘中触达标签与 D+2 收盘退出错配",
            "production_alignment": {
                "pearson": 0.85,
                "pearson_pvalue": 0.001,
                "spearman": 0.72,
                "spearman_pvalue": 0.001,
            },
        }
    )
    assert "仅训练集" in report
    assert "Top4" in report
    assert "Top10" in report
    assert "盘中触达标签与 D+2 收盘退出错配" in report
