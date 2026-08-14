from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.gp000_loss_attribution as audit_module
from helix.config import BacktestConfig
from helix.gp.library import FactorLibrary, FactorSpec
from scripts.gp000_loss_attribution import (
    OutputPaths,
    apply_cost_by_d0,
    audit_adjustment_chain,
    build_daily_artifact,
    build_price_lookup,
    deduplicate_or_fail,
    evaluate_ex_right_samples,
    evaluate_horizon_decay,
    evaluate_monthly_returns,
    evaluate_quintiles,
    evaluate_style_neutral_book,
    evaluate_top_k_book,
    load_audit_config,
    outcome_complete_dates,
    rank_root_causes,
    render_decay_svg,
    render_equity_svg,
    render_report,
    replay_formal_factor,
    summarize_quintile_monotonicity,
    validate_formal_factor,
    validate_training_calendar,
    write_outputs,
)


def test_outcome_complete_dates_never_cross_training_end() -> None:
    calendar = np.array(
        [
            "2024-08-21",
            "2024-08-22",
            "2024-08-23",
            "2024-08-26",
            "2024-08-27",
            "2024-08-28",
            "2024-08-29",
            "2024-08-30",
            "2024-09-02",
            "2024-09-03",
            "2024-09-04",
        ]
    )

    d2 = outcome_complete_dates(calendar, calendar, 2)
    d10 = outcome_complete_dates(calendar, calendar, 10)

    assert d2.tolist() == calendar[:-2].tolist()
    assert d10.tolist() == ["2024-08-21"]


def test_validate_formal_factor_rejects_other_gp000_library() -> None:
    library = FactorLibrary(
        factors=[FactorSpec("gp_000", "neg(x)", 1.0)],
        field_names=["x"],
        windows=[],
        kind="event",
    )

    with pytest.raises(ValueError, match="expression"):
        validate_formal_factor(library)


def _market_for_adjustment_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-05-10", "2024-05-13", "2024-05-14"],
            "ts_code": ["000001.SZ"] * 3,
            "open": [9.8, 10.0, 8.9],
            "high": [10.0, 10.2, 9.2],
            "close": [9.9, 10.0, 9.0],
            "adj_factor": [1.0, 1.0, 1.12],
        }
    )


def test_adjusted_return_removes_ex_right_gap() -> None:
    events = pd.DataFrame(
        {
            "trade_date": ["2024-05-10"],
            "stock_code": ["000001.SZ"],
            "label_px_d1_open": [10.0],
            "label_px_d2_high": [9.2],
            "label_px_d2_close": [9.0],
            "label_d2_return": [-0.1],
            "label_d2_hit_8pct": [0.0],
        }
    )
    prices = build_price_lookup(
        _market_for_adjustment_test(),
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )

    audit, aligned = audit_adjustment_chain(events, prices)

    assert aligned.loc[0, "raw_return"] == pytest.approx(-0.1)
    assert aligned.loc[0, "hfq_return"] == pytest.approx(0.008)
    assert audit["return_mismatch_count"] == 1
    assert audit["event_prices_match_raw"] is True


def test_ex_right_detection_uses_adj_factor_change_on_same_stock() -> None:
    prices = build_price_lookup(
        _market_for_adjustment_test(),
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )

    assert prices.ex_right[:, 0].tolist() == [False, False, True]


def test_hit_rounding_mismatch_is_separate_from_adjustment_flip() -> None:
    events = pd.DataFrame(
        {
            "trade_date": ["2024-05-10"],
            "stock_code": ["000001.SZ"],
            "label_px_d1_open": [10.0],
            "label_px_d2_high": [9.2],
            "label_px_d2_close": [9.0],
            "label_d2_return": [-0.1],
            "label_d2_hit_8pct": [1.0],
        }
    )
    prices = build_price_lookup(
        _market_for_adjustment_test(),
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )

    audit, _ = audit_adjustment_chain(events, prices)

    assert audit["event_raw_hit_mismatch_count"] == 1
    assert audit["hit_flip_count"] == 0
    assert audit["event_label_to_hfq_hit_difference_count"] == 1


def test_constant_adjustment_factor_cannot_create_threshold_hit_flip() -> None:
    events = pd.DataFrame(
        {
            "trade_date": ["2024-05-10"],
            "stock_code": ["000001.SZ"],
            "label_px_d1_open": [10.0],
            "label_px_d2_high": [10.8],
            "label_px_d2_close": [10.0],
            "label_d2_return": [0.0],
            "label_d2_hit_8pct": [1.0],
        }
    )
    market = _market_for_adjustment_test().assign(
        open=[9.8, 10.0, 10.0],
        high=[10.0, 10.2, 10.8],
        close=[9.9, 10.0, 10.0],
        adj_factor=1.03,
    )
    prices = build_price_lookup(
        market,
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )

    audit, _ = audit_adjustment_chain(events, prices)

    assert audit["hit_flip_count"] == 0


def test_ex_right_diagnostics_include_robust_control_and_top4_contribution() -> None:
    rows = []
    for day, date in enumerate(("2024-01-02", "2024-01-03")):
        for name in range(5):
            rows.append(
                {
                    "trade_date": date,
                    "stock_code": f"S{name}",
                    "factor_score": float(name + day),
                    "raw_return": name / 100.0,
                    "hfq_return": name / 100.0 + (0.02 if day == 1 and name == 4 else 0),
                    "return_delta": 0.02 if day == 1 and name == 4 else 0.0,
                    "raw_hit": False,
                    "reconstructed_raw_hit": False,
                    "hfq_hit": day == 1 and name == 4,
                    "d0_ex_right": day == 1 and name == 4,
                    "entry_ex_right": False,
                    "exit_ex_right": day == 1 and name == 4,
                    "d0_ex_right_observable": True,
                    "entry_ex_right_observable": True,
                    "exit_ex_right_observable": True,
                    "entry_date": date,
                    "exit_date": date,
                }
            )
    aligned = pd.DataFrame(rows)

    result = evaluate_ex_right_samples(aligned, BacktestConfig(top_k=2))

    d0_count = result["counts"].query("stage == 'D0'").iloc[0]
    assert d0_count["n_events"] == 1
    assert d0_count["n_stocks"] == 1
    assert set(result["factor_diagnostics"]["sample"]) == {"D0除权", "D0非除权"}
    assert "factor_robust_tail_rate" in result["factor_diagnostics"]
    assert result["top4_summary"]["selected_any_ex_right"] == 1
    assert result["top4_summary"]["hfq_minus_raw_book_return_sum"] > 0


def _synthetic_factor_returns(days: int = 2, names: int = 10) -> pd.DataFrame:
    rows = []
    for day in range(days):
        for name in range(names):
            rows.append(
                {
                    "trade_date": f"2024-01-{day + 2:02d}",
                    "stock_code": f"S{name:02d}",
                    "factor_score": float(name),
                    "gross_return": name / 100.0,
                    "hit_hfq": float(name >= 8),
                }
            )
    return pd.DataFrame(rows)


def test_quintiles_are_daily_and_ordered_low_to_high() -> None:
    result = evaluate_quintiles(_synthetic_factor_returns(), BacktestConfig())

    assert result["quintile"].tolist() == [1, 2, 3, 4, 5]
    assert result["n"].tolist() == [4, 4, 4, 4, 4]
    assert result.loc[4, "gross_return"] > result.loc[0, "gross_return"]


def test_quintiles_are_formed_before_future_outcome_filtering() -> None:
    frame = _synthetic_factor_returns(days=1, names=10)
    frame.loc[frame["factor_score"] == 9.0, "gross_return"] = np.nan

    result = evaluate_quintiles(frame, BacktestConfig())
    monotonicity = summarize_quintile_monotonicity(result)

    assert result["n"].tolist() == [2, 2, 2, 2, 2]
    assert result["n_observed"].tolist() == [2, 2, 2, 2, 1]
    assert result.loc[4, "gross_return"] == pytest.approx(0.08)
    assert monotonicity["q5_minus_q1_gross"] == pytest.approx(0.075)
    assert monotonicity["gross_spearman"] == pytest.approx(1.0)


def test_top4_missing_exit_stays_cash_without_replacement() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 5,
            "stock_code": list("ABCDE"),
            "factor_score": [5, 4, 3, 2, 1],
            "gross_return": [0.1, np.nan, 0.03, 0.02, 9.0],
        }
    )

    _, daily = evaluate_top_k_book(
        frame,
        BacktestConfig(top_k=4),
        gross=True,
        overlap=2,
    )

    assert daily.loc[0, "n_executed"] == 3
    assert daily.loc[0, "portfolio_return"] == pytest.approx(
        (0.1 + 0.03 + 0.02) / 4 / 2
    )


def test_monthly_returns_compound_daily_returns() -> None:
    daily = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-02-01"],
            "gross_portfolio_return": [0.1, -0.1, 0.2],
            "net_portfolio_return": [0.08, -0.12, 0.18],
        }
    )

    monthly = evaluate_monthly_returns(daily)

    assert monthly.loc[0, "gross_return"] == pytest.approx(1.1 * 0.9 - 1.0)
    assert monthly.loc[1, "net_return"] == pytest.approx(0.18)


def test_cost_model_normalizes_dates_across_stamp_duty_cut() -> None:
    config = BacktestConfig(
        commission_bps=0.0,
        transfer_bps=0.0,
        slippage_bps=0.0,
        stamp_sell_bps=5.0,
        stamp_sell_bps_before_cut=10.0,
    )

    result = apply_cost_by_d0(
        np.zeros(2),
        np.array(["2023-08-25", "2023-08-28"]),
        config,
    )

    assert result.tolist() == pytest.approx([-0.001, -0.0005])


def test_cost_model_rejects_noncanonical_or_impossible_dates() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD or YYYYMMDD"):
        apply_cost_by_d0([0.0], ["prefix-2023-08-28"], BacktestConfig())
    with pytest.raises(ValueError, match="valid calendar dates"):
        apply_cost_by_d0([0.0], ["2023-02-30"], BacktestConfig())


def test_audit_requires_formal_top4_but_loads_effective_costs(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        "backtest:\n"
        "  top_k: 4\n"
        "  exit_rule: close\n"
        "  commission_bps: 3.0\n",
        encoding="utf-8",
    )

    config = load_audit_config(config_path)

    assert config.top_k == 4
    assert config.commission_bps == 3.0

    config_path.write_text("backtest:\n  top_k: 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Top4"):
        load_audit_config(config_path)


def test_horizon_decay_uses_horizon_as_overlap_and_truncates_exit() -> None:
    calendar = [
        "2024-08-29",
        "2024-08-30",
        "2024-09-02",
        "2024-09-03",
        "2024-09-04",
    ]
    market = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": code,
                "open": 10.0 + day,
                "high": 10.2 + day,
                "close": 10.1 + day,
                "adj_factor": 1.0,
            }
            for day, date in enumerate(calendar)
            for code in ("A", "B")
        ]
    )
    prices = build_price_lookup(market, calendar, ["A", "B"])
    events = pd.DataFrame(
        [
            {"trade_date": date, "stock_code": code, "factor_score": score}
            for date in calendar
            for score, code in enumerate(("A", "B"), start=1)
        ]
    )

    evidence = evaluate_horizon_decay(
        events,
        prices,
        BacktestConfig(top_k=2),
        horizons=range(1, 4),
        min_ic_samples=2,
    )

    assert evidence["summary"]["horizon"].tolist() == [1, 2, 3]
    assert evidence["summary"].loc[2, "d0_end"] == "2024-08-30"
    assert evidence["daily"].query("horizon == 3")["exit_date"].max() <= "2024-09-04"
    assert evidence["daily"].query("horizon == 3")["overlap"].eq(3).all()


def _style_test_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = []
    styles = []
    members = []
    for name in range(10):
        code = f"S{name:02d}"
        industry = "I1" if name < 5 else "I2"
        members.append(
            {
                "index_code": industry,
                "industry_name": industry,
                "stock_code": code,
                "in_date": "2020-01-01",
                "out_date": np.nan,
            }
        )
        for day, date in enumerate(("2024-01-02", "2024-01-03")):
            events.append(
                {
                    "trade_date": date,
                    "stock_code": code,
                    "factor_score": float((name - 4.5) ** 2 + day * name),
                    "gross_return": name / 100.0,
                }
            )
            styles.append(
                {
                    "trade_date": date,
                    "stock_code": code,
                    "log_total_mv": float(name),
                    "momentum_20d": float(name % 3),
                    "volatility_20d": float(name % 4),
                    "turnover_mean_20d": float(name % 5),
                }
            )
    return pd.DataFrame(events), pd.DataFrame(styles), pd.DataFrame(members)


def test_style_neutral_book_uses_common_mask_and_is_orthogonal() -> None:
    events, styles, members = _style_test_inputs()

    result = evaluate_style_neutral_book(
        events,
        styles,
        members,
        BacktestConfig(top_k=2),
        min_ic_samples=2,
    )

    assert result["raw"]["n_days"] == result["style_neutral"]["n_days"]
    assert result["orthogonality"]["max_abs_normalized_exposure"] < 1e-10


def test_daily_artifact_contains_each_horizon_score_and_cost_arm() -> None:
    calendar = ["2024-08-29", "2024-08-30", "2024-09-02", "2024-09-03"]
    market = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": code,
                "open": 10.0 + day,
                "high": 10.2 + day,
                "close": 10.1 + day,
                "adj_factor": 1.0,
            }
            for day, date in enumerate(calendar)
            for code in ("A", "B")
        ]
    )
    prices = build_price_lookup(market, calendar, ["A", "B"])
    events = pd.DataFrame(
        [
            {"trade_date": date, "stock_code": code, "factor_score": float(rank)}
            for date in calendar
            for rank, code in enumerate(("A", "B"), start=1)
        ]
    )
    scores = events.rename(columns={"factor_score": "raw_score"}).copy()
    scores["style_neutral_score"] = -scores["raw_score"]

    artifact = build_daily_artifact(
        events,
        prices,
        scores,
        BacktestConfig(top_k=1),
        horizons=(1, 2),
    )

    assert set(artifact["horizon"]) == {1, 2}
    assert set(artifact["score_basis"]) == {"raw_common", "style_neutral"}
    assert set(artifact["cost_basis"]) == {"gross", "net"}
    assert artifact["exit_date"].max() <= "2024-09-03"


def test_conflicting_duplicate_event_keys_fail_closed() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "stock_code": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="conflicting duplicate"):
        deduplicate_or_fail(frame, ["trade_date", "stock_code"], source="event")


def _minimal_evidence() -> dict[str, object]:
    daily = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "gross_portfolio_return": [0.01, -0.02],
            "net_portfolio_return": [0.008, -0.022],
            "style_neutral_return": [0.004, -0.006],
        }
    )
    decay_daily = pd.DataFrame(
        [
            {
                "date": date,
                "horizon": horizon,
                "net_portfolio_return": value,
            }
            for horizon in range(1, 11)
            for date, value in (("2024-01-02", 0.005), ("2024-01-03", -0.004))
        ]
    )
    daily_artifact = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "exit_date": "2024-01-03",
                "horizon": horizon,
                "score_basis": score_basis,
                "cost_basis": cost_basis,
                "n_selected": 4,
                "n_executed": 4,
                "portfolio_return": 0.001,
            }
            for horizon in range(1, 11)
            for score_basis in ("raw_common", "style_neutral")
            for cost_basis in ("gross", "net")
        ]
    )
    root_contract = {
        "severity": "高",
        "主导亏损": "否",
        "修复文件/接口": "module.py/interface",
        "回归测试": "test_contract",
        "预期指标变化": "消除错配",
        "不承诺效果": "不承诺转正",
    }
    finite_factor_diagnostics = pd.DataFrame(
        [
            {
                "sample": sample,
                "n_events": 1,
                "n_stocks": 1,
                "factor_percentile_median": 0.5,
                "factor_abs_robust_z_p95": 1.0,
                "factor_abs_robust_z_max": 1.0,
                "factor_robust_tail_rate": 0.0,
                "prior_event_jump_abs_median": 0.1,
                "prior_event_jump_abs_p95": 0.1,
            }
            for sample in ("D0除权", "D0非除权")
        ]
    )
    finite_return_errors = pd.DataFrame(
        [
            {
                "sample": sample,
                "n_events": 1,
                "n_stocks": 1,
                "return_delta_mean": 0.0,
                "return_delta_median": 0.0,
                "return_delta_abs_p95": 0.0,
                "return_delta_abs_max": 0.0,
                "hit_flip_count": 0,
                "adjustment_hit_flip_count": 0,
                "equal_factor_hit_flip_count": 0,
            }
            for sample in ("全样本", "D+2除权", "D+2非除权")
        ]
    )
    return {
        "metadata": {
            "command": "PYTHONPATH=. .venv/bin/python scripts/gp000_loss_attribution.py",
            "train_start": "2022-01-04",
            "train_end": "2024-09-04",
            "formal_factor": "gp_000",
            "formal_expression": "formal expression",
            "formal_sign": 1.0,
            "calendar_digest": "a" * 64,
            "input_sha256": "b" * 64,
            "library_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "price_cache_sha256": "e" * 64,
            "style_market_sha256": "f" * 64,
            "industries_sha256": "0" * 64,
            "effective_backtest": BacktestConfig().model_dump(),
        },
        "summary": "复权口径存在跨路径错配，但不是亏损核心原因。",
        "adjustment_matrix": pd.DataFrame(
            [{"节点": "数据源层", "口径": "原始价+点时复权因子", "未来函数": "未发现"}]
        ),
        "adjustment_stats": pd.DataFrame([{"指标": "收益错配数", "值": 1}]),
        "ex_right_samples": pd.DataFrame([{"trade_date": "2024-01-03", "n": 1}]),
        "ex_right_counts": pd.DataFrame(
            [
                {"stage": stage, "n_events": 1, "n_stocks": 1}
                for stage in ("D0", "D+1", "D+2")
            ]
        ),
        "ex_right_factor_diagnostics": finite_factor_diagnostics,
        "ex_right_return_errors": finite_return_errors,
        "ex_right_top4_summary": {
            "selected_trades": 4,
            "selected_any_ex_right": 1,
            "selected_holding_ex_right": 1,
            "hfq_minus_raw_book_return_sum": 0.0,
        },
        "ex_right_portfolio_comparison": pd.DataFrame(
            [
                {
                    "价格口径": price_basis,
                    "成本口径": cost_basis,
                    "CAGR": -0.1,
                    "夏普": -0.2,
                    "单笔收益": -0.001,
                    "累计净值": 0.8,
                    "最大回撤": -0.2,
                    "执行率": 1.0,
                    "交易日": 2,
                }
                for price_basis in ("raw", "HFQ")
                for cost_basis in ("毛收益", "净收益")
            ]
        ),
        "quintiles": pd.DataFrame(
            [
                {
                    "quintile": quintile,
                    "n": 2,
                    "gross_return": -0.001 * quintile,
                    "net_return": -0.002 * quintile,
                }
                for quintile in range(1, 6)
            ]
        ),
        "quintile_monotonicity": {
            "q5_minus_q1_gross": -0.004,
            "q5_minus_q1_net": -0.008,
            "gross_spearman": -1.0,
            "net_spearman": -1.0,
        },
        "cost_split": pd.DataFrame(
            [
                {
                    "口径": "毛收益",
                    "CAGR": -0.1,
                    "夏普": -0.2,
                    "单笔收益": -0.001,
                    "累计净值": 0.8,
                    "最大回撤": -0.2,
                    "执行率": 1.0,
                    "交易日": 2,
                }
            ]
        ),
        "decay": {
            "summary": pd.DataFrame(
                [
                    {
                        "horizon": horizon,
                        "n_d0": 1,
                        "d0_end": "2024-01-02",
                        "exit_end": "2024-01-03",
                        "ic_mean": -0.1,
                        "net_per_trade": -0.001,
                        "net_cagr": -0.1,
                        "net_sharpe": -0.2,
                        "net_final_equity": 0.8,
                    }
                    for horizon in range(1, 11)
                ]
            ),
            "daily": decay_daily,
        },
        "monthly": pd.DataFrame(
            [{"month": "2024-01", "gross_return": -0.008, "net_return": -0.01}]
        ),
        "daily": daily,
        "daily_artifact": daily_artifact,
        "style_table": pd.DataFrame(
            [
                {
                    "组合": "原始",
                    "D+2 IC": -0.1,
                    "ICIR": -0.2,
                    "CAGR": -0.1,
                    "夏普": -0.2,
                    "单笔净收益": -0.001,
                    "累计净值": 0.8,
                    "最大回撤": -0.2,
                    "交易日": 2,
                }
            ]
        ),
        "root_causes": [
            {
                "category": "因子 alpha",
                "priority": 1,
                "cause": "弱",
                "evidence": "IC<0",
                **root_contract,
            },
            {
                "category": "工程 bug",
                "priority": 1,
                "cause": "错配",
                "evidence": "raw/hfq",
                **root_contract,
            },
            {
                "category": "参数配置",
                "priority": 1,
                "cause": "目标",
                "evidence": "hit/close",
                **root_contract,
            },
        ],
        "repairs": [
            {"类别": "工程 bug", "修复路径": "统一 HFQ", "预期效果": "消除错配"}
        ],
    }


def test_root_causes_rank_engineering_before_config_before_alpha() -> None:
    causes = rank_root_causes(_minimal_evidence())

    assert [cause["category"] for cause in causes] == [
        "工程 bug",
        "参数配置",
        "因子 alpha",
    ]


def test_report_contains_every_required_section() -> None:
    report = render_report(_minimal_evidence())

    for heading in (
        "复权全链路审计",
        "除权日专项校验",
        "五分位单调性",
        "成本拆分",
        "收益衰减",
        "时间分布",
        "风格中性收益",
        "根因优先级",
        "修复路径与预期效果",
        "复现方式",
    ):
        assert heading in report


def test_svg_and_machine_outputs_are_well_formed(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    ET.fromstring(render_equity_svg(evidence["daily"]))
    ET.fromstring(render_decay_svg(evidence["decay"]))
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    write_outputs(evidence, paths)

    assert "NaN" not in paths.json.read_text(encoding="utf-8")
    json.loads(paths.json.read_text(encoding="utf-8"))
    ET.parse(paths.equity_svg)
    ET.parse(paths.decay_svg)
    assert pd.read_parquet(paths.daily).shape == evidence["daily_artifact"].shape


def test_invalid_evidence_writes_no_partial_outputs(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    evidence["daily"].loc[0, "net_portfolio_return"] = np.nan
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    with pytest.raises(ValueError, match="finite"):
        write_outputs(evidence, paths)

    assert not any(path.exists() for path in vars(paths).values())


@pytest.mark.parametrize(
    "missing_section",
    [
        "ex_right_counts",
        "ex_right_factor_diagnostics",
        "ex_right_return_errors",
        "ex_right_top4_summary",
        "ex_right_portfolio_comparison",
        "quintile_monotonicity",
    ],
)
def test_missing_core_audit_contract_refuses_publish(
    tmp_path: Path,
    missing_section: str,
) -> None:
    evidence = _minimal_evidence()
    del evidence[missing_section]
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    with pytest.raises(ValueError, match="required"):
        write_outputs(evidence, paths)


def test_daily_artifact_requires_four_matching_arms_per_horizon(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    artifact = evidence["daily_artifact"]
    evidence["daily_artifact"] = artifact.loc[
        ~(
            artifact["horizon"].eq(3)
            & artifact["score_basis"].eq("style_neutral")
            & artifact["cost_basis"].eq("net")
        )
    ]
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    with pytest.raises(ValueError, match="four arms"):
        write_outputs(evidence, paths)


def test_daily_artifact_arm_dates_must_match_decay_boundary(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    artifact = evidence["daily_artifact"]
    extra = artifact.loc[
        artifact["horizon"].eq(2)
        & artifact["score_basis"].eq("raw_common")
        & artifact["cost_basis"].eq("gross")
    ].copy()
    extra["date"] = "2024-01-03"
    evidence["daily_artifact"] = pd.concat([artifact, extra], ignore_index=True)
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    with pytest.raises(ValueError, match="arm dates"):
        write_outputs(evidence, paths)


def test_missing_input_hash_refuses_publish(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    del evidence["metadata"]["config_sha256"]
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )

    with pytest.raises(ValueError, match="input hashes"):
        write_outputs(evidence, paths)


def test_output_publish_failure_restores_previous_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )
    evidence = _minimal_evidence()
    write_outputs(evidence, paths)
    before = {path: path.read_bytes() for path in vars(paths).values()}
    original_replace = audit_module.os.replace
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(audit_module.os, "replace", fail_second_replace)
    evidence["summary"] = "new summary"

    with pytest.raises(OSError, match="simulated"):
        write_outputs(evidence, paths)

    assert {path: path.read_bytes() for path in vars(paths).values()} == before


def test_output_rollback_failure_retains_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = OutputPaths(
        report=tmp_path / "report.md",
        json=tmp_path / "evidence.json",
        daily=tmp_path / "daily.parquet",
        equity_svg=tmp_path / "equity.svg",
        decay_svg=tmp_path / "decay.svg",
    )
    evidence = _minimal_evidence()
    write_outputs(evidence, paths)
    original_replace = audit_module.os.replace
    calls = 0

    def fail_publish_and_restore(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("simulated replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(audit_module.os, "replace", fail_publish_and_restore)
    evidence["summary"] = "new summary"

    with pytest.raises(RuntimeError, match="backups retained"):
        write_outputs(evidence, paths)

    assert list(tmp_path.glob("*.bak"))


def test_training_calendar_requires_approved_bounds_count_and_digest() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    digest = hashlib.sha256("\n".join(dates).encode()).hexdigest()

    result = validate_training_calendar(
        dates,
        train_start=dates[0],
        train_end=dates[-1],
        expected_count=3,
        expected_digest=digest,
    )

    assert result.tolist() == dates
    with pytest.raises(ValueError, match="approved"):
        validate_training_calendar(
            dates[:-1],
            train_start=dates[0],
            train_end=dates[-1],
            expected_count=3,
            expected_digest=digest,
        )


def test_replay_formal_factor_uses_canonical_expression() -> None:
    library = FactorLibrary(
        factors=[
            FactorSpec(
                "gp_000",
                (
                    "add(add(stock_intra_amp_d1d3_mean, "
                    "div(stock_vwap_dev_d1, vol_burst_count_20d)), "
                    "stock_intra_amp_d0)"
                ),
                1.0,
            )
        ],
        field_names=[
            "stock_intra_amp_d1d3_mean",
            "stock_vwap_dev_d1",
            "vol_burst_count_20d",
            "stock_intra_amp_d0",
        ],
        windows=[],
        kind="event",
    )
    events = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "stock_code": ["A", "B"],
            "stock_intra_amp_d1d3_mean": [1.0, 2.0],
            "stock_vwap_dev_d1": [2.0, 3.0],
            "vol_burst_count_20d": [4.0, 3.0],
            "stock_intra_amp_d0": [0.1, 0.2],
        }
    )

    replayed = replay_formal_factor(events, library)

    assert replayed["factor_score"].tolist() == pytest.approx([1.6, 3.2])
