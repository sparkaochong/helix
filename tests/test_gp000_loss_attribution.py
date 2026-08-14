from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helix.config import BacktestConfig
from helix.gp.library import FactorLibrary, FactorSpec
from scripts.gp000_loss_attribution import (
    OutputPaths,
    audit_adjustment_chain,
    build_price_lookup,
    evaluate_horizon_decay,
    evaluate_monthly_returns,
    evaluate_quintiles,
    evaluate_style_neutral_book,
    evaluate_top_k_book,
    outcome_complete_dates,
    rank_root_causes,
    render_decay_svg,
    render_equity_svg,
    render_report,
    validate_formal_factor,
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
        {
            "date": ["2024-01-02", "2024-01-03"] * 2,
            "horizon": [1, 1, 2, 2],
            "net_portfolio_return": [0.01, -0.01, 0.005, -0.005],
        }
    )
    return {
        "metadata": {
            "command": "PYTHONPATH=. .venv/bin/python scripts/gp000_loss_attribution.py",
            "train_start": "2022-01-04",
            "train_end": "2024-09-04",
        },
        "summary": "复权口径存在跨路径错配，但不是亏损核心原因。",
        "adjustment_matrix": pd.DataFrame(
            [{"节点": "数据源层", "口径": "原始价+点时复权因子", "未来函数": "未发现"}]
        ),
        "adjustment_stats": pd.DataFrame([{"指标": "收益错配数", "值": 1}]),
        "ex_right_samples": pd.DataFrame([{"trade_date": "2024-01-03", "n": 1}]),
        "quintiles": pd.DataFrame([{"quintile": 1, "n": 2}]),
        "cost_split": pd.DataFrame([{"口径": "毛收益", "CAGR": -0.1}]),
        "decay": {
            "summary": pd.DataFrame([{"horizon": 1, "ic_mean": -0.1}]),
            "daily": decay_daily,
        },
        "monthly": pd.DataFrame([{"month": "2024-01", "net_return": -0.01}]),
        "daily": daily,
        "style_table": pd.DataFrame([{"组合": "原始", "CAGR": -0.1}]),
        "root_causes": [
            {"category": "因子 alpha", "priority": 1, "cause": "弱", "evidence": "IC<0"},
            {"category": "工程 bug", "priority": 1, "cause": "错配", "evidence": "raw/hfq"},
            {"category": "参数配置", "priority": 1, "cause": "目标", "evidence": "hit/close"},
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
    assert pd.read_parquet(paths.daily).shape == evidence["daily"].shape
