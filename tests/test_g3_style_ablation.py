"""Contracts for the training-only G3 style ablation experiment."""

from __future__ import annotations

import hashlib
import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import scripts.g3_style_ablation as g3
from helix.config import BacktestConfig
from helix.gp.library import FactorLibrary, FactorSpec
from scripts.g3_style_ablation import (
    TRAIN_END,
    TRAIN_START,
    backtest_top_k,
    bootstrap_metric_summary,
    circular_block_bootstrap_indices,
    compute_trailing_styles,
    decide_go,
    load_placebo_icir_p95,
    render_report,
    split_evaluation_windows,
    split_training_outcomes,
    validate_formal_library,
    validate_seed_contract,
    validate_training_calendar,
)


def test_training_window_is_exact_and_oos_is_returned_separately():
    frame = pd.DataFrame(
        {
            "trade_date": ["2022-01-03", TRAIN_START, TRAIN_END, "2024-09-05"],
            "stock_code": ["A", "B", "C", "D"],
            "value": [0, 1, 2, 3],
        }
    )

    train, oos = split_evaluation_windows(frame)

    assert train["trade_date"].tolist() == [TRAIN_START, TRAIN_END]
    assert oos["trade_date"].tolist() == ["2024-09-05"]
    train.loc[:, "value"] = 99
    assert frame["value"].tolist() == [0, 1, 2, 3]


def test_training_calendar_requires_bounds_count_and_digest():
    dates = np.array(["2022-01-04", "2022-01-05", "2022-01-06"])
    digest = hashlib.sha256("\n".join(dates).encode()).hexdigest()
    validated = validate_training_calendar(
        dates,
        train_start=dates[0],
        train_end=dates[-1],
        expected_count=3,
        expected_digest=digest,
    )
    np.testing.assert_array_equal(validated, dates)

    with pytest.raises(ValueError, match="calendar"):
        validate_training_calendar(
            dates[:-1],
            train_start=dates[0],
            train_end=dates[-1],
            expected_count=3,
            expected_digest=digest,
        )


def test_training_decision_receives_no_oos_values():
    assert list(inspect.signature(decide_go).parameters) == [
        "neutral_icir",
        "placebo_icir_p95",
        "raw_net_return",
        "neutral_net_return",
    ]


def test_training_outcomes_use_d2_date_and_move_boundary_rows_to_appendix():
    calendar = np.array(
        [
            "2024-09-02",
            "2024-09-03",
            "2024-09-04",
            "2024-09-05",
            "2024-09-06",
        ]
    )
    frame = pd.DataFrame(
        {
            "trade_date": calendar[:3],
            "stock_code": ["A", "B", "C"],
        }
    )

    decision_train, boundary = split_training_outcomes(
        frame,
        calendar,
        train_end=TRAIN_END,
    )

    assert decision_train["trade_date"].tolist() == ["2024-09-02"]
    assert decision_train["label_d2_date"].tolist() == [TRAIN_END]
    assert boundary["trade_date"].tolist() == ["2024-09-03", TRAIN_END]
    assert boundary["label_d2_date"].tolist() == ["2024-09-05", "2024-09-06"]


def test_training_outcomes_reject_unresolvable_d2_dates():
    frame = pd.DataFrame(
        {"trade_date": [TRAIN_END], "stock_code": ["A"]}
    )

    with pytest.raises(ValueError, match=r"D\+2"):
        split_training_outcomes(frame, np.array([TRAIN_END]), train_end=TRAIN_END)


def test_run_experiment_routes_d2_boundary_rows_only_to_oos_appendix(
    monkeypatch, tmp_path
):
    calendar = np.array(
        [
            "2024-09-02",
            "2024-09-03",
            "2024-09-04",
            "2024-09-05",
            "2024-09-06",
            "2024-09-09",
            "2024-09-10",
        ]
    )
    frame = pd.DataFrame(
        {
            "trade_date": calendar[:5],
            "stock_code": ["A", "B", "C", "D", "E"],
            "feat": 1.0,
            "label_d2_hit_8pct_hfq": 0.0,
            "label_px_d1_open_hfq": 10.0,
            "label_px_d2_close_hfq": 11.0,
        }
    )
    library = FactorLibrary(
        factors=[FactorSpec("gp_000", "feat", 1.0)],
        field_names=["feat"],
        windows=[],
        kind="event",
    )
    market = pd.DataFrame({"trade_date": calendar, "stock_code": "A"})
    calls = []
    metrics = {"icir": 0.4, "net_return": 0.2, "net_per_trade": 0.01}

    monkeypatch.setattr(g3, "load_factors", lambda _: library)
    monkeypatch.setattr(g3.pd, "read_parquet", lambda *_, **__: frame.copy())
    monkeypatch.setattr(
        g3,
        "validate_training_calendar",
        lambda dates: np.unique(np.asarray(dates).astype(str)),
    )
    monkeypatch.setattr(
        g3,
        "_load_caches",
        lambda *_, **__: (market.copy(), pd.DataFrame(), calendar),
    )
    monkeypatch.setattr(g3, "compute_trailing_styles", lambda *_: pd.DataFrame())
    monkeypatch.setattr(g3, "_price_lookup", lambda *_: object())

    def fake_evaluate(evaluation_frame, *_, with_bootstrap, **__):
        calls.append(evaluation_frame.copy())
        return {
            "deterministic": {"raw": metrics.copy(), "style_neutral": metrics.copy()},
            "bootstrap": {},
            "orthogonality": {},
            "decay": [],
        }

    monkeypatch.setattr(g3, "_evaluate_window", fake_evaluate)
    monkeypatch.setattr(g3, "load_placebo_icir_p95", lambda _: 0.3)
    monkeypatch.setattr(g3, "render_report", lambda _: "report")
    monkeypatch.setattr(g3, "_sha256", lambda _: "sha256")
    monkeypatch.setattr(g3, "_atomic_text", lambda *_: None)

    g3.run_experiment(
        input_path=tmp_path / "input.parquet",
        library_path=tmp_path / "library.json",
        placebo_path=tmp_path / "placebo.parquet",
        market_cache=tmp_path / "market.parquet",
        industry_cache=tmp_path / "industry.parquet",
        result_path=tmp_path / "result.json",
        report_path=tmp_path / "report.md",
        seeds=(1, 2, 3),
        horizons=(1, 2),
    )

    assert calls[0]["trade_date"].tolist() == ["2024-09-02"]
    assert calls[0]["label_d2_date"].tolist() == [TRAIN_END]
    assert calls[1]["trade_date"].tolist() == [
        "2024-09-02",
        "2024-09-03",
        TRAIN_END,
    ]
    assert calls[2]["trade_date"].tolist() == [
        "2024-09-03",
        TRAIN_END,
        "2024-09-05",
        "2024-09-06",
    ]


def test_trailing_styles_use_d0_and_exactly_19_prior_sessions():
    calendar = pd.date_range("2024-01-02", periods=20, freq="B").strftime("%Y-%m-%d")
    market = pd.DataFrame(
        {
            "trade_date": calendar,
            "stock_code": "A",
            "pct_chg": np.arange(1.0, 21.0),
            "total_mv": 1000.0,
            "turnover_rate_f": np.arange(20.0),
        }
    )

    styles = compute_trailing_styles(market, np.asarray(calendar))

    assert styles.iloc[:19]["momentum_20d"].isna().all()
    last = styles.iloc[-1]
    assert last["momentum_20d"] == pytest.approx(
        float(np.prod(1.0 + np.arange(1.0, 21.0) / 100.0) - 1.0)
    )
    assert last["volatility_20d"] == pytest.approx(
        float(np.std(np.arange(1.0, 21.0) / 100.0, ddof=1))
    )
    assert last["turnover_mean_20d"] == pytest.approx(9.5)
    assert last["log_total_mv"] == pytest.approx(np.log(1000.0))


def test_trailing_styles_do_not_bridge_missing_stock_sessions():
    calendar = pd.date_range("2024-01-02", periods=20, freq="B").strftime("%Y-%m-%d")
    market = pd.DataFrame(
        {
            "trade_date": np.delete(calendar, 7),
            "stock_code": "A",
            "pct_chg": 1.0,
            "total_mv": 1000.0,
            "turnover_rate_f": 2.0,
        }
    )

    styles = compute_trailing_styles(market, np.asarray(calendar))

    last = styles.loc[styles["trade_date"] == calendar[-1]].iloc[0]
    assert np.isnan(last["momentum_20d"])
    assert np.isnan(last["volatility_20d"])
    assert np.isnan(last["turnover_mean_20d"])


@pytest.mark.parametrize("seeds", [[], [1], [1, 2], [1, 1, 2]])
def test_at_least_three_unique_seeds_are_required(seeds):
    with pytest.raises(ValueError, match="three unique"):
        validate_seed_contract(seeds)


def test_circular_block_bootstrap_is_reproducible_and_keeps_whole_dates():
    first = circular_block_bootstrap_indices(11, block_length=4, seed=13)
    second = circular_block_bootstrap_indices(11, block_length=4, seed=13)
    other = circular_block_bootstrap_indices(11, block_length=4, seed=42)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert len(first) == 11
    assert ((np.diff(first).reshape(-1)[:3] % 11) == 1).all()
    assert first.min() >= 0 and first.max() < 11


def test_bootstrap_summary_uses_sample_standard_deviation():
    def metric(index: np.ndarray) -> dict[str, float]:
        return {"score": float(index[0])}

    seeds = [7, 13, 42]
    summary = bootstrap_metric_summary(9, 3, seeds, metric)
    values = np.array(
        [circular_block_bootstrap_indices(9, 3, seed)[0] for seed in seeds],
        dtype=float,
    )

    assert summary["score"]["mean"] == pytest.approx(values.mean())
    assert summary["score"]["std"] == pytest.approx(values.std(ddof=1))
    assert summary["score"]["values"] == values.tolist()


@pytest.mark.parametrize(
    ("neutral_icir", "raw_return", "neutral_return", "expected"),
    [
        (0.4, 0.01, 0.001, "GO"),
        (-0.4, -0.01, -0.001, "GO"),
        (0.3, 0.01, 0.001, "NO-GO"),
        (0.4, 0.01, -0.001, "NO-GO"),
        (0.4, 0.0, 0.001, "NO-GO"),
    ],
)
def test_go_requires_strict_icir_p95_exceedance_and_same_nonzero_return_sign(
    neutral_icir, raw_return, neutral_return, expected
):
    assert decide_go(neutral_icir, 0.3, raw_return, neutral_return)["decision"] == expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_decision_inputs_are_no_go(value):
    for args in (
        (value, 0.3, 0.01, 0.01),
        (0.4, value, 0.01, 0.01),
        (0.4, 0.3, value, 0.01),
        (0.4, 0.3, 0.01, value),
    ):
        assert decide_go(*args)["decision"] == "NO-GO"


def test_formal_library_selects_the_named_event_factor():
    valid = FactorLibrary(
        factors=[FactorSpec("gp_000", "feat", 1.0)],
        field_names=["feat"],
        windows=[],
        kind="event",
    )
    assert validate_formal_library(valid, "gp_000").name == "gp_000"

    # A library with several factors is fine as long as exactly one matches the name.
    multi = FactorLibrary(
        factors=[FactorSpec("gp_000", "feat", 1.0), FactorSpec("gp_001", "feat", 1.0)],
        field_names=["feat"],
        windows=[],
        kind="event",
    )
    assert validate_formal_library(multi, "gp_001").name == "gp_001"

    with pytest.raises(ValueError, match="gp_002"):
        validate_formal_library(multi, "gp_002")
    with pytest.raises(ValueError, match="exactly one"):
        validate_formal_library(
            FactorLibrary(
                factors=[valid.factors[0], FactorSpec("gp_000", "feat", 1.0)],
                field_names=["feat"],
                windows=[],
                kind="event",
            ),
            "gp_000",
        )


def test_placebo_p95_is_recomputed_and_training_metadata_is_validated(tmp_path):
    path = tmp_path / "placebo.parquet"
    pd.DataFrame(
        {
            "icir": [0.1, 0.2, 0.3, 0.4],
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
        }
    ).to_parquet(path, index=False)

    assert load_placebo_icir_p95(path) == pytest.approx(0.385)

    bad = pd.read_parquet(path).assign(train_end="2024-09-05")
    bad.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="training window"):
        load_placebo_icir_p95(path)


def test_top_k_backtest_never_replaces_an_invalid_selected_name():
    score = np.array([[4.0, 3.0, 2.0, 1.0]])
    label = np.array([[1.0, 1.0, 0.0, 0.0]])
    entry = np.full_like(score, 10.0)
    exit_ = np.array([[11.0, np.nan, 9.0, 20.0]])
    mask = np.ones_like(score, dtype=bool)
    metrics, daily = backtest_top_k(
        score,
        label,
        entry,
        exit_,
        mask,
        np.array(["2024-01-02"]),
        BacktestConfig(top_k=2, slippage_bps=0.0),
        overlap=2,
    )

    assert daily.loc[0, "n_selected"] == 2
    assert daily.loc[0, "n_executed"] == 1
    assert metrics["top10_hit_rate"] == pytest.approx(1.0)
    assert metrics["net_per_trade"] > 0


def _report_payload() -> dict:
    metrics = {
        "ic_mean": 0.1,
        "icir": 0.4,
        "gini": 0.2,
        "top10_hit_rate": 0.3,
        "base_rate": 0.1,
        "lift": 3.0,
        "net_return": 0.2,
        "net_per_trade": 0.01,
        "cagr": 0.08,
        "sharpe": 1.0,
        "max_drawdown": -0.1,
        "n_days": 649.0,
        "coverage": 1.0,
    }
    return {
        "metadata": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "n_train_dates": 649,
            "n_decision_dates": 647,
            "decision_d0_end": "2024-09-02",
            "boundary_d0_dates": ["2024-09-03", TRAIN_END],
            "boundary_rows_moved_to_oos": 592,
            "seeds": [7, 13, 42],
            "command": "python scripts/g3_style_ablation.py --cache-only",
        },
        "decision": decide_go(0.4, 0.3, 0.01, 0.005),
        "placebo_icir_p95": 0.3,
        "deterministic": {"raw": metrics, "style_neutral": metrics | {"net_per_trade": 0.005}},
        "bootstrap": {
            "raw": {key: {"mean": value, "std": 0.01} for key, value in metrics.items()},
            "style_neutral": {
                key: {"mean": value, "std": 0.01} for key, value in metrics.items()
            },
        },
        "orthogonality": {"max_abs_exposure": 1e-12},
        "decay": [
            {"horizon": 1, "arm": "raw", "ic_mean": 0.1, "icir": 0.4, "net_per_trade": 0.01}
        ],
        "c1_boundary_comparison": {
            "before": {
                "neutral_icir": 0.41,
                "raw_net_return": 0.19,
                "raw_net_per_trade": 0.009,
                "neutral_net_return": 0.18,
                "neutral_net_per_trade": 0.004,
            },
            "after": {
                "neutral_icir": 0.4,
                "raw_net_return": 0.2,
                "raw_net_per_trade": 0.01,
                "neutral_net_return": 0.2,
                "neutral_net_per_trade": 0.005,
            },
            "delta": {
                "neutral_icir": -0.01,
                "raw_net_return": 0.01,
                "raw_net_per_trade": 0.001,
                "neutral_net_return": 0.02,
                "neutral_net_per_trade": 0.001,
            },
            "decision_before": "GO",
            "decision_after": "GO",
        },
        "oos": {"raw": metrics, "style_neutral": metrics},
    }


def test_report_has_training_tables_decision_decay_reproduction_and_oos_appendix():
    report = render_report(_report_payload())

    for text in (
        "# G3 Style Ablation",
        "GO",
        "2022-01-04",
        "2024-09-04",
        "确定性核心指标",
        "种子稳健性",
        "C1 边界修复前后对比",
        "2024-09-02",
        "647",
        "因子收益衰减",
        "样本外附录",
        "不参与 GO/NO-GO",
        "复现命令",
    ):
        assert text in report


def test_oos_numbers_cannot_change_rendered_decision():
    first = _report_payload()
    second = deepcopy(first)
    second["oos"]["style_neutral"]["icir"] = -999.0
    second["oos"]["style_neutral"]["net_per_trade"] = -999.0

    first_report = render_report(first)
    second_report = render_report(second)

    assert first_report.splitlines()[2] == second_report.splitlines()[2]
    assert "**GO**" in first_report.splitlines()[2]


def test_boundary_reproduction_is_contained_in_oos_appendix():
    report = render_report(_report_payload())

    assert report.index("## 样本外附录") < report.index("C1 边界修复前后对比")
