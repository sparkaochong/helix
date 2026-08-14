"""Point-in-time mask wiring at pipeline boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from helix import pipeline
from helix.config import Config, DataConfig, GPConfig, SplitConfig
from helix.data.panel import Panel
from helix.eval.backtest import BacktestResult
from helix.gp.library import FactorLibrary, FactorSpec
from helix.labels.touch_label import LabelSet


def make_prepared() -> pipeline.Prepared:
    shape = (8, 3)
    panel = Panel(
        dates=np.array([f"2024010{i}" for i in range(1, 9)]),
        codes=np.array(["000001.SZ", "000002.SZ", "000003.SZ"]),
        fields={
            "amount": np.arange(24.0).reshape(shape),
            "high_hfq": np.ones(shape),
        },
    )
    universe = np.array(
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    valid = ~universe
    values = np.ones(shape)
    labels = LabelSet(
        y=np.where(valid, 0.0, np.nan),
        valid=valid,
        touch_tradable=valid.copy(),
        entry_price=np.where(valid, values, np.nan),
        target_price=np.where(valid, values * 1.08, np.nan),
        exit_price=np.where(valid, values, np.nan),
    )
    return pipeline.Prepared(
        panel=panel,
        universe=universe,
        fields={"signal": values},
        names=["signal"],
        labels=labels,
    )


def test_mine_uses_the_d0_universe_for_liquidity_sampling(monkeypatch, tmp_path):
    prepared = make_prepared()
    cfg = Config(
        data=DataConfig(root=tmp_path),
        split=SplitConfig(train_days=5, valid_days=1, test_days=1, step_days=1),
        gp=GPConfig(search_max_stocks=3),
    )
    captured: dict[str, object] = {}

    def fake_liquidity_top_columns(amount, mask, max_stocks):
        captured["mask"] = mask.copy()
        return np.arange(amount.shape[1])

    monkeypatch.setattr(pipeline, "liquidity_top_columns", fake_liquidity_top_columns)
    def fake_run_search(**kwargs):
        captured["search"] = kwargs
        return SimpleNamespace(library=SimpleNamespace())

    monkeypatch.setattr(pipeline, "run_search", fake_run_search)
    monkeypatch.setattr(pipeline, "save_factors", lambda *args: None)

    pipeline.mine(cfg, prepared)

    np.testing.assert_array_equal(captured["mask"], prepared.universe[:3])
    search = captured["search"]
    np.testing.assert_array_equal(search["candidate_mask"], prepared.universe[:3])
    np.testing.assert_array_equal(search["dates"], prepared.panel.dates[:3])
    expected_gross = np.where(
        prepared.labels.valid[:3],
        prepared.labels.exit_price[:3] / prepared.labels.entry_price[:3] - 1.0,
        np.nan,
    )
    np.testing.assert_allclose(search["gross_returns"], expected_gross, equal_nan=True)
    assert search["backtest_cfg"].top_k == 4


def test_backtest_pipeline_passes_d0_masks_to_ranking(monkeypatch, tmp_path):
    prepared = make_prepared()
    cfg = Config(data=DataConfig(root=tmp_path))
    predictions = np.arange(24.0).reshape(8, 3)
    predictions[0, 0] = np.nan
    captured: dict[str, object] = {}

    def fake_run_backtest(**kwargs):
        captured["candidate_mask"] = kwargs.get("candidate_mask")
        return BacktestResult(daily=pd.DataFrame(), summary={"n_days": 0.0})

    lift_masks: list[np.ndarray] = []

    def fake_lift_at_k(score, y, mask, k):
        lift_masks.append(mask.copy())
        return 1.0

    monkeypatch.setattr(pipeline, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(pipeline, "lift_at_k", fake_lift_at_k)

    pipeline.backtest(cfg, prepared, predictions)

    np.testing.assert_array_equal(captured["candidate_mask"], prepared.universe)
    expected_lift_mask = prepared.universe & np.isfinite(predictions)
    assert len(lift_masks) == 4
    for mask in lift_masks:
        np.testing.assert_array_equal(mask, expected_lift_mask)


def test_train_separates_label_observability_from_d0_scoring(monkeypatch, tmp_path):
    prepared = make_prepared()
    cfg = Config(
        data=DataConfig(root=tmp_path),
        split=SplitConfig(train_days=3, valid_days=1, test_days=1, step_days=1),
    )
    normalized = np.ones((8, 3, 2), dtype=np.float32)
    normalized[4, 1] = np.nan
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(
        pipeline,
        "_normalized_factors",
        lambda cfg, prepared, library: (
            ["a", "b"],
            normalized,
            np.ones(prepared.panel.shape, dtype=np.float32),
        ),
    )

    def fake_train_walk_forward(**kwargs):
        captured["label_mask"] = kwargs.get("label_mask")
        captured["prediction_mask"] = kwargs.get("prediction_mask")
        return np.zeros(prepared.panel.shape), []

    monkeypatch.setattr(pipeline, "train_walk_forward", fake_train_walk_forward)
    monkeypatch.setattr(pipeline, "walk_forward", lambda *args: [])

    pipeline.train(cfg, prepared, SimpleNamespace())

    np.testing.assert_array_equal(captured["label_mask"], prepared.labels.valid)
    expected = prepared.universe & (np.isfinite(normalized).mean(axis=-1) >= 0.5)
    np.testing.assert_array_equal(captured["prediction_mask"], expected)


def test_factor_report_cannot_see_rows_after_the_training_boundary(monkeypatch, tmp_path):
    prepared = make_prepared()
    cfg = Config(
        data=DataConfig(root=tmp_path),
        split=SplitConfig(train_days=5, valid_days=1, test_days=1, step_days=1),
    )
    values = np.ones((*prepared.panel.shape, 1), dtype=np.float32)
    values[3:, :, 0] = 999.0
    captured: dict[str, object] = {}
    empty_block = {
        "production_objective": {"net": {"mean": 0.0}},
    }

    monkeypatch.setattr(
        pipeline,
        "compute_factors",
        lambda *args: (["gp_000"], values),
    )

    def fake_monitors(**kwargs):
        captured.update(kwargs)
        return {
            "fit": empty_block,
            "selection": empty_block,
            "training_full": empty_block,
        }

    monkeypatch.setattr(pipeline, "evaluate_training_monitors", fake_monitors)
    library = FactorLibrary(
        factors=[FactorSpec("gp_000", "signal", 1.0)],
        field_names=["signal"],
        windows=[],
    )

    report = pipeline.evaluate_factors(cfg, prepared, library)

    assert captured["score"].shape == (3, 3)
    assert not np.any(captured["score"] == 999.0)
    np.testing.assert_array_equal(captured["dates"], prepared.panel.dates[:3])
    assert "out_of_sample" not in report["gp_000"]
