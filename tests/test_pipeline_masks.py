"""Point-in-time mask wiring at pipeline boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from helix import pipeline
from helix.config import Config, DataConfig, GPConfig, SplitConfig
from helix.data.panel import Panel
from helix.eval.backtest import BacktestResult
from helix.labels.touch_label import LabelSet


def make_prepared() -> pipeline.Prepared:
    shape = (8, 3)
    panel = Panel(
        dates=np.array([f"2024010{i}" for i in range(1, 9)]),
        codes=np.array(["000001.SZ", "000002.SZ", "000003.SZ"]),
        fields={"amount": np.arange(24.0).reshape(shape)},
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
        split=SplitConfig(train_days=3, valid_days=1, test_days=1, step_days=1),
        gp=GPConfig(search_max_stocks=3),
    )
    captured: dict[str, np.ndarray] = {}

    def fake_liquidity_top_columns(amount, mask, max_stocks):
        captured["mask"] = mask.copy()
        return np.arange(amount.shape[1])

    monkeypatch.setattr(pipeline, "liquidity_top_columns", fake_liquidity_top_columns)
    monkeypatch.setattr(
        pipeline,
        "run_search",
        lambda **kwargs: SimpleNamespace(library=SimpleNamespace()),
    )
    monkeypatch.setattr(pipeline, "save_factors", lambda *args: None)

    pipeline.mine(cfg, prepared)

    np.testing.assert_array_equal(captured["mask"], prepared.universe[:3])


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
