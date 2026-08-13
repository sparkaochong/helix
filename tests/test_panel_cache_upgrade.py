"""Panel caches must be invalidated when newly required provenance fields are absent."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from helix import pipeline
from helix.config import Config, DataConfig
from helix.data.panel import Panel


def test_prepare_rebuilds_a_legacy_panel_without_limit_provenance(monkeypatch, tmp_path):
    shape = (2, 1)
    dates = np.asarray(["20240101", "20240102"])
    codes = np.asarray(["000001.SZ"])
    legacy = Panel(dates=dates, codes=codes, fields={"open": np.ones(shape)})
    panel_path = tmp_path / "cache" / "panel.npz"
    legacy.save(panel_path)

    upgraded = Panel(
        dates=dates,
        codes=codes,
        fields={
            "open": np.ones(shape),
            "limit_price_observed": np.ones(shape),
        },
    )
    rebuilds: list[bool] = []

    def fake_build_panel(*args, **kwargs):
        rebuilds.append(True)
        return upgraded

    monkeypatch.setattr(pipeline, "build_panel", fake_build_panel)
    monkeypatch.setattr(
        pipeline,
        "compute_base_fields",
        lambda panel: {"signal": np.ones(panel.shape)},
    )
    monkeypatch.setattr(
        pipeline,
        "build_universe",
        lambda panel, store, cfg: np.ones(panel.shape, dtype=bool),
    )
    monkeypatch.setattr(
        pipeline,
        "build_touch_label",
        lambda panel, universe, cfg: SimpleNamespace(),
    )

    prepared = pipeline.prepare(Config(data=DataConfig(root=tmp_path)))

    assert rebuilds == [True]
    assert "limit_price_observed" in prepared.panel
    assert "limit_price_observed" in Panel.load(panel_path)
