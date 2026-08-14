"""Panel caches must be invalidated when newly required provenance fields are absent."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from helix import pipeline
from helix.config import Config, DataConfig
from helix.data.panel import Panel
from helix.data.price_lineage import make_hfq_lineage


def test_prepare_rebuilds_a_legacy_panel_without_required_provenance(monkeypatch, tmp_path):
    shape = (2, 1)
    dates = np.asarray(["20240101", "20240102"])
    codes = np.asarray(["000001.SZ"])
    legacy = Panel(
        dates=dates,
        codes=codes,
        fields={
            "open": np.ones(shape),
            **{field: np.ones(shape) for field in pipeline.PANEL_CACHE_REQUIRED_FIELDS},
            **{field: np.ones(shape) for field in pipeline.PANEL_CACHE_REQUIRED_ADJUSTED_FIELDS},
        },
    )
    panel_path = tmp_path / "cache" / "panel.npz"
    legacy.save(panel_path)
    np.savez_compressed(panel_path.parent / "base_fields.npz", stale=np.zeros(shape))

    upgraded = Panel(
        dates=dates,
        codes=codes,
        fields={"open": np.ones(shape), "limit_price_observed": np.ones(shape)},
    )
    lineage = make_hfq_lineage(dates, "raw-times-same-day-adj-v1:abc")
    for field in ("open_hfq", "high_hfq", "low_hfq", "close_hfq"):
        upgraded.add(field, np.ones(shape), price_lineage=lineage)
    rebuilds: list[bool] = []
    base_field_builds: list[bool] = []

    def fake_build_panel(*args, **kwargs):
        rebuilds.append(True)
        return upgraded

    monkeypatch.setattr(pipeline, "build_panel", fake_build_panel)

    def fake_compute_base_fields(panel):
        base_field_builds.append(True)
        return {"fresh": np.ones(panel.shape)}

    monkeypatch.setattr(pipeline, "compute_base_fields", fake_compute_base_fields)
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
    assert base_field_builds == [True]
    assert set(prepared.fields) == {"fresh"}
    assert "limit_price_observed" in prepared.panel
    assert "limit_price_observed" in Panel.load(panel_path)
    assert Panel.load(panel_path).price_lineage["open_hfq"] == lineage
    with np.load(panel_path.parent / "base_fields.npz", allow_pickle=False) as cached_fields:
        assert set(cached_fields.files) == {"fresh"}
