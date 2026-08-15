from __future__ import annotations

import json

import numpy as np
import pytest

from helix.config import Config
from helix.data.event_lineage import load_event_lineage, validate_event_fields
from helix.data.panel import Panel
from helix.data.price_lineage import make_hfq_lineage
from scripts.build_argus_hfq_event_table import (
    ADDITIONAL_HIT_RATIOS,
    _as_of,
    _hyphenate,
    _lead_dates,
    build_event_frame,
    build_lineage_manifest,
)

VERSION = "raw-times-same-day-adj-v1:" + "0" * 64
DATES = np.asarray(
    ["20240102", "20240103", "20240104", "20240105", "20240108", "20240109"]
)
CODES = np.asarray(["000001.SZ", "000002.SZ"])


def _fixture_panel() -> Panel:
    T, N = len(DATES), len(CODES)
    open_hfq = np.full((T, N), 10.0)
    open_hfq[:, 1] = 20.0
    high_hfq = open_hfq + 2.0
    close_hfq = open_hfq + 1.0
    low_hfq = open_hfq - 1.0
    # Stock 0 touches +8% by D+2 high on every D0; stock 1 never does.
    high_hfq[:, 0] = open_hfq[:, 0] * 1.09
    high_hfq[:, 1] = open_hfq[:, 1] * 1.02

    lineage = make_hfq_lineage(DATES, VERSION)
    panel = Panel(dates=DATES, codes=CODES)
    for name, values in {
        "open_hfq": open_hfq, "high_hfq": high_hfq,
        "low_hfq": low_hfq, "close_hfq": close_hfq,
    }.items():
        panel.add(name, values, price_lineage=lineage)
    panel.add("close", open_hfq)  # raw close; magnitude irrelevant to label math
    panel.add("amount", np.full((T, N), 50_000.0))
    panel.add("up_limit", open_hfq * 1.2)
    return panel


def test_lead_dates_shifts_like_the_lead_operator() -> None:
    shifted = _lead_dates(DATES, 2)
    assert shifted[0] == "20240104"
    assert shifted[-1] == ""
    assert shifted[-2] == ""


def test_hyphenate_and_as_of() -> None:
    assert _hyphenate("20240102") == "2024-01-02"
    assert _as_of("2024-01-02") == "2024-01-02T15:00:00+08:00"


def test_build_event_frame_computes_the_touch_label_from_d1_open_and_d2_high() -> None:
    panel = _fixture_panel()
    cfg = Config.load()
    universe_mask = np.ones(panel.shape, dtype=bool)

    frame = build_event_frame(panel, universe_mask, cfg)

    # Only D0 rows with a full D+2 window survive: 6 dates - 2 = 4, x2 stocks.
    assert len(frame) == 4 * 2
    assert set(frame["stock_code"]) == {"000001.SZ", "000002.SZ"}

    row0 = frame[(frame["trade_date"] == "2024-01-02") & (frame["stock_code"] == "000001.SZ")].iloc[0]
    assert row0["label_px_d1_open_hfq"] == pytest.approx(10.0)
    assert row0["label_px_d2_high_hfq"] == pytest.approx(10.0 * 1.09)
    assert row0["label_d2_peak_return_hfq"] == pytest.approx(0.09)
    assert row0["label_d2_hit_8pct_hfq"] == 1.0
    assert row0["label_d2_hit_5pct_hfq"] == 1.0
    assert row0["label_d2_hit_3pct_hfq"] == 1.0

    row1 = frame[(frame["trade_date"] == "2024-01-02") & (frame["stock_code"] == "000002.SZ")].iloc[0]
    assert row1["label_d2_hit_8pct_hfq"] == 0.0
    assert row1["label_d2_hit_5pct_hfq"] == 0.0
    assert row1["label_d2_hit_3pct_hfq"] == 0.0

    # D+1/D+2 audit dates track the true trading-session offsets, not calendar days.
    assert row0["audit_d1_source_date"] == "2024-01-03"
    assert row0["audit_d2_source_date"] == "2024-01-04"
    assert row0["audit_f_source_date"] == "2024-01-02"
    assert row0["audit_f_price_basis"] == "hfq"
    assert row0["audit_f_adj_factor_version"] == VERSION


def test_build_event_frame_excludes_cells_without_a_complete_d2_window() -> None:
    panel = _fixture_panel()
    cfg = Config.load()
    universe_mask = np.ones(panel.shape, dtype=bool)

    frame = build_event_frame(panel, universe_mask, cfg)

    assert "2024-01-08" not in set(frame["trade_date"])
    assert "2024-01-09" not in set(frame["trade_date"])


def test_lineage_manifest_validates_against_the_built_frame(tmp_path) -> None:
    panel = _fixture_panel()
    cfg = Config.load()
    universe_mask = np.ones(panel.shape, dtype=bool)
    frame = build_event_frame(panel, universe_mask, cfg)

    feature_names = [
        name for name in frame.columns
        if not name.startswith(("label_", "audit_", "trade_date", "stock_code"))
    ]
    manifest = build_lineage_manifest(feature_names)

    # Round-trip through JSON, exactly as mine_argus.py's --lineage flag consumes it.
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_event_lineage(manifest_path)

    outcome_columns = [
        "label_px_d1_open_hfq", "label_px_d2_high_hfq", "label_px_d2_close_hfq",
        "label_d2_peak_return_hfq", "label_d2_return_hfq", "label_d2_hit_8pct_hfq",
        *ADDITIONAL_HIT_RATIOS,
    ]
    # Must not raise: every governed feature + outcome column carries a valid audit trail.
    validate_event_fields(
        frame,
        loaded,
        feature_names,
        outcome_fields=outcome_columns,
        calendar=[str(d) for d in DATES],
    )
