"""build_panel must fail closed when the store has fewer trading days than declared,
instead of silently starting later than requested (the exact shape of the bug found
in the 2026-08-15 data baseline audit: config declared 20180101, data started 20211201)."""

from __future__ import annotations

import pandas as pd
import pytest

from helix.data import schema
from helix.data.panel import PanelCoverageError, _validate_panel_coverage, build_panel
from helix.data.store import ParquetStore

DATES = ["20240102", "20240103", "20240104", "20240105", "20240108"]


def _write_calendar(store: ParquetStore, open_dates: list[str]) -> None:
    cal = pd.DataFrame({"exchange": ["SSE"] * len(open_dates), "cal_date": open_dates, "is_open": [1] * len(open_dates)})
    store.write_static(schema.TRADE_CAL, cal)


def _write_price_tables(store: ParquetStore, dates: list[str], codes: list[str]) -> None:
    rows = [{"ts_code": c, "trade_date": d, "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "pre_close": 10.0, "vol": 100.0, "amount": 1000.0} for d in dates for c in codes]
    store.append_dated(schema.DAILY, pd.DataFrame(rows))
    adj_rows = [{"ts_code": c, "trade_date": d, "adj_factor": 1.0} for d in dates for c in codes]
    store.append_dated(schema.ADJ_FACTOR, pd.DataFrame(adj_rows))


def test_validate_panel_coverage_passes_when_dates_match_calendar_exactly(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    dates = pd.array(DATES).astype(str).to_numpy()

    _validate_panel_coverage(dates, store, "20240102", "20240108", "SSE")  # must not raise


def test_validate_panel_coverage_raises_on_a_middle_gap(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    dates_with_gap = pd.array([d for d in DATES if d != "20240104"]).astype(str).to_numpy()

    with pytest.raises(PanelCoverageError, match="20240104"):
        _validate_panel_coverage(dates_with_gap, store, "20240102", "20240108", "SSE")


def test_validate_panel_coverage_raises_on_front_truncation_like_the_real_bug(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    dates_missing_front = pd.array(DATES[2:]).astype(str).to_numpy()

    with pytest.raises(PanelCoverageError, match="20240102"):
        _validate_panel_coverage(dates_missing_front, store, "20240102", "20240108", "SSE")


def test_validate_panel_coverage_skips_when_trade_cal_is_empty(tmp_path):
    store = ParquetStore(tmp_path)  # no trade_cal written at all
    dates_with_gap = pd.array([d for d in DATES if d != "20240104"]).astype(str).to_numpy()

    _validate_panel_coverage(dates_with_gap, store, "20240102", "20240108", "SSE")  # must not raise


def test_validate_panel_coverage_with_empty_start_end_checks_only_internal_gaps(tmp_path):
    """No explicit start/end means 'whatever the store has', not 'since the dawn of the exchange'."""
    store = ParquetStore(tmp_path)
    _write_calendar(store, ["20060101", *DATES])  # calendar goes back decades further than the data
    dates = pd.array(DATES).astype(str).to_numpy()

    _validate_panel_coverage(dates, store, "", "", "SSE")  # must not demand coverage back to 2006


def test_build_panel_raises_panel_coverage_error_end_to_end(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    _write_price_tables(store, [d for d in DATES if d != "20240104"], ["000001.SZ"])

    with pytest.raises(PanelCoverageError, match="20240104"):
        build_panel(store, "20240102", "20240108")


def test_build_panel_succeeds_with_complete_coverage(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    _write_price_tables(store, DATES, ["000001.SZ"])

    panel = build_panel(store, "20240102", "20240108")

    assert list(panel.dates) == DATES
