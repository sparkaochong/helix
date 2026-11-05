"""scripts/check_data_freshness.py: fast local gap check + optional --deep live re-verification."""

from __future__ import annotations

import pandas as pd

from helix.data import schema
from helix.data.store import ParquetStore
from scripts.check_data_freshness import (
    check_date_tables,
    check_static_tables_deep,
    expected_recent_open_days,
)

DATES = ["20240102", "20240103", "20240104", "20240105", "20240108"]


def _write_calendar(store: ParquetStore, open_dates: list[str]) -> None:
    cal = pd.DataFrame({"exchange": ["SSE"] * len(open_dates), "cal_date": open_dates, "is_open": [1] * len(open_dates)})
    store.write_static(schema.TRADE_CAL, cal)


def _write_table(store: ParquetStore, spec, dates: list[str], columns: list[str]) -> None:
    rows = [{**{"ts_code": "000001.SZ", "trade_date": d}, **{c: 1.0 for c in columns}} for d in dates]
    store.append_dated(spec, pd.DataFrame(rows))


def test_expected_recent_open_days_respects_lookback(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)

    assert expected_recent_open_days(store, "SSE", lookback=2) == ["20240105", "20240108"]
    assert expected_recent_open_days(store, "SSE", lookback=0) == DATES


def test_expected_recent_open_days_excludes_future_calendar_dates(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, [*DATES, "20990101"])  # far future, must never count as "expected by now"

    assert "20990101" not in expected_recent_open_days(store, "SSE", lookback=0)


def test_check_date_tables_reports_no_gaps_when_complete(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    for spec, cols in [
        (schema.DAILY, ["open", "high", "low", "close", "pre_close", "vol", "amount"]),
        (schema.ADJ_FACTOR, ["adj_factor"]),
        (schema.DAILY_BASIC, ["turnover_rate_f", "volume_ratio", "pe_ttm", "pb", "circ_mv"]),
        (schema.STK_LIMIT, ["up_limit", "down_limit"]),
    ]:
        _write_table(store, spec, DATES, cols)

    gaps = check_date_tables(store, "SSE", lookback=5)

    assert gaps == {}


def test_check_date_tables_reports_a_missing_date(tmp_path):
    store = ParquetStore(tmp_path)
    _write_calendar(store, DATES)
    _write_table(store, schema.DAILY, [d for d in DATES if d != "20240104"], ["open", "high", "low", "close", "pre_close", "vol", "amount"])
    _write_table(store, schema.ADJ_FACTOR, DATES, ["adj_factor"])
    _write_table(store, schema.DAILY_BASIC, DATES, ["turnover_rate_f", "volume_ratio", "pe_ttm", "pb", "circ_mv"])
    _write_table(store, schema.STK_LIMIT, DATES, ["up_limit", "down_limit"])

    gaps = check_date_tables(store, "SSE", lookback=5)

    assert gaps == {"daily": ["20240104"]}


class _FakeTushareSource:
    def __init__(self, cfg):
        self.store = ParquetStore(cfg.data.root)

    def _call_paginated(self, api, **kwargs):
        if api == "namechange":
            return pd.DataFrame({"ts_code": ["000001.SZ"] * 3, "name": ["a"] * 3, "start_date": ["20200101"] * 3})
        if api == "stock_basic":
            n = {"L": 2, "D": 1, "P": 0}[kwargs["list_status"]]
            return pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(n)]})
        raise AssertionError(f"unexpected api {api!r}")


def test_check_static_tables_deep_flags_a_row_count_mismatch(tmp_path, monkeypatch):
    from helix.config import Config, DataConfig

    store = ParquetStore(tmp_path)
    store.write_static(schema.NAMECHANGE, pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["a"], "start_date": ["20200101"], "end_date": [pd.NA]}))
    store.write_static(schema.STOCK_BASIC, pd.DataFrame({
        "ts_code": [f"{i:06d}.SZ" for i in range(3)], "name": ["a", "b", "c"],
        "list_date": ["20200101"] * 3, "delist_date": [pd.NA] * 3,
        "market": ["主板"] * 3, "exchange": ["SZSE"] * 3,
    }))
    monkeypatch.setattr("helix.data.tushare_source.TushareSource", _FakeTushareSource)

    findings = check_static_tables_deep(Config(data=DataConfig(root=tmp_path)))

    assert findings["namechange"] == {"local_rows": 1, "live_rows": 3}
    assert "stock_basic" not in findings  # 2+1+0 == 3, matches local -> not flagged


def test_check_static_tables_deep_reports_nothing_when_counts_match(tmp_path, monkeypatch):
    from helix.config import Config, DataConfig

    store = ParquetStore(tmp_path)
    store.write_static(
        schema.NAMECHANGE,
        pd.DataFrame({"ts_code": ["000001.SZ"] * 3, "name": ["a"] * 3, "start_date": ["20200101"] * 3, "end_date": [pd.NA] * 3}),
    )
    store.write_static(schema.STOCK_BASIC, pd.DataFrame({
        "ts_code": [f"{i:06d}.SZ" for i in range(3)], "name": ["a", "b", "c"],
        "list_date": ["20200101"] * 3, "delist_date": [pd.NA] * 3,
        "market": ["主板"] * 3, "exchange": ["SZSE"] * 3,
    }))
    monkeypatch.setattr("helix.data.tushare_source.TushareSource", _FakeTushareSource)

    findings = check_static_tables_deep(Config(data=DataConfig(root=tmp_path)))

    assert findings == {}
