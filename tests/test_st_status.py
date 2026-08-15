"""Point-in-time ST detection, and the ST 5% override in panel.py's rule-based
limit-price fallback (used only where stk_limit itself has gaps).

The 5% band is scoped to main-board risk-warning names only -- real-data
validation against the local store found STAR/ChiNext/BSE ST names keep their
board's normal rate, and delisting-consolidation ("退") names follow neither."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.data import schema
from helix.data.panel import Panel, _fallback_limit_prices, _limit_pct
from helix.data.st_status import (
    looks_st,
    point_in_time_risk_warning_mask,
    point_in_time_st_mask,
)
from helix.data.store import ParquetStore

DATES = np.array(["20240102", "20240103", "20240104", "20240105", "20240108"])


@pytest.mark.parametrize(
    "name,expected",
    [("*ST海润", True), ("ST生态", True), ("XX退", True), ("平安银行", False), ("退市博元", True)],
)
def test_looks_st(name, expected):
    assert looks_st(name) is expected


def _write_namechange(store: ParquetStore, rows: list[dict]) -> None:
    store.write_static(schema.NAMECHANGE, pd.DataFrame(rows))


def test_point_in_time_st_mask_only_flags_the_st_named_window(tmp_path):
    store = ParquetStore(tmp_path)
    codes = np.array(["000001.SZ"])
    _write_namechange(
        store,
        [
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20200101", "end_date": "20240103"},
            {"ts_code": "000001.SZ", "name": "*ST平安", "start_date": "20240104", "end_date": "20240105"},
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20240108", "end_date": pd.NaT},
        ],
    )

    mask = point_in_time_st_mask(DATES, codes, store)

    # end_date is inclusive: the *ST row covers 20240104 and 20240105 both.
    assert mask[:, 0].tolist() == [False, False, True, True, False]


def test_risk_warning_mask_excludes_delisting_names(tmp_path):
    store = ParquetStore(tmp_path)
    codes = np.array(["000001.SZ", "000002.SZ"])
    _write_namechange(
        store,
        [
            {"ts_code": "000001.SZ", "name": "*ST平安", "start_date": "20240102", "end_date": pd.NaT},
            {"ts_code": "000002.SZ", "name": "退市博元", "start_date": "20240102", "end_date": pd.NaT},
        ],
    )

    st_mask = point_in_time_st_mask(DATES, codes, store)
    risk_mask = point_in_time_risk_warning_mask(DATES, codes, store)

    assert st_mask[:, 0].all() and st_mask[:, 1].all()  # both count as "not normal" for universe exclusion
    assert risk_mask[:, 0].all()  # *ST is a risk-warning name
    assert not risk_mask[:, 1].any()  # 退 is delisting consolidation, not the 5% ST band


def test_limit_pct_applies_5pct_only_to_main_board_risk_warning_names(tmp_path):
    store = ParquetStore(tmp_path)
    codes = np.array(["000001.SZ", "688001.SH", "300001.SZ", "000002.SZ"])
    _write_namechange(
        store,
        [
            # main board: ST -> 5%
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20200101", "end_date": "20240103"},
            {"ts_code": "000001.SZ", "name": "*ST平安", "start_date": "20240104", "end_date": "20240105"},
            {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20240108", "end_date": pd.NaT},
            # STAR board: ST-named but must stay at the board's 20%, not drop to 5%
            {"ts_code": "688001.SH", "name": "*ST某科创", "start_date": "20240102", "end_date": pd.NaT},
            # ChiNext: same
            {"ts_code": "300001.SZ", "name": "ST某创业", "start_date": "20240102", "end_date": pd.NaT},
            # main board: delisting consolidation -> stays at the board default (10%), not 5%
            {"ts_code": "000002.SZ", "name": "退市某某", "start_date": "20240102", "end_date": pd.NaT},
        ],
    )

    pct = _limit_pct(DATES, codes, store)

    assert pct[:, 0].tolist() == [0.10, 0.10, 0.05, 0.05, 0.10]  # main-board ST window
    assert (pct[:, 1] == 0.20).all()  # STAR board unaffected by its own ST status
    assert (pct[:, 2] == 0.20).all()  # ChiNext board unaffected by its own ST status
    assert (pct[:, 3] == 0.10).all()  # delisting name is not the 5% ST rule


def test_fallback_limit_prices_reflects_st_5pct(tmp_path):
    store = ParquetStore(tmp_path)
    codes = np.array(["000001.SZ"])
    _write_namechange(
        store,
        [{"ts_code": "000001.SZ", "name": "*ST平安", "start_date": "20240102", "end_date": pd.NaT}],
    )
    pre_close = np.full((len(DATES), 1), 10.0)
    panel = Panel(dates=DATES, codes=codes, fields={"pre_close": pre_close})

    up, down = _fallback_limit_prices(panel, store)

    assert np.allclose(up[:, 0], 10.5)  # 10 * 1.05
    assert np.allclose(down[:, 0], 9.5)  # 10 * 0.95
