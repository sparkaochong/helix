from __future__ import annotations

import numpy as np
import pytest

from helix.data.panel import Panel
from helix.data.price_lineage import HFQ_BASIS, PriceLineage, PriceLineageError, make_hfq_lineage
from helix.features.base_fields import compute_base_fields

VERSION = "raw-times-same-day-adj-v1:" + "0" * 64


def _split_panel() -> Panel:
    dates = np.datetime_as_string(np.datetime64("2024-01-01") + np.arange(20), unit="D")
    codes = np.asarray(["000001.SZ"])
    hfq_ohlc = np.asarray(
        [
            [10.0, 12.0, 9.0, 11.0],
            [12.0, 13.0, 10.0, 12.0],
            [11.0, 14.0, 10.0, 13.0],
            [11.0, 15.0, 10.0, 12.0],
            [12.0, 14.0, 11.0, 13.0],
            [14.0, 15.0, 13.0, 14.0],
            *([[14.0, 15.0, 13.0, 14.0]] * 14),
        ]
    )
    factor = np.asarray([1.0, 1.0, 1.0, *([2.0] * 17)])
    raw_ohlc = hfq_ohlc / factor[:, None]
    raw_open, raw_high, raw_low, raw_close = (raw_ohlc[:, index : index + 1] for index in range(4))
    open_hfq, high_hfq, low_hfq, close_hfq = (
        hfq_ohlc[:, index : index + 1] for index in range(4)
    )
    pre_close = np.vstack([raw_close[:1], raw_close[:-1]])
    lineage = make_hfq_lineage(dates, VERSION)
    panel = Panel(dates=dates, codes=codes)
    for name, values in {
        "open": raw_open,
        "high": raw_high,
        "low": raw_low,
        "close": raw_close,
        "pre_close": pre_close,
        "amount": np.full((20, 1), 1_000_000.0),
        "up_limit": np.where(
            np.arange(20)[:, None] == 3, raw_close, raw_close + 0.5
        ),
    }.items():
        panel.add(name, values)
    for name, values in {
        "open_hfq": open_hfq,
        "high_hfq": high_hfq,
        "low_hfq": low_hfq,
        "close_hfq": close_hfq,
    }.items():
        panel.add(name, values, price_lineage=lineage)
    return panel


def test_base_price_shapes_use_governed_hfq_prices_across_a_split() -> None:
    fields = compute_base_fields(_split_panel())

    for name in ("ret1", "gap", "hl_range", "upper_shadow", "lower_shadow"):
        assert np.isnan(fields[name][0, 0])
    assert np.isfinite(fields["intraday"][0, 0])
    assert np.isfinite(fields["close_pos"][0, 0])
    assert np.isnan(fields["open_gap_mean5"][:5, 0]).all()
    assert np.flatnonzero(np.isfinite(fields["open_gap_mean5"][:, 0]))[0] == 5

    expected = {
        3: {
            "ret1": 12.0 / 13.0 - 1.0,
            "gap": 11.0 / 13.0 - 1.0,
            "intraday": 12.0 / 11.0 - 1.0,
            "hl_range": 5.0 / 13.0,
            "close_pos": 2.0 / 5.0,
            "upper_shadow": 3.0 / 13.0,
            "lower_shadow": 1.0 / 13.0,
        },
        4: {
            "ret1": 13.0 / 12.0 - 1.0,
            "gap": 0.0,
            "intraday": 13.0 / 12.0 - 1.0,
            "hl_range": 3.0 / 12.0,
            "close_pos": 2.0 / 3.0,
            "upper_shadow": 1.0 / 12.0,
            "lower_shadow": 1.0 / 12.0,
        },
    }
    for row, values in expected.items():
        for name, value in values.items():
            assert fields[name][row, 0] == pytest.approx(value)
    assert fields["open_gap_mean5"][5, 0] == pytest.approx(
        (1.0 / 11.0 - 1.0 / 12.0 - 1.0 / 13.0) / 5.0
    )


def test_base_fields_requires_hfq_lineage() -> None:
    panel = _split_panel()
    panel.price_lineage.pop("close_hfq")

    with pytest.raises(PriceLineageError, match=r"compute_base_fields: missing price lineage"):
        compute_base_fields(panel)


@pytest.mark.parametrize(
    ("basis", "version", "match"),
    [
        ("raw", VERSION, "price basis"),
        (HFQ_BASIS, "raw-times-same-day-adj-v1:" + "1" * 64, "inconsistent adjustment version"),
    ],
)
def test_base_fields_rejects_ungoverned_or_inconsistent_hfq_prices(
    basis: str, version: str, match: str
) -> None:
    panel = _split_panel()
    original = panel.price_lineage["close_hfq"]
    panel.price_lineage["close_hfq"] = PriceLineage(
        original.source_date, original.as_of_time, basis, version
    )

    with pytest.raises(PriceLineageError, match=match):
        compute_base_fields(panel)


def test_limit_state_fields_remain_raw_price_based_when_hfq_scale_changes() -> None:
    panel = _split_panel()
    baseline = compute_base_fields(panel)
    assert baseline["to_up_limit"][3, 0] == 0.0
    assert baseline["limitup_cnt20"][19, 0] == 1.0
    for name in ("open_hfq", "high_hfq", "low_hfq", "close_hfq"):
        panel.fields[name] *= 7.0
    scaled = compute_base_fields(panel)

    np.testing.assert_allclose(scaled["to_up_limit"], baseline["to_up_limit"], equal_nan=True)
    np.testing.assert_allclose(scaled["limitup_cnt20"], baseline["limitup_cnt20"], equal_nan=True)


def test_base_fields_do_not_access_raw_ohlc_or_pre_close() -> None:
    baseline = compute_base_fields(_split_panel())
    panel = _split_panel()
    for name in ("open", "high", "low", "pre_close"):
        panel.fields.pop(name)

    without_raw_ohlc = compute_base_fields(panel)

    for name, values in baseline.items():
        if name not in ("to_up_limit", "limitup_cnt20"):
            np.testing.assert_allclose(without_raw_ohlc[name], values, equal_nan=True)
    assert without_raw_ohlc["to_up_limit"][3, 0] == 0.0
    assert without_raw_ohlc["limitup_cnt20"][19, 0] == 1.0
