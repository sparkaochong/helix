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
    raw_close = np.asarray([10.0, 10.0, 10.0, *([5.0] * 17)])[:, None]
    raw_open = np.asarray([10.0, 11.0, 9.0, 5.0, 6.0, 4.0, *([5.0] * 14)])[:, None]
    raw_high = np.asarray([11.0, 12.0, 10.0, 8.0, 7.0, 6.0, *([6.0] * 14)])[:, None]
    raw_low = np.asarray([9.0, 8.0, 8.0, 3.0, 4.0, 3.0, *([4.0] * 14)])[:, None]
    pre_close = np.vstack([raw_close[:1], raw_close[:-1]])
    close_hfq = np.full((20, 1), 10.0)
    open_hfq = np.asarray([10.0, 11.0, 9.0, 10.0, 12.0, 8.0, *([10.0] * 14)])[:, None]
    high_hfq = np.asarray([11.0, 12.0, 10.0, 14.0, 13.0, 12.0, *([11.0] * 14)])[:, None]
    low_hfq = np.asarray([9.0, 8.0, 8.0, 8.0, 9.0, 7.0, *([9.0] * 14)])[:, None]
    lineage = make_hfq_lineage(dates, VERSION)
    panel = Panel(dates=dates, codes=codes)
    for name, values in {
        "open": raw_open,
        "high": raw_high,
        "low": raw_low,
        "close": raw_close,
        "pre_close": pre_close,
        "amount": np.full((20, 1), 1_000_000.0),
        "up_limit": np.asarray([11.0, 11.0, 11.0, 5.0, *([5.5] * 16)])[:, None],
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

    split_day = 3
    assert fields["ret1"][split_day, 0] == 0.0
    assert fields["gap"][split_day, 0] == 0.0
    assert fields["intraday"][split_day, 0] == 0.0
    assert fields["hl_range"][split_day, 0] == pytest.approx(0.6)
    assert fields["close_pos"][split_day, 0] == pytest.approx(1 / 3)
    assert fields["upper_shadow"][split_day, 0] == pytest.approx(0.4)
    assert fields["lower_shadow"][split_day, 0] == pytest.approx(0.2)
    assert fields["open_gap_mean5"][5, 0] == pytest.approx(0.0)


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
