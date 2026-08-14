from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.data.panel import PRICE_COLUMNS, Panel, build_adjusted_price_fields
from helix.data.price_lineage import (
    HFQ_BASIS,
    PriceLineage,
    PriceLineageError,
    adjustment_factor_version,
    make_hfq_lineage,
)

VERSION = "raw-times-same-day-adj-v1:" + "0" * 64


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240101", "20240102"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            **{column: [10.0, 5.0] for column in PRICE_COLUMNS},
        }
    )


def _adj(values: list[float], dates: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates or ["20240101", "20240102"],
            "ts_code": ["000001.SZ"] * len(values),
            "adj_factor": values,
        }
    )


def test_build_adjusted_prices_uses_same_trade_date_factor_and_lineage():
    dates = np.asarray(["20240101", "20240102"])
    fields, lineage = build_adjusted_price_fields(
        _daily(), _adj([1.0, 2.0]), dates, np.asarray(["000001.SZ"])
    )

    np.testing.assert_allclose(fields["open_hfq"], [[10.0], [10.0]])
    assert lineage["open_hfq"].source_date.tolist() == ["20240101", "20240102"]
    assert lineage["open_hfq"].as_of_time.tolist() == [
        "2024-01-01T15:00:00+08:00",
        "2024-01-02T15:00:00+08:00",
    ]
    assert lineage["open_hfq"].price_basis == HFQ_BASIS


def test_build_adjusted_prices_rejects_shifted_factor_date():
    with pytest.raises(PriceLineageError, match=r"20240101.*000001\.SZ.*same-date"):
        build_adjusted_price_fields(
            _daily(),
            _adj([1.0], ["20240102"]),
            np.asarray(["20240101", "20240102"]),
            np.asarray(["000001.SZ"]),
        )


@pytest.mark.parametrize("factor", [np.nan, 0.0, -1.0])
def test_build_adjusted_prices_rejects_missing_or_nonpositive_factor(factor):
    with pytest.raises(PriceLineageError, match=r"20240102.*000001\.SZ.*same-date"):
        build_adjusted_price_fields(
            _daily(),
            _adj([1.0, factor]),
            np.asarray(["20240101", "20240102"]),
            np.asarray(["000001.SZ"]),
        )


def test_build_adjusted_prices_rejects_conflicting_duplicate_factors():
    adj = pd.concat([_adj([1.0, 2.0]), _adj([3.0], ["20240101"])], ignore_index=True)

    with pytest.raises(PriceLineageError, match=r"20240101.*000001\.SZ.*conflicting"):
        build_adjusted_price_fields(
            _daily(), adj, np.asarray(["20240101", "20240102"]), np.asarray(["000001.SZ"])
        )


def test_build_adjusted_prices_accepts_identical_duplicate_factors():
    adj = pd.concat([_adj([1.0, 2.0]), _adj([1.0], ["20240101"])], ignore_index=True)

    fields, _ = build_adjusted_price_fields(
        _daily(), adj, np.asarray(["20240101", "20240102"]), np.asarray(["000001.SZ"])
    )

    np.testing.assert_allclose(fields["close_hfq"], [[10.0], [10.0]])


def test_adjustment_factor_version_is_stable_across_row_order():
    adj = _adj([1.0, 2.0])

    assert adjustment_factor_version(adj) == adjustment_factor_version(adj.iloc[::-1])


def test_adjustment_version_excludes_factor_rows_outside_panel_dates_and_codes():
    dates = np.asarray(["20240101", "20240102"])
    codes = np.asarray(["000001.SZ"])
    inside = _adj([1.0, 2.0])
    outside = pd.DataFrame(
        {
            "trade_date": ["20240103", "20240101"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "adj_factor": [99.0, 88.0],
        }
    )

    _, expected = build_adjusted_price_fields(_daily(), inside, dates, codes)
    _, with_outside = build_adjusted_price_fields(
        _daily(), pd.concat([inside, outside], ignore_index=True).iloc[::-1], dates, codes
    )

    assert expected["open_hfq"].adj_factor_version == with_outside["open_hfq"].adj_factor_version


@pytest.mark.parametrize(
    ("source_date", "as_of_time"),
    [
        ("20240101", "2024-01-01T15:00:00+08:00"),
        ("2024-01-01", "2024-01-01T14:59:59+08:00"),
    ],
)
def test_price_lineage_accepts_valid_compact_and_hyphenated_source_dates(source_date, as_of_time):
    lineage = PriceLineage(np.asarray([source_date]), np.asarray([as_of_time]), HFQ_BASIS, VERSION)

    assert lineage.source_date.tolist() == [source_date]


@pytest.mark.parametrize(
    "source_date,as_of_time",
    [
        ("20240230", "2024-02-30T15:00:00+08:00"),
        ("20240101", "2024-01-01"),
        ("20240101", "not-a-timestamp"),
        ("20240101", "2024-01-01T15:00:00+00:00"),
        ("20240101", "2024-01-01T15:00:01+08:00"),
    ],
)
def test_price_lineage_rejects_malformed_or_non_point_in_time_dates(source_date, as_of_time):
    with pytest.raises(PriceLineageError):
        PriceLineage(np.asarray([source_date]), np.asarray([as_of_time]), HFQ_BASIS, VERSION)


@pytest.mark.parametrize(
    "version",
    ["", "raw-times-same-day-adj-v1:abc", "raw-times-same-day-adj-v1:" + "A" * 64],
)
def test_price_lineage_rejects_unsupported_adjustment_versions(version):
    with pytest.raises(PriceLineageError):
        make_hfq_lineage(np.asarray(["20240101"]), version)


def test_price_lineage_arrays_are_copied_and_immutable():
    source_date = np.asarray(["20240101"])
    as_of_time = np.asarray(["2024-01-01T15:00:00+08:00"])
    lineage = PriceLineage(source_date, as_of_time, HFQ_BASIS, VERSION)
    source_date[0] = "19990101"

    assert lineage.source_date.tolist() == ["20240101"]
    with pytest.raises(ValueError):
        lineage.source_date[0] = "19990101"


def test_build_adjusted_prices_rejects_conflicting_duplicate_daily_rows():
    daily = pd.concat(
        [_daily(), _daily().assign(open=lambda frame: frame.open + 1)], ignore_index=True
    )

    with pytest.raises(PriceLineageError, match=r"daily.*20240101.*000001\.SZ.*conflicting"):
        build_adjusted_price_fields(
            daily, _adj([1.0, 2.0]), np.asarray(["20240101", "20240102"]), np.asarray(["000001.SZ"])
        )


def test_build_adjusted_prices_accepts_identical_duplicate_daily_rows():
    daily = pd.concat([_daily(), _daily()], ignore_index=True)

    fields, _ = build_adjusted_price_fields(
        daily, _adj([1.0, 2.0]), np.asarray(["20240101", "20240102"]), np.asarray(["000001.SZ"])
    )

    np.testing.assert_allclose(fields["open_hfq"], [[10.0], [10.0]])


def test_build_adjusted_prices_requires_factor_for_any_finite_raw_price():
    daily = _daily()
    daily.loc[1, "close"] = np.nan

    with pytest.raises(PriceLineageError, match=r"20240102.*000001\.SZ.*same-date"):
        build_adjusted_price_fields(
            daily,
            _adj([1.0, np.nan]),
            np.asarray(["20240101", "20240102"]),
            np.asarray(["000001.SZ"]),
        )


def test_build_adjusted_prices_normalizes_mixed_key_types_and_versions_stably():
    daily = _daily().replace(
        {"trade_date": {"20240101": 20240101, "20240102": 20240102}, "ts_code": {"000001.SZ": 1}}
    )
    adj = pd.DataFrame(
        {"trade_date": ["20240101", "20240102"], "ts_code": ["1", "1"], "adj_factor": [1.0, 2.0]}
    )

    _, lineage = build_adjusted_price_fields(
        daily, adj, np.asarray([20240101, 20240102]), np.asarray([1])
    )
    _, reversed_lineage = build_adjusted_price_fields(
        daily, adj.iloc[::-1], np.asarray([20240101, 20240102]), np.asarray([1])
    )

    assert lineage["open_hfq"].adj_factor_version == reversed_lineage["open_hfq"].adj_factor_version


def test_panel_cache_round_trips_lineage_and_subsets_it(tmp_path):
    dates = np.asarray(["20240101", "20240102"])
    codes = np.asarray(["000001.SZ", "000002.SZ"])
    lineage = make_hfq_lineage(dates, VERSION)
    panel = Panel(dates, codes)
    panel.add("open_hfq", np.ones((2, 2), dtype=np.float32), price_lineage=lineage)
    path = tmp_path / "panel.npz"
    panel.save(path)

    loaded = Panel.load(path)
    assert "__lineage__open_hfq__source_date" not in loaded.fields
    assert loaded.price_lineage["open_hfq"] == lineage
    assert loaded.slice_dates("20240102").price_lineage["open_hfq"].source_date.tolist() == [
        "20240102"
    ]
    assert loaded.select_codes(np.asarray([1])).price_lineage["open_hfq"] == lineage


def test_generated_lineage_passes_adjusted_price_guard_after_cache_round_trip(tmp_path):
    dates = np.asarray(["20240101", "20240102"])
    codes = np.asarray(["000001.SZ"])
    lineage = make_hfq_lineage(dates, VERSION)
    panel = Panel(dates, codes)
    for field in ("open_hfq", "high_hfq", "low_hfq", "close_hfq"):
        panel.add(field, np.ones((2, 1), dtype=np.float32), price_lineage=lineage)
    path = tmp_path / "panel.npz"
    panel.save(path)

    panel.require_adjusted_prices(("open_hfq", "high_hfq", "low_hfq", "close_hfq"), "test")
    Panel.load(path).require_adjusted_prices(
        ("open_hfq", "high_hfq", "low_hfq", "close_hfq"), "cached test"
    )
