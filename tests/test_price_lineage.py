from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.data.panel import PRICE_COLUMNS, Panel, build_adjusted_price_fields
from helix.data.price_lineage import (
    HFQ_BASIS,
    PriceLineageError,
    adjustment_factor_version,
    make_hfq_lineage,
)


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


def test_panel_cache_round_trips_lineage_and_subsets_it(tmp_path):
    dates = np.asarray(["20240101", "20240102"])
    codes = np.asarray(["000001.SZ", "000002.SZ"])
    lineage = make_hfq_lineage(dates, "raw-times-same-day-adj-v1:abc")
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
    lineage = make_hfq_lineage(dates, "raw-times-same-day-adj-v1:abc")
    panel = Panel(dates, codes)
    for field in ("open_hfq", "high_hfq", "low_hfq", "close_hfq"):
        panel.add(field, np.ones((2, 1), dtype=np.float32), price_lineage=lineage)
    path = tmp_path / "panel.npz"
    panel.save(path)

    panel.require_adjusted_prices(("open_hfq", "high_hfq", "low_hfq", "close_hfq"), "test")
    Panel.load(path).require_adjusted_prices(
        ("open_hfq", "high_hfq", "low_hfq", "close_hfq"), "cached test"
    )
