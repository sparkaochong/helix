"""normalize_trade_date: the raw-store (YYYYMMDD) <-> legacy event table (YYYY-MM-DD)
format bridge."""

from __future__ import annotations

import pytest

from helix.data.dates import normalize_trade_date


@pytest.mark.parametrize(
    "value,expected",
    [
        ("20260731", "20260731"),
        ("2026-07-31", "20260731"),
        ("2022-01-04", "20220104"),
    ],
)
def test_normalizes_both_formats(value, expected):
    assert normalize_trade_date(value) == expected


@pytest.mark.parametrize("bad", ["2026/07/31", "not-a-date", "", "202607", "2026-7-31"])
def test_rejects_anything_else(bad):
    with pytest.raises(ValueError, match="unrecognized trade_date format"):
        normalize_trade_date(bad)


def test_tolerates_surrounding_whitespace():
    assert normalize_trade_date(" 20260731 ") == "20260731"
