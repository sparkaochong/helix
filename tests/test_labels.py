"""The label is the specification. These tests pin the exact D0/D+1/D+2 semantics."""

from __future__ import annotations

import numpy as np
import pytest

from helix.config import LabelConfig
from helix.data.panel import Panel
from helix.data.price_lineage import (
    AdjustmentStamp,
    PriceLineage,
    PriceLineageError,
    make_hfq_lineage,
)
from helix.labels.touch_label import build_touch_label

TEST_ADJ_VERSION = "raw-times-same-day-adj-v1:" + "0" * 64


def make_panel(n_dates: int = 6, n_codes: int = 1, **overrides) -> Panel:
    """Flat 10.0 prices with an unreachable up-limit unless overridden."""
    shape = (n_dates, n_codes)
    fields = {
        "open": np.full(shape, 10.0),
        "high": np.full(shape, 10.0),
        "low": np.full(shape, 10.0),
        "close": np.full(shape, 10.0),
        "pre_close": np.full(shape, 10.0),
        "open_hfq": np.full(shape, 10.0),
        "high_hfq": np.full(shape, 10.0),
        "low_hfq": np.full(shape, 10.0),
        "close_hfq": np.full(shape, 10.0),
        "up_limit": np.full(shape, 11.0),
        "limit_price_observed": np.ones(shape),
        "is_trading": np.ones(shape),
    }
    fields.update(overrides)
    dates = np.array([f"2024010{i}" for i in range(1, n_dates + 1)])
    return Panel(
        dates=dates,
        codes=np.array([f"00000{j}.SZ" for j in range(n_codes)]),
        fields={k: np.asarray(v, dtype=np.float64) for k, v in fields.items()},
        price_lineage={
            name: make_hfq_lineage(dates, TEST_ADJ_VERSION)
            for name in ("open_hfq", "high_hfq", "close_hfq")
        },
    )


@pytest.fixture
def cfg() -> LabelConfig:
    return LabelConfig(entry_offset=1, touch_offset=2, target_ratio=1.08)


def test_touch_is_measured_against_the_next_open(cfg):
    high = np.full((6, 1), 10.0)
    high[3, 0] = 10.9  # D+2 for the decision made on row 1 (entry open 10.0 -> target 10.8)
    panel = make_panel(high_hfq=high)
    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)

    assert labels.y[1, 0] == 1.0
    assert labels.y[0, 0] == 0.0
    np.testing.assert_allclose(labels.entry_price[1, 0], 10.0)
    np.testing.assert_allclose(labels.target_price[1, 0], 10.8)


def test_just_below_the_target_is_a_miss(cfg):
    high = np.full((6, 1), 10.0)
    high[3, 0] = 10.79
    panel = make_panel(high_hfq=high)
    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)
    assert labels.y[1, 0] == 0.0


def test_last_rows_have_no_label(cfg):
    panel = make_panel()
    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)
    assert not labels.valid[-1, 0]
    assert not labels.valid[-2, 0]
    assert np.isnan(labels.y[-1, 0])
    assert labels.adjustment == AdjustmentStamp("hfq", TEST_ADJ_VERSION)


def test_label_requires_governed_hfq_lineage(cfg):
    panel = make_panel()
    del panel.price_lineage["high_hfq"]

    with pytest.raises(PriceLineageError, match=r"build_touch_label: missing.*high_hfq"):
        build_touch_label(panel, np.ones(panel.shape, dtype=bool), cfg)


def test_label_rejects_raw_adjustment_basis(cfg):
    panel = make_panel()
    matching = panel.price_lineage["high_hfq"]
    panel.price_lineage["high_hfq"] = PriceLineage(
        matching.source_date,
        matching.as_of_time,
        "raw",
        TEST_ADJ_VERSION,
    )

    with pytest.raises(PriceLineageError, match=r"build_touch_label: field 'high_hfq'.*basis 'raw'"):
        build_touch_label(panel, np.ones(panel.shape, dtype=bool), cfg)


def test_label_rejects_inconsistent_adjustment_versions(cfg):
    panel = make_panel()
    panel.price_lineage["close_hfq"] = make_hfq_lineage(
        panel.dates, "raw-times-same-day-adj-v1:" + "1" * 64
    )

    with pytest.raises(PriceLineageError, match=r"build_touch_label: field 'close_hfq'.*inconsistent adjustment version"):
        build_touch_label(panel, np.ones(panel.shape, dtype=bool), cfg)


def test_label_carries_exact_adjustment_stamp(cfg):
    panel = make_panel()
    expected = AdjustmentStamp("hfq", TEST_ADJ_VERSION)

    labels = build_touch_label(panel, np.ones(panel.shape, dtype=bool), cfg)

    assert labels.adjustment == expected


def test_entry_at_the_up_limit_is_dropped_not_labelled(cfg):
    """A D+1 open at the limit cannot be filled, so the sample must be unusable."""
    open_raw = np.full((6, 1), 10.0)
    open_raw[2, 0] = 11.0  # D+1 for row 1 opens exactly at the up-limit
    high = np.full((6, 1), 10.0)
    high[3, 0] = 20.0  # would otherwise be an easy hit
    panel = make_panel(open=open_raw, high_hfq=high)

    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)
    assert not labels.valid[1, 0]
    assert np.isnan(labels.y[1, 0])


def test_limit_up_samples_survive_when_the_filter_is_off():
    open_raw = np.full((6, 1), 10.0)
    open_raw[2, 0] = 11.0
    high = np.full((6, 1), 10.0)
    high[3, 0] = 20.0
    panel = make_panel(open=open_raw, high_hfq=high)

    cfg = LabelConfig(exclude_entry_limit_up=False)
    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)
    assert labels.valid[1, 0]
    assert labels.y[1, 0] == 1.0


def test_suspension_makes_the_outcome_undefined(cfg):
    trading = np.ones((6, 1))
    trading[3, 0] = 0.0  # D+2 suspended for the row-1 decision
    panel = make_panel(is_trading=trading)
    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)
    assert not labels.touch_tradable[1, 0]
    assert not labels.valid[1, 0]
    assert np.isnan(labels.y[1, 0])


def test_d2_suspension_preserves_d1_entry_observability(cfg):
    """A filled D+1 entry remains a position even when its planned exit is suspended."""
    trading = np.ones((6, 1))
    trading[3, 0] = 0.0  # D+2 for the decision made on row 1
    panel = make_panel(is_trading=trading)

    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)

    assert labels.entry_valid[1, 0]
    assert labels.entry_price[1, 0] == pytest.approx(10.0)
    assert not labels.valid[1, 0]
    assert np.isnan(labels.y[1, 0])


def test_universe_exclusion_propagates(cfg):
    panel = make_panel()
    universe = np.ones((6, 1), dtype=bool)
    universe[1, 0] = False
    labels = build_touch_label(panel, universe, cfg)
    assert not labels.valid[1, 0]


def test_ratio_uses_adjusted_prices_so_dividends_do_not_fake_a_miss(cfg):
    """Raw prices halve on a 2:1 split; back-adjusted prices must not."""
    raw_high = np.full((6, 1), 10.0)
    raw_high[3, 0] = 5.4          # post-split raw price
    hfq_high = np.full((6, 1), 10.0)
    hfq_high[3, 0] = 10.8         # same economics, adjusted
    panel = make_panel(high=raw_high, high_hfq=hfq_high)

    labels = build_touch_label(panel, np.ones((6, 1), dtype=bool), cfg)
    assert labels.y[1, 0] == 1.0
