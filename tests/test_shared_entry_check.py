"""All D+1 entry consumers must share one observed-limit fillability contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from helix.config import LabelConfig
from helix.data.panel import Panel
from helix.eval import backtest as backtest_module
from helix.eval.shared_entry_check import entry_is_fillable
from helix.labels import touch_label as touch_label_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import d2_limit_down_bias as report  # noqa: E402


def make_panel() -> Panel:
    shape = (5, 3)
    fields = {
        "open": np.full(shape, 10.0),
        "open_hfq": np.full(shape, 10.0),
        "high_hfq": np.full(shape, 10.0),
        "close_hfq": np.full(shape, 10.0),
        "up_limit": np.full(shape, 11.0),
        "limit_price_observed": np.ones(shape),
        "is_trading": np.ones(shape),
    }
    fields["open"][2] = [10.5, 10.5, 11.0]
    fields["limit_price_observed"][2, 1] = 0.0
    return Panel(
        dates=np.asarray([f"2024010{i}" for i in range(1, 6)]),
        codes=np.asarray(["A", "B", "C"]),
        fields=fields,
    )


def test_entry_check_uses_configured_offset_and_requires_observed_limit():
    panel = make_panel()
    cfg = LabelConfig(entry_offset=2, touch_offset=3)

    assert entry_is_fillable(panel, 0, 0, cfg)
    assert not entry_is_fillable(panel, 0, 1, cfg)
    assert not entry_is_fillable(panel, 0, 2, cfg)


def test_disabling_limit_up_filter_still_requires_observed_market_data():
    panel = make_panel()
    cfg = LabelConfig(
        entry_offset=2,
        touch_offset=3,
        exclude_entry_limit_up=False,
    )

    assert entry_is_fillable(panel, 0, 2, cfg)
    assert not entry_is_fillable(panel, 0, 1, cfg)


def test_label_executable_entry_matches_the_shared_contract():
    panel = make_panel()
    cfg = LabelConfig(entry_offset=2, touch_offset=3)
    universe = np.ones(panel.shape, dtype=bool)
    universe[0, 0] = False

    labels = touch_label_module.build_touch_label(panel, universe, cfg)
    expected = np.asarray(
        [
            [entry_is_fillable(panel, d0, stock, cfg) for stock in range(panel.shape[1])]
            for d0 in range(panel.shape[0])
        ]
    )

    np.testing.assert_array_equal(labels.executable_entry, universe & expected)


def test_all_three_consumers_reference_the_shared_function():
    assert touch_label_module.entry_is_fillable is entry_is_fillable
    assert backtest_module.entry_is_fillable is entry_is_fillable
    assert report.entry_is_fillable is entry_is_fillable
