from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helix.config import BacktestConfig, LabelConfig
from helix.data.panel import PRICE_COLUMNS, Panel, build_adjusted_price_fields
from helix.data.price_lineage import HFQ_BASIS, PriceLineageError
from helix.eval.backtest import run_backtest
from helix.features.base_fields import compute_base_fields
from helix.labels.touch_label import build_touch_label

DATES = np.asarray(
    ["20240102", "20240103", "20240104", "20240105", "20240108", "20240109"]
)
CODES = np.asarray(["000001.SZ"])


def _split_daily() -> pd.DataFrame:
    raw_prices = {
        "open": [10.0, 5.0, 5.0, 5.5, 5.5, 5.5],
        "high": [10.2, 5.1, 5.6, 5.6, 5.6, 5.6],
        "low": [9.8, 4.9, 4.9, 5.4, 5.4, 5.4],
        "close": [10.0, 5.0, 5.5, 5.5, 5.5, 5.5],
        "pre_close": [10.0, 5.0, 5.0, 5.5, 5.5, 5.5],
    }
    return pd.DataFrame(
        {
            "trade_date": DATES,
            "ts_code": CODES.repeat(len(DATES)),
            **raw_prices,
        }
    )


def _split_adjustments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": DATES,
            "ts_code": CODES.repeat(len(DATES)),
            "adj_factor": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


def _build_split_panel() -> Panel:
    daily = _split_daily()
    adjusted, lineage = build_adjusted_price_fields(
        daily, _split_adjustments(), DATES, CODES
    )
    panel = Panel(dates=DATES.copy(), codes=CODES.copy())
    for field in PRICE_COLUMNS:
        panel.add(field, daily[field].to_numpy(dtype=np.float64)[:, None])
    for field, values in adjusted.items():
        panel.add(field, values, price_lineage=lineage.get(field))
    panel.add("amount", np.full(panel.shape, 1_000_000.0))
    panel.add("up_limit", np.asarray([11.0, 5.5, 6.05, 6.05, 6.05, 6.05])[:, None])
    panel.add("down_limit", np.asarray([9.0, 4.5, 4.95, 4.95, 4.95, 4.95])[:, None])
    panel.add("limit_price_observed", np.ones(panel.shape))
    panel.add("is_trading", np.ones(panel.shape))
    return panel


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        top_k=1,
        exit_rule="close",
        enable_realistic_exit=True,
        commission_bps=2.0,
        transfer_bps=0.0,
        stamp_sell_bps=5.0,
        stamp_sell_bps_before_cut=5.0,
        slippage_bps=0.0,
    )


def _one_trade_inputs(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(panel.shape, np.nan)
    candidates = np.zeros(panel.shape, dtype=bool)
    predictions[0, 0] = 1.0
    candidates[0, 0] = True
    return predictions, candidates


def test_split_chain_uses_exact_hfq_lineage_and_keeps_raw_out_of_accounting(
    tmp_path: Path,
) -> None:
    panel = _build_split_panel()
    cache = tmp_path / "split-panel.npz"
    panel.save(cache)
    loaded = Panel.load(cache)

    stamp = loaded.require_adjusted_prices(
        ("open_hfq", "high_hfq", "low_hfq", "close_hfq"), "full-chain regression"
    )
    assert stamp.price_basis == HFQ_BASIS
    np.testing.assert_allclose(loaded["adj_factor"][:, 0], [1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert loaded.price_lineage["close_hfq"].source_date.tolist() == DATES.tolist()
    assert loaded.price_lineage["close_hfq"].as_of_time[1] == (
        "2024-01-03T15:00:00+08:00"
    )

    fields = compute_base_fields(loaded)
    assert fields["ret1"][1, 0] == pytest.approx(0.0)
    assert fields["gap"][1, 0] == pytest.approx(0.0)

    label_cfg = LabelConfig(entry_offset=1, touch_offset=2, target_ratio=1.08)
    labels = build_touch_label(loaded, np.ones(loaded.shape, dtype=bool), label_cfg)
    assert labels.adjustment == stamp
    assert labels.entry_price[0, 0] == pytest.approx(10.0)
    assert labels.target_price[0, 0] == pytest.approx(10.8)
    assert labels.y[0, 0] == pytest.approx(1.0)
    assert labels.exit_price[0, 0] == pytest.approx(11.0)
    assert loaded.dates[-1] == "20240109"
    # The 2024-01-08 D0 needs 2024-01-10 for exact D+2, beyond train_end.
    assert not labels.valid[-2:, 0].any()
    assert np.isnan(labels.y[-2:, 0]).all()

    predictions, candidates = _one_trade_inputs(loaded)
    result = run_backtest(
        predictions,
        labels,
        candidates,
        loaded.dates,
        label_cfg,
        _backtest_config(),
        panel=loaded,
    )
    expected_gross = 0.10
    expected_net = (1.0 + expected_gross) * (1.0 - 0.0007) / (1.0 + 0.0002) - 1.0
    assert result.daily["n_executed"].tolist() == [1]
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(10.0)
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(11.0)
    assert result.trades.iloc[0]["realistic_gross_return"] == pytest.approx(expected_gross)
    assert result.summary["mean_trade_return_net"] == pytest.approx(expected_net)

    raw_poisoned = Panel.load(cache)
    for field in (*PRICE_COLUMNS, "up_limit", "down_limit"):
        raw_poisoned.fields[field] *= 100.0
    poisoned_fields = compute_base_fields(raw_poisoned)
    for field in ("ret1", "gap", "intraday", "hl_range"):
        np.testing.assert_allclose(poisoned_fields[field], fields[field], equal_nan=True)
    poisoned_labels = build_touch_label(
        raw_poisoned, np.ones(raw_poisoned.shape, dtype=bool), label_cfg
    )
    poisoned = run_backtest(
        predictions,
        poisoned_labels,
        candidates,
        raw_poisoned.dates,
        label_cfg,
        _backtest_config(),
        panel=raw_poisoned,
    )
    assert poisoned.daily["n_executed"].tolist() == result.daily["n_executed"].tolist()
    assert poisoned.trades.iloc[0]["realistic_gross_return"] == pytest.approx(expected_gross)
    assert poisoned.summary["mean_trade_return_net"] == pytest.approx(expected_net)


def test_full_chain_fails_closed_at_first_price_node_without_lineage() -> None:
    panel = _build_split_panel()
    panel.price_lineage.pop("close_hfq")

    with pytest.raises(
        PriceLineageError, match=r"compute_base_fields: missing price lineage.*close_hfq"
    ):
        compute_base_fields(panel)


def test_d10_and_adjustment_report_keep_exact_reciprocal_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = (root / "docs/factor-governance.md").read_text(encoding="utf-8")
    report = (root / "docs/risk/adjustment_unification_fix.md").read_text(encoding="utf-8")
    historical = (root / "docs/risk/gp000_loss_attribution.md").read_text(encoding="utf-8")
    d10 = next(line for line in ledger.splitlines() if line.startswith("| **D10**"))

    for required in (
        "**修复完成（2026-08-14）**",
        "[专项报告](risk/gp000_loss_attribution.md)",
        "[修复说明](risk/adjustment_unification_fix.md)",
        "-0.0627748063907745",
        "-0.062899974234733",
        "-0.005455654320765759",
        "-0.005233397934459387",
        "+0.00022225638630637198",
        "+0.022226 个百分点",
        "-1.4420300457461805",
        "-1.3882776746645582",
        "647 个 D+2 完整日期",
        "2024-09-02",
        "2024-09-04",
        "legacy",
        "未经验证",
        "不可追溯",
        "不改写",
        "非核心/非主导亏损",
        "目标错配仍是主导亏损原因",
    ):
        assert required in d10
    assert "[治理台账 D10](../factor-governance.md)" in report
    assert "[治理台账 D10](../factor-governance.md)" in historical
