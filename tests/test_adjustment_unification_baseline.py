from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.adjustment_unification_baseline as baseline_module
from helix.config import BacktestConfig
from scripts.adjustment_unification_baseline import (
    EXPECTED_D0_DATES,
    EXPECTED_HFQ_CAGR,
    EXPECTED_HFQ_FINAL_EQUITY,
    EXPECTED_HFQ_IC,
    EXPECTED_HFQ_NET_PER_TRADE,
    EXPECTED_HFQ_SHARPE,
    EXPECTED_RAW_CAGR,
    EXPECTED_RAW_FINAL_EQUITY,
    EXPECTED_RAW_IC,
    EXPECTED_RAW_NET_PER_TRADE,
    EXPECTED_RAW_SHARPE,
    build_structured_output,
    compare_fixed_scores,
    publish_report,
    render_report,
    validate_expected_impact,
    validate_frozen_boundary,
)


def _compact_aligned_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw_by_day = (
        (0.00, 0.01, 0.02, 0.03),
        (0.03, 0.02, 0.01, 0.00),
    )
    for day, (trade_date, exit_date) in enumerate(
        (("2024-01-02", "2024-01-04"), ("2024-01-03", "2024-01-05"))
    ):
        for stock, raw_return in enumerate(raw_by_day[day]):
            rows.append(
                {
                    "trade_date": trade_date,
                    "exit_date": exit_date,
                    "stock_code": f"S{stock}",
                    "factor_score": float(stock),
                    "raw_return": raw_return,
                    "hfq_return": raw_return + 0.001,
                }
            )
    return pd.DataFrame(rows)


def _reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "price_basis": "raw",
                "d2_close_ic": EXPECTED_RAW_IC,
                "net_per_trade": EXPECTED_RAW_NET_PER_TRADE,
                "cagr": EXPECTED_RAW_CAGR,
                "sharpe": EXPECTED_RAW_SHARPE,
                "final_equity": EXPECTED_RAW_FINAL_EQUITY,
                "n_days": EXPECTED_D0_DATES,
                "selected_score_digest": "same-selection",
            },
            {
                "price_basis": "hfq",
                "d2_close_ic": EXPECTED_HFQ_IC,
                "net_per_trade": EXPECTED_HFQ_NET_PER_TRADE,
                "cagr": EXPECTED_HFQ_CAGR,
                "sharpe": EXPECTED_HFQ_SHARPE,
                "final_equity": EXPECTED_HFQ_FINAL_EQUITY,
                "n_days": EXPECTED_D0_DATES,
                "selected_score_digest": "same-selection",
            },
        ]
    )


def test_compare_fixed_scores_reuses_one_selection_without_mutating_input() -> None:
    aligned = _compact_aligned_frame()
    before = aligned.copy(deep=True)
    config = BacktestConfig(
        top_k=4,
        commission_bps=0,
        transfer_bps=0,
        stamp_sell_bps=0,
        stamp_sell_bps_before_cut=0,
        slippage_bps=0,
    )

    comparison = compare_fixed_scores(aligned, config, min_ic_samples=4)

    pd.testing.assert_frame_equal(aligned, before)
    assert comparison["price_basis"].tolist() == ["raw", "hfq"]
    assert comparison["selected_score_digest"].nunique() == 1
    assert comparison.set_index("price_basis").loc["raw", "d2_close_ic"] == pytest.approx(0.0)
    assert comparison.set_index("price_basis").loc["hfq", "d2_close_ic"] == pytest.approx(0.0)
    assert comparison.set_index("price_basis").loc["raw", "net_per_trade"] == pytest.approx(0.015)
    assert comparison.set_index("price_basis").loc["hfq", "net_per_trade"] == pytest.approx(0.016)


def test_validate_expected_impact_accepts_only_the_frozen_golden_values() -> None:
    validate_expected_impact(_reference_table())

    drifted = _reference_table()
    drifted.loc[drifted["price_basis"] == "hfq", "net_per_trade"] = 0.01
    with pytest.raises(AssertionError, match="historical audit tolerance"):
        validate_expected_impact(drifted)


def test_validate_frozen_boundary_fails_closed_on_count_or_exit_drift() -> None:
    frame = _compact_aligned_frame()
    boundary = validate_frozen_boundary(
        frame,
        expected_dates=2,
        expected_d0_end="2024-01-03",
        expected_exit_end="2024-01-05",
    )
    assert boundary == {
        "d2_complete_dates": 2,
        "d0_end": "2024-01-03",
        "d2_exit_end": "2024-01-05",
    }

    with pytest.raises(AssertionError, match=r"D\+2-complete date count"):
        validate_frozen_boundary(
            frame,
            expected_dates=3,
            expected_d0_end="2024-01-03",
            expected_exit_end="2024-01-05",
        )
    with pytest.raises(AssertionError, match=r"D\+2 exit boundary"):
        validate_frozen_boundary(
            frame,
            expected_dates=2,
            expected_d0_end="2024-01-03",
            expected_exit_end="2024-01-04",
        )


def test_structured_output_marks_legacy_and_unchanged_loss() -> None:
    payload = build_structured_output(
        _reference_table(),
        boundary={
            "d2_complete_dates": EXPECTED_D0_DATES,
            "d0_end": "2024-09-02",
            "d2_exit_end": "2024-09-04",
        },
        inputs={"event_table": {"path": "/immutable/events.parquet", "sha256": "a" * 64}},
    )

    assert payload["legacy_unverified_lineage"] is True
    assert payload["historical_reports_rewritten"] is False
    assert payload["loss_conclusion_unchanged"] is True
    assert payload["target_mismatch_remains_dominant"] is True
    assert payload["comparison"][1]["net_per_trade"] == pytest.approx(
        EXPECTED_HFQ_NET_PER_TRADE
    )


def test_render_is_read_only_and_publish_only_accepts_designated_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = build_structured_output(
        _reference_table(),
        boundary={
            "d2_complete_dates": EXPECTED_D0_DATES,
            "d0_end": "2024-09-02",
            "d2_exit_end": "2024-09-04",
        },
        inputs={},
    )

    before = set(tmp_path.iterdir())
    report = render_report(payload)
    assert set(tmp_path.iterdir()) == before
    assert "目标错配是主导亏损原因" in report
    assert "复权口径问题存在，但不是核心或主导亏损原因" in report

    report_dir = tmp_path / "worktree" / "docs" / "risk"
    report_dir.mkdir(parents=True)
    output = report_dir / "adjustment_unification_fix.md"
    monkeypatch.setattr(baseline_module, "DESIGNATED_REPORT", output)

    publish_report(report, output)
    assert output.read_text(encoding="utf-8") == report

    historical = report_dir / "gp000_loss_attribution.md"
    historical.write_bytes(b"historical-sentinel")
    other_doc = report_dir / "other.md"
    outside = tmp_path / "main-data" / "artifact.md"
    traversal = report_dir / ".." / "risk" / output.name
    for rejected in (other_doc, outside, historical, traversal):
        with pytest.raises(ValueError, match="designated adjustment report"):
            publish_report("should-not-write", rejected)

    monkeypatch.setattr(
        baseline_module,
        "build_baseline_evidence",
        lambda **_kwargs: payload,
    )
    with pytest.raises(ValueError, match="designated adjustment report"):
        baseline_module.main(["--report", str(outside)])

    assert historical.read_bytes() == b"historical-sentinel"
    assert not other_doc.exists()
    assert not outside.exists()
    assert output.read_text(encoding="utf-8") == report

    alias = tmp_path / "worktree" / "risk-link"
    alias.symlink_to(report_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="designated adjustment report"):
        publish_report("should-not-write", alias / output.name)
    assert output.read_text(encoding="utf-8") == report

    output.unlink()
    escaped = tmp_path / "escaped-report.md"
    escaped.write_bytes(b"outside-sentinel")
    output.symlink_to(escaped)
    with pytest.raises(ValueError, match="designated adjustment report"):
        publish_report("should-not-write", output)
    assert escaped.read_bytes() == b"outside-sentinel"

    parsed = json.loads(json.dumps(payload, allow_nan=False))
    assert parsed["historical_reports_rewritten"] is False
