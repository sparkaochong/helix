from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, cast

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
    InputManifest,
    build_structured_output,
    collect_provenance,
    compare_fixed_scores,
    freeze_top_k_selection,
    prevalidate_report_target,
    publish_report,
    render_report,
    require_posix_capabilities,
    snapshot_file,
    validate_expected_impact,
    validate_frozen_boundary,
    validate_legacy_reconstruction,
    validate_selection_identity,
    verify_input_manifest,
)
from scripts.gp000_loss_attribution import audit_adjustment_chain, build_price_lookup


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


def _test_provenance() -> dict[str, object]:
    return {
        "git": {
            "head": "a" * 40,
            "head_tree": "b" * 40,
            "dirty_excluding_generated_report": True,
            "diff_sha256": "c" * 64,
            "status_sha256": "d" * 64,
        },
        "source_modules": {"script.py": "e" * 64},
        "runtime": {"python": "test", "numpy": "test", "pandas": "test"},
    }


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
        legacy_reconstruction={
            "event_returns_match_raw": True,
            "event_return_rounding_error_max": 5e-7,
        },
        provenance=_test_provenance(),
    )

    assert payload["legacy_unverified_lineage"] is True
    assert payload["historical_reports_rewritten"] is False
    assert payload["loss_conclusion_unchanged"] is True
    assert payload["target_mismatch_remains_dominant"] is True
    assert payload["input_manifest_verified_after_computation"] is True
    assert payload["legacy_reconstruction"]["event_returns_match_raw"] is True
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
        legacy_reconstruction={
            "event_returns_match_raw": True,
            "event_return_rounding_error_max": 5e-7,
        },
        provenance=_test_provenance(),
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
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    output.chmod(0o640)
    publish_report(report, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o640

    historical = report_dir / "gp000_loss_attribution.md"
    historical.write_bytes(b"historical-sentinel")
    other_doc = report_dir / "other.md"
    outside = tmp_path / "main-data" / "artifact.md"
    traversal = report_dir / ".." / "risk" / output.name
    for rejected in (other_doc, outside, historical, traversal):
        with pytest.raises(ValueError, match="designated adjustment report"):
            publish_report("should-not-write", rejected)

    audit_called = False

    def forbidden_audit(**_kwargs: object) -> dict[str, object]:
        nonlocal audit_called
        audit_called = True
        raise AssertionError("audit must not run for an invalid report target")

    monkeypatch.setattr(baseline_module, "build_baseline_evidence", forbidden_audit)
    with pytest.raises(ValueError, match="designated adjustment report"):
        baseline_module.main(["--report", str(outside)])
    assert audit_called is False

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


def test_publish_is_anchored_against_parent_symlink_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = tmp_path / "worktree" / "docs" / "risk"
    report_dir.mkdir(parents=True)
    designated = report_dir / "adjustment_unification_fix.md"
    moved_dir = tmp_path / "moved-risk"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.setattr(baseline_module, "DESIGNATED_REPORT", designated)

    original_open = os.open
    substituted = False

    def substituting_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal substituted
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not substituted and os.fsdecode(path) == report_dir.name and dir_fd is not None:
            report_dir.rename(moved_dir)
            report_dir.symlink_to(outside_dir, target_is_directory=True)
            substituted = True
        return descriptor

    monkeypatch.setattr(baseline_module.os, "open", substituting_open)
    with pytest.raises((ValueError, OSError), match="designated adjustment report|symlink"):
        publish_report("must-not-escape", designated)

    assert substituted is True
    assert not (outside_dir / designated.name).exists()
    assert not (moved_dir / designated.name).exists()


def test_publish_rejects_docs_ancestor_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "worktree"
    report_dir = worktree / "docs" / "risk"
    report_dir.mkdir(parents=True)
    designated = report_dir / "adjustment_unification_fix.md"
    moved_docs = tmp_path / "moved-docs"
    outside = tmp_path / "outside"
    (outside / "risk").mkdir(parents=True)
    monkeypatch.setattr(baseline_module, "DESIGNATED_REPORT", designated)
    validated = prevalidate_report_target(designated)

    original_open = os.open
    substituted = False

    def substituting_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal substituted
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not substituted and os.fsdecode(path) == worktree.name and dir_fd is not None:
            (worktree / "docs").rename(moved_docs)
            (worktree / "docs").symlink_to(outside, target_is_directory=True)
            substituted = True
        return descriptor

    monkeypatch.setattr(baseline_module.os, "open", substituting_open)
    with pytest.raises((ValueError, OSError), match="designated adjustment report|symlink"):
        publish_report("must-not-escape", validated)

    assert substituted is True
    assert not (outside / "risk" / designated.name).exists()
    assert not (moved_docs / "risk" / designated.name).exists()


def _manifest_for_drift_test(tmp_path: Path) -> InputManifest:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    market = cache / "20240102.parquet"
    market.write_bytes(b"market-v1")
    event = tmp_path / "events.parquet"
    library = tmp_path / "library.json"
    config = tmp_path / "config.yaml"
    script = tmp_path / "script.py"
    for path, content in (
        (event, b"events-v1"),
        (library, b"library-v1"),
        (config, b"config-v1"),
        (script, b"script-v1"),
    ):
        path.write_bytes(content)
    snapshot_root = tmp_path / "snapshots"
    listing, cache_chain = baseline_module._cache_listing(cache)
    return InputManifest(
        event_table=snapshot_file(event, snapshot_root=snapshot_root),
        factor_library=snapshot_file(library, snapshot_root=snapshot_root),
        backtest_config=snapshot_file(config, snapshot_root=snapshot_root),
        comparison_script=snapshot_file(script, snapshot_root=snapshot_root),
        price_cache=cache,
        price_cache_listing=listing,
        price_cache_chain=cache_chain,
        market_files=(snapshot_file(market, snapshot_root=snapshot_root),),
        market_calendar=("2024-01-02",),
    )


def test_input_manifest_detects_content_mutation_and_cache_addition(
    tmp_path: Path,
) -> None:
    manifest = _manifest_for_drift_test(tmp_path)
    Path(manifest.event_table.path).write_bytes(b"events-v2")
    with pytest.raises(RuntimeError, match="event_table.*changed"):
        verify_input_manifest(manifest)

    stable = _manifest_for_drift_test(tmp_path / "stable")
    (stable.price_cache / "20240103.parquet").write_bytes(b"added")
    with pytest.raises(RuntimeError, match="price cache listing changed"):
        verify_input_manifest(stable)


def test_private_snapshot_binds_parsed_bytes_during_source_swap_restore(
    tmp_path: Path,
) -> None:
    source = tmp_path / "library.json"
    source.write_text('{"version": 1}', encoding="utf-8")
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    record = snapshot_file(source, snapshot_root=snapshot_root)
    assert stat.S_IMODE(Path(record.snapshot_path).stat().st_mode) == 0o400

    original = tmp_path / "library.original"
    replacement = tmp_path / "library.replacement"
    replacement.write_text('{"version": 2}', encoding="utf-8")
    source.rename(original)
    replacement.rename(source)
    parsed = json.loads(Path(record.snapshot_path).read_text(encoding="utf-8"))
    source.rename(replacement)
    original.rename(source)

    assert parsed == {"version": 1}
    assert json.loads(source.read_text(encoding="utf-8")) == {"version": 1}
    listing, cache_chain = baseline_module._cache_listing(tmp_path)
    manifest = InputManifest(
        event_table=record,
        factor_library=record,
        backtest_config=record,
        comparison_script=record,
        price_cache=tmp_path,
        price_cache_listing=listing,
        price_cache_chain=cache_chain,
        market_files=(),
        market_calendar=(),
    )
    verify_input_manifest(manifest)


def test_posix_capability_gate_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_posix_capabilities()
    provenance = cast(dict[str, Any], collect_provenance())
    assert provenance["git"]["head"]
    assert len(provenance["git"]["diff_sha256"]) == 64
    assert len(provenance["source_modules"]) >= 5
    assert provenance["runtime"]["python"]
    assert provenance["runtime"]["numpy"] == baseline_module.np.__version__
    assert provenance["runtime"]["pandas"] == pd.__version__

    monkeypatch.setattr(baseline_module.os, "name", "nt")
    with pytest.raises(RuntimeError, match="POSIX"):
        require_posix_capabilities()


def test_mismatched_legacy_return_label_fails_reconstruction_gate() -> None:
    events = pd.DataFrame(
        {
            "trade_date": ["2024-05-10"],
            "stock_code": ["000001.SZ"],
            "label_px_d1_open": [10.0],
            "label_px_d2_high": [11.0],
            "label_px_d2_close": [10.5],
            "label_d2_return": [0.50],
            "label_d2_hit_8pct": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": ["2024-05-10", "2024-05-13", "2024-05-14"],
            "ts_code": ["000001.SZ"] * 3,
            "open": [9.8, 10.0, 10.4],
            "high": [10.0, 10.2, 11.0],
            "close": [9.9, 10.0, 10.5],
            "adj_factor": [1.0, 1.0, 1.0],
        }
    )
    prices = build_price_lookup(
        market,
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )
    summary, _ = audit_adjustment_chain(events, prices)

    with pytest.raises(AssertionError, match="legacy event returns do not match raw"):
        validate_legacy_reconstruction(summary)


def test_frozen_selection_handles_ties_and_detects_identity_drift() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 6,
            "exit_date": ["2024-01-04"] * 6,
            "stock_code": [f"S{index}" for index in range(6)],
            "factor_score": [3.0, 2.0, 2.0, 2.0, 1.0, 0.0],
            "raw_return": [0.30, -0.20, 0.10, -0.05, 0.40, -0.40],
            "hfq_return": [-0.30, 0.20, -0.10, 0.05, -0.40, 0.40],
        }
    )
    config = BacktestConfig(
        top_k=4,
        commission_bps=0,
        transfer_bps=0,
        stamp_sell_bps=0,
        stamp_sell_bps_before_cut=0,
        slippage_bps=0,
    )
    frozen = freeze_top_k_selection(frame, config)
    assert frozen["stock_code"].tolist() == ["S0", "S1", "S2", "S3"]

    raw_arm = frozen.assign(gross_return=frozen["raw_return"])
    hfq_arm = frozen.assign(gross_return=frozen["hfq_return"])
    raw_digest, hfq_digest = validate_selection_identity(raw_arm, hfq_arm)
    assert raw_digest == hfq_digest
    comparison = compare_fixed_scores(frame, config, min_ic_samples=4).set_index(
        "price_basis"
    )
    assert comparison.loc["raw", "net_per_trade"] == pytest.approx(0.0375)
    assert comparison.loc["hfq", "net_per_trade"] == pytest.approx(-0.0375)
    assert comparison["selected_score_digest"].nunique() == 1

    drifted = hfq_arm.copy()
    drifted.loc[0, "factor_score"] = -99.0
    with pytest.raises(AssertionError, match="frozen Top-K selection identity drifted"):
        validate_selection_identity(raw_arm, drifted)
