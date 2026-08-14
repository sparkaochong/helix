"""The generated apply script must reproduce Helix's own factor values exactly.

If these drift, the IC measured on the training host is not the IC that was mined.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helix.data.event_table import build_event_panel
from helix.gp.export import render_apply_script, write_apply_script
from helix.gp.library import FactorLibrary, FactorSpec, compute_factors

VERSION = "raw-times-same-day-adj-v1:" + "a" * 64


def _govern_export_input(tmp_path, frame: pd.DataFrame) -> tuple[Path, Path]:
    decisions = frame["trade_date"].astype(str)
    unique_dates = sorted(decisions.unique())
    calendar_dates = pd.bdate_range("2024-01-01", periods=len(unique_dates) + 2)
    replacement = dict(
        zip(unique_dates, calendar_dates[: len(unique_dates)].strftime("%Y%m%d"), strict=True)
    )
    frame["trade_date"] = decisions.map(replacement)
    positions = {day: position for position, day in enumerate(calendar_dates.strftime("%Y%m%d"))}
    outcome_source = frame["trade_date"].map(
        lambda day: calendar_dates[positions[day] + 2].strftime("%Y%m%d")
    )
    frame["feat_a_source"] = frame["trade_date"]
    frame["feat_a_asof"] = frame["trade_date"].map(
        lambda day: f"{day[:4]}-{day[4:6]}-{day[6:]}T14:30:00+08:00"
    )
    frame["feat_b_source"] = frame["trade_date"]
    frame["feat_b_asof"] = frame["trade_date"].map(
        lambda day: f"{day[:4]}-{day[4:6]}-{day[6:]}T14:45:00+08:00"
    )
    frame["feature_basis"] = "hfq"
    frame["feature_version"] = VERSION
    frame["outcome_source"] = outcome_source
    frame["outcome_asof"] = outcome_source.map(
        lambda day: f"{day[:4]}-{day[4:6]}-{day[6:]}T15:00:00+08:00"
    )
    manifest = {
        "schema_version": 1,
        "fields": {
            "feat_a": {
                "source_date": "feat_a_source",
                "as_of_time": "feat_a_asof",
                "price_basis": "feature_basis",
                "adj_factor_version": "feature_version",
                "horizon": 0,
            },
            "feat_b": {
                "source_date": "feat_b_source",
                "as_of_time": "feat_b_asof",
                "price_basis": "feature_basis",
                "adj_factor_version": "feature_version",
                "horizon": 0,
            },
            **{
                label: {
                    "source_date": "outcome_source",
                    "as_of_time": "outcome_asof",
                    "price_basis": "feature_basis",
                    "adj_factor_version": "feature_version",
                    "horizon": 2,
                }
                for label in ("label_d2_peak_return", "label_d2_hit_8pct")
            },
        },
    }
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps(manifest), encoding="utf-8")
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({"cal_date": calendar_dates.strftime("%Y%m%d"), "is_open": 1}).to_parquet(
        calendar, index=False
    )
    return lineage, calendar


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n_dates, per_day = 40, 25
    rows = []
    for d in range(n_dates):
        for s in range(per_day):
            rows.append(
                {
                    "trade_date": f"2024{d // 28 + 1:02d}{d % 28 + 1:02d}",
                    "stock_code": f"{600000 + s:06d}.SH",
                    "feat_a": rng.normal(),
                    "feat_b": rng.normal(),
                    "label_d2_peak_return": rng.normal(0.03, 0.05),
                    "label_d2_hit_8pct": float(rng.uniform() < 0.15),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def library() -> FactorLibrary:
    return FactorLibrary(
        factors=[
            FactorSpec(name="gp_000", expression="cs_rank(feat_a)", sign=1.0),
            FactorSpec(
                name="gp_001",
                expression="div(cs_zscore(feat_a), add(abs(feat_b), cs_rank(feat_b)))",
                sign=-1.0,
            ),
        ],
        field_names=["feat_a", "feat_b"],
        windows=[],
        kind="event",
    )


def test_export_refuses_a_panel_library(library):
    panel_library = FactorLibrary(
        factors=library.factors, field_names=["feat_a"], windows=[5], kind="panel"
    )
    with pytest.raises(ValueError, match="event libraries"):
        render_apply_script(panel_library, ["label_d2_peak_return"])


def test_export_refuses_column_names_that_are_not_identifiers(library):
    bad = FactorLibrary(
        factors=library.factors, field_names=["feat a", "2feat"], windows=[], kind="event"
    )
    with pytest.raises(ValueError, match="not valid Python identifiers"):
        render_apply_script(bad, ["label_d2_peak_return"])


def test_generated_script_matches_helix_values(tmp_path, frame, library):
    lineage, calendar = _govern_export_input(tmp_path, frame)
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)

    script = write_apply_script(tmp_path / "apply_factors.py", library, ["label_d2_peak_return"])
    output = tmp_path / "output.parquet"
    report = tmp_path / "report.json"

    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(source),
         "--lineage", str(lineage), "--calendar", str(calendar),
         "--output", str(output), "--report", str(report), "--min-samples", "5"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    produced = pd.read_parquet(output)
    assert "gp_000" in produced.columns and "gp_001" in produced.columns
    assert len(produced) == len(frame)

    # Same expressions evaluated through Helix itself.
    panel = build_event_panel(
        frame, ["feat_a", "feat_b"], ["label_d2_peak_return", "label_d2_hit_8pct"]
    )
    names, values = compute_factors(library, panel.fields)
    expected = panel.to_long({n: values[:, :, k] for k, n in enumerate(names)})

    merged = produced.merge(expected, on=["trade_date", "stock_code"], suffixes=("", "_helix"))
    assert len(merged) == len(frame)
    for name in names:
        both = merged[name].notna() & merged[f"{name}_helix"].notna()
        assert both.sum() > 0
        # Helix stores factor values at float32; the script computes in float64 from
        # float32 inputs, so agreement is expected at float32 precision, not exactly.
        np.testing.assert_allclose(
            merged.loc[both, name], merged.loc[both, f"{name}_helix"], rtol=1e-5, atol=1e-8
        )


def test_generated_script_reports_ic(tmp_path, frame, library):
    lineage, calendar = _govern_export_input(tmp_path, frame)
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    script = write_apply_script(
        tmp_path / "apply_factors.py", library, ["label_d2_peak_return", "label_d2_hit_8pct"]
    )
    report = tmp_path / "report.json"

    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(source),
         "--lineage", str(lineage), "--calendar", str(calendar),
         "--report", str(report), "--min-samples", "5"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(report.read_text())
    assert set(payload) == {"_meta", "gp_000", "gp_001"}
    stats = payload["gp_000"]["label_d2_peak_return"]
    assert -1.0 <= stats["ic_mean"] <= 1.0
    assert 0.0 <= stats["positive_rate"] <= 1.0
    assert stats["n_days"] > 0
    assert payload["gp_001"]["sign"] == -1.0
    assert payload["_meta"]["n_dates_scored"] > 0


def test_ic_since_restricts_scoring_to_out_of_sample_dates(tmp_path, frame, library):
    """The default must be the mining cut-off, so the headline IC is not part in-sample."""
    lineage, calendar = _govern_export_input(tmp_path, frame)
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    cut = sorted(frame["trade_date"].unique())[len(frame["trade_date"].unique()) // 2]

    script = write_apply_script(
        tmp_path / "apply_factors.py", library, ["label_d2_peak_return"], search_end=cut
    )
    assert f'SEARCH_END = {cut!r}' in script.read_text()

    report = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(source),
         "--lineage", str(lineage), "--calendar", str(calendar),
         "--report", str(report), "--min-samples", "5"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(report.read_text())
    n_after = sum(d > cut for d in frame["trade_date"].unique())
    assert payload["_meta"]["ic_since"] == cut
    assert payload["_meta"]["n_dates_scored"] == n_after
    assert payload["gp_000"]["label_d2_peak_return"]["n_days"] <= n_after


def test_script_fails_loudly_on_a_table_missing_features(tmp_path, frame, library):
    lineage, calendar = _govern_export_input(tmp_path, frame)
    frame.drop(columns=["feat_b"]).to_parquet(tmp_path / "bad.parquet", index=False)
    script = write_apply_script(tmp_path / "apply_factors.py", library, ["label_d2_peak_return"])
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(tmp_path / "bad.parquet"),
            "--lineage",
            str(lineage),
            "--calendar",
            str(calendar),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "missing required feature columns" in proc.stdout + proc.stderr


def test_generated_script_requires_lineage_and_calendar(tmp_path, frame, library):
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    script = write_apply_script(tmp_path / "apply_factors.py", library, [])

    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(source)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 2
    assert "--lineage" in proc.stderr
    assert "--calendar" in proc.stderr


def test_generated_output_emits_governed_factor_lineage(tmp_path, frame, library):
    lineage, calendar = _govern_export_input(tmp_path, frame)
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    script = write_apply_script(tmp_path / "apply_factors.py", library, [])
    output = tmp_path / "output.parquet"
    output_lineage = tmp_path / "output-lineage.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--lineage",
            str(lineage),
            "--calendar",
            str(calendar),
            "--output",
            str(output),
            "--output-lineage",
            str(output_lineage),
            "--report",
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, proc.stderr
    produced = pd.read_parquet(output)
    factor_audits = {
        "gp_factor_source_date",
        "gp_factor_as_of_time",
        "gp_factor_price_basis",
        "gp_factor_adj_factor_version",
    }
    assert factor_audits <= set(produced)
    assert produced["gp_factor_source_date"].equals(produced["trade_date"])
    assert produced["gp_factor_as_of_time"].str.contains(
        "T14:45:00+08:00", regex=False
    ).all()
    assert produced["gp_factor_price_basis"].eq("hfq").all()
    assert produced["gp_factor_adj_factor_version"].eq(VERSION).all()
    emitted = json.loads(output_lineage.read_text())
    for factor in ("gp_000", "gp_001"):
        assert emitted["fields"][factor] == {
            "source_date": "gp_factor_source_date",
            "as_of_time": "gp_factor_as_of_time",
            "price_basis": "gp_factor_price_basis",
            "adj_factor_version": "gp_factor_adj_factor_version",
            "horizon": 0,
        }


def test_generated_script_rejects_inconsistent_source_adjustment_audits(
    tmp_path, frame, library
):
    lineage, calendar = _govern_export_input(tmp_path, frame)
    frame.loc[frame.index[-1], "feature_version"] = VERSION[:-1] + "b"
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    script = write_apply_script(tmp_path / "apply_factors.py", library, [])

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--lineage",
            str(lineage),
            "--calendar",
            str(calendar),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode != 0
    assert "inconsistent adjustment version" in proc.stdout + proc.stderr


def test_generated_script_rejects_future_horizon_library_feature(tmp_path, frame, library):
    lineage, calendar = _govern_export_input(tmp_path, frame)
    payload = json.loads(lineage.read_text())
    payload["fields"]["feat_a"]["horizon"] = 2
    lineage.write_text(json.dumps(payload), encoding="utf-8")
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    script = write_apply_script(tmp_path / "apply_factors.py", library, [])

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--lineage",
            str(lineage),
            "--calendar",
            str(calendar),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode != 0
    assert "feature field 'feat_a' must declare horizon=0" in proc.stdout + proc.stderr
