"""The generated apply script must reproduce Helix's own factor values exactly.

If these drift, the IC measured on the training host is not the IC that was mined.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from helix.data.event_table import build_event_panel
from helix.gp.export import render_apply_script, write_apply_script
from helix.gp.library import FactorLibrary, FactorSpec, compute_factors


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
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)

    script = write_apply_script(tmp_path / "apply_factors.py", library, ["label_d2_peak_return"])
    output = tmp_path / "output.parquet"
    report = tmp_path / "report.json"

    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(source),
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
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    script = write_apply_script(
        tmp_path / "apply_factors.py", library, ["label_d2_peak_return", "label_d2_hit_8pct"]
    )
    report = tmp_path / "report.json"

    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(source),
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
    frame.drop(columns=["feat_b"]).to_parquet(tmp_path / "bad.parquet", index=False)
    script = write_apply_script(tmp_path / "apply_factors.py", library, ["label_d2_peak_return"])
    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(tmp_path / "bad.parquet")],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "missing required feature columns" in proc.stdout + proc.stderr
