#!/usr/bin/env python3
"""Calibrate placebo IC/Gini thresholds on the fixed formal training window."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from helix.config import PROJECT_ROOT, Config, PlaceboThresholdConfig
from helix.data.event_table import build_event_panel, is_label_column
from helix.eval.placebo import (
    METRICS,
    factor_metrics,
    metric_quantiles,
    placebo_distribution,
    screen_factor_metrics,
)
from helix.gp.library import FactorLibrary, compute_factors, load_factors

FORMAL_TRAIN_END = "2024-09-04"
FORMAL_TRAIN_START = "2022-01-04"
FORMAL_TRAIN_DATES = 649
# SHA-256 of newline-joined sorted unique dates from the filtered formal input window.
FORMAL_TRAIN_DATE_DIGEST = "df8186eafc50efa3e7ae9432e6e6327a333f7050b677f130838a17b03571e381"
TARGET = "label_d2_hit_8pct_hfq"

DEFAULT_INPUT = Path("data/raw/argus_quant_working_hfq.parquet")
DEFAULT_FORMAL_LIBRARY = Path("data/artifacts/argus/event_factors.json")
DEFAULT_N40_LIBRARY = Path("data/artifacts/argus_n40/event_factors.json")
DEFAULT_MULTI_LIBRARY = Path("data/artifacts/argus_multi/event_factors.json")
DEFAULT_DISTRIBUTION = Path("data/artifacts/placebo_ic_distribution.parquet")
DEFAULT_SCREENING = Path("data/artifacts/placebo_factor_screening.parquet")
DEFAULT_REPORT = Path("docs/risk/placebo_ic_calibration.md")

BEGIN_MARKER = "# BEGIN PLACEBO CALIBRATION"
END_MARKER = "# END PLACEBO CALIBRATION"


def validate_train_end(train_end: str | None) -> str:
    """Accept only the cutoff used to mine the formal factor library."""
    if train_end != FORMAL_TRAIN_END:
        raise ValueError(f"train_end must equal formal cutoff {FORMAL_TRAIN_END}")
    return train_end


def load_training_frame(
    path: str | Path,
    columns: Sequence[str],
    train_end: str,
) -> pd.DataFrame:
    """Read only projected training rows, applying the cutoff in the Parquet scan."""
    cutoff = validate_train_end(train_end)
    projected = list(dict.fromkeys(["trade_date", "stock_code", TARGET, *columns]))
    frame = pd.read_parquet(
        Path(path),
        columns=projected,
        filters=[("trade_date", "<=", cutoff)],
    )
    missing = [column for column in projected if column not in frame.columns]
    if missing:
        raise ValueError(f"filtered input is missing requested columns: {missing}")
    frame = frame.loc[:, projected].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    if frame.empty:
        raise ValueError("training filter returned no rows")
    if frame["trade_date"].max() > cutoff:
        raise ValueError("training filter admitted an out-of-sample row")
    if cutoff not in set(frame["trade_date"]):
        raise ValueError("train_end is absent from the filtered input")
    if (frame["trade_date"] > cutoff).any():
        raise ValueError("training frame contains an out-of-sample row")
    return frame


def validate_training_dates(dates: Sequence[object], train_end: str) -> np.ndarray:
    """Require the complete formal mining window, not merely its final date."""
    cutoff = validate_train_end(train_end)
    unique_dates = np.unique(np.asarray(dates).astype(str))
    valid = (
        unique_dates.size == FORMAL_TRAIN_DATES
        and unique_dates[0] == FORMAL_TRAIN_START
        and unique_dates[-1] == cutoff
    )
    if not valid:
        raise ValueError(
            "formal training window must contain exactly "
            f"{FORMAL_TRAIN_DATES} dates from {FORMAL_TRAIN_START} through {cutoff}"
        )
    digest = hashlib.sha256("\n".join(unique_dates).encode()).hexdigest()
    if digest != FORMAL_TRAIN_DATE_DIGEST:
        raise ValueError("formal training calendar does not match the approved date sequence")
    return unique_dates


def validate_training_labels(labels: pd.Series) -> None:
    """Reject malformed labels before panel packing can coerce them to missing."""
    observed = labels.dropna()
    if not pd.api.types.is_numeric_dtype(observed.dtype):
        raise ValueError(f"{TARGET} must contain only binary values or NaN")
    values = observed.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"{TARGET} must contain only binary values or NaN")


def validate_library_fields(field_names: Sequence[str]) -> list[str]:
    """Keep every outcome-prefixed column out of factor replay."""
    names = list(field_names)
    leaked = [name for name in names if is_label_column(name)]
    if leaked:
        raise ValueError(f"library fields contain label or outcome columns: {leaked}")
    return names


def compute_complete_library(
    library: FactorLibrary,
    fields: dict[str, np.ndarray],
    scope: str,
) -> tuple[list[str], np.ndarray]:
    """Replay every saved expression or fail instead of silently using a subset."""
    names, values = compute_factors(library, fields)
    expected_names = [factor.name for factor in library.factors]
    if names != expected_names or values.ndim != 3 or values.shape[2] != len(expected_names):
        raise RuntimeError(f"{scope} library did not replay every saved factor")
    return names, values


def calibrate_scopes(
    formal_names: Sequence[str],
    formal_values: np.ndarray,
    supplemental_scopes: Mapping[
        str, tuple[Sequence[str], np.ndarray, str | Path]
    ],
    labels: np.ndarray,
    mask: np.ndarray,
    n_permutations: int,
    seed: int,
    min_samples: int,
    formal_library_path: str | Path = DEFAULT_FORMAL_LIBRARY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate on formal values, then grade every scope against those thresholds."""
    formal_array = np.asarray(formal_values)
    if not formal_names or formal_array.ndim != 3 or formal_array.shape[2] == 0:
        raise ValueError("at least one formal factor is required")
    if len(formal_names) != formal_array.shape[2]:
        raise ValueError("formal_names must contain one entry per formal factor")

    distribution = placebo_distribution(
        formal_array,
        labels,
        mask,
        n_permutations=n_permutations,
        seed=seed,
        min_samples=min_samples,
    )
    quantiles = metric_quantiles(distribution)

    formal_metrics = factor_metrics(
        formal_names,
        formal_array,
        labels,
        mask,
        min_samples=min_samples,
    )
    screened = [
        screen_factor_metrics(
            formal_metrics,
            quantiles,
            scope="formal",
            library_path=str(formal_library_path),
        )
    ]

    for scope, (names, values, library_path) in supplemental_scopes.items():
        if scope == "formal":
            raise ValueError("supplemental scopes cannot use the formal scope")
        metrics = factor_metrics(
            names,
            values,
            labels,
            mask,
            min_samples=min_samples,
        )
        supplemental = screen_factor_metrics(
            metrics,
            quantiles,
            scope=scope,
            library_path=str(library_path),
        )
        if supplemental["candidate_eligible"].astype(bool).any():
            raise AssertionError("supplemental factors cannot become formal candidates")
        screened.append(supplemental)

    return distribution, quantiles, pd.concat(screened, ignore_index=True)


def _format_markdown_value(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return str(value)
        return f"{float(value):.12g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    if frame.empty:
        return "_No factors._"
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(_format_markdown_value(value) for value in row)
        + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    raise ValueError(f"report metadata is missing {names[0]!r}")


def _screening_details(screening: pd.DataFrame, scope: str) -> pd.DataFrame:
    columns = [
        "library_path",
        "factor_name",
        "ic_mean_signed",
        "ic_mean",
        "icir_signed",
        "icir",
        "gini_signed",
        "gini",
        "ic_level",
        "icir_level",
        "gini_level",
        "overall_level",
        "candidate_eligible",
        "suggest_evict",
    ]
    details = screening.loc[screening["scope"] == scope, columns].copy()
    details["library_path"] = details["library_path"].map(_report_path)
    return details


def _report_path(value: object) -> str:
    """Render repository paths portably without rewriting external paths."""
    original = str(value)
    path = Path(original).expanduser()
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return original


def _level_counts(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(columns=["overall_level", "count"])
    return (
        details["overall_level"]
        .value_counts(dropna=False)
        .rename_axis("overall_level")
        .reset_index(name="count")
    )


def render_report(
    distribution: pd.DataFrame,
    quantiles: pd.DataFrame,
    screening: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    """Render the auditable training-only calibration report."""
    input_path = _report_path(_metadata_value(metadata, "input", "input_path"))
    target = _metadata_value(metadata, "target")
    seed = _metadata_value(metadata, "seed")
    n_permutations = _metadata_value(metadata, "n_permutations", "permutations")
    min_samples = _metadata_value(metadata, "min_samples")
    train_start = _metadata_value(metadata, "train_start")
    train_end = _metadata_value(metadata, "train_end")
    n_train_dates = _metadata_value(metadata, "n_train_dates")
    formal_factor_count = _metadata_value(metadata, "formal_factor_count")
    supplemental_factor_count = _metadata_value(metadata, "supplemental_factor_count")

    summary = (
        distribution.loc[:, list(METRICS)]
        .agg(["mean", "std", "min", "median", "max"])
        .T.rename_axis("metric")
        .reset_index()
    )
    threshold_table = quantiles.loc[:, ["metric", "p95", "p99", "p999"]].rename(
        columns={"p999": "p99.9"}
    )
    formal = _screening_details(screening, "formal")
    n40 = _screening_details(screening, "argus_n40")
    multi = _screening_details(screening, "argus_multi")

    return f"""# Placebo IC/Gini Calibration

## Calibration Inputs and Training Window

- Input: `{input_path}`
- Target: `{target}`
- Seed: `{seed}`
- Permutations: `{n_permutations}`
- Minimum daily samples: `{min_samples}`
- Training window: `{train_start}` through `{train_end}`
- Training dates: `{n_train_dates}`
- Formal factors: `{formal_factor_count}`
- Supplemental factors: `{supplemental_factor_count}`

## Null Distribution Summary

{_markdown_table(summary)}

## Core Thresholds

{_markdown_table(threshold_table)}

## Formal Library Screening

### Formal factor details

{_markdown_table(formal)}

### Formal level counts

{_markdown_table(_level_counts(formal))}

## Supplemental: argus_n40

Factors: {len(n40)}

{_markdown_table(n40)}

### argus_n40 level counts

{_markdown_table(_level_counts(n40))}

## Supplemental: argus_multi

Factors: {len(multi)}

{_markdown_table(multi)}

### argus_multi level counts

{_markdown_table(_level_counts(multi))}

## Governance Caveats

Only the formal library and training dates generated these thresholds. Out-of-sample rows
never entered the panel, factor computation, permutation RNG, null statistics, or thresholds.

Supplemental scopes never participated in threshold generation and are never eligible for formal admission.
Their levels are counterfactual comparisons against the formal thresholds only.

The canonical `daily_ic` currently applies ordinal ranks to the binary label. Consequently,
these IC placebo thresholds depend on the current event-slot layout, missingness structure,
and formal factor family. They are only the G1 baseline for the current event-table formal library.
A change to the dataset, event-slot layout, missingness structure, or formal factor family
requires recalibration.

Passing G1 does not override G3. A factor rejected by G3 remains rejected even when its G1
statistics exceed the placebo thresholds.
"""


def _threshold_values(quantiles: pd.DataFrame) -> dict[str, float]:
    if "metric" not in quantiles or "p99" not in quantiles:
        raise ValueError("quantiles must contain metric and p99 columns")
    if quantiles["metric"].duplicated().any():
        raise ValueError("quantiles must contain exactly one row per metric")
    indexed = quantiles.set_index("metric")
    missing = [metric for metric in METRICS if metric not in indexed.index]
    if missing:
        raise ValueError(f"quantiles are missing metrics: {missing}")
    values = {metric: float(indexed.loc[metric, "p99"]) for metric in METRICS}
    if not all(np.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("formal p99 thresholds must be finite and non-negative")
    return values


def _threshold_block(
    values: Mapping[str, float], train_start: str, train_end: str
) -> str:
    return (
        f"{BEGIN_MARKER}\n"
        "factor_admission:\n"
        "  placebo_threshold:\n"
        f"    ic_mean: {repr(values['ic_mean'])}\n"
        f"    icir: {repr(values['icir'])}\n"
        f"    gini: {repr(values['gini'])}\n"
        "    quantile: 0.99\n"
        f'    train_start: "{train_start}"\n'
        f'    train_end: "{train_end}"\n'
        f"{END_MARKER}\n"
    )


def write_threshold_config(
    path: str | Path,
    quantiles: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> None:
    """Atomically append or replace the delimited calibrated-threshold YAML block."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    original = config_path.read_text(encoding="utf-8")
    candidate = _threshold_config_candidate(
        original, quantiles, train_start, train_end
    )
    _atomic_write_text(config_path, candidate)


def _threshold_config_candidate(
    original: str,
    quantiles: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> str:
    """Build and validate complete config text without mutating the filesystem."""
    if train_start != FORMAL_TRAIN_START:
        raise ValueError(f"train_start must equal formal start {FORMAL_TRAIN_START}")
    validate_train_end(train_end)
    values = _threshold_values(quantiles)
    threshold = PlaceboThresholdConfig(
        **values,
        quantile=0.99,
        train_start=train_start,
        train_end=train_end,
    )
    block = _threshold_block(values, train_start, train_end)
    begin_count = original.count(BEGIN_MARKER)
    end_count = original.count(END_MARKER)
    if begin_count != end_count or begin_count > 1:
        raise ValueError("config contains malformed placebo calibration markers")
    pattern = re.compile(
        rf"(?m)^{re.escape(BEGIN_MARKER)}\n.*?^{re.escape(END_MARKER)}(?:\n|$)",
        re.DOTALL,
    )
    outside_block, replacements = pattern.subn("", original)
    if replacements != begin_count:
        raise ValueError("config contains malformed placebo calibration block")
    outside_raw = yaml.safe_load(outside_block) or {}
    if isinstance(outside_raw, dict) and "factor_admission" in outside_raw:
        raise ValueError("factor_admission exists outside the delimited block")
    if begin_count == 0:
        separator = "" if not original or original.endswith("\n") else "\n"
        candidate = original + separator + block
    else:
        candidate, replacements = pattern.subn(block, original)
        if replacements != 1:
            raise ValueError("config contains malformed placebo calibration block")

    raw = yaml.safe_load(candidate) or {}
    if not isinstance(raw, dict):
        raise ValueError("config YAML must contain a mapping")
    configured = Config.model_validate(raw).factor_admission.placebo_threshold
    if configured is None or configured.model_dump() != threshold.model_dump():
        raise ValueError("generated placebo threshold configuration failed validation")
    return candidate


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def _set_staged_mode(staged: Path, target: Path) -> None:
    mode = target.stat().st_mode & 0o7777 if target.exists() else 0o644
    os.chmod(staged, mode)


def _atomic_write_text(target: Path, text: str) -> None:
    staged = _temporary_path(target)
    try:
        staged.write_text(text, encoding="utf-8")
        _set_staged_mode(staged, target)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _load_library(path: Path, scope: str, require_factors: bool = True) -> FactorLibrary:
    library = load_factors(path)
    if library.kind != "event":
        raise ValueError(f"{scope} library must be an event factor library")
    if require_factors and not library.factors:
        raise ValueError(f"{scope} factor library is empty")
    return library


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _preflight_paths(
    *,
    input_path: str | Path,
    formal_library_path: str | Path,
    n40_library_path: str | Path,
    multi_library_path: str | Path,
    distribution_path: str | Path,
    screening_path: str | Path,
    report_path: str | Path,
    write_config_path: str | Path | None,
) -> dict[str, Path | None]:
    resolved: dict[str, Path | None] = {
        "input_path": Path(input_path).expanduser().resolve(),
        "formal_library_path": Path(formal_library_path).expanduser().resolve(),
        "n40_library_path": Path(n40_library_path).expanduser().resolve(),
        "multi_library_path": Path(multi_library_path).expanduser().resolve(),
        "distribution_path": Path(distribution_path).expanduser().resolve(),
        "screening_path": Path(screening_path).expanduser().resolve(),
        "report_path": Path(report_path).expanduser().resolve(),
        "write_config_path": (
            None
            if write_config_path is None
            else Path(write_config_path).expanduser().resolve()
        ),
    }
    inputs = {
        name: resolved[name]
        for name in (
            "input_path",
            "formal_library_path",
            "n40_library_path",
            "multi_library_path",
        )
    }
    outputs = {
        name: resolved[name]
        for name in ("distribution_path", "screening_path", "report_path")
    }
    seen_outputs: dict[Path, str] = {}
    for name, path in outputs.items():
        assert path is not None
        if path in seen_outputs:
            raise ValueError(f"path collision: {name} equals {seen_outputs[path]}")
        seen_outputs[path] = name
        for input_name, input_value in inputs.items():
            if path == input_value:
                raise ValueError(f"path collision: {name} equals {input_name}")

    config_path = resolved["write_config_path"]
    if config_path is not None:
        for name, path in {**inputs, **outputs}.items():
            if config_path == path:
                raise ValueError(f"path collision: write_config_path equals {name}")
    return resolved


def _preflight_publication(
    distribution_path: Path,
    screening_path: Path,
    report_path: Path,
    write_config_path: Path | None,
) -> None:
    for target in (distribution_path, screening_path, report_path):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_file():
            raise ValueError(f"output target is not a file: {target}")
    if write_config_path is not None:
        if not write_config_path.is_file():
            raise FileNotFoundError(f"config not found: {write_config_path}")
        if not write_config_path.parent.is_dir():
            raise ValueError(f"config parent is not a directory: {write_config_path.parent}")


def _publish_calibration_outputs(
    distribution: pd.DataFrame,
    screening: pd.DataFrame,
    report: str,
    distribution_path: Path,
    screening_path: Path,
    report_path: Path,
    config_candidate: tuple[Path, str] | None,
) -> None:
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path | None]] = []
    retain_backups = False
    try:
        staged_distribution = _temporary_path(distribution_path)
        staged.append((staged_distribution, distribution_path))
        distribution.to_parquet(staged_distribution, index=False)

        staged_screening = _temporary_path(screening_path)
        staged.append((staged_screening, screening_path))
        screening.to_parquet(staged_screening, index=False)

        staged_report = _temporary_path(report_path)
        staged.append((staged_report, report_path))
        staged_report.write_text(report, encoding="utf-8")

        if config_candidate is not None:
            config_path, config_text = config_candidate
            staged_config = _temporary_path(config_path)
            staged.append((staged_config, config_path))
            staged_config.write_text(config_text, encoding="utf-8")

        for staged_path, target_path in staged:
            _set_staged_mode(staged_path, target_path)
        for _, target_path in staged:
            backup = None
            if target_path.exists():
                backup = _temporary_path(target_path)
                backups.append((target_path, backup))
                shutil.copy2(target_path, backup)
            else:
                backups.append((target_path, None))

        try:
            for staged_path, target_path in staged:
                os.replace(staged_path, target_path)
        except Exception as publish_error:
            rollback_errors = []
            for target_path, backup in backups:
                try:
                    if backup is None:
                        target_path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target_path)
                except Exception as rollback_error:
                    rollback_errors.append((target_path, backup, rollback_error))
            if rollback_errors:
                retain_backups = True
                recovery = ", ".join(
                    f"{target} <- {backup}"
                    for target, backup, _ in rollback_errors
                )
                raise RuntimeError(
                    f"calibration publication failed and rollback was incomplete; "
                    f"recovery files retained: {recovery}"
                ) from publish_error
            raise
    finally:
        for staged_path, _ in staged:
            staged_path.unlink(missing_ok=True)
        if not retain_backups:
            for _, backup in backups:
                if backup is not None:
                    backup.unlink(missing_ok=True)


def run_calibration(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    formal_library_path: str | Path = DEFAULT_FORMAL_LIBRARY,
    n40_library_path: str | Path = DEFAULT_N40_LIBRARY,
    multi_library_path: str | Path = DEFAULT_MULTI_LIBRARY,
    train_end: str,
    seed: int = 20260813,
    n_permutations: int = 1000,
    min_samples: int = 50,
    distribution_path: str | Path = DEFAULT_DISTRIBUTION,
    screening_path: str | Path = DEFAULT_SCREENING,
    report_path: str | Path = DEFAULT_REPORT,
    write_config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay all libraries on filtered data and write calibration artifacts."""
    cutoff = validate_train_end(train_end)
    n_permutations = _positive_integer(n_permutations, "n_permutations")
    min_samples = _positive_integer(min_samples, "min_samples")
    paths = _preflight_paths(
        input_path=input_path,
        formal_library_path=formal_library_path,
        n40_library_path=n40_library_path,
        multi_library_path=multi_library_path,
        distribution_path=distribution_path,
        screening_path=screening_path,
        report_path=report_path,
        write_config_path=write_config_path,
    )
    input_path = paths["input_path"]
    formal_library_path = paths["formal_library_path"]
    n40_library_path = paths["n40_library_path"]
    multi_library_path = paths["multi_library_path"]
    distribution_path = paths["distribution_path"]
    screening_path = paths["screening_path"]
    report_path = paths["report_path"]
    write_config_path = paths["write_config_path"]
    assert isinstance(input_path, Path)
    assert isinstance(formal_library_path, Path)
    assert isinstance(n40_library_path, Path)
    assert isinstance(multi_library_path, Path)
    assert isinstance(distribution_path, Path)
    assert isinstance(screening_path, Path)
    assert isinstance(report_path, Path)
    assert write_config_path is None or isinstance(write_config_path, Path)
    _preflight_publication(
        distribution_path,
        screening_path,
        report_path,
        write_config_path,
    )

    formal_library = _load_library(formal_library_path, "formal")
    n40_library = _load_library(n40_library_path, "argus_n40")
    multi_library = _load_library(multi_library_path, "argus_multi")
    libraries = (formal_library, n40_library, multi_library)
    field_names = validate_library_fields(
        dict.fromkeys(name for library in libraries for name in library.field_names)
    )
    if not field_names:
        raise ValueError("factor libraries require no input fields")

    training_frame = load_training_frame(input_path, field_names, cutoff)
    validate_training_labels(training_frame[TARGET])
    validate_training_dates(training_frame["trade_date"], cutoff)
    panel = build_event_panel(training_frame, field_names, [TARGET])
    panel_dates = validate_training_dates(panel.dates, cutoff)

    formal_names, formal_values = compute_complete_library(
        formal_library, panel.fields, "formal"
    )
    n40_names, n40_values = compute_complete_library(
        n40_library, panel.fields, "argus_n40"
    )
    multi_names, multi_values = compute_complete_library(
        multi_library, panel.fields, "argus_multi"
    )
    formal_factor_count = int(formal_values.shape[2])
    if formal_factor_count == 0:
        raise ValueError("formal factor replay produced no factors")

    labels = panel.f64(TARGET)
    mask = panel.occupied
    supplemental = {
        "argus_n40": (n40_names, n40_values, n40_library_path),
        "argus_multi": (multi_names, multi_values, multi_library_path),
    }
    distribution, quantiles, screening = calibrate_scopes(
        formal_names,
        formal_values,
        supplemental,
        labels,
        mask,
        n_permutations,
        seed,
        min_samples,
        formal_library_path=formal_library_path,
    )
    if formal_factor_count != len(formal_names):
        raise AssertionError("formal factor metadata does not match replayed formal values")

    train_start = str(panel_dates[0])
    n_train_dates = int(panel_dates.size)
    distribution = distribution.assign(
        seed=seed,
        train_start=train_start,
        train_end=cutoff,
        n_train_dates=n_train_dates,
        formal_factor_count=formal_factor_count,
    )
    if not distribution["formal_factor_count"].eq(formal_values.shape[2]).all():
        raise AssertionError("formal factor count contains supplemental factors")

    metadata = {
        "input": str(input_path),
        "target": TARGET,
        "seed": seed,
        "n_permutations": n_permutations,
        "min_samples": min_samples,
        "train_start": train_start,
        "train_end": cutoff,
        "n_train_dates": n_train_dates,
        "formal_factor_count": formal_factor_count,
        "supplemental_factor_count": len(n40_names) + len(multi_names),
    }
    report = render_report(distribution, quantiles, screening, metadata)
    config_candidate = None
    if write_config_path is not None:
        config_text = _threshold_config_candidate(
            write_config_path.read_text(encoding="utf-8"),
            quantiles,
            train_start,
            cutoff,
        )
        config_candidate = (write_config_path, config_text)
    _publish_calibration_outputs(
        distribution,
        screening,
        report,
        distribution_path,
        screening_path,
        report_path,
        config_candidate,
    )
    return distribution, quantiles, screening


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--formal-library", type=Path, default=DEFAULT_FORMAL_LIBRARY)
    parser.add_argument("--n40-library", type=Path, default=DEFAULT_N40_LIBRARY)
    parser.add_argument("--multi-library", type=Path, default=DEFAULT_MULTI_LIBRARY)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--screening", type=Path, default=DEFAULT_SCREENING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, quantiles, screening = run_calibration(
        input_path=args.input,
        formal_library_path=args.formal_library,
        n40_library_path=args.n40_library,
        multi_library_path=args.multi_library,
        train_end=args.train_end,
        seed=args.seed,
        n_permutations=args.permutations,
        min_samples=args.min_samples,
        distribution_path=args.distribution,
        screening_path=args.screening,
        report_path=args.report,
        write_config_path=args.write_config,
    )
    print("Formal placebo thresholds:")
    print(_markdown_table(quantiles.rename(columns={"p999": "p99.9"})))
    print("Scope counts:")
    print(screening["scope"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
