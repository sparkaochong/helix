#!/usr/bin/env python3
"""Read-only fixed-score comparison for the legacy ``gp_000`` adjustment repair."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helix.config import BacktestConfig
from helix.eval.ic import daily_ic, summarize_ic
from helix.gp.library import load_factors
from scripts.gp000_loss_attribution import (
    FORMAL_EXPRESSION,
    FORMAL_FACTOR,
    TRAIN_END,
    TRAIN_START,
    _hash_file,
    _hash_frame,
    _hyphenated,
    _top_k_selected_rows,
    audit_adjustment_chain,
    build_price_lookup,
    evaluate_top_k_book,
    event_grids,
    json_ready,
    load_audit_config,
    load_training_events,
    validate_training_calendar,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/argus_quant_working.parquet"
DEFAULT_LIBRARY = ROOT / "data/artifacts/argus/event_factors.json"
DEFAULT_PRICE_CACHE = ROOT / "data/raw/d2_exit_cache"
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
DESIGNATED_REPORT = ROOT / "docs/risk/adjustment_unification_fix.md"
DEFAULT_REPORT = DESIGNATED_REPORT

EXPECTED_RAW_IC = -0.0627748063907745
EXPECTED_HFQ_IC = -0.062899974234733
EXPECTED_RAW_NET_PER_TRADE = -0.005455654320765759
EXPECTED_HFQ_NET_PER_TRADE = -0.005233397934459387
EXPECTED_NET_DELTA = 0.0002222563863063718
EXPECTED_RAW_CAGR = -0.5517349330358576
EXPECTED_HFQ_CAGR = -0.5385714016648523
EXPECTED_RAW_SHARPE = -1.4420300457461805
EXPECTED_HFQ_SHARPE = -1.3882776746645582
EXPECTED_RAW_FINAL_EQUITY = 0.1274470164745505
EXPECTED_HFQ_FINAL_EQUITY = 0.1372782241838809
EXPECTED_D0_DATES = 647
EXPECTED_D0_END = "2024-09-02"
EXPECTED_EXIT_END = "2024-09-04"
ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DirectoryIdentity:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class DirectoryChain:
    path: str
    components: tuple[str, ...]
    identities: tuple[DirectoryIdentity, ...]


@dataclass(frozen=True)
class ReportTarget:
    path: Path
    parent_chain: DirectoryChain


@dataclass(frozen=True)
class FileSnapshot:
    """Content and identity captured before any analytical input is consumed."""

    path: str
    snapshot_path: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    mode: int
    sha256: str
    source_chain: DirectoryChain

    @property
    def source_parent_identity(self) -> DirectoryIdentity:
        return self.source_chain.identities[-1]


@dataclass(frozen=True)
class InputManifest:
    """One immutable manifest shared by loading, hashing, and drift checks."""

    event_table: FileSnapshot
    factor_library: FileSnapshot
    backtest_config: FileSnapshot
    comparison_script: FileSnapshot
    price_cache: Path
    price_cache_listing: tuple[str, ...]
    price_cache_chain: DirectoryChain
    market_files: tuple[FileSnapshot, ...]
    market_calendar: tuple[str, ...]


def require_posix_capabilities() -> None:
    """Fail clearly when no-follow descriptor-relative integrity is unavailable."""
    required_constants = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required_constants if not hasattr(os, name)]
    replace_parameters = inspect.signature(os.replace).parameters
    replace_dir_fd = {"src_dir_fd", "dst_dir_fd"} <= set(replace_parameters)
    open_dir_fd = "dir_fd" in inspect.signature(os.open).parameters
    stat_parameters = inspect.signature(os.stat).parameters
    stat_dir_fd = "dir_fd" in stat_parameters and "follow_symlinks" in stat_parameters
    unlink_dir_fd = "dir_fd" in inspect.signature(os.unlink).parameters
    if (
        os.name != "posix"
        or missing
        or not replace_dir_fd
        or not open_dir_fd
        or not stat_dir_fd
        or not unlink_dir_fd
        or not hasattr(os, "fchmod")
    ):
        detail = f"missing constants={missing}, replace_dir_fd={replace_dir_fd}"
        raise RuntimeError(
            "adjustment baseline integrity requires POSIX dir_fd/O_NOFOLLOW " + detail
        )


def _absolute_lexical(path: Path) -> Path:
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ValueError(f"absolute anchored path rejects traversal: {path}")
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


def _directory_identity(descriptor: int) -> DirectoryIdentity:
    value = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError("anchored path component is not a directory")
    return DirectoryIdentity(int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _open_absolute_directory(
    path: Path,
    *,
    expected: DirectoryChain | None = None,
) -> tuple[int, DirectoryChain]:
    """Open every absolute path component from trusted ``/`` without symlinks."""
    require_posix_capabilities()
    absolute = _absolute_lexical(path)
    components = tuple(absolute.parts[1:])
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    identities = [_directory_identity(descriptor)]
    try:
        for component in components:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            identities.append(_directory_identity(descriptor))
        chain = DirectoryChain(str(absolute), components, tuple(identities))
        if expected is not None and chain != expected:
            raise ValueError("anchored directory identity changed")
        return descriptor, chain
    except OSError as error:
        os.close(descriptor)
        raise ValueError(
            "anchored path component cannot be opened without following a symlink"
        ) from error
    except Exception:
        os.close(descriptor)
        raise


def _verify_directory_chain(expected: DirectoryChain) -> None:
    descriptor, _ = _open_absolute_directory(Path(expected.path), expected=expected)
    os.close(descriptor)


_SELECTION_IDENTITY_COLUMNS = ("trade_date", "stock_code", "factor_score")


def freeze_top_k_selection(
    aligned: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Select once from scores so neither outcome arm can change membership."""
    frozen = _top_k_selected_rows(aligned, config).reset_index(drop=True)
    if frozen.empty:
        raise ValueError("fixed-score comparison produced no Top-K selections")
    return frozen


def _selection_digest(frame: pd.DataFrame) -> str:
    missing = set(_SELECTION_IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise KeyError(f"selection identity is missing: {sorted(missing)}")
    return _hash_frame(frame, _SELECTION_IDENTITY_COLUMNS)


def validate_selection_identity(
    raw_arm: pd.DataFrame,
    hfq_arm: pd.DataFrame,
) -> tuple[str, str]:
    """Derive each consumed arm's identity independently and require equality."""
    raw_digest = _selection_digest(raw_arm)
    hfq_digest = _selection_digest(hfq_arm)
    if raw_digest != hfq_digest:
        raise AssertionError("frozen Top-K selection identity drifted between arms")
    return raw_digest, hfq_digest


def compare_fixed_scores(
    aligned: pd.DataFrame,
    config: BacktestConfig,
    *,
    min_ic_samples: int = 30,
) -> pd.DataFrame:
    """Compare two outcome bases while freezing scores and Top-K selections."""
    required = {
        "trade_date",
        "stock_code",
        "factor_score",
        "raw_return",
        "hfq_return",
    }
    missing = required - set(aligned.columns)
    if missing:
        raise KeyError(f"fixed-score comparison is missing: {sorted(missing)}")
    if aligned.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("fixed-score comparison contains duplicate event keys")

    frozen = freeze_top_k_selection(aligned, config)
    arms = {
        "raw": frozen.assign(gross_return=frozen["raw_return"]),
        "hfq": frozen.assign(gross_return=frozen["hfq_return"]),
    }
    raw_digest, hfq_digest = validate_selection_identity(arms["raw"], arms["hfq"])
    selection_digests = {"raw": raw_digest, "hfq": hfq_digest}
    rows: list[dict[str, object]] = []
    for basis, return_column in (("raw", "raw_return"), ("hfq", "hfq_return")):
        _, score, target, mask = event_grids(
            aligned,
            "factor_score",
            return_column,
        )
        ic = summarize_ic(
            daily_ic(score, target, mask, min_samples=min_ic_samples)
        )
        metrics, _ = evaluate_top_k_book(
            arms[basis],
            config,
            gross=False,
            overlap=2,
        )
        rows.append(
            {
                "price_basis": basis,
                "d2_close_ic": ic["ic_mean"],
                "net_per_trade": metrics["mean_trade_return"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "final_equity": metrics["final_equity"],
                "n_days": int(metrics["n_days"]),
                "selected_score_digest": selection_digests[basis],
            }
        )
    return pd.DataFrame(rows)


def validate_frozen_boundary(
    aligned: pd.DataFrame,
    *,
    expected_dates: int = EXPECTED_D0_DATES,
    expected_d0_end: str = EXPECTED_D0_END,
    expected_exit_end: str = EXPECTED_EXIT_END,
) -> dict[str, object]:
    """Fail closed if the approved D+2-complete training boundary drifts."""
    required = {"trade_date", "exit_date"}
    missing = required - set(aligned.columns)
    if missing:
        raise KeyError(f"boundary frame is missing: {sorted(missing)}")
    d0_dates = aligned["trade_date"].astype(str)
    exit_dates = aligned["exit_date"].astype(str)
    actual_dates = int(d0_dates.nunique())
    if actual_dates != expected_dates:
        raise AssertionError(
            "D+2-complete date count drifted: "
            f"expected {expected_dates}, got {actual_dates}"
        )
    actual_d0_end = str(d0_dates.max())
    if actual_d0_end != expected_d0_end:
        raise AssertionError(
            f"D0 boundary drifted: expected {expected_d0_end}, got {actual_d0_end}"
        )
    actual_exit_end = str(exit_dates.max())
    if actual_exit_end != expected_exit_end or actual_exit_end > TRAIN_END:
        raise AssertionError(
            "D+2 exit boundary drifted: "
            f"expected {expected_exit_end}, got {actual_exit_end}"
        )
    return {
        "d2_complete_dates": actual_dates,
        "d0_end": actual_d0_end,
        "d2_exit_end": actual_exit_end,
    }


def validate_expected_impact(
    comparison: pd.DataFrame,
    *,
    absolute_tolerance: float = ABS_TOLERANCE,
) -> None:
    """Reject results that differ from the independently audited frozen values."""
    required = {
        "price_basis",
        "d2_close_ic",
        "net_per_trade",
        "cagr",
        "sharpe",
        "final_equity",
        "n_days",
        "selected_score_digest",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise AssertionError(
            f"historical audit tolerance cannot be checked; missing {sorted(missing)}"
        )
    if comparison["price_basis"].tolist() != ["raw", "hfq"]:
        raise AssertionError(
            "historical audit tolerance requires one raw row followed by one hfq row"
        )
    if comparison["selected_score_digest"].nunique() != 1:
        raise AssertionError(
            "historical audit tolerance failed: score selection digest changed by basis"
        )

    expected = {
        ("raw", "d2_close_ic"): EXPECTED_RAW_IC,
        ("hfq", "d2_close_ic"): EXPECTED_HFQ_IC,
        ("raw", "net_per_trade"): EXPECTED_RAW_NET_PER_TRADE,
        ("hfq", "net_per_trade"): EXPECTED_HFQ_NET_PER_TRADE,
        ("raw", "cagr"): EXPECTED_RAW_CAGR,
        ("hfq", "cagr"): EXPECTED_HFQ_CAGR,
        ("raw", "sharpe"): EXPECTED_RAW_SHARPE,
        ("hfq", "sharpe"): EXPECTED_HFQ_SHARPE,
        ("raw", "final_equity"): EXPECTED_RAW_FINAL_EQUITY,
        ("hfq", "final_equity"): EXPECTED_HFQ_FINAL_EQUITY,
    }
    indexed = comparison.set_index("price_basis")
    for (basis, metric), reference in expected.items():
        actual = float(indexed.loc[basis, metric])
        if not np.isclose(
            actual,
            reference,
            rtol=0.0,
            atol=absolute_tolerance,
        ):
            raise AssertionError(
                "historical audit tolerance failed for "
                f"{basis}.{metric}: expected {reference!r}, got {actual!r}"
            )
    if not (indexed["n_days"] == EXPECTED_D0_DATES).all():
        raise AssertionError("historical audit tolerance failed for D+2 date count")
    delta = float(
        indexed.loc["hfq", "net_per_trade"]
        - indexed.loc["raw", "net_per_trade"]
    )
    if not np.isclose(delta, EXPECTED_NET_DELTA, rtol=0.0, atol=absolute_tolerance):
        raise AssertionError(
            "historical audit tolerance failed for net-per-trade delta: "
            f"expected {EXPECTED_NET_DELTA!r}, got {delta!r}"
        )
    if not (indexed["net_per_trade"] < 0).all():
        raise AssertionError(
            "historical audit tolerance failed: adjustment unexpectedly reversed loss"
        )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
    )


def _open_source_file(
    path: Path,
    *,
    expected_chain: DirectoryChain | None = None,
) -> tuple[int, Path, DirectoryChain, os.stat_result]:
    absolute = _absolute_lexical(path)
    parent_descriptor, chain = _open_absolute_directory(
        absolute.parent,
        expected=expected_chain,
    )
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"input snapshot requires a regular file: {absolute}")
    return descriptor, absolute, chain, metadata


def _write_all(descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("private snapshot write made no progress")
        remaining = remaining[written:]


def snapshot_file(path: Path, *, snapshot_root: Path) -> FileSnapshot:
    """Copy and hash the exact bytes read from one anchored source descriptor."""
    source, absolute, chain, before = _open_source_file(path)
    snapshot_path: Path | None = None
    destination: int | None = None
    digest = hashlib.sha256()
    try:
        snapshot_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        snapshot_path = snapshot_root / (
            f"{len(tuple(snapshot_root.iterdir())):04d}-{absolute.name}-"
            f"{secrets.token_hex(6)}"
        )
        destination = os.open(
            snapshot_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        while True:
            chunk = os.read(source, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(destination, chunk)
        os.fsync(destination)
        os.fchmod(destination, 0o400)
        after = os.fstat(source)
    except Exception:
        if snapshot_path is not None:
            with suppress(FileNotFoundError):
                snapshot_path.unlink()
        raise
    finally:
        if destination is not None:
            os.close(destination)
        os.close(source)
    if snapshot_path is None:
        raise RuntimeError("private input snapshot path was not created")
    if not stat.S_ISREG(before.st_mode):
        with suppress(FileNotFoundError):
            snapshot_path.unlink()
        raise RuntimeError(f"input snapshot requires a regular file: {absolute}")
    if _stat_identity(before) != _stat_identity(after):
        snapshot_path.unlink()
        raise RuntimeError(f"input changed while being snapshotted: {absolute}")
    return FileSnapshot(
        path=str(absolute),
        snapshot_path=str(snapshot_path),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mode=int(after.st_mode),
        sha256=digest.hexdigest(),
        source_chain=chain,
    )


def _cache_listing(
    directory: Path,
    *,
    expected_chain: DirectoryChain | None = None,
) -> tuple[tuple[str, ...], DirectoryChain]:
    descriptor, chain = _open_absolute_directory(directory, expected=expected_chain)
    try:
        return tuple(sorted(os.listdir(descriptor))), chain
    finally:
        os.close(descriptor)


def build_input_manifest(
    *,
    input_path: Path,
    library_path: Path,
    price_cache: Path,
    config_path: Path,
    snapshot_root: Path,
) -> InputManifest:
    """Resolve and fingerprint the only file set the audit may consume."""
    cache = _absolute_lexical(price_cache)
    listing, cache_chain = _cache_listing(cache)
    parquet_paths = [
        cache / name
        for name in listing
        if name.endswith(".parquet")
    ]
    training_paths = [
        path
        for path in parquet_paths
        if TRAIN_START.replace("-", "") <= path.stem <= TRAIN_END.replace("-", "")
    ]
    training_calendar = np.asarray(
        [_hyphenated(path.stem) for path in training_paths],
        dtype=str,
    )
    validate_training_calendar(training_calendar)
    prior_paths = [path for path in parquet_paths if path.stem < TRAIN_START.replace("-", "")]
    market_paths = [*prior_paths[-1:], *training_paths]
    calendar = tuple(_hyphenated(path.stem) for path in market_paths)
    manifest = InputManifest(
        event_table=snapshot_file(input_path, snapshot_root=snapshot_root),
        factor_library=snapshot_file(library_path, snapshot_root=snapshot_root),
        backtest_config=snapshot_file(config_path, snapshot_root=snapshot_root),
        comparison_script=snapshot_file(Path(__file__), snapshot_root=snapshot_root),
        price_cache=cache,
        price_cache_listing=listing,
        price_cache_chain=cache_chain,
        market_files=tuple(
            snapshot_file(path, snapshot_root=snapshot_root) for path in market_paths
        ),
        market_calendar=calendar,
    )
    final_listing, _ = _cache_listing(cache, expected_chain=cache_chain)
    if final_listing != listing:
        raise RuntimeError("price cache listing changed while manifest was built")
    return manifest


def _assert_snapshot_unchanged(label: str, expected: FileSnapshot) -> None:
    try:
        descriptor, _, _, metadata = _open_source_file(
            Path(expected.path),
            expected_chain=expected.source_chain,
        )
        digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
    except (FileNotFoundError, RuntimeError, OSError) as error:
        raise RuntimeError(f"{label} changed after input snapshot") from error
    actual_identity = _stat_identity(metadata)
    expected_identity = (
        expected.size,
        expected.mtime_ns,
        expected.device,
        expected.inode,
        expected.mode,
    )
    if actual_identity != expected_identity or digest.hexdigest() != expected.sha256:
        raise RuntimeError(f"{label} changed after input snapshot")


def verify_input_manifest(manifest: InputManifest) -> None:
    """Re-stat and re-hash every consumed byte, including cache membership."""
    initial_listing, _ = _cache_listing(
        manifest.price_cache,
        expected_chain=manifest.price_cache_chain,
    )
    if initial_listing != manifest.price_cache_listing:
        raise RuntimeError("price cache listing changed after input snapshot")
    for label, record in (
        ("event_table", manifest.event_table),
        ("factor_library", manifest.factor_library),
        ("backtest_config", manifest.backtest_config),
        ("comparison_script", manifest.comparison_script),
    ):
        _assert_snapshot_unchanged(label, record)
    for record in manifest.market_files:
        _assert_snapshot_unchanged(f"price cache file {Path(record.path).name}", record)
    final_listing, _ = _cache_listing(
        manifest.price_cache,
        expected_chain=manifest.price_cache_chain,
    )
    if final_listing != manifest.price_cache_listing:
        raise RuntimeError("price cache listing changed during post-computation verification")


def _load_training_market_manifest(
    manifest: InputManifest,
    event_codes: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load exactly the market files frozen in ``manifest``; never re-enumerate."""
    columns = ["trade_date", "ts_code", "open", "high", "close", "adj_factor"]
    frames = []
    for record in manifest.market_files:
        frame = pd.read_parquet(record.snapshot_path, columns=columns)
        frames.append(frame.loc[frame["ts_code"].astype(str).isin(event_codes)])
    market = pd.concat(frames, ignore_index=True)
    if market.empty:
        raise ValueError("training market cache has no matching event stocks")
    return market, np.asarray(manifest.market_calendar, dtype=str)


def _snapshot_payload(record: FileSnapshot) -> dict[str, object]:
    return {
        "path": record.path,
        "sha256": record.sha256,
        "size": record.size,
        "mtime_ns": record.mtime_ns,
        "device": record.device,
        "inode": record.inode,
    }


def collect_input_fingerprints(
    manifest: InputManifest,
) -> dict[str, dict[str, object]]:
    """Render fingerprints from the pre-consumption immutable manifest only."""
    market_digest = hashlib.sha256()
    for record in manifest.market_files:
        market_digest.update(Path(record.path).name.encode())
        market_digest.update(b"\0")
        market_digest.update(record.sha256.encode())
        market_digest.update(b"\n")
    listing_digest = hashlib.sha256(
        "\n".join(manifest.price_cache_listing).encode()
    ).hexdigest()
    return {
        "event_table": _snapshot_payload(manifest.event_table),
        "factor_library": _snapshot_payload(manifest.factor_library),
        "backtest_config": _snapshot_payload(manifest.backtest_config),
        "price_cache": {
            "path": str(manifest.price_cache),
            "sha256": market_digest.hexdigest(),
            "file_count": len(manifest.market_files),
            "listing_sha256": listing_digest,
            "listing_count": len(manifest.price_cache_listing),
        },
        "comparison_script": _snapshot_payload(manifest.comparison_script),
    }


def validate_legacy_reconstruction(summary: dict[str, object]) -> dict[str, object]:
    """Require the persisted legacy return label to match reconstructed raw prices."""
    if summary.get("event_returns_match_raw") is not True:
        raise AssertionError("legacy event returns do not match raw price reconstruction")
    rounding_value = summary.get("event_return_rounding_error_max")
    if not isinstance(rounding_value, (int, float, np.integer, np.floating)):
        raise AssertionError("legacy event return rounding error is missing or invalid")
    rounding_error = float(rounding_value)
    if not np.isfinite(rounding_error) or rounding_error > 1e-6:
        raise AssertionError(
            "legacy event return rounding error exceeds 1e-6 reconstruction tolerance"
        )
    return summary


def _git_output(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def collect_provenance() -> dict[str, object]:
    """Record executable source, repository state, and numerical runtime versions."""
    report_relative = DESIGNATED_REPORT.relative_to(ROOT).as_posix()
    head = _git_output("rev-parse", "HEAD").decode().strip()
    head_tree = _git_output("rev-parse", "HEAD^{tree}").decode().strip()
    diff = _git_output(
        "diff",
        "--binary",
        "HEAD",
        "--",
        ".",
        f":(exclude){report_relative}",
    )
    status_lines = _git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).decode().splitlines()
    status_lines = [line for line in status_lines if report_relative not in line]
    status = "\n".join(status_lines).encode()
    source_paths = (
        Path(__file__),
        ROOT / "scripts/gp000_loss_attribution.py",
        ROOT / "helix/config.py",
        ROOT / "helix/eval/backtest.py",
        ROOT / "helix/eval/ic.py",
        ROOT / "helix/gp/library.py",
    )
    return {
        "git": {
            "head": head,
            "head_tree": head_tree,
            "dirty_excluding_generated_report": bool(status_lines),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        },
        "source_modules": {
            str(path.relative_to(ROOT)): _hash_file(path) for path in source_paths
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def build_structured_output(
    comparison: pd.DataFrame,
    *,
    boundary: dict[str, object],
    inputs: dict[str, dict[str, object]],
    legacy_reconstruction: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, Any]:
    """Build strict JSON evidence without writing any artifact."""
    indexed = comparison.set_index("price_basis")
    raw_net = float(indexed.loc["raw", "net_per_trade"])
    hfq_net = float(indexed.loc["hfq", "net_per_trade"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "fixed-score legacy-before versus governed-HFQ-outcome after",
        "formal_factor": {
            "name": FORMAL_FACTOR,
            "expression": FORMAL_EXPRESSION,
            "retrained": False,
            "score_basis_changed_between_arms": False,
        },
        "training_window": {
            "nominal_start": TRAIN_START,
            "nominal_end": TRAIN_END,
            **boundary,
        },
        "inputs": inputs,
        "input_manifest_verified_after_computation": True,
        "provenance": provenance,
        "legacy_reconstruction": json_ready(legacy_reconstruction),
        "legacy_unverified_lineage": True,
        "historical_reports_rewritten": False,
        "loss_conclusion_unchanged": bool(raw_net < 0 and hfq_net < 0),
        "target_mismatch_remains_dominant": True,
        "adjustment_mismatch_is_core_loss_cause": False,
        "net_per_trade_delta": hfq_net - raw_net,
        "comparison": json_ready(comparison),
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def build_baseline_evidence(
    *,
    input_path: Path,
    library_path: Path,
    price_cache: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Run the read-only legacy adapter and return validated structured evidence."""
    with tempfile.TemporaryDirectory(prefix="helix-adjustment-snapshot-") as temporary:
        manifest = build_input_manifest(
            input_path=input_path,
            library_path=library_path,
            price_cache=price_cache,
            config_path=config_path,
            snapshot_root=Path(temporary),
        )
        config = load_audit_config(Path(manifest.backtest_config.snapshot_path))
        library = load_factors(Path(manifest.factor_library.snapshot_path))
        events = load_training_events(Path(manifest.event_table.snapshot_path), library)
        market, calendar = _load_training_market_manifest(
            manifest,
            set(events["stock_code"].astype(str)),
        )
        prices = build_price_lookup(market, calendar, events["stock_code"].unique())
        adjustment_summary, aligned = audit_adjustment_chain(events, prices)
        validate_legacy_reconstruction(adjustment_summary)
        boundary = validate_frozen_boundary(aligned)
        comparison = compare_fixed_scores(aligned, config)
        validate_expected_impact(comparison)
        verify_input_manifest(manifest)
        inputs = collect_input_fingerprints(manifest)
        return build_structured_output(
            comparison,
            boundary=boundary,
            inputs=inputs,
            legacy_reconstruction=adjustment_summary,
            provenance=collect_provenance(),
        )


def _metric_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparison = payload["comparison"]
    return {str(row["price_basis"]): row for row in comparison}


def render_report(payload: dict[str, Any]) -> str:
    """Render the validated fixed-score evidence as the new repair document."""
    rows = _metric_rows(payload)
    raw = rows["raw"]
    hfq = rows["hfq"]
    boundary = payload["training_window"]
    inputs = payload["inputs"]
    reconstruction = payload["legacy_reconstruction"]
    provenance = payload["provenance"]
    git_provenance = provenance["git"]
    runtime = provenance["runtime"]
    digest = str(raw["selected_score_digest"])

    def pct(value: float) -> str:
        return f"{value:.6%}"

    def input_row(name: str, label: str) -> str:
        item = inputs.get(name)
        if item is None:
            return f"| {label} | 测试内存输入 | 不适用 |"
        return f"| {label} | `{item['path']}` | `{item['sha256']}` |"

    report = f"""# 后复权基线统一修复说明

**生成日期：** 2026-08-14

**生成入口：** `scripts/adjustment_unification_baseline.py`

**证据性质：** 固定 `gp_000` 分数与选股的只读 before/after 对照；不重训、不重新挖掘、不覆盖历史产物。

## 执行摘要

**复权口径问题存在，但不是核心或主导亏损原因。** 同一组冻结的 `gp_000` 分数和 Top4 选择从 legacy raw outcome 切换为同日点时 HFQ outcome 后，单笔净收益仅从 {pct(float(raw['net_per_trade']))} 改善至 {pct(float(hfq['net_per_trade']))}，变化 {pct(float(payload['net_per_trade_delta']))}，收益仍为负。

**目标错配是主导亏损原因。** `gp_000` 的历史准入目标与 D+2 收盘净收益目标错配；本次工作只建立可审计的合规复权基线，不修复老因子的盈利能力。后续新一代 GP 因子必须从带四元血缘的 HFQ 新链路生成。

现有 `gp_000` 成品特征与 event 表仅作为“修复前 legacy 基线”保留。其上游价格口径和 `source_date/as_of_time/price_basis/adj_factor_version` 不可追溯，本报告不将其补写为已验证 raw 或 HFQ，也不改写任何既有实验结论。详见[治理台账 D10](../factor-governance.md)与[既有专项审计](gp000_loss_attribution.md)。

## 修复合同

- 新链路中，跨日价格因子、标签、成交计价与收益核算只接受带四元血缘的点时 HFQ 价格，校验失败即终止。
- 原始 OHLC 只用于同日涨跌停状态特征（涨跌停距离与计数）以及涨跌停/可成交性校验，不参与跨日价格因子、标签值或持仓收益核算。
- 本对照是 legacy baseline adapter：只读取既有 event 特征、正式因子库和行情缓存，并复用专项审计的纯计算函数；不调用历史报告写入器。
- 两个对照臂共享同一分数、同一 Top4 选择和同一成本/滑点配置，仅 outcome 价格口径不同。

## 固定窗口与 D+2 边界

| 项目 | 值 |
| --- | ---: |
| 名义训练窗 | {boundary['nominal_start']} 至 {boundary['nominal_end']} |
| D+2 完整 D0 数 | {boundary['d2_complete_dates']} |
| 最后 D0 | {boundary['d0_end']} |
| 最后 D+2 退出日 | {boundary['d2_exit_end']} |
| Top4 冻结选择摘要 | `{digest}` |

最后两个没有完整 D+2 outcome 的 D0 被严格排除；任何 D0 数量、最后 D0 或退出日变化都会触发 fail-closed，不会自动更新基线。

## Legacy outcome 重建闸门

| 校验 | 原始统计值 | 要求 |
| --- | ---: | --- |
| event return 与 raw 重建一致 | `{str(reconstruction['event_returns_match_raw']).lower()}` | 必须为 `true` |
| 最大舍入误差 | {float(reconstruction['event_return_rounding_error_max']):.12g} | 不超过 `1e-6` |

该闸门只证明本次 legacy event outcome 能由指定 raw 行情缓存重建；不证明 legacy 因子特征的上游复权口径或四元血缘。

## gp_000 修复前后核心指标

| 指标 | 修复前：legacy raw outcome | 修复后：点时 HFQ outcome | 变化 |
| --- | ---: | ---: | ---: |
| D+2 close IC | {float(raw['d2_close_ic']):.10f} | {float(hfq['d2_close_ic']):.10f} | {float(hfq['d2_close_ic']) - float(raw['d2_close_ic']):+.10f} |
| Top4 单笔净收益 | {pct(float(raw['net_per_trade']))} | {pct(float(hfq['net_per_trade']))} | {pct(float(hfq['net_per_trade']) - float(raw['net_per_trade']))} |
| 年化 Sharpe | {float(raw['sharpe']):.6f} | {float(hfq['sharpe']):.6f} | {float(hfq['sharpe']) - float(raw['sharpe']):+.6f} |
| CAGR（补充） | {pct(float(raw['cagr']))} | {pct(float(hfq['cagr']))} | {pct(float(hfq['cagr']) - float(raw['cagr']))} |
| 期末净值（补充） | {float(raw['final_equity']):.6f} | {float(hfq['final_equity']):.6f} | {float(hfq['final_equity']) - float(raw['final_equity']):+.6f} |

该变化与既有专项审计预估一致：Top4 单笔净收益改善约 `0.022226` 个百分点，修复后约 `-0.5233%`，未逆转亏损结论。

## 输入追溯

| 输入 | 绝对路径 | SHA-256/集合摘要 |
| --- | --- | --- |
{input_row('event_table', 'legacy event 表')}
{input_row('factor_library', '正式 gp_000 因子库')}
{input_row('price_cache', 'D+2 行情缓存')}
{input_row('backtest_config', '成本与 Top4 配置')}
{input_row('comparison_script', '本报告生成脚本')}

所有输入通过从可信 `/` 开始、逐组件 `O_NOFOLLOW` 的源 fd 复制到私有只读快照；SHA-256 与解析器消费的是同一份复制字节。行情只从该 manifest 的快照文件集合加载。计算结束后脚本重新枚举源缓存，并对全部源文件重新 stat 与 SHA-256 校验；任何内容、身份或成员变化都会终止发布。

## 运行来源

| 项目 | 值 |
| --- | --- |
| Git HEAD | `{git_provenance['head']}` |
| Git HEAD tree | `{git_provenance['head_tree']}` |
| 排除生成报告后的工作树是否脏 | `{str(git_provenance['dirty_excluding_generated_report']).lower()}` |
| 工作树 diff SHA-256 | `{git_provenance['diff_sha256']}` |
| 工作树 status SHA-256 | `{git_provenance['status_sha256']}` |
| Python | `{runtime['python']}` |
| NumPy | `{runtime['numpy']}` |
| pandas | `{runtime['pandas']}` |

严格 JSON 还记录了比较脚本、专项审计、配置、回测、IC 与因子库模块的逐文件 SHA-256。

CLI 标准输出同时提供严格 JSON，其中 `legacy_unverified_lineage=true`、`historical_reports_rewritten=false`、`loss_conclusion_unchanged=true`。这些标志防止把 outcome 修正误解为对 legacy 特征血缘或盈利能力的认证。

## 限制与后续基线

- “修复前 raw”只描述 event outcome 与历史行情缓存的重建关系，不代表 legacy 特征的上游价格口径已获认证。
- 本次没有重新训练、调参或改变正式因子方向；分数由冻结的正式表达式和既有成品特征确定，两个对照臂不重新选股。
- 本报告不替代、不覆盖 `docs/risk/gp000_loss_attribution.md`，也不修改历史 artifacts。
- 新一代 GP 因子不得通过 legacy adapter 进入正式训练；必须使用新的 HFQ 血缘强契约链路。
- 报告发布与输入完整性实现依赖 POSIX `dir_fd`、`O_DIRECTORY`、`O_NOFOLLOW` 和 descriptor-relative rename；不具备这些能力的平台会在读取或写入前明确终止。
"""
    return report


def prevalidate_report_target(path: Path) -> ReportTarget:
    """Bind the only allowed report path to a no-follow ancestor identity chain."""
    if ".." in path.expanduser().parts:
        raise ValueError(
            "report output must be the designated adjustment report without traversal"
        )
    target = _absolute_lexical(path)
    designated = _absolute_lexical(DESIGNATED_REPORT)
    if target != designated:
        raise ValueError("report output must be the designated adjustment report")
    parent_descriptor, parent_chain = _open_absolute_directory(target.parent)
    try:
        try:
            existing = os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("designated adjustment report must be a regular file")
    finally:
        os.close(parent_descriptor)
    return ReportTarget(target, parent_chain)


def _existing_report_mode(parent_descriptor: int, filename: str) -> int:
    try:
        existing = os.stat(
            filename,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return 0o644
    if not stat.S_ISREG(existing.st_mode):
        raise ValueError("designated adjustment report must be a regular file")
    return stat.S_IMODE(existing.st_mode)


def publish_report(report: str, path: Path | ReportTarget) -> None:
    """Atomically publish only the one designated worktree repair report."""
    validated = path if isinstance(path, ReportTarget) else prevalidate_report_target(path)
    target = validated.path
    parent_descriptor, _ = _open_absolute_directory(
        target.parent,
        expected=validated.parent_chain,
    )
    temporary_name = f".{target.name}.{secrets.token_hex(12)}.tmp"
    temporary_descriptor: int | None = None
    try:
        mode = _existing_report_mode(parent_descriptor, target.name)
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(temporary_descriptor, mode)
        with os.fdopen(temporary_descriptor, "w", encoding="utf-8") as handle:
            temporary_descriptor = None
            handle.write(report)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_directory_chain(validated.parent_chain)
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--price-cache", type=Path, default=DEFAULT_PRICE_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report_target = prevalidate_report_target(args.report)
    payload = build_baseline_evidence(
        input_path=args.input,
        library_path=args.library,
        price_cache=args.price_cache,
        config_path=args.config,
    )
    publish_report(render_report(payload), report_target)
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
