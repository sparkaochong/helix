#!/usr/bin/env python3
"""Build a governed, point-in-time HFQ event table for the next-generation GP mining round.

The legacy ``data/raw/argus_quant_working.parquet`` carries 459 feature columns with no
recoverable generation formula or price-basis lineage, so it cannot satisfy
``helix.data.event_table.open_event_source``'s fail-closed lineage gate: every governed
column (features *and* labels) must declare a verifiable ``source_date`` / ``as_of_time`` /
``price_basis="hfq"`` / ``adj_factor_version`` audit, and the version must agree across the
whole table. Rather than fabricate that provenance for an opaque legacy artifact, this script
builds a small, fully self-consistent event table from scratch, reusing the same governed
building blocks the ordinary panel pipeline already uses:

* :func:`helix.data.panel.build_panel` -- the HFQ price panel and its adjustment stamp.
* :func:`helix.data.universe.build_universe` -- the point-in-time eligible-stock mask.
* :func:`helix.features.base_fields.compute_base_fields` -- ~25 D0-close-observable features.

Labels follow the same D+1-open-entry / D+2-high-touch / D+2-close-exit convention as
``helix.labels.touch_label.build_touch_label``: ``target_ratio`` is read straight from
``LabelConfig`` for the primary hit column, with 3%/5% companions at fixed ratios purely
for parity with the legacy schema (leakage-guarded by the ``label_`` prefix either way).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from helix.config import PROJECT_ROOT, Config
from helix.data.panel import build_panel
from helix.data.store import ParquetStore
from helix.data.universe import build_universe
from helix.features.base_fields import compute_base_fields
from helix.features.operators import lead
from helix.logging_setup import get_logger, setup_logging

log = get_logger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/argus_quant_working_hfq.parquet"
DEFAULT_LINEAGE = PROJECT_ROOT / "data/raw/argus_quant_working_hfq_lineage.json"
DEFAULT_CALENDAR = PROJECT_ROOT / "data/raw/trade_cal.parquet"

ADDITIONAL_HIT_RATIOS = {"label_d2_hit_3pct_hfq": 0.03, "label_d2_hit_5pct_hfq": 0.05}


def _hyphenate(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def _as_of(source_date: str) -> str:
    return f"{source_date}T15:00:00+08:00"


def _lead_dates(dates: np.ndarray, offset: int) -> np.ndarray:
    """Same shift as ``helix.features.operators.lead``, applied to the date axis itself."""
    out = np.full(dates.shape, "", dtype=object)
    if offset < len(dates):
        out[: len(dates) - offset] = dates[offset:]
    return out


def build_event_frame(panel, universe_mask: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Flatten the panel's universe-eligible cells into one row per (D0 date, stock)."""
    adjustment = panel.require_adjusted_prices(
        ("open_hfq", "high_hfq", "close_hfq"), "build_argus_hfq_event_table"
    )
    entry_off, touch_off = cfg.label.entry_offset, cfg.label.touch_offset
    target_ratio = cfg.label.target_ratio

    open_entry = lead(panel.f64("open_hfq"), entry_off)
    high_touch = lead(panel.f64("high_hfq"), touch_off)
    close_exit = lead(panel.f64("close_hfq"), touch_off)
    peak_return = high_touch / open_entry - 1.0
    close_return = close_exit / open_entry - 1.0

    base_fields = compute_base_fields(panel)
    feature_names = sorted(base_fields)

    d1_dates = _lead_dates(panel.dates, entry_off)
    d2_dates = _lead_dates(panel.dates, touch_off)

    rows_t, rows_n = np.nonzero(universe_mask)
    label_valid = (
        np.isfinite(open_entry[rows_t, rows_n])
        & (open_entry[rows_t, rows_n] > 0)
        & np.isfinite(high_touch[rows_t, rows_n])
        & np.isfinite(close_exit[rows_t, rows_n])
        & (d1_dates[rows_t] != "")
        & (d2_dates[rows_t] != "")
    )
    rows_t, rows_n = rows_t[label_valid], rows_n[label_valid]
    log.info("universe cells: %d eligible, %d with a complete D+%d label",
              int(universe_mask.sum()), len(rows_t), touch_off)

    d0 = panel.dates[rows_t]
    d1 = d1_dates[rows_t]
    d2 = d2_dates[rows_t]
    entry = open_entry[rows_t, rows_n]
    high = high_touch[rows_t, rows_n]
    close = close_exit[rows_t, rows_n]
    peak = peak_return[rows_t, rows_n]
    ret = close_return[rows_t, rows_n]

    data: dict[str, np.ndarray] = {
        "trade_date": np.array([_hyphenate(d) for d in d0], dtype=object),
        "stock_code": panel.codes[rows_n],
    }
    for name in feature_names:
        data[name] = base_fields[name][rows_t, rows_n]

    data["label_px_d1_open_hfq"] = entry
    data["label_px_d2_high_hfq"] = high
    data["label_px_d2_close_hfq"] = close
    data["label_d2_peak_return_hfq"] = peak
    data["label_d2_return_hfq"] = ret
    data["label_d2_hit_8pct_hfq"] = (peak >= target_ratio - 1.0).astype(np.float64)
    for column, ratio in ADDITIONAL_HIT_RATIOS.items():
        data[column] = (peak >= ratio).astype(np.float64)

    d1_hyphen = np.array([_hyphenate(d) for d in d1], dtype=object)
    d2_hyphen = np.array([_hyphenate(d) for d in d2], dtype=object)
    version = adjustment.adj_factor_version

    data["audit_f_source_date"] = data["trade_date"]
    data["audit_f_as_of_time"] = np.array([_as_of(d) for d in data["trade_date"]], dtype=object)
    data["audit_f_price_basis"] = np.full(len(rows_t), "hfq", dtype=object)
    data["audit_f_adj_factor_version"] = np.full(len(rows_t), version, dtype=object)

    data["audit_d1_source_date"] = d1_hyphen
    data["audit_d1_as_of_time"] = np.array([_as_of(d) for d in d1_hyphen], dtype=object)
    data["audit_d1_price_basis"] = np.full(len(rows_t), "hfq", dtype=object)
    data["audit_d1_adj_factor_version"] = np.full(len(rows_t), version, dtype=object)

    data["audit_d2_source_date"] = d2_hyphen
    data["audit_d2_as_of_time"] = np.array([_as_of(d) for d in d2_hyphen], dtype=object)
    data["audit_d2_price_basis"] = np.full(len(rows_t), "hfq", dtype=object)
    data["audit_d2_adj_factor_version"] = np.full(len(rows_t), version, dtype=object)

    frame = pd.DataFrame(data).sort_values(["trade_date", "stock_code"], kind="stable")
    return frame.reset_index(drop=True)


def build_lineage_manifest(feature_names: list[str]) -> dict[str, object]:
    f_group = {
        "source_date": "audit_f_source_date", "as_of_time": "audit_f_as_of_time",
        "price_basis": "audit_f_price_basis", "adj_factor_version": "audit_f_adj_factor_version",
        "horizon": 0,
    }
    d1_group = {
        "source_date": "audit_d1_source_date", "as_of_time": "audit_d1_as_of_time",
        "price_basis": "audit_d1_price_basis", "adj_factor_version": "audit_d1_adj_factor_version",
        "horizon": 1,
    }
    d2_group = {
        "source_date": "audit_d2_source_date", "as_of_time": "audit_d2_as_of_time",
        "price_basis": "audit_d2_price_basis", "adj_factor_version": "audit_d2_adj_factor_version",
        "horizon": 2,
    }
    d2_labels = (
        "label_px_d2_high_hfq", "label_px_d2_close_hfq",
        "label_d2_peak_return_hfq", "label_d2_return_hfq",
        "label_d2_hit_8pct_hfq", *ADDITIONAL_HIT_RATIOS,
    )
    fields = {name: dict(f_group) for name in feature_names}
    fields["label_px_d1_open_hfq"] = dict(d1_group)
    for name in d2_labels:
        fields[name] = dict(d2_group)
    return {"schema_version": 1, "fields": fields}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--start-date", default=None, help="Overrides data.start_date for the panel build")
    ap.add_argument("--end-date", default=None, help="Overrides data.end_date for the panel build")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    args = ap.parse_args()

    setup_logging()
    cfg = Config.load(args.config)
    start = args.start_date or cfg.data.start_date
    end = args.end_date or cfg.data.end_date
    store = ParquetStore(cfg.data.root)

    log.info("building HFQ panel %s..%s from %s", start, end or "(latest)", cfg.data.root)
    panel = build_panel(store, start, end)
    universe_mask = build_universe(panel, store, cfg.universe)
    frame = build_event_frame(panel, universe_mask, cfg)

    feature_names = sorted(compute_base_fields(panel))
    manifest = build_lineage_manifest(feature_names)

    _atomic_parquet(args.output, frame)
    _atomic_text(args.lineage, json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info(
        "wrote %d rows x %d columns to %s (lineage: %s); calendar: %s",
        len(frame), frame.shape[1], args.output, args.lineage, DEFAULT_CALENDAR,
    )
    print(json.dumps({
        "output": str(args.output),
        "lineage": str(args.lineage),
        "calendar": str(DEFAULT_CALENDAR),
        "rows": len(frame),
        "feature_columns": len(feature_names),
        "date_range": [str(frame["trade_date"].min()), str(frame["trade_date"].max())],
    }, indent=2))


if __name__ == "__main__":
    main()
