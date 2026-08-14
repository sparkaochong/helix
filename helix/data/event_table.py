"""Long-format event tables reshaped into a slot panel.

The argus_quant dataset is one row per ``(trade_date, stock_code)`` that passed a
pre-market screen -- roughly 500 of 5000 names on any given day. Materialising that as
a true ``(dates x all stocks)`` panel would be ~95% NaN and cost gigabytes per field,
so instead each date's rows are packed into slots ``0..n_t-1`` of a ``(T, N_max)`` grid.

**The slot index is not a stock.** Slot 3 is a different company on different dates.
That makes every cross-sectional operation valid (they reduce along axis 1, within a
date) and every time-series operation meaningless -- ``ts_mean`` over a slot column
would average unrelated companies. :func:`helix.gp.event_primitives.build_event_pset`
is the only sanctioned way to build a primitive set for this layout, and it refuses to
include windowed operators for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from .event_lineage import (
    EventLineageError,
    audit_column_names,
    load_event_calendar,
    load_event_lineage,
    require_independent_event_calendar,
    validate_event_fields,
    validate_event_schema,
)

log = get_logger(__name__)

DATE_COLUMN = "trade_date"
CODE_COLUMN = "stock_code"


@dataclass
class EventPanel:
    """Date-major slot grid. Every array is ``(T, N_max)`` and aligned to ``occupied``."""

    dates: np.ndarray                  # (T,) sorted trade dates
    codes: np.ndarray                  # (T, N_max) stock code per slot, "" where empty
    occupied: np.ndarray               # (T, N_max) bool, True where a real row sits
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    @property
    def n_rows(self) -> int:
        return int(self.occupied.sum())

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.fields:
            return self.fields[name]
        if name in self.labels:
            return self.labels[name]
        raise KeyError(f"unknown column {name!r}")

    def f64(self, name: str) -> np.ndarray:
        return np.asarray(self[name], dtype=np.float64)

    def field_names(self) -> list[str]:
        """Stable ordering -- GP terminals are positional, so this must never be a set."""
        return sorted(self.fields)

    def select_fields(self, names: list[str]) -> EventPanel:
        missing = [n for n in names if n not in self.fields]
        if missing:
            raise KeyError(f"unknown fields: {missing}")
        return EventPanel(
            dates=self.dates,
            codes=self.codes,
            occupied=self.occupied,
            fields={n: self.fields[n] for n in names},
            labels=dict(self.labels),
        )

    def to_long(self, columns: dict[str, np.ndarray]) -> pd.DataFrame:
        """Flatten slot grids back to one row per event, keyed by date and code."""
        rows, slots = np.nonzero(self.occupied)
        out = {
            DATE_COLUMN: self.dates[rows],
            CODE_COLUMN: self.codes[rows, slots],
        }
        for name, grid in columns.items():
            out[name] = grid[rows, slots]
        return pd.DataFrame(out)


@dataclass
class SlotIndex:
    """Row-to-slot mapping, built once and reused to pack columns one at a time.

    Packing every column of a wide table at once is what makes this layout expensive
    (459 columns x 1083 dates x 2656 slots is ~5GB at float32). Holding just the index
    lets callers stream columns through, keep the few they want, and drop the rest.
    """

    dates: np.ndarray
    codes: np.ndarray
    occupied: np.ndarray
    row_order: np.ndarray   # positions into the *deduplicated, sorted* frame
    date_pos: np.ndarray
    slot: np.ndarray
    audit_columns: frozenset[str] = field(default_factory=frozenset)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    def pack(self, values: np.ndarray) -> np.ndarray:
        """Pack one column (already in deduplicated, sorted row order) into a slot grid."""
        grid = np.full(self.shape, np.nan, dtype=np.float32)
        grid[self.date_pos, self.slot] = values
        return grid


def normalize_frame(
    frame: pd.DataFrame, date_column: str = DATE_COLUMN, code_column: str = CODE_COLUMN
) -> pd.DataFrame:
    """Deduplicate on ``(date, code)`` keeping the last row, then sort deterministically."""
    df = frame.copy()
    df[date_column] = df[date_column].astype(str)
    df[code_column] = df[code_column].astype(str)
    before = len(df)
    df = df.drop_duplicates([date_column, code_column], keep="last")
    if before != len(df):
        log.info("dropped %d duplicate (date, code) rows", before - len(df))
    return df.sort_values([date_column, code_column], kind="stable").reset_index(drop=True)


def build_slot_index(
    frame: pd.DataFrame, date_column: str = DATE_COLUMN, code_column: str = CODE_COLUMN
) -> SlotIndex:
    """Compute the slot layout from an already-normalised frame."""
    dates, date_pos = np.unique(frame[date_column].to_numpy(), return_inverse=True)
    counts = np.bincount(date_pos, minlength=len(dates))
    n_max = int(counts.max())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(frame)) - np.repeat(starts, counts)

    occupied = np.zeros((len(dates), n_max), dtype=bool)
    occupied[date_pos, slot] = True
    codes = np.full((len(dates), n_max), "", dtype=object)
    codes[date_pos, slot] = frame[code_column].to_numpy()

    log.info(
        "slot layout: %d dates x %d slots (median %d rows/day, %d rows, %.1f%% occupied)",
        len(dates), n_max, int(np.median(counts)), len(frame),
        100 * occupied.mean(),
    )
    return SlotIndex(
        dates=dates, codes=codes, occupied=occupied,
        row_order=np.arange(len(frame)), date_pos=date_pos, slot=slot,
    )


def build_event_panel(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    date_column: str = DATE_COLUMN,
    code_column: str = CODE_COLUMN,
) -> EventPanel:
    """Pack a long dataframe into an :class:`EventPanel`.

    Duplicate ``(date, code)`` rows are dropped keeping the last occurrence -- the merged
    argus_quant table carries a few hundred of them from overlapping rolling windows.
    """
    missing = [c for c in (*feature_columns, *label_columns) if c not in frame.columns]
    if missing:
        raise KeyError(f"columns absent from the frame: {missing[:10]}")

    df = normalize_frame(frame, date_column, code_column)
    index = build_slot_index(df, date_column, code_column)

    def pack(column: str) -> np.ndarray:
        return index.pack(pd.to_numeric(df[column], errors="coerce").to_numpy())

    return EventPanel(
        dates=index.dates,
        codes=index.codes,
        occupied=index.occupied,
        fields={c: pack(c) for c in feature_columns},
        labels={c: pack(c) for c in label_columns},
    )


def load_event_panel(
    path: Path,
    label_columns: list[str],
    feature_columns: list[str] | None = None,
    meta_columns: tuple[str, ...] = (DATE_COLUMN, CODE_COLUMN),
    *,
    lineage_path: Path | str | None = None,
    calendar_path: Path | str | None = None,
    train_end: str | None = None,
) -> EventPanel:
    """Read and govern a parquet event table before packing selected numeric fields."""
    require_independent_event_calendar(path, calendar_path)
    manifest = load_event_lineage(lineage_path)
    audit_columns = audit_column_names(manifest)
    if feature_columns is not None:
        leaked_audits = sorted(set(feature_columns) & audit_columns)
        if leaked_audits:
            raise EventLineageError(
                f"feature_columns cannot include event audit columns: {leaked_audits}"
            )
    if feature_columns is None:
        feature_columns = numeric_feature_columns(
            path,
            label_columns,
            meta_columns,
            extra_excluded=tuple(audit_columns),
        )
    assert_no_label_columns(feature_columns)
    requested = list(dict.fromkeys([*feature_columns, *label_columns]))
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    validate_event_schema(schema_names, manifest, requested)
    required_audits = audit_column_names({field: manifest[field] for field in requested})
    needs_calendar = any(manifest[field].horizon > 0 for field in requested)
    calendar = (
        load_event_calendar(calendar_path)
        if calendar_path is not None or needs_calendar
        else None
    )
    columns = list(dict.fromkeys([*meta_columns, *requested, *sorted(required_audits)]))
    frame = pd.read_parquet(path, columns=columns)
    validate_event_fields(
        frame, manifest, requested, calendar=calendar, train_end=train_end
    )
    return build_event_panel(frame, feature_columns, label_columns)


#: Any column whose name starts with one of these is an outcome, never an input.
LABEL_PREFIXES: tuple[str, ...] = ("label", "target", "y_", "fwd_", "future_")


def is_label_column(name: str, prefixes: tuple[str, ...] = LABEL_PREFIXES) -> bool:
    """Outcome columns are excluded by **prefix**, not by an enumerated list.

    Listing them individually is how a leak gets in: the argus_quant table carries
    ``label_d2_hit_3pct`` and ``label_d2_hit_5pct`` alongside the 8% target, and any one
    of them left in the feature set predicts the others almost perfectly. A factor built
    on one scores IC > 0.6 and is worth exactly nothing.
    """
    lowered = name.lower()
    return any(lowered.startswith(p) for p in prefixes)


def assert_no_label_columns(names: list[str], prefixes: tuple[str, ...] = LABEL_PREFIXES) -> None:
    leaked = [n for n in names if is_label_column(n, prefixes)]
    if leaked:
        raise AssertionError(
            f"outcome columns reached the feature set: {leaked}. "
            "Any label-derived input makes the reported IC meaningless."
        )


def numeric_feature_columns(
    path: Path,
    label_columns: list[str] | None = None,
    meta_columns: tuple[str, ...] = (DATE_COLUMN, CODE_COLUMN),
    extra_excluded: tuple[str, ...] = (),
    label_prefixes: tuple[str, ...] = LABEL_PREFIXES,
) -> list[str]:
    """Numeric feature column names in a parquet file, without reading the data."""
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    excluded = set(meta_columns) | set(label_columns or ()) | set(extra_excluded)
    names = [
        name
        for name in schema.names
        if name not in excluded
        and not is_label_column(name, label_prefixes)
        and any(k in str(schema.field(name).type) for k in ("int", "float", "double"))
    ]
    assert_no_label_columns(names, label_prefixes)
    return names


def open_event_source(
    path: Path,
    label_columns: list[str],
    date_column: str = DATE_COLUMN,
    code_column: str = CODE_COLUMN,
    *,
    lineage_path: Path | str | None = None,
    calendar_path: Path | str | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[SlotIndex, dict[str, np.ndarray], pd.DataFrame]:
    """Build the slot index and label grids, returning the key frame for streaming reads.

    The returned frame holds only the keys, in the deduplicated/sorted order that
    :meth:`SlotIndex.pack` expects, so feature columns can be read and aligned later
    without keeping the whole wide table in memory.
    """
    import pyarrow.parquet as pq

    require_independent_event_calendar(path, calendar_path)
    manifest = load_event_lineage(lineage_path)
    all_audits = audit_column_names(manifest)
    leaked_audits = sorted(set(feature_columns or ()) & all_audits)
    if leaked_audits:
        raise EventLineageError(
            f"feature_columns cannot include event audit columns: {leaked_audits}"
        )
    governed = list(dict.fromkeys([*(feature_columns or []), *label_columns]))
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    validate_event_schema(schema_names, manifest, governed)
    audits = audit_column_names({field: manifest[field] for field in governed})
    needs_calendar = any(manifest[field].horizon > 0 for field in governed)
    calendar = (
        load_event_calendar(calendar_path)
        if calendar_path is not None or needs_calendar
        else None
    )
    projected = list(dict.fromkeys([date_column, code_column, *label_columns, *sorted(audits)]))
    keys = pq.read_table(path, columns=projected).to_pandas()
    validate_event_fields(keys, manifest, governed, calendar=calendar)
    keys = keys[[date_column, code_column, *label_columns]].copy()
    keys["_row"] = np.arange(len(keys))
    keys = normalize_frame(keys, date_column, code_column)
    index = build_slot_index(keys, date_column, code_column)
    index.audit_columns = frozenset(all_audits)
    labels = {
        c: index.pack(pd.to_numeric(keys[c], errors="coerce").to_numpy()) for c in label_columns
    }
    return index, labels, keys


def stream_feature_grids(
    path: Path,
    keys: pd.DataFrame,
    index: SlotIndex,
    columns: list[str],
    batch_size: int = 40,
):
    """Yield ``(name, grid)`` for each column, reading the parquet in column batches."""
    import pyarrow.parquet as pq

    leaked_audits = sorted(set(columns) & index.audit_columns)
    if leaked_audits:
        raise EventLineageError(
            f"stream cannot emit event audit columns: {leaked_audits}"
        )
    take = keys["_row"].to_numpy()
    for start in range(0, len(columns), batch_size):
        batch = columns[start : start + batch_size]
        table = pq.read_table(path, columns=batch)
        for name in batch:
            values = table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
            yield name, index.pack(values[take])
        del table
