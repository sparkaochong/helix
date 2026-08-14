"""Fail-closed field lineage for long-format event tables."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from .price_lineage import ADJUSTMENT_ALGORITHM, HFQ_BASIS

_ENTRY_KEYS = {
    "source_date",
    "as_of_time",
    "price_basis",
    "adj_factor_version",
    "horizon",
}
_VERSION_RE = re.compile(rf"{re.escape(ADJUSTMENT_ALGORITHM)}:[0-9a-f]{{64}}")
_AS_OF_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}"
)


class EventLineageError(ValueError):
    """Raised when an event field's point-in-time price provenance is untrusted."""


@dataclass(frozen=True)
class EventAuditColumns:
    """Actual audit-column names and declared trading-day horizon for one field."""

    source_date: str
    as_of_time: str
    price_basis: str
    adj_factor_version: str
    horizon: int

    def __post_init__(self) -> None:
        for name in ("source_date", "as_of_time", "price_basis", "adj_factor_version"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise EventLineageError(
                    f"event lineage manifest entry {name!r} must be a non-empty string"
                )
        if type(self.horizon) is not int or self.horizon < 0:
            raise EventLineageError(
                "event lineage manifest entry 'horizon' must be an integer >= 0"
            )


EventLineageManifest = dict[str, EventAuditColumns]


@dataclass
class EventLineageValidationState:
    """The sole row-independent invariant retained across bounded audit batches."""

    adjustment_version: str | None = None


def require_independent_event_calendar(
    event_path: Path | str, calendar_path: Path | str | None
) -> None:
    """Reject using the governed event table as its own calendar authority."""
    if calendar_path is not None and Path(event_path).resolve() == Path(calendar_path).resolve():
        raise EventLineageError("event trading calendar must be supplied independently")


def _manifest_error(message: str, exc: Exception | None = None) -> EventLineageError:
    error = EventLineageError(f"invalid event lineage manifest: {message}")
    if exc is not None:
        error.__cause__ = exc
    return error


def load_event_lineage(path: Path | str | None) -> EventLineageManifest:
    """Load and strictly validate a schema-version-1 event lineage manifest."""
    if path is None:
        raise EventLineageError("event lineage manifest is required")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise _manifest_error(f"cannot read {path!s}: {exc}", exc) from exc
    if type(payload) is not dict:
        raise _manifest_error("root must be an object")
    if set(payload) != {"schema_version", "fields"}:
        raise _manifest_error("root must contain exactly 'schema_version' and 'fields'")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise _manifest_error("schema_version must be integer 1")
    fields = payload["fields"]
    if type(fields) is not dict:
        raise _manifest_error("fields must be an object")

    manifest: EventLineageManifest = {}
    for field, raw_entry in fields.items():
        if type(field) is not str or not field:
            raise _manifest_error("field names must be non-empty strings")
        if type(raw_entry) is not dict or set(raw_entry) != _ENTRY_KEYS:
            raise _manifest_error(
                f"field {field!r} must contain exactly {sorted(_ENTRY_KEYS)}"
            )
        try:
            manifest[field] = EventAuditColumns(**raw_entry)
        except (EventLineageError, TypeError) as exc:
            raise _manifest_error(f"field {field!r}: {exc}", exc) from exc
    return manifest


def load_event_calendar(path: Path | str | None) -> tuple[str, ...]:
    """Load an independently supplied, ordered exchange trading calendar."""
    if path is None:
        raise EventLineageError("authoritative event trading calendar is required")
    calendar_path = Path(path)
    try:
        if calendar_path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            schema = pq.ParquetFile(calendar_path).schema_arrow.names
            date_column = "cal_date" if "cal_date" in schema else "trade_date"
            if date_column not in schema:
                raise EventLineageError(
                    "event trading calendar needs a 'cal_date' or 'trade_date' column"
                )
            columns = [date_column] + (["is_open"] if "is_open" in schema else [])
            frame = pd.read_parquet(calendar_path, columns=columns)
        elif calendar_path.suffix.lower() == ".csv":
            frame = pd.read_csv(calendar_path, dtype=str)
            date_column = "cal_date" if "cal_date" in frame else "trade_date"
            if date_column not in frame:
                raise EventLineageError(
                    "event trading calendar needs a 'cal_date' or 'trade_date' column"
                )
        else:
            raise EventLineageError(
                "event trading calendar must be a parquet or CSV file"
            )
        if "is_open" in frame:
            flags: list[int] = []
            for session, value in zip(
                frame[date_column].tolist(), frame["is_open"].tolist(), strict=True
            ):
                valid_string = isinstance(value, str) and re.fullmatch(r"[01]", value)
                valid_integer = isinstance(value, Integral) and not isinstance(value, bool)
                valid_real = (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value).is_integer()
                )
                if not (valid_string or valid_integer or valid_real) or int(value) not in {0, 1}:
                    raise EventLineageError(
                        f"invalid event trading calendar is_open value {value!r} "
                        f"for session {session!r}; expected 0 or 1"
                    )
                flags.append(int(value))
            frame = frame[[flag == 1 for flag in flags]]
        values = frame[date_column].tolist()
    except EventLineageError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise EventLineageError(
            f"invalid event trading calendar {calendar_path}: {exc}"
        ) from exc

    parsed = [
        _parse_date(
            value, field="calendar", rule="session", context=f"calendar row {position}"
        )
        for position, value in enumerate(values)
    ]
    if not parsed:
        raise EventLineageError("event trading calendar is empty")
    if any(right <= left for left, right in zip(parsed, parsed[1:], strict=False)):
        raise EventLineageError(
            "event trading calendar sessions must be strictly increasing and unique"
        )
    return tuple(values)


def audit_column_names(manifest: Mapping[str, EventAuditColumns]) -> set[str]:
    """Return all physical audit columns, deduplicating entries shared by fields."""
    return {
        column
        for item in manifest.values()
        for column in (
            item.source_date,
            item.as_of_time,
            item.price_basis,
            item.adj_factor_version,
        )
    }


def validate_event_schema(
    columns: Sequence[str],
    manifest: Mapping[str, EventAuditColumns],
    fields: Sequence[str],
) -> None:
    """Validate governed fields and their physical audits before projected file reads."""
    available = set(columns)
    for field in dict.fromkeys(fields):
        item = manifest.get(field)
        if item is None:
            raise EventLineageError(f"field {field!r} has no event lineage manifest entry")
        if field not in available:
            raise EventLineageError(f"governed field {field!r} is missing from the event table")
        for rule, column in (
            ("source_date", item.source_date),
            ("as_of_time", item.as_of_time),
            ("price_basis", item.price_basis),
            ("adj_factor_version", item.adj_factor_version),
        ):
            if column not in available:
                raise EventLineageError(
                    f"field {field!r} {rule} audit column {column!r} is missing"
                )


def _parse_date(value: Any, *, field: str, rule: str, context: Any) -> date:
    if type(value) is not str:
        raise EventLineageError(
            f"field {field!r} {rule} must be literal YYYYMMDD or YYYY-MM-DD at {context}"
        )
    try:
        if re.fullmatch(r"\d{8}", value):
            return datetime.strptime(value, "%Y%m%d").date()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        pass
    raise EventLineageError(
        f"field {field!r} invalid {rule} {value!r} at {context}"
    )


def _parse_as_of(value: Any, source: date, *, field: str, context: Any) -> datetime:
    if type(value) is not str or not _AS_OF_RE.fullmatch(value):
        raise EventLineageError(
            f"field {field!r} invalid as_of_time {value!r} at {context}"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise EventLineageError(
            f"field {field!r} invalid as_of_time {value!r} at {context}"
        ) from exc
    if parsed.utcoffset() != timedelta(hours=8):
        raise EventLineageError(
            f"field {field!r} as_of_time must use +08:00 at {context}: {value!r}"
        )
    if parsed.date() != source:
        raise EventLineageError(
            f"field {field!r} as_of_time is not local to source_date at {context}"
        )
    if parsed.timetz().replace(tzinfo=None) > time(15):
        raise EventLineageError(
            f"field {field!r} as_of_time is after market close at {context}: {value!r}"
        )
    return parsed


def _row_context(position: int, trade_date: Any, stock_code: Any | None) -> str:
    context = f"row {position} trade_date={trade_date!r}"
    if stock_code is not None:
        context += f" stock_code={stock_code!r}"
    return context


def validate_event_fields(
    frame: pd.DataFrame,
    manifest: Mapping[str, EventAuditColumns],
    fields: Sequence[str],
    *,
    calendar: Sequence[str] | None = None,
    train_end: str | None = None,
    state: EventLineageValidationState | None = None,
    row_offset: int = 0,
    row_positions: Sequence[int] | None = None,
) -> None:
    """Validate governed HFQ lineage for every requested field and row.

    Positive horizons are checked against an independently supplied authoritative
    trading calendar. Event rows and their declared outcome dates are never allowed
    to define that calendar themselves.
    """
    requested = list(dict.fromkeys(fields))
    needs_calendar = any(
        manifest.get(field) is not None and manifest[field].horizon > 0
        for field in requested
    )
    if needs_calendar and calendar is None:
        raise EventLineageError("authoritative event trading calendar is required")
    if "trade_date" not in frame:
        raise EventLineageError("event lineage validation requires trade_date")
    row_dates_raw = frame["trade_date"].tolist()
    stock_codes = (
        frame["stock_code"].tolist()
        if "stock_code" in frame
        else [None] * len(row_dates_raw)
    )
    if row_positions is not None and len(row_positions) != len(row_dates_raw):
        raise EventLineageError("event lineage row_positions length does not match frame")
    row_contexts = [
        _row_context(
            row_positions[position] if row_positions is not None else row_offset + position,
            value,
            stock_codes[position],
        )
        for position, value in enumerate(row_dates_raw)
    ]
    row_dates = [
        _parse_date(value, field="trade_date", rule="trade_date", context=row_contexts[position])
        for position, value in enumerate(row_dates_raw)
    ]
    cutoff = (
        _parse_date(
            train_end,
            field="train_end",
            rule="train_end",
            context=f"train_end={train_end!r}",
        )
        if train_end is not None
        else None
    )
    calendar_dates = (
        [
            _parse_date(
                value,
                field="calendar",
                rule="session",
                context=f"calendar row {position}",
            )
            for position, value in enumerate(calendar)
        ]
        if calendar is not None
        else []
    )
    if any(
        right <= left
        for left, right in zip(calendar_dates, calendar_dates[1:], strict=False)
    ):
        raise EventLineageError(
            "event trading calendar sessions must be strictly increasing and unique"
        )
    calendar_positions = {day: position for position, day in enumerate(calendar_dates)}
    if cutoff is not None and needs_calendar and cutoff not in calendar_positions:
        raise EventLineageError(f"train_end {train_end!r} is not an event trading session")

    parsed_sources: dict[str, list[date]] = {}
    parsed_audits: dict[EventAuditColumns, list[date]] = {}
    validation_state = state if state is not None else EventLineageValidationState()
    for field in requested:
        item = manifest.get(field)
        if item is None:
            raise EventLineageError(f"field {field!r} has no event lineage manifest entry")
        for rule, column in (
            ("source_date", item.source_date),
            ("as_of_time", item.as_of_time),
            ("price_basis", item.price_basis),
            ("adj_factor_version", item.adj_factor_version),
        ):
            if column not in frame:
                raise EventLineageError(
                    f"field {field!r} {rule} audit column {column!r} is missing"
                )
        cached_sources = parsed_audits.get(item)
        if cached_sources is not None:
            parsed_sources[field] = cached_sources
            continue

        sources: list[date] = []
        audit_rows = zip(
            frame[item.price_basis].tolist(),
            frame[item.adj_factor_version].tolist(),
            frame[item.source_date].tolist(),
            frame[item.as_of_time].tolist(),
            strict=True,
        )
        for position, (basis, version, source_value, as_of_value) in enumerate(audit_rows):
            context = row_contexts[position]
            if type(basis) is not str or basis != HFQ_BASIS:
                raise EventLineageError(
                    f"field {field!r} price_basis must be literal 'hfq' at {context}, "
                    f"got {basis!r}"
                )
            if type(version) is not str or not _VERSION_RE.fullmatch(version):
                raise EventLineageError(
                    f"field {field!r} unsupported adj_factor_version at {context}: "
                    f"{version!r}"
                )
            if validation_state.adjustment_version is None:
                validation_state.adjustment_version = version
            elif version != validation_state.adjustment_version:
                raise EventLineageError(
                    f"field {field!r} has inconsistent adjustment version at {context}: "
                    f"{version!r} != {validation_state.adjustment_version!r}"
                )
            source = _parse_date(
                source_value, field=field, rule="source_date", context=context
            )
            _parse_as_of(as_of_value, source, field=field, context=context)
            sources.append(source)
            if cutoff is not None and item.horizon > 0 and source > cutoff:
                raise EventLineageError(
                    f"field {field!r} outcome exceeds train_end {train_end!r} at {context}: "
                    f"source_date={source.isoformat()}"
                )
        parsed_audits[item] = sources
        parsed_sources[field] = sources

    for field in requested:
        item = manifest[field]
        for position, (decision, source) in enumerate(
            zip(row_dates, parsed_sources[field], strict=True)
        ):
            context = row_contexts[position]
            if item.horizon == 0:
                if source != decision:
                    raise EventLineageError(
                        f"field {field!r} horizon=0 source_date mismatch at {context}: "
                        f"got {source.isoformat()}, expected {decision.isoformat()}"
                    )
                continue
            decision_position = calendar_positions.get(decision)
            if decision_position is None:
                raise EventLineageError(
                    f"field {field!r} decision date at {context} is not in the event trading calendar"
                )
            expected_position = decision_position + item.horizon
            expected = (
                calendar_dates[expected_position]
                if expected_position < len(calendar_dates)
                else None
            )
            if source != expected:
                expected_text = expected.isoformat() if expected is not None else "calendar end"
                raise EventLineageError(
                    f"field {field!r} horizon={item.horizon} source mismatch at {context}: "
                    f"got {source.isoformat()}, expected {expected_text}"
                )
