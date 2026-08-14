"""Point-in-time provenance for back-adjusted price fields."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from hashlib import sha256

import numpy as np
import pandas as pd

HFQ_BASIS = "hfq"
ADJUSTMENT_ALGORITHM = "raw-times-same-day-adj-v1"
_VERSION_RE = re.compile(rf"{ADJUSTMENT_ALGORITHM}:[0-9a-f]{{64}}")
_AS_OF_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+08:00")


class PriceLineageError(ValueError):
    """Raised when adjusted-price provenance cannot be trusted."""


def _parse_source_date(value: str) -> date:
    text = str(value)
    try:
        if re.fullmatch(r"\d{8}", text):
            return datetime.strptime(text, "%Y%m%d").date()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        pass
    raise PriceLineageError(f"invalid source date {value!r}")


def _parse_as_of_time(value: str, source_date: date) -> None:
    text = str(value)
    if not _AS_OF_RE.fullmatch(text):
        raise PriceLineageError(f"invalid as_of_time {value!r}")
    try:
        as_of_time = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PriceLineageError(f"invalid as_of_time {value!r}") from exc
    if as_of_time.utcoffset() != timedelta(hours=8):
        raise PriceLineageError(f"as_of_time must use +08:00, got {value!r}")
    if as_of_time.date() != source_date:
        raise PriceLineageError(f"as_of_time is not local to source date {source_date.isoformat()}")
    if as_of_time.timetz().replace(tzinfo=None) > time(15):
        raise PriceLineageError(f"as_of_time is after market close {value!r}")


@dataclass(frozen=True)
class AdjustmentStamp:
    price_basis: str
    adj_factor_version: str


def require_hfq_adjustment_stamp(stamp: AdjustmentStamp, purpose: str) -> AdjustmentStamp:
    """Fail closed unless a stamp names the supported governed HFQ adjustment."""
    price_basis = str(stamp.price_basis)
    version = str(stamp.adj_factor_version)
    if price_basis != HFQ_BASIS or not _VERSION_RE.fullmatch(version):
        raise PriceLineageError(
            f"{purpose}: unsupported adjustment stamp "
            f"price_basis={price_basis!r}, adj_factor_version={version!r}"
        )
    return AdjustmentStamp(price_basis, version)


@dataclass(frozen=True, eq=False)
class PriceLineage:
    source_date: np.ndarray
    as_of_time: np.ndarray
    price_basis: str
    adj_factor_version: str

    def __post_init__(self) -> None:
        source_date = np.array(self.source_date, dtype=str, copy=True)
        as_of_time = np.array(self.as_of_time, dtype=str, copy=True)
        if source_date.ndim != 1 or as_of_time.ndim != 1:
            raise PriceLineageError("price lineage dates must be one-dimensional")
        if source_date.shape != as_of_time.shape:
            raise PriceLineageError("price lineage source_date and as_of_time shapes differ")
        version = str(self.adj_factor_version)
        if not _VERSION_RE.fullmatch(version):
            raise PriceLineageError(f"unsupported adj_factor_version {version!r}")
        for source, as_of in zip(source_date, as_of_time, strict=True):
            _parse_as_of_time(as_of, _parse_source_date(source))
        source_date.setflags(write=False)
        as_of_time.setflags(write=False)
        object.__setattr__(self, "source_date", source_date)
        object.__setattr__(self, "as_of_time", as_of_time)
        object.__setattr__(self, "price_basis", str(self.price_basis))
        object.__setattr__(self, "adj_factor_version", version)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PriceLineage):
            return NotImplemented
        return (
            np.array_equal(self.source_date, other.source_date)
            and np.array_equal(self.as_of_time, other.as_of_time)
            and self.price_basis == other.price_basis
            and self.adj_factor_version == other.adj_factor_version
        )


def adjustment_factor_version(frame: pd.DataFrame) -> str:
    """Return a stable content hash for adjustment-factor rows."""
    required = ("trade_date", "ts_code", "adj_factor")
    missing = [name for name in required if name not in frame]
    if missing:
        raise PriceLineageError(f"adjustment factors missing columns {missing}")
    stable = frame.loc[:, required].copy()
    stable["trade_date"] = stable["trade_date"].astype(str)
    stable["ts_code"] = stable["ts_code"].astype(str)
    stable["adj_factor"] = pd.to_numeric(stable["adj_factor"], errors="coerce")
    stable = stable.sort_values(["trade_date", "ts_code", "adj_factor"], kind="mergesort")
    digest = sha256()
    for row in stable.itertuples(index=False):
        digest.update(
            f"{row.trade_date}\x1f{row.ts_code}\x1f{float(row.adj_factor).hex()}\n".encode()
        )
    return f"{ADJUSTMENT_ALGORITHM}:{digest.hexdigest()}"


def make_hfq_lineage(dates: Sequence[str] | np.ndarray, version: str) -> PriceLineage:
    source_date = np.asarray(dates).astype(str)
    return PriceLineage(
        source_date=source_date,
        as_of_time=np.asarray(
            [f"{_parse_source_date(date).isoformat()}T15:00:00+08:00" for date in source_date]
        ),
        price_basis=HFQ_BASIS,
        adj_factor_version=version,
    )


def require_hfq_lineage(
    dates: Sequence[str] | np.ndarray,
    lineage: Mapping[str, PriceLineage],
    fields: Sequence[str],
    purpose: str,
) -> AdjustmentStamp:
    """Fail closed unless every requested field has matching HFQ provenance."""
    expected_dates = np.asarray(dates).astype(str)
    stamp: AdjustmentStamp | None = None
    for field in fields:
        item = lineage.get(field)
        if item is None:
            raise PriceLineageError(f"{purpose}: missing price lineage for field {field!r}")
        if item.price_basis != HFQ_BASIS:
            raise PriceLineageError(
                f"{purpose}: field {field!r} has price basis {item.price_basis!r}, expected {HFQ_BASIS!r}"
            )
        if not np.array_equal(item.source_date, expected_dates):
            first = next(
                (
                    index
                    for index in range(max(len(item.source_date), len(expected_dates)))
                    if index >= len(item.source_date)
                    or index >= len(expected_dates)
                    or item.source_date[index] != expected_dates[index]
                ),
                0,
            )
            date = expected_dates[first] if first < len(expected_dates) else item.source_date[first]
            raise PriceLineageError(
                f"{purpose}: field {field!r} source_date does not match panel date {date}"
            )
        for source_date, as_of_time in zip(item.source_date, item.as_of_time, strict=True):
            try:
                _parse_as_of_time(as_of_time, _parse_source_date(source_date))
            except PriceLineageError as exc:
                raise PriceLineageError(
                    f"{purpose}: field {field!r} has invalid lineage: {exc}"
                ) from exc
        current = AdjustmentStamp(item.price_basis, item.adj_factor_version)
        if stamp is not None and current != stamp:
            raise PriceLineageError(
                f"{purpose}: field {field!r} has inconsistent adjustment version {current.adj_factor_version!r}"
            )
        stamp = current
    if stamp is None:
        raise PriceLineageError(f"{purpose}: no adjusted price fields were requested")
    return stamp
