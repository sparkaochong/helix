"""Point-in-time provenance for back-adjusted price fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

HFQ_BASIS = "hfq"
ADJUSTMENT_ALGORITHM = "raw-times-same-day-adj-v1"


class PriceLineageError(ValueError):
    """Raised when adjusted-price provenance cannot be trusted."""


@dataclass(frozen=True)
class AdjustmentStamp:
    price_basis: str
    adj_factor_version: str


@dataclass(frozen=True, eq=False)
class PriceLineage:
    source_date: np.ndarray
    as_of_time: np.ndarray
    price_basis: str
    adj_factor_version: str

    def __post_init__(self) -> None:
        source_date = np.asarray(self.source_date).astype(str)
        as_of_time = np.asarray(self.as_of_time).astype(str)
        if source_date.ndim != 1 or as_of_time.ndim != 1:
            raise PriceLineageError("price lineage dates must be one-dimensional")
        if source_date.shape != as_of_time.shape:
            raise PriceLineageError("price lineage source_date and as_of_time shapes differ")
        if not self.adj_factor_version:
            raise PriceLineageError("price lineage adj_factor_version must not be empty")
        object.__setattr__(self, "source_date", source_date)
        object.__setattr__(self, "as_of_time", as_of_time)
        object.__setattr__(self, "price_basis", str(self.price_basis))
        object.__setattr__(self, "adj_factor_version", str(self.adj_factor_version))

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
        as_of_time=np.asarray([f"{date}T15:00:00+08:00" for date in source_date]),
        price_basis=HFQ_BASIS,
        adj_factor_version=version,
    )


def _local_date(value: str) -> str:
    return "".join(char for char in str(value)[:10] if char.isdigit())


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
            if _local_date(as_of_time) != _local_date(source_date):
                raise PriceLineageError(
                    f"{purpose}: field {field!r} as_of_time is not local to source date {source_date}"
                )
        current = AdjustmentStamp(item.price_basis, item.adj_factor_version)
        if stamp is not None and current != stamp:
            raise PriceLineageError(
                f"{purpose}: field {field!r} has inconsistent adjustment version {current.adj_factor_version!r}"
            )
        stamp = current
    if stamp is None:
        raise PriceLineageError(f"{purpose}: no adjusted price fields were requested")
    return stamp
