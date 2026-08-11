"""Parquet-backed local store for raw Tushare tables.

Layout::

    {root}/raw/{table}/{year}.parquet   # date-partitioned tables
    {root}/raw/{table}.parquet          # static tables

Date-partitioned writes are idempotent: rows for a trade date already present are
replaced, so re-running a download never duplicates data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_setup import get_logger
from .schema import TableSpec, validate

log = get_logger(__name__)


class ParquetStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths --
    def _static_path(self, spec: TableSpec) -> Path:
        return self.raw_dir / f"{spec.name}.parquet"

    def _year_path(self, spec: TableSpec, year: str) -> Path:
        return self.raw_dir / spec.name / f"{year}.parquet"

    # ----------------------------------------------------------------- read --
    def read_static(self, spec: TableSpec) -> pd.DataFrame:
        path = self._static_path(spec)
        if not path.exists():
            return pd.DataFrame(columns=list(spec.columns))
        return pd.read_parquet(path)

    def read_dated(
        self, spec: TableSpec, start_date: str = "", end_date: str = ""
    ) -> pd.DataFrame:
        table_dir = self.raw_dir / spec.name
        if not table_dir.exists():
            return pd.DataFrame(columns=list(spec.columns))
        frames = []
        for path in sorted(table_dir.glob("*.parquet")):
            year = path.stem
            if start_date and year < start_date[:4]:
                continue
            if end_date and year > end_date[:4]:
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=list(spec.columns))
        df = pd.concat(frames, ignore_index=True)
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]
        return df.reset_index(drop=True)

    def existing_dates(self, spec: TableSpec) -> set[str]:
        """Trade dates already stored for a date-partitioned table."""
        table_dir = self.raw_dir / spec.name
        if not table_dir.exists():
            return set()
        dates: set[str] = set()
        for path in sorted(table_dir.glob("*.parquet")):
            col = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
            dates.update(col.astype(str).tolist())
        return dates

    # ---------------------------------------------------------------- write --
    def write_static(self, spec: TableSpec, df: pd.DataFrame) -> None:
        path = self._static_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        validate(df, spec).to_parquet(path, index=False, compression="zstd")
        log.info("wrote %s rows to %s", len(df), path.name)

    def append_dated(self, spec: TableSpec, df: pd.DataFrame) -> None:
        """Merge ``df`` into the year partitions it touches, replacing same-date rows."""
        if df.empty:
            return
        df = validate(df, spec)
        df["trade_date"] = df["trade_date"].astype(str)
        for year, chunk in df.groupby(df["trade_date"].str[:4]):
            path = self._year_path(spec, str(year))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pd.read_parquet(path)
                replaced = set(chunk["trade_date"])
                existing = existing[~existing["trade_date"].astype(str).isin(replaced)]
                chunk = pd.concat([existing, chunk], ignore_index=True)
            chunk = chunk.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
            chunk.to_parquet(path, index=False, compression="zstd")
