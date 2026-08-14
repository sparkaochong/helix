"""The wide ``(T, N)`` panel every downstream stage operates on.

Rows are trade dates ascending, columns are ts_codes. Storing data this way makes
time-series operators an axis-0 reduction and cross-sectional operators an axis-1
reduction, which is what keeps GP evaluation fast enough to be practical.

Price fields come in two flavours and mixing them is a correctness bug:

* ``*_hfq`` (back-adjusted) -- use for **anything comparing prices across days**,
  including returns and the label's touch ratio. Immune to ex-dividend jumps.
* raw ``open/high/low/close`` -- use only for same-day comparisons against
  ``up_limit`` / ``down_limit``, which are quoted in raw prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from . import schema
from .price_lineage import (
    AdjustmentStamp,
    PriceLineage,
    PriceLineageError,
    adjustment_factor_version,
    make_hfq_lineage,
    require_hfq_lineage,
)
from .store import ParquetStore

log = get_logger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close")


@dataclass
class Panel:
    dates: np.ndarray  # (T,) YYYYMMDD strings, ascending
    codes: np.ndarray  # (N,) ts_codes, sorted
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    price_lineage: dict[str, PriceLineage] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.codes)

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(f"unknown field {name!r}; have {sorted(self.fields)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def add(
        self, name: str, values: np.ndarray, *, price_lineage: PriceLineage | None = None
    ) -> None:
        if values.shape != self.shape:
            raise ValueError(f"field {name!r} has shape {values.shape}, expected {self.shape}")
        self.fields[name] = values
        if price_lineage is None:
            self.price_lineage.pop(name, None)
        else:
            self.price_lineage[name] = price_lineage

    def f64(self, name: str) -> np.ndarray:
        return np.asarray(self[name], dtype=np.float64)

    def require_adjusted_prices(self, fields: tuple[str, ...], purpose: str) -> AdjustmentStamp:
        missing = [field for field in fields if field not in self]
        if missing:
            raise PriceLineageError(f"{purpose}: missing adjusted price fields {missing}")
        return require_hfq_lineage(self.dates, self.price_lineage, fields, purpose)

    # ------------------------------------------------------------- subsetting --
    def slice_dates(self, start: str = "", end: str = "") -> Panel:
        lo = int(np.searchsorted(self.dates, start, "left")) if start else 0
        hi = int(np.searchsorted(self.dates, end, "right")) if end else len(self.dates)
        return Panel(
            dates=self.dates[lo:hi],
            codes=self.codes,
            fields={k: v[lo:hi] for k, v in self.fields.items()},
            price_lineage={
                name: PriceLineage(
                    source_date=item.source_date[lo:hi],
                    as_of_time=item.as_of_time[lo:hi],
                    price_basis=item.price_basis,
                    adj_factor_version=item.adj_factor_version,
                )
                for name, item in self.price_lineage.items()
            },
        )

    def select_codes(self, col_index: np.ndarray) -> Panel:
        return Panel(
            dates=self.dates,
            codes=self.codes[col_index],
            fields={k: v[:, col_index] for k, v in self.fields.items()},
            price_lineage=dict(self.price_lineage),
        )

    # ----------------------------------------------------------------- cache --
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lineage = {}
        for name, item in self.price_lineage.items():
            prefix = f"__lineage__{name}__"
            lineage[f"{prefix}source_date"] = np.asarray(item.source_date, dtype=str)
            lineage[f"{prefix}as_of_time"] = np.asarray(item.as_of_time, dtype=str)
            lineage[f"{prefix}price_basis"] = np.asarray(item.price_basis, dtype=str)
            lineage[f"{prefix}adj_factor_version"] = np.asarray(item.adj_factor_version, dtype=str)
        np.savez_compressed(path, dates=self.dates, codes=self.codes, **self.fields, **lineage)
        log.info("panel cached to %s (%d dates x %d codes)", path, *self.shape)

    @classmethod
    def load(cls, path: Path) -> Panel:
        with np.load(path, allow_pickle=False) as z:
            dates = z["dates"].astype(str)
            codes = z["codes"].astype(str)
            lineage_keys = [key for key in z.files if key.startswith("__lineage__")]
            fields = {k: z[k] for k in z.files if k not in ("dates", "codes", *lineage_keys)}
            lineage: dict[str, PriceLineage] = {}
            suffix = "__source_date"
            for key in lineage_keys:
                if not key.endswith(suffix):
                    continue
                name = key[len("__lineage__") : -len(suffix)]
                prefix = f"__lineage__{name}__"
                required = ("source_date", "as_of_time", "price_basis", "adj_factor_version")
                if not all(f"{prefix}{part}" in z.files for part in required):
                    raise PriceLineageError(f"malformed cached price lineage for field {name!r}")
                lineage[name] = PriceLineage(
                    source_date=z[f"{prefix}source_date"],
                    as_of_time=z[f"{prefix}as_of_time"],
                    price_basis=str(z[f"{prefix}price_basis"]),
                    adj_factor_version=str(z[f"{prefix}adj_factor_version"]),
                )
        return cls(dates=dates, codes=codes, fields=fields, price_lineage=lineage)


def _pivot(df: pd.DataFrame, value: str, dates: np.ndarray, codes: np.ndarray) -> np.ndarray:
    wide = df.pivot(index="trade_date", columns="ts_code", values=value)
    wide = wide.reindex(index=dates, columns=codes)
    return wide.to_numpy(dtype=np.float32)


def _deduplicate_rows(frame: pd.DataFrame, columns: tuple[str, ...], row_type: str) -> pd.DataFrame:
    keys = ["trade_date", "ts_code"]
    duplicate_rows = frame[frame.duplicated(keys, keep=False)]
    for (date, code), group in duplicate_rows.groupby(keys, sort=True):
        first = group.iloc[0]
        for _, row in group.iloc[1:].iterrows():
            for column in columns:
                left, right = row[column], first[column]
                if not ((pd.isna(left) and pd.isna(right)) or left == right):
                    raise PriceLineageError(
                        f"{row_type} rows conflict for {date} {code}: conflicting values"
                    )
    return frame.drop_duplicates(keys, keep="first")


def build_adjusted_price_fields(
    daily: pd.DataFrame, adj: pd.DataFrame, dates: np.ndarray, codes: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, PriceLineage]]:
    """Build HFQ price fields using only same-date adjustment factors."""
    dates = np.asarray(dates).astype(str)
    codes = np.asarray(codes).astype(str)
    daily = daily.copy()
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily = _deduplicate_rows(daily, PRICE_COLUMNS, "daily")
    adj = adj.copy()
    adj["trade_date"] = adj["trade_date"].astype(str)
    adj["ts_code"] = adj["ts_code"].astype(str)
    adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
    adj = _deduplicate_rows(adj, ("adj_factor",), "adj_factor")
    adj = adj[adj["trade_date"].isin(dates) & adj["ts_code"].isin(codes)]
    adj_factor = _pivot(adj, "adj_factor", dates, codes)
    prices = {column: _pivot(daily, column, dates, codes) for column in PRICE_COLUMNS}
    raw_price_present = np.logical_or.reduce([np.isfinite(values) for values in prices.values()])
    invalid = raw_price_present & (~np.isfinite(adj_factor) | (adj_factor <= 0))
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise PriceLineageError(
            f"missing or invalid adj_factor for {dates[row]} {codes[column]}: same-date factor required"
        )
    version = adjustment_factor_version(adj)
    lineage = make_hfq_lineage(dates, version)
    fields = {"adj_factor": adj_factor}
    field_lineage = {}
    for column in PRICE_COLUMNS:
        fields[f"{column}_hfq"] = prices[column] * adj_factor
        field_lineage[f"{column}_hfq"] = lineage
    return fields, field_lineage


def build_panel(store: ParquetStore, start_date: str = "", end_date: str = "") -> Panel:
    """Assemble the raw panel from the local store. No derived features yet."""
    daily = store.read_dated(schema.DAILY, start_date, end_date)
    if daily.empty:
        raise RuntimeError("no rows in the daily table; run `helix download` first")

    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    dates = np.array(sorted(daily["trade_date"].unique()), dtype=object).astype(str)
    codes = np.array(sorted(daily["ts_code"].unique()), dtype=object).astype(str)
    log.info("building panel: %d dates x %d codes", len(dates), len(codes))

    adj = store.read_dated(schema.ADJ_FACTOR, start_date, end_date)
    if adj.empty:
        raise RuntimeError("adj_factor table is empty; back-adjusted prices are required")
    adj["trade_date"] = adj["trade_date"].astype(str)
    adj["ts_code"] = adj["ts_code"].astype(str)
    adjusted_fields, price_lineage = build_adjusted_price_fields(daily, adj, dates, codes)
    daily = daily.drop_duplicates(["trade_date", "ts_code"], keep="last")
    panel = Panel(dates=dates, codes=codes)
    for col in (*PRICE_COLUMNS, "vol", "amount"):
        panel.add(col, _pivot(daily, col, dates, codes))
    for name, values in adjusted_fields.items():
        panel.add(name, values, price_lineage=price_lineage.get(name))

    limits = store.read_dated(schema.STK_LIMIT, start_date, end_date)
    if limits.empty:
        log.warning("stk_limit is empty; falling back to rule-based limit prices")
        limit_price_observed = np.zeros(panel.shape, dtype=bool)
        up, down = _fallback_limit_prices(panel, store)
    else:
        limits["trade_date"] = limits["trade_date"].astype(str)
        limits["ts_code"] = limits["ts_code"].astype(str)
        limits = limits.drop_duplicates(["trade_date", "ts_code"], keep="last")
        up = _pivot(limits, "up_limit", dates, codes)
        down = _pivot(limits, "down_limit", dates, codes)
        limit_price_observed = np.isfinite(up) & np.isfinite(down)
        rule_up, rule_down = _fallback_limit_prices(panel, store)
        up = np.where(np.isnan(up), rule_up, up)
        down = np.where(np.isnan(down), rule_down, down)
    panel.add("up_limit", up.astype(np.float32))
    panel.add("down_limit", down.astype(np.float32))
    panel.add("limit_price_observed", limit_price_observed.astype(np.float32))

    basic = store.read_dated(schema.DAILY_BASIC, start_date, end_date)
    if not basic.empty:
        basic["trade_date"] = basic["trade_date"].astype(str)
        basic["ts_code"] = basic["ts_code"].astype(str)
        basic = basic.drop_duplicates(["trade_date", "ts_code"], keep="last")
        for col in ("turnover_rate_f", "volume_ratio", "pe_ttm", "pb", "circ_mv"):
            panel.add(col, _pivot(basic, col, dates, codes))
    else:
        log.warning("daily_basic is empty; valuation and turnover fields will be missing")

    # A stock trades on a date iff it has a price and non-zero volume there.
    is_trading = (~np.isnan(panel["close"])) & (np.nan_to_num(panel["vol"]) > 0)
    panel.add("is_trading", is_trading.astype(np.float32))
    return panel


def _limit_pct(codes: np.ndarray, store: ParquetStore) -> np.ndarray:
    """Per-stock daily limit as a fraction, from board rules. Shape ``(N,)``."""
    basic = store.read_static(schema.STOCK_BASIC)
    market = (
        basic.set_index(basic["ts_code"].astype(str))["market"].astype(str).to_dict()
        if not basic.empty
        else {}
    )
    pct = np.full(len(codes), 0.10)
    for j, code in enumerate(codes):
        mkt = market.get(code, "")
        if code.startswith("688") or code.startswith("689"):
            pct[j] = 0.20  # 科创板
        elif code.startswith("30"):
            pct[j] = 0.20  # 创业板 (post-2020-08; conservative for the whole sample)
        elif code.endswith(".BJ") or mkt == "北交所":
            pct[j] = 0.30
    return pct


def _fallback_limit_prices(panel: Panel, store: ParquetStore) -> tuple[np.ndarray, np.ndarray]:
    """Rule-based limit prices from ``pre_close``, used where stk_limit has gaps."""
    pct = _limit_pct(panel.codes, store)[None, :]
    pre = panel["pre_close"].astype(np.float64)
    return np.round(pre * (1 + pct), 2), np.round(pre * (1 - pct), 2)
