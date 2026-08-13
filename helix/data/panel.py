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
from .store import ParquetStore

log = get_logger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close")


@dataclass
class Panel:
    dates: np.ndarray  # (T,) YYYYMMDD strings, ascending
    codes: np.ndarray  # (N,) ts_codes, sorted
    fields: dict[str, np.ndarray] = field(default_factory=dict)

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

    def add(self, name: str, values: np.ndarray) -> None:
        if values.shape != self.shape:
            raise ValueError(f"field {name!r} has shape {values.shape}, expected {self.shape}")
        self.fields[name] = values

    def f64(self, name: str) -> np.ndarray:
        return np.asarray(self[name], dtype=np.float64)

    # ------------------------------------------------------------- subsetting --
    def slice_dates(self, start: str = "", end: str = "") -> Panel:
        lo = int(np.searchsorted(self.dates, start, "left")) if start else 0
        hi = int(np.searchsorted(self.dates, end, "right")) if end else len(self.dates)
        return Panel(
            dates=self.dates[lo:hi],
            codes=self.codes,
            fields={k: v[lo:hi] for k, v in self.fields.items()},
        )

    def select_codes(self, col_index: np.ndarray) -> Panel:
        return Panel(
            dates=self.dates,
            codes=self.codes[col_index],
            fields={k: v[:, col_index] for k, v in self.fields.items()},
        )

    # ----------------------------------------------------------------- cache --
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dates=self.dates, codes=self.codes, **self.fields)
        log.info("panel cached to %s (%d dates x %d codes)", path, *self.shape)

    @classmethod
    def load(cls, path: Path) -> Panel:
        with np.load(path, allow_pickle=False) as z:
            dates = z["dates"].astype(str)
            codes = z["codes"].astype(str)
            fields = {k: z[k] for k in z.files if k not in ("dates", "codes")}
        return cls(dates=dates, codes=codes, fields=fields)


def _pivot(df: pd.DataFrame, value: str, dates: np.ndarray, codes: np.ndarray) -> np.ndarray:
    wide = df.pivot(index="trade_date", columns="ts_code", values=value)
    wide = wide.reindex(index=dates, columns=codes)
    return wide.to_numpy(dtype=np.float32)


def build_panel(store: ParquetStore, start_date: str = "", end_date: str = "") -> Panel:
    """Assemble the raw panel from the local store. No derived features yet."""
    daily = store.read_dated(schema.DAILY, start_date, end_date)
    if daily.empty:
        raise RuntimeError("no rows in the daily table; run `helix download` first")

    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily = daily.drop_duplicates(["trade_date", "ts_code"], keep="last")

    dates = np.array(sorted(daily["trade_date"].unique()), dtype=object).astype(str)
    codes = np.array(sorted(daily["ts_code"].unique()), dtype=object).astype(str)
    log.info("building panel: %d dates x %d codes", len(dates), len(codes))

    panel = Panel(dates=dates, codes=codes)
    for col in (*PRICE_COLUMNS, "vol", "amount"):
        panel.add(col, _pivot(daily, col, dates, codes))

    adj = store.read_dated(schema.ADJ_FACTOR, start_date, end_date)
    if adj.empty:
        raise RuntimeError("adj_factor table is empty; back-adjusted prices are required")
    adj["trade_date"] = adj["trade_date"].astype(str)
    adj["ts_code"] = adj["ts_code"].astype(str)
    adj = adj.drop_duplicates(["trade_date", "ts_code"], keep="last")
    adj_factor = _pivot(adj, "adj_factor", dates, codes)
    panel.add("adj_factor", adj_factor)
    for col in PRICE_COLUMNS:
        panel.add(f"{col}_hfq", panel[col] * adj_factor)

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
