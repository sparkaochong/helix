"""Column contracts for every raw Tushare table Helix depends on.

Keeping these in one place means a Tushare schema change surfaces as a loud
validation error at download time instead of silently poisoning the factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    """A raw table: which Tushare API to call and which columns must come back."""

    name: str
    api: str
    columns: tuple[str, ...]
    by_trade_date: bool  # True -> fetched one trade date at a time


DAILY = TableSpec(
    name="daily",
    api="daily",
    columns=("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"),
    by_trade_date=True,
)

ADJ_FACTOR = TableSpec(
    name="adj_factor",
    api="adj_factor",
    columns=("ts_code", "trade_date", "adj_factor"),
    by_trade_date=True,
)

DAILY_BASIC = TableSpec(
    name="daily_basic",
    api="daily_basic",
    columns=("ts_code", "trade_date", "turnover_rate_f", "volume_ratio", "pe_ttm", "pb", "circ_mv"),
    by_trade_date=True,
)

STK_LIMIT = TableSpec(
    name="stk_limit",
    api="stk_limit",
    columns=("ts_code", "trade_date", "up_limit", "down_limit"),
    by_trade_date=True,
)

STOCK_BASIC = TableSpec(
    name="stock_basic",
    api="stock_basic",
    columns=("ts_code", "name", "list_date", "delist_date", "market", "exchange"),
    by_trade_date=False,
)

NAMECHANGE = TableSpec(
    name="namechange",
    api="namechange",
    columns=("ts_code", "name", "start_date", "end_date"),
    by_trade_date=False,
)

TRADE_CAL = TableSpec(
    name="trade_cal",
    api="trade_cal",
    columns=("exchange", "cal_date", "is_open"),
    by_trade_date=False,
)

DATE_TABLES: tuple[TableSpec, ...] = (DAILY, ADJ_FACTOR, DAILY_BASIC, STK_LIMIT)
STATIC_TABLES: tuple[TableSpec, ...] = (STOCK_BASIC, NAMECHANGE, TRADE_CAL)
ALL_TABLES: tuple[TableSpec, ...] = DATE_TABLES + STATIC_TABLES

#: Numeric columns coerced to float on load; anything unparseable becomes NaN.
NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "open", "high", "low", "close", "pre_close", "vol", "amount",
        "adj_factor", "turnover_rate_f", "volume_ratio", "pe_ttm", "pb", "circ_mv",
        "up_limit", "down_limit", "is_open",
    }
)


def validate(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Return ``df`` restricted to the spec columns, raising if any are missing."""
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(f"table {spec.name!r} is missing columns {missing}; got {list(df.columns)}")
    out = df.loc[:, list(spec.columns)].copy()
    for col in out.columns:
        if col in NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = out[col].astype("string")
    return out
