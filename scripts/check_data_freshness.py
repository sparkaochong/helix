#!/usr/bin/env python3
"""Daily data-freshness / completeness spot check for the local Tushare store.

Fast default mode (safe to run from any external scheduler, e.g. once a day): for
each date-partitioned table (daily/adj_factor/daily_basic/stk_limit), checks the
most recent ``--lookback`` trading days per the local ``trade_cal`` against what is
actually stored, and reports any gap. Purely local -- no network, no Tushare token
required.

``--deep`` mode (heavier, live API calls): additionally re-verifies ``namechange``
and ``stock_basic`` against a live paginated pull, to catch a recurrence of the
single-call silent-truncation bug found in the 2026-08-15 data baseline audit
(see ``helix/data/tushare_source.py::_call_paginated``). Not meant to run daily --
periodic (weekly/monthly) is enough to catch a regression before it does damage.

Exit codes: 0 = clean, 1 = gap/mismatch found, 2 = execution error.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from helix.config import Config, load_dotenv
from helix.data import schema
from helix.data.store import ParquetStore


def expected_recent_open_days(store: ParquetStore, calendar_exchange: str, lookback: int) -> list[str]:
    cal = store.read_static(schema.TRADE_CAL)
    if cal.empty:
        raise SystemExit("trade_cal is empty; run `helix download` first")
    cal = cal[cal["exchange"].astype(str) == calendar_exchange]
    open_days = sorted(
        cal.loc[pd.to_numeric(cal["is_open"], errors="coerce") == 1, "cal_date"].astype(str)
    )
    today = pd.Timestamp.today().strftime("%Y%m%d")
    open_days = [d for d in open_days if d <= today]
    return open_days[-lookback:] if lookback > 0 else open_days


def check_date_tables(store: ParquetStore, calendar_exchange: str, lookback: int) -> dict[str, list[str]]:
    """Return ``{table_name: [missing dates]}`` for each of ``schema.DATE_TABLES``."""
    expected = expected_recent_open_days(store, calendar_exchange, lookback)
    gaps: dict[str, list[str]] = {}
    for spec in schema.DATE_TABLES:
        have = store.existing_dates(spec)
        missing = [d for d in expected if d not in have]
        if missing:
            gaps[spec.name] = missing
    return gaps


def check_static_tables_deep(cfg: Config) -> dict[str, dict[str, int]]:
    """Re-pull namechange/stock_basic live and compare row counts against local."""
    from helix.data.tushare_source import TushareSource

    src = TushareSource(cfg)
    findings: dict[str, dict[str, int]] = {}

    live_names = src._call_paginated("namechange", fields=",".join(schema.NAMECHANGE.columns))
    local_names = src.store.read_static(schema.NAMECHANGE)
    if len(live_names) != len(local_names):
        findings["namechange"] = {"local_rows": len(local_names), "live_rows": len(live_names)}

    live_total = 0
    for status in ("L", "D", "P"):
        live_total += len(
            src._call_paginated(
                "stock_basic", exchange="", list_status=status, fields=",".join(schema.STOCK_BASIC.columns)
            )
        )
    local_basic = src.store.read_static(schema.STOCK_BASIC)
    if live_total != len(local_basic):
        findings["stock_basic"] = {"local_rows": len(local_basic), "live_rows": live_total}

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--lookback", type=int, default=10,
        help="How many of the most recent trading days to check per date-partitioned table.",
    )
    ap.add_argument(
        "--deep", action="store_true",
        help="Also re-verify namechange/stock_basic against a live paginated pull (network, needs a token).",
    )
    args = ap.parse_args()

    load_dotenv()
    cfg = Config.load(args.config)
    store = ParquetStore(cfg.data.root)

    ok = True
    gaps = check_date_tables(store, cfg.data.calendar_exchange, args.lookback)
    if gaps:
        ok = False
        print(f"GAPS FOUND in date-partitioned tables (most recent {args.lookback} trading days):")
        for name, missing in gaps.items():
            print(f"  {name}: missing {len(missing)} date(s): {', '.join(missing)}")
    else:
        print(f"OK: date-partitioned tables complete for the most recent {args.lookback} trading days")

    if args.deep:
        print("running --deep static-table re-verification (live API calls)...")
        deep_findings = check_static_tables_deep(cfg)
        if deep_findings:
            ok = False
            print("STATIC TABLE MISMATCH (local vs live):")
            for name, info in deep_findings.items():
                print(f"  {name}: local={info['local_rows']} live={info['live_rows']}")
        else:
            print("OK: namechange/stock_basic match live row counts")

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top-level guard, must not leak a traceback as "exit 1"
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
