"""Incremental Tushare Pro downloader.

Date-partitioned tables are fetched one trade date at a time (one call returns the
whole market), which is both the cheapest way to use the API quota and the easiest
to resume: dates already in the store are skipped.
"""

from __future__ import annotations

import time
from collections import deque

import pandas as pd
from tqdm import tqdm

from ..config import Config, require_tushare_token
from ..logging_setup import get_logger
from . import schema
from .store import ParquetStore

log = get_logger(__name__)


class RateLimiter:
    """Sliding-window limiter: at most ``per_minute`` calls in any 60s window."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60.0:
            self._calls.popleft()
        if len(self._calls) >= self.per_minute:
            sleep_for = 60.0 - (now - self._calls[0]) + 0.05
            if sleep_for > 0:
                log.debug("rate limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
            return self.acquire()
        self._calls.append(now)


class TushareSource:
    def __init__(self, cfg: Config):
        import tushare as ts

        self.cfg = cfg
        self.pro = ts.pro_api(require_tushare_token())
        self.limiter = RateLimiter(cfg.data.requests_per_minute)
        self.store = ParquetStore(cfg.data.root)

    # ------------------------------------------------------------------ api --
    def _call(self, api: str, **kwargs) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(self.cfg.data.max_retries + 1):
            self.limiter.acquire()
            try:
                return getattr(self.pro, api)(**kwargs)
            except Exception as err:  # tushare raises bare Exception on quota/network
                last_err = err
                backoff = min(60.0, 2.0**attempt)
                log.warning("%s(%s) failed (%s); retry in %.0fs", api, kwargs, err, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"{api} failed after {self.cfg.data.max_retries} retries") from last_err

    def _call_paginated(self, api: str, page_size: int = 5000, **kwargs) -> pd.DataFrame:
        """Page through an interface that silently caps single-call rows.

        ``namechange`` returns at most ~10,000 rows per call with no error when the
        true result set is larger -- a request for "everything" silently becomes
        "the first page of everything". Looping on ``offset``/``limit`` until a
        short page comes back is the only way to get the true total.
        """
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            page = self._call(api, offset=offset, limit=page_size, **kwargs)
            if page is None or page.empty:
                break
            frames.append(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # -------------------------------------------------------------- calendar --
    def trade_dates(self) -> list[str]:
        """Open trade dates in the configured range, ascending."""
        cal = self.store.read_static(schema.TRADE_CAL)
        if cal.empty:
            cal = self._call(
                "trade_cal",
                exchange=self.cfg.data.calendar_exchange,
                start_date="19900101",
                end_date="20991231",
            )
            self.store.write_static(schema.TRADE_CAL, cal)
        cal = cal[cal["exchange"] == self.cfg.data.calendar_exchange]
        open_days = cal[pd.to_numeric(cal["is_open"], errors="coerce") == 1]["cal_date"].astype(str)
        today = pd.Timestamp.today().strftime("%Y%m%d")
        end = self.cfg.data.end_date or today
        end = min(end, today)
        sel = open_days[(open_days >= self.cfg.data.start_date) & (open_days <= end)]
        return sorted(sel.tolist())

    def refresh_calendar(self) -> None:
        cal = self._call(
            "trade_cal",
            exchange=self.cfg.data.calendar_exchange,
            start_date="19900101",
            end_date="20991231",
        )
        self.store.write_static(schema.TRADE_CAL, cal)

    # -------------------------------------------------------------- download --
    def download_static(self) -> None:
        basic_live = self._call_paginated("stock_basic", exchange="", list_status="L", fields=",".join(schema.STOCK_BASIC.columns))
        basic_dead = self._call_paginated("stock_basic", exchange="", list_status="D", fields=",".join(schema.STOCK_BASIC.columns))
        basic_pause = self._call_paginated("stock_basic", exchange="", list_status="P", fields=",".join(schema.STOCK_BASIC.columns))
        basic = pd.concat([basic_live, basic_dead, basic_pause], ignore_index=True)
        if "delist_date" not in basic.columns:
            basic["delist_date"] = pd.NA
        self.store.write_static(schema.STOCK_BASIC, basic)
        log.info("stock_basic: %d symbols (L/D/P combined)", len(basic))

        names = self._call_paginated("namechange", fields=",".join(schema.NAMECHANGE.columns))
        self.store.write_static(schema.NAMECHANGE, names)
        log.info("namechange: %d rows", len(names))

    def download_dated(self, tables: tuple[schema.TableSpec, ...] | None = None) -> None:
        tables = tables or schema.DATE_TABLES
        wanted = self.trade_dates()
        if not wanted:
            raise RuntimeError("no trade dates in the configured range; check data.start_date")
        for spec in tables:
            have = self.store.existing_dates(spec)
            todo = [d for d in wanted if d not in have]
            if not todo:
                log.info("%s: already complete (%d dates)", spec.name, len(wanted))
                continue
            log.info("%s: fetching %d/%d dates", spec.name, len(todo), len(wanted))
            buffer: list[pd.DataFrame] = []
            for i, date in enumerate(tqdm(todo, desc=spec.name, unit="d")):
                df = self._call(spec.api, trade_date=date)
                if df is not None and not df.empty:
                    buffer.append(df)
                # flush periodically so a crash keeps most of the work
                if buffer and (len(buffer) >= 60 or i == len(todo) - 1):
                    self.store.append_dated(spec, pd.concat(buffer, ignore_index=True))
                    buffer.clear()

    def download_all(self) -> None:
        self.refresh_calendar()
        self.download_static()
        self.download_dated()
