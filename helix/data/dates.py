"""Plain string normalization for trade-date formats.

``data/raw/*`` (the ingested Tushare tables) uniformly use ``YYYYMMDD``. The legacy
``data/raw/argus_quant_working.parquet`` event table uses ``YYYY-MM-DD`` instead --
joining the two without normalizing first silently returns nothing (every row misses).

This is deliberately separate from the date parsing already in ``price_lineage.py``
(``_parse_source_date``) and ``event_lineage.py`` (``_parse_date``): those are
private, return ``datetime.date`` objects, and exist specifically to feed their
modules' fail-closed lineage validators. This module returns a canonical ``YYYYMMDD``
*string* -- the raw-store convention -- for the different, currently-hypothetical use
case of joining the legacy event table against the raw tables. It does not rewrite
``argus_quant_working.parquet`` itself; that file is a frozen legacy artifact per
``docs/factor-governance.md``.
"""

from __future__ import annotations

import re

_YYYYMMDD = re.compile(r"\d{8}")
_YYYY_MM_DD = re.compile(r"\d{4}-\d{2}-\d{2}")


def normalize_trade_date(value: str) -> str:
    """Return ``value`` as ``YYYYMMDD``, accepting either ``YYYYMMDD`` or ``YYYY-MM-DD``."""
    text = str(value).strip()
    if _YYYYMMDD.fullmatch(text):
        return text
    if _YYYY_MM_DD.fullmatch(text):
        return text.replace("-", "")
    raise ValueError(f"unrecognized trade_date format: {value!r}")
