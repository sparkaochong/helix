#!/usr/bin/env python3
"""Give every column that lacks one a per-date cross-sectional rank -- plus a placebo.

The one thing GP can build that a row-wise tree cannot replicate is a cross-sectional
operator: ``cs_rank`` needs the whole day's cross-section, a tree sees one row at a time.
The argus_quant table has 459 numeric features of which only 39 are rank-shaped, so 402
are consumed as absolute levels while the daily pool swings between 174 and 2656 names --
a threshold learned on a thin day is not the same threshold on a fat one.

Whether that gap is worth anything is far cheaper to settle by handing the model the
answer than by spending GP budget hunting for it. Rank all 402 within their date, ablate,
and if the ranks themselves buy nothing then searching for cross-sectional structure is
pointless.

A null result from that alone would be unreadable, though. Adding 402 columns to a 459
column feature set dilutes ``colsample_bytree``, so "the information is worthless" and
"the signal was diluted away" produce the same number. Hence the second output: a placebo
arm holding the same 402 rank columns, except every row within a date receives some other
row's vector. Same marginal distributions, same inter-column correlations, same column
count, no alignment to the target. The real arm has to beat the placebo, not merely beat
the base. This is the same discipline as the positive control in ``check_suspension.py``:
a zero that no control validates is not evidence.

Depends only on numpy / pandas / pyarrow, so it runs on whichever host holds the table --
the ablation itself needs a gradient-boosting library that the mining host may not have.

    python make_cs_columns.py --input train.parquet --out-dir artifacts/cs
    python ablate_factors.py --input artifacts/cs/train_cs.parquet \
        --features artifacts/cs/base_features.json \
        --factors  artifacts/cs/cs_real.json \
        --split-date 2024-09-04 --embargo-days 3 --seeds 7,17,27 \
        --report ablation_cs_real.json
    # ...and again with --factors cs_placebo.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATE_COLUMN = "trade_date"
CODE_COLUMN = "stock_code"

#: Outcome columns are excluded by prefix, never by an enumerated list -- the table
#: carries ``label_d2_hit_3pct`` and ``label_d2_hit_5pct`` beside the 8% target and any
#: one of them left in the feature set predicts the others almost perfectly.
#: Mirrors ``helix.data.event_table.LABEL_PREFIXES``; duplicated rather than imported so
#: this file stays runnable on a host without helix installed, and
#: ``tests/test_cs_columns.py`` asserts the two agree on the real schema.
LABEL_PREFIXES: tuple[str, ...] = ("label", "target", "y_", "fwd_", "future_")

#: A column whose name ends in one of these already is a cross-sectional rank.
RANK_SUFFIXES: tuple[str, ...] = ("_rank", "_pctl", "_rank_pct", "_pct_rank")

#: Doubled underscore: no source column can collide with a generated name.
REAL_SUFFIX = "__csr"
PLACEBO_SUFFIX = "__csr_shuf"


def is_label_column(name: str, prefixes: tuple[str, ...] = LABEL_PREFIXES) -> bool:
    lowered = name.lower()
    return any(lowered.startswith(p) for p in prefixes)


def numeric_feature_names(
    schema: pa.Schema,
    meta_columns: tuple[str, ...] = (DATE_COLUMN, CODE_COLUMN),
) -> list[str]:
    """Numeric non-outcome columns -- the feature set a model would actually be handed."""
    names = [
        name
        for name in schema.names
        if name not in meta_columns
        and not is_label_column(name)
        and any(k in str(schema.field(name).type) for k in ("int", "float", "double"))
    ]
    leaked = [n for n in names if is_label_column(n)]
    if leaked:
        raise AssertionError(f"outcome columns reached the feature set: {leaked}")
    return names


def select_columns(features: list[str]) -> list[str]:
    """The feature columns with no cross-sectional version anywhere in the table.

    Two exclusions. A name ending in ``_rank`` / ``_pctl`` already is one -- the
    ``_pool_rank`` family is ranked inside the same daily pool this script would rank
    over, so re-ranking it produces a copy. And a column with a sibling ``{name}_rank``
    has one sitting next to it, which is the same objection one column removed.
    """
    known = set(features)
    return [c for c in features if not c.endswith(RANK_SUFFIXES) and f"{c}_rank" not in known]


def cs_rank_block(block: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Percentile rank of each column within one date's rows.

    NaN stays NaN rather than becoming a middling rank: gradient-boosting libraries route
    missing values down their own branch, and imputing a rank would destroy that. Ranks
    live in [0, 1] so float32 is lossless at any precision this experiment can resolve.
    """
    return block[columns].rank(pct=True).to_numpy(dtype=np.float32)


def placebo_block(ranks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same day's rank vectors, handed to the wrong rows.

    One permutation shared by all columns, not one per column. That keeps every placebo
    row a real and internally coherent vector and preserves the inter-column correlation
    structure, so the arm differs from the real one in exactly one respect: which row it
    describes. Per-column permutations would also destroy the correlations, and the arm
    would then be a weaker control than the thing it is controlling for.
    """
    return ranks[rng.permutation(len(ranks))]


def iter_date_blocks(path: Path, batch_size: int):
    """Yield ``(date, rows)`` one whole trade date at a time.

    The table is a single row group, so there is no row-group-at-a-time read; batching is
    the only way to avoid materialising 617k x 470 float64 at once. A date can straddle a
    batch boundary and a cross-sectional rank needs the whole date, so partial dates are
    held back. Global date ordering is asserted rather than assumed -- out-of-order rows
    would silently rank a date against a fragment of itself.
    """
    pending: list[pd.DataFrame] = []
    pending_date = None
    emitted: str | None = None

    def _check(date) -> None:
        nonlocal emitted
        if emitted is not None and not date > emitted:
            raise SystemExit(
                f"{DATE_COLUMN} is not sorted ({date!r} follows {emitted!r}); "
                "a cross-sectional rank would be computed over part of a date"
            )
        emitted = date

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        for date, block in batch.to_pandas().groupby(DATE_COLUMN, sort=True):
            if pending_date is not None and date != pending_date:
                _check(pending_date)
                yield pending_date, pd.concat(pending, ignore_index=True)
                pending = []
            pending_date = date
            pending.append(block)
    if pending_date is not None:
        _check(pending_date)
        yield pending_date, pd.concat(pending, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="artifacts/cs")
    ap.add_argument("--output", default="", help="Defaults to <out-dir>/<stem>_cs.parquet")
    ap.add_argument("--seed", type=int, default=20260812, help="Placebo permutations only.")
    ap.add_argument("--batch-size", type=int, default=65536, help="Rows per read batch.")
    ap.add_argument("--row-group-rows", type=int, default=50000,
                    help="Approximate rows per output row group.")
    ap.add_argument("--limit-dates", type=int, default=0,
                    help="Stop after this many dates. 0 processes all; a small value is "
                         "the cheap end-to-end check before committing an hour to the "
                         "full table.")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = Path(args.output) if args.output else out_dir / f"{src.stem}_cs.parquet"

    schema = pq.ParquetFile(src).schema_arrow
    features = numeric_feature_names(schema)
    columns = select_columns(features)
    real_names = [f"{c}{REAL_SUFFIX}" for c in columns]
    placebo_names = [f"{c}{PLACEBO_SUFFIX}" for c in columns]

    clash = sorted(set(real_names + placebo_names) & set(schema.names))
    if clash:
        raise SystemExit(f"generated names collide with source columns: {clash[:5]}")

    print(f"{len(features)} numeric features | {len(columns)} lack a cross-sectional "
          f"version | adding {2 * len(columns)} columns")
    for name, payload in (("base_features.json", features),
                          ("cs_real.json", real_names),
                          ("cs_placebo.json", placebo_names)):
        (out_dir / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"  wrote {out_dir / name} ({len(payload)} columns)")

    rng = np.random.default_rng(args.seed)
    added_names = real_names + placebo_names
    writer: pq.ParquetWriter | None = None
    buffered: list[pd.DataFrame] = []
    buffered_rows = 0
    n_dates = n_rows = 0

    def flush() -> None:
        nonlocal writer, buffered, buffered_rows
        if not buffered:
            return
        table = pa.Table.from_pandas(pd.concat(buffered, ignore_index=True),
                                     schema=writer.schema if writer else None,
                                     preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dest, table.schema, compression=args.compression)
        writer.write_table(table)
        buffered, buffered_rows = [], 0

    try:
        for _date, block in iter_date_blocks(src, args.batch_size):
            real = cs_rank_block(block, columns)
            added = pd.DataFrame(np.hstack([real, placebo_block(real, rng)]),
                                 columns=added_names)
            buffered.append(pd.concat([block.reset_index(drop=True), added], axis=1))
            buffered_rows += len(block)
            n_dates += 1
            n_rows += len(block)
            if buffered_rows >= args.row_group_rows:
                flush()
            if n_dates % 100 == 0:
                print(f"  {n_dates} dates / {n_rows:,} rows", flush=True)
            if args.limit_dates and n_dates >= args.limit_dates:
                print(f"  stopping at --limit-dates {args.limit_dates}")
                break
        flush()
    finally:
        if writer is not None:
            writer.close()

    size_mb = dest.stat().st_size / 1e6 if dest.exists() else 0.0
    print(f"wrote {dest} | {n_dates} dates | {n_rows:,} rows | "
          f"{len(schema.names) + len(added_names)} columns | {size_mb:,.0f} MB")


if __name__ == "__main__":
    main()
