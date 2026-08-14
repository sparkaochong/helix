#!/usr/bin/env python3
"""Mine GP factors from the argus_quant event table and export the apply script.

Runs in three passes so a 459-column table never sits in memory as slot grids at once:

1. stream every feature column, score its univariate IC on the search window;
2. keep the top uncorrelated ones and pack only those;
3. evolve factors, measure IC/IC_IR after the search window, write the apply script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from helix.config import Config
from helix.data.event_lineage import audit_column_names, load_event_lineage
from helix.data.event_table import (
    EventPanel,
    numeric_feature_columns,
    open_event_source,
    stream_feature_grids,
)
from helix.eval.ic import daily_ic, summarize_ic
from helix.features.operators import cs_rank
from helix.gp.engine import run_search
from helix.gp.event_primitives import build_event_pset
from helix.gp.feature_select import _max_abs_corr
from helix.gp.library import FactorLibrary, compute_factors
from helix.gp.neutralize import build_basis
from helix.logging_setup import get_logger, setup_logging
from helix.pipeline_events import (
    DEFAULT_LABELS,
    PRIMARY_TARGET,
    RETURN_TARGET,
    EventRun,
    evaluate_ic,
    save,
)
from helix.splits import complete_outcome_window

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--out", default="data/artifacts/argus")
    ap.add_argument("--config", default=None)
    ap.add_argument("--search-fraction", type=float, default=0.6)
    ap.add_argument("--n-features", type=int, default=80)
    ap.add_argument("--feature-corr", type=float, default=0.85)
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=1,
                    help="Mining rounds; each neutralises against all factors found so far.")
    ap.add_argument("--neutralize-base", type=int, default=0,
                    help="Neutralise against this many of the top-IC source columns. "
                         "0 disables neutralisation entirely (raw-IC mining).")
    args = ap.parse_args()

    setup_logging()
    cfg = Config.load(args.config)
    path = Path(args.input)
    labels = list(DEFAULT_LABELS)
    manifest = load_event_lineage(args.lineage)
    features = numeric_feature_columns(
        path, labels, extra_excluded=tuple(audit_column_names(manifest))
    )

    # ---------------------------------------------------------------- pass 1 --
    index, label_grids, keys = open_event_source(
        path,
        labels,
        lineage_path=args.lineage,
        feature_columns=features,
    )
    n_dates = len(index.dates)
    rows = complete_outcome_window(
        slice(0, max(int(n_dates * args.search_fraction), 1)), cfg.label.touch_offset
    )
    log.info(
        "search window %s ~ %s (%d/%d dates); IC is reported on the %d dates after it",
        index.dates[rows][0], index.dates[rows][-1], len(index.dates[rows]), n_dates,
        n_dates - rows.stop,
    )

    log.info("streaming %d feature columns for univariate screening", len(features))

    mask = index.occupied[rows]
    target = label_grids[PRIMARY_TARGET][rows].astype(np.float64)
    scored: list[tuple[str, float, float]] = []
    for name, grid in stream_feature_grids(path, keys, index, features):
        stats = summarize_ic(daily_ic(grid[rows].astype(np.float64), target, mask, args.min_samples))
        if np.isfinite(stats["ic_mean"]):
            scored.append((name, stats["ic_mean"], stats["icir"]))
    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    log.info("top univariate features:")
    for name, ic, icir in scored[:15]:
        log.info("  %-36s IC %+.5f  ICIR %+.3f", name, ic, icir)

    # ---------------------------------------------------------------- pass 2 --
    kept: list[str] = []
    kept_ranks: list[np.ndarray] = []
    wanted = {n for n, _, _ in scored}
    grids: dict[str, np.ndarray] = {}
    for name, grid in stream_feature_grids(path, keys, index, [n for n, _, _ in scored]):
        if name not in wanted or len(kept) >= args.n_features:
            continue
        ranks = cs_rank(np.where(mask, grid[rows].astype(np.float64), np.nan)).ravel()
        if _max_abs_corr(ranks, kept_ranks) > args.feature_corr:
            continue
        kept.append(name)
        kept_ranks.append(ranks)
        grids[name] = grid
    del kept_ranks
    log.info("selected %d features after correlation dedup", len(kept))

    panel = EventPanel(
        dates=index.dates, codes=index.codes, occupied=index.occupied,
        fields={k: grids[k] for k in kept}, labels=label_grids,
    )

    # ---------------------------------------------------------------- pass 3 --
    # Rounds. Each one neutralises against the strongest existing columns plus every
    # factor already found, so a later factor can only score by explaining something
    # none of them can. Without this the search just keeps rediscovering the same
    # linear blend of the same few volatility columns.
    search_fields = {k: panel.fields[k][rows].astype(np.float64) for k in kept}
    gross_returns = panel.f64(RETURN_TARGET)[rows]
    pset = build_event_pset(kept)

    base_grids = [search_fields[n] for n in kept[: args.neutralize_base]]
    log.info("neutralising against %d base columns", len(base_grids))

    all_specs: list = []
    for round_index in range(args.rounds):
        basis = build_basis(base_grids, mask) if base_grids else None
        log.info("=== round %d/%d | basis rank %d ===",
                 round_index + 1, args.rounds, 0 if basis is None else basis.shape[2])
        cfg.gp.seed = cfg.gp.seed + round_index
        result = run_search(
            fields=search_fields,
            field_names=kept,
            gross_returns=gross_returns,
            candidate_mask=mask,
            dates=panel.dates[rows],
            cfg=cfg.gp,
            backtest_cfg=cfg.backtest,
            entry_offset=cfg.label.entry_offset,
            touch_offset=cfg.label.touch_offset,
            embargo_days=cfg.split.embargo_days,
            pset=pset, kind="event", basis=basis,
        )
        if not result.library.factors:
            log.warning("round %d found nothing that survived selection; stopping",
                        round_index + 1)
            break
        for spec in result.library.factors:
            spec.name = f"gp_{len(all_specs):03d}"
            all_specs.append(spec)
        # Feed this round's factors into the next round's basis.
        found = FactorLibrary(factors=result.library.factors, field_names=kept,
                              windows=[], kind="event")
        _, values = compute_factors(found, search_fields)
        base_grids.extend(values[:, :, k].astype(np.float64) for k in range(values.shape[2]))

    library = FactorLibrary(factors=all_specs, field_names=kept, windows=[], kind="event")
    run = EventRun(panel=panel, library=library, selected_features=kept,
                   search_rows=rows, config=cfg, report={})
    evaluate_ic(run, min_samples=args.min_samples)
    paths = save(run, Path(args.out))
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
