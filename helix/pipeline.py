"""Stage wiring: panel -> universe -> fields -> labels -> GP -> DL -> backtest.

Each stage caches its output under ``{data.root}/cache`` so the expensive parts
(downloading, pivoting the panel, computing base fields) run once and the parts you
actually iterate on -- mining and training -- start in seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .data.panel import Panel, build_panel
from .data.store import ParquetStore
from .data.universe import build_universe
from .dl.dataset import normalize_factors
from .dl.train import train_walk_forward
from .eval.backtest import run_backtest
from .eval.metrics import daily_gini, lift_at_k, summarize_daily
from .features.base_fields import compute_base_fields, field_names
from .gp.engine import liquidity_top_columns, run_search
from .gp.library import FactorLibrary, compute_factors, load_factors, save_factors
from .labels.touch_label import LabelSet, build_touch_label
from .logging_setup import get_logger
from .splits import search_window, walk_forward

log = get_logger(__name__)


@dataclass
class Prepared:
    panel: Panel
    universe: np.ndarray
    fields: dict[str, np.ndarray]
    names: list[str]
    labels: LabelSet


def artifacts_dir(cfg: Config) -> Path:
    path = cfg.data.root / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(cfg: Config) -> Path:
    path = cfg.data.root / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare(cfg: Config, rebuild: bool = False) -> Prepared:
    """Build (or reload) the panel, universe mask, base fields and labels."""
    store = ParquetStore(cfg.data.root)
    panel_path = cache_dir(cfg) / "panel.npz"
    if panel_path.exists() and not rebuild:
        panel = Panel.load(panel_path)
        log.info("loaded cached panel %d dates x %d codes", *panel.shape)
    else:
        panel = build_panel(store, cfg.data.start_date, cfg.data.end_date)
        panel.save(panel_path)

    fields_path = cache_dir(cfg) / "base_fields.npz"
    if fields_path.exists() and not rebuild:
        with np.load(fields_path, allow_pickle=False) as z:
            fields = {k: z[k] for k in z.files}
        log.info("loaded %d cached base fields", len(fields))
    else:
        fields = compute_base_fields(panel)
        np.savez_compressed(fields_path, **{k: v.astype(np.float32) for k, v in fields.items()})

    universe = build_universe(panel, store, cfg.universe)
    labels = build_touch_label(panel, universe, cfg.label)
    return Prepared(panel=panel, universe=universe, fields=fields, names=field_names(fields), labels=labels)


def mine(cfg: Config, prepared: Prepared) -> FactorLibrary:
    """Run GP discovery on the oldest training block only."""
    panel, labels = prepared.panel, prepared.labels
    rows = search_window(len(panel.dates), cfg.split)
    log.info(
        "GP search window: %s ~ %s (%d dates)",
        panel.dates[rows][0], panel.dates[rows][-1], len(panel.dates[rows]),
    )

    mask_rows = labels.valid[rows]
    cols = liquidity_top_columns(panel.f64("amount")[rows], mask_rows, cfg.gp.search_max_stocks)
    log.info("evolving on %d liquidity-ranked stocks", len(cols))

    sub_fields = {k: np.asarray(v[rows][:, cols], dtype=np.float64) for k, v in prepared.fields.items()}
    result = run_search(
        fields=sub_fields,
        field_names=prepared.names,
        y=labels.y[rows][:, cols],
        mask=mask_rows[:, cols],
        cfg=cfg.gp,
        embargo_days=cfg.split.embargo_days,
    )
    save_factors(artifacts_dir(cfg) / "factors.json", result.library)
    return result.library


def evaluate_factors(cfg: Config, prepared: Prepared, library: FactorLibrary) -> dict[str, dict]:
    """Per-factor daily gini on the full panel, split into search and post-search rows."""
    names, values = compute_factors(library, prepared.fields)
    labels = prepared.labels
    rows = search_window(len(prepared.panel.dates), cfg.split)
    after = slice(rows.stop, len(prepared.panel.dates))

    report: dict[str, dict] = {}
    for k, name in enumerate(names):
        layer = values[:, :, k].astype(np.float64)
        in_sample = summarize_daily(
            daily_gini(layer[rows], labels.y[rows], labels.valid[rows], cfg.gp.min_daily_samples)
        )
        out_sample = summarize_daily(
            daily_gini(layer[after], labels.y[after], labels.valid[after], cfg.gp.min_daily_samples)
        )
        report[name] = {"in_sample": in_sample, "out_of_sample": out_sample}
        log.info(
            "%s | in-sample gini %.4f (IR %.2f) | out-of-sample gini %.4f (IR %.2f)",
            name, in_sample["mean"], in_sample["ir"], out_sample["mean"], out_sample["ir"],
        )
    return report


def train(cfg: Config, prepared: Prepared, library: FactorLibrary) -> tuple[np.ndarray, list]:
    """Walk-forward train the combiner and return the stitched out-of-sample scores."""
    names, values = compute_factors(library, prepared.fields)
    log.info("combining %d factors: %s", len(names), ", ".join(names))

    labels = prepared.labels
    normalized = normalize_factors(values, labels.valid, cfg.dl.clip_sigma)
    traded = (prepared.panel["is_trading"] > 0).astype(np.float32)

    folds = walk_forward(len(prepared.panel.dates), cfg.split)
    for fold in folds:
        log.info(fold.describe(prepared.panel.dates))

    predictions, results = train_walk_forward(
        folds=folds,
        values=normalized,
        traded=traded,
        y=labels.y,
        mask=labels.valid,
        cfg=cfg.dl,
    )
    np.savez_compressed(
        artifacts_dir(cfg) / "predictions.npz",
        predictions=predictions,
        dates=prepared.panel.dates,
        codes=prepared.panel.codes,
    )
    return predictions, results


def backtest(cfg: Config, prepared: Prepared, predictions: np.ndarray) -> dict:
    result = run_backtest(
        predictions=predictions,
        labels=prepared.labels,
        dates=prepared.panel.dates,
        label_cfg=cfg.label,
        cfg=cfg.backtest,
    )
    out = artifacts_dir(cfg)
    if not result.daily.empty:
        result.daily.to_csv(out / "backtest_daily.csv", index=False)

    tested = np.isfinite(predictions)
    summary = dict(result.summary)
    for k in (5, 10, 20, 50):
        summary[f"lift_at_{k}"] = lift_at_k(
            predictions, prepared.labels.y, prepared.labels.valid & tested, k
        )
    (out / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_all(cfg: Config, rebuild: bool = False) -> dict:
    prepared = prepare(cfg, rebuild=rebuild)
    library_path = artifacts_dir(cfg) / "factors.json"
    library = load_factors(library_path) if library_path.exists() else mine(cfg, prepared)
    predictions, _ = train(cfg, prepared, library)
    return backtest(cfg, prepared, predictions)
