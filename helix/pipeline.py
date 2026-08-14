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
import pandas as pd

from .config import Config
from .data.panel import Panel, build_panel
from .data.price_lineage import PriceLineageError
from .data.store import ParquetStore
from .data.universe import build_universe
from .dl.checkpoint import latest_checkpoint, load_checkpoint, require_matching_factors
from .dl.dataset import SequenceDataset, normalize_factors
from .dl.models import pick_device
from .dl.train import predict_scores, train_walk_forward
from .eval.backtest import run_backtest
from .eval.factor_monitor import evaluate_training_monitors
from .eval.metrics import lift_at_k
from .features.base_fields import compute_base_fields, field_names
from .features.operators import lead
from .gp.engine import liquidity_top_columns, run_search
from .gp.library import FactorLibrary, compute_factors, load_factors, save_factors
from .labels.touch_label import LabelSet, build_touch_label
from .logging_setup import get_logger
from .splits import complete_outcome_window, search_window, walk_forward

log = get_logger(__name__)

PANEL_CACHE_REQUIRED_FIELDS = ("limit_price_observed",)
PANEL_CACHE_REQUIRED_ADJUSTED_FIELDS = ("open_hfq", "high_hfq", "low_hfq", "close_hfq")


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


def models_dir(cfg: Config) -> Path:
    path = artifacts_dir(cfg) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare(cfg: Config, rebuild: bool = False) -> Prepared:
    """Build (or reload) the panel, universe mask, base fields and labels."""
    store = ParquetStore(cfg.data.root)
    panel_path = cache_dir(cfg) / "panel.npz"
    rebuild_panel = rebuild or not panel_path.exists()
    if not rebuild_panel:
        try:
            panel = Panel.load(panel_path)
            missing = [field for field in PANEL_CACHE_REQUIRED_FIELDS if field not in panel]
            if missing:
                raise PriceLineageError(f"missing required fields {missing}")
            panel.require_adjusted_prices(PANEL_CACHE_REQUIRED_ADJUSTED_FIELDS, "cached panel")
        except PriceLineageError as exc:
            log.warning("cached panel lacks valid provenance: %s; rebuilding", exc)
            rebuild_panel = True
            rebuild = True
        else:
            log.info("loaded cached panel %d dates x %d codes", *panel.shape)
    if rebuild_panel:
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
    rows = complete_outcome_window(
        search_window(len(panel.dates), cfg.split), cfg.label.touch_offset
    )
    log.info(
        "GP search window: %s ~ %s (%d dates)",
        panel.dates[rows][0], panel.dates[rows][-1], len(panel.dates[rows]),
    )

    population_rows = prepared.universe[rows]
    cols = liquidity_top_columns(
        panel.f64("amount")[rows], population_rows, cfg.gp.search_max_stocks
    )
    log.info("evolving on %d liquidity-ranked stocks", len(cols))

    sub_fields = {k: np.asarray(v[rows][:, cols], dtype=np.float64) for k, v in prepared.fields.items()}
    gross = np.where(
        labels.valid[rows],
        labels.exit_price[rows] / labels.entry_price[rows] - 1.0,
        np.nan,
    )
    result = run_search(
        fields=sub_fields,
        field_names=prepared.names,
        gross_returns=gross[:, cols],
        candidate_mask=prepared.universe[rows][:, cols],
        dates=panel.dates[rows],
        cfg=cfg.gp,
        backtest_cfg=cfg.backtest,
        entry_offset=cfg.label.entry_offset,
        touch_offset=cfg.label.touch_offset,
        embargo_days=cfg.split.embargo_days,
    )
    save_factors(artifacts_dir(cfg) / "factors.json", result.library)
    return result.library


def evaluate_factors(cfg: Config, prepared: Prepared, library: FactorLibrary) -> dict[str, dict]:
    """Report economic and label monitors using outcome-complete training rows only."""
    names, values = compute_factors(library, prepared.fields)
    labels = prepared.labels
    panel = prepared.panel
    rows = complete_outcome_window(
        search_window(len(panel.dates), cfg.split), cfg.label.touch_offset
    )
    gross = np.where(
        labels.valid[rows],
        labels.exit_price[rows] / labels.entry_price[rows] - 1.0,
        np.nan,
    )
    high_at_exit = lead(panel.f64("high_hfq"), cfg.label.touch_offset)
    peak = np.where(
        labels.valid[rows],
        high_at_exit[rows] / labels.entry_price[rows] - 1.0,
        np.nan,
    )
    specs = {factor.name: factor for factor in library.factors}

    report: dict[str, dict] = {}
    for k, name in enumerate(names):
        monitors = evaluate_training_monitors(
            score=values[rows, :, k].astype(np.float64),
            hit_label=labels.y[rows],
            peak_return=peak,
            gross_return=gross,
            candidate_mask=prepared.universe[rows],
            dates=panel.dates[rows],
            config=cfg.backtest,
            entry_offset=cfg.label.entry_offset,
            touch_offset=cfg.label.touch_offset,
            embargo_days=cfg.split.embargo_days,
            min_samples=cfg.gp.min_daily_samples,
        )
        spec = specs[name]
        report[name] = {"expression": spec.expression, "sign": spec.sign, **monitors}
        fit_net = monitors["fit"]["production_objective"]["net"]["mean"]
        selection_net = monitors["selection"]["production_objective"]["net"]["mean"]
        log.info(
            "%s | training fit Top%d net %.4f%% | selection net %.4f%%",
            name,
            cfg.backtest.top_k,
            100 * fit_net,
            100 * selection_net,
        )
    return report


def _normalized_factors(cfg: Config, prepared: Prepared, library: FactorLibrary):
    """Cross-sectionally normalise the factor panel against the D0 tradable universe.

    The universe -- not ``labels.valid`` -- is the right population here. ``labels.valid``
    additionally depends on whether D+1 opened limit-up and whether D+1/D+2 traded, so
    normalising against it would let future information shape the statistics applied to
    D0 features. It also leaves the most recent rows entirely undefined, which is exactly
    where live scoring needs values.
    """
    names, values = compute_factors(library, prepared.fields)
    normalized = normalize_factors(values, prepared.universe, cfg.dl.clip_sigma)
    traded = (prepared.panel["is_trading"] > 0).astype(np.float32)
    return names, normalized, traded


def train(cfg: Config, prepared: Prepared, library: FactorLibrary) -> tuple[np.ndarray, list]:
    """Walk-forward train the combiner and return the stitched out-of-sample scores."""
    names, normalized, traded = _normalized_factors(cfg, prepared, library)
    log.info("combining %d factors: %s", len(names), ", ".join(names))
    labels = prepared.labels

    folds = walk_forward(len(prepared.panel.dates), cfg.split)
    for fold in folds:
        log.info(fold.describe(prepared.panel.dates))

    predictions, results = train_walk_forward(
        folds=folds,
        values=normalized,
        traded=traded,
        y=labels.y,
        label_mask=labels.valid,
        prediction_mask=(
            prepared.universe & (np.isfinite(normalized).mean(axis=-1) >= 0.5)
        ),
        cfg=cfg.dl,
        checkpoint_dir=models_dir(cfg),
        factor_names=names,
        dates=prepared.panel.dates,
    )
    np.savez_compressed(
        artifacts_dir(cfg) / "predictions.npz",
        predictions=predictions,
        dates=prepared.panel.dates,
        codes=prepared.panel.codes,
    )
    return predictions, results


def score(
    cfg: Config,
    prepared: Prepared,
    library: FactorLibrary,
    date: str = "",
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Rank the tradable universe for one D0, using the most recently trained fold.

    Scoring the *latest* date is the whole point: its features are complete at the D0
    close while its label is necessarily undefined (D+2 has not happened). That is why
    the candidate mask here is the D0 universe rather than ``labels.valid``, which
    encodes outcomes that do not exist yet.
    """
    names, normalized, traded = _normalized_factors(cfg, prepared, library)
    panel = prepared.panel
    dates = panel.dates

    if date:
        pos = int(np.searchsorted(dates, date, "left"))
        if pos >= len(dates) or dates[pos] != date:
            raise ValueError(f"{date} is not a trade date in the panel")
        t = pos
    else:
        t = len(dates) - 1

    path = Path(checkpoint_path) if checkpoint_path else latest_checkpoint(models_dir(cfg))
    ckpt = load_checkpoint(path)
    require_matching_factors(ckpt, names)
    if t < ckpt.seq_len - 1:
        raise ValueError(f"need {ckpt.seq_len} dates of history to score {dates[t]}")
    log.info(
        "scoring %s with fold %d (trained through %s)", dates[t], ckpt.fold, ckpt.train_end or "?"
    )

    # A name needs most of its factors defined; the dataset zero-fills the rest.
    defined = np.isfinite(normalized[t]).mean(axis=-1) >= 0.5
    candidates = np.flatnonzero(prepared.universe[t] & defined)
    if candidates.size == 0:
        raise RuntimeError(f"no tradable candidates on {dates[t]}")

    index = np.column_stack([np.full(candidates.size, t), candidates]).astype(np.int64)
    dataset = SequenceDataset(normalized, traded, np.zeros(panel.shape), index, ckpt.seq_len)
    device = pick_device()
    probabilities = predict_scores(ckpt.build(device), dataset, device, cfg.dl.batch_size)

    frame = pd.DataFrame(
        {
            "date": dates[t],
            "ts_code": panel.codes[candidates],
            "probability": probabilities,
            "close": panel.f64("close")[t, candidates],
            "amount_kcny": panel.f64("amount")[t, candidates],
            "to_up_limit": (
                panel.f64("up_limit")[t, candidates] / panel.f64("close")[t, candidates] - 1.0
            ),
        }
    ).sort_values("probability", ascending=False, ignore_index=True)
    frame.insert(0, "rank", frame.index + 1)

    out = artifacts_dir(cfg) / f"scores_{dates[t]}.csv"
    frame.to_csv(out, index=False)
    log.info("scored %d candidates on %s -> %s", len(frame), dates[t], out.name)
    return frame


def backtest(cfg: Config, prepared: Prepared, predictions: np.ndarray) -> dict:
    result = run_backtest(
        predictions=predictions,
        labels=prepared.labels,
        candidate_mask=prepared.universe,
        dates=prepared.panel.dates,
        label_cfg=cfg.label,
        cfg=cfg.backtest,
        panel=prepared.panel if cfg.backtest.enable_realistic_exit else None,
    )
    out = artifacts_dir(cfg)
    if not result.daily.empty:
        result.daily.to_csv(out / "backtest_daily.csv", index=False)

    tested = np.isfinite(predictions)
    summary = dict(result.summary)
    for k in (5, 10, 20, 50):
        summary[f"lift_at_{k}"] = lift_at_k(
            predictions, prepared.labels.y, prepared.universe & tested, k
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
