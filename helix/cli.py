"""Command line entry point: ``helix <stage>``.

Stages are separately runnable so you can iterate on mining without re-downloading,
and on training without re-mining. ``helix run`` chains them end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from . import pipeline
from .config import Config
from .logging_setup import get_logger, setup_logging

app = typer.Typer(add_completion=False, help="Genetic factor mining for A-share short-horizon signals.")
log = get_logger(__name__)

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to a YAML config (default: configs/default.yaml).")
RebuildOpt = typer.Option(False, "--rebuild", help="Ignore cached panel/base fields and rebuild them.")


def _load(config: Path | None) -> Config:
    setup_logging()
    cfg = Config.load(config)
    log.info("data root: %s", cfg.data.root)
    return cfg


@app.command()
def download(config: Path | None = ConfigOpt) -> None:
    """Fetch every Tushare table Helix needs into the local parquet store."""
    from .data.tushare_source import TushareSource

    cfg = _load(config)
    TushareSource(cfg).download_all()


@app.command()
def prepare(config: Path | None = ConfigOpt, rebuild: bool = RebuildOpt) -> None:
    """Build the panel, universe mask, base fields and labels; report the base rate."""
    cfg = _load(config)
    prepared = pipeline.prepare(cfg, rebuild=rebuild)
    typer.echo(
        f"panel {prepared.panel.shape[0]} dates x {prepared.panel.shape[1]} codes | "
        f"{len(prepared.names)} base fields | "
        f"{int(prepared.labels.valid.sum()):,} usable samples | "
        f"base rate {100 * prepared.labels.base_rate:.3f}%"
    )


@app.command()
def mine(config: Path | None = ConfigOpt, rebuild: bool = RebuildOpt) -> None:
    """Evolve factor expressions on the oldest training block."""
    cfg = _load(config)
    library = pipeline.mine(cfg, pipeline.prepare(cfg, rebuild=rebuild))
    typer.echo(f"kept {len(library.factors)} factors -> {pipeline.artifacts_dir(cfg)/'factors.json'}")


@app.command()
def evaluate(config: Path | None = ConfigOpt) -> None:
    """Score each mined factor in-sample vs. after the search window."""
    from .gp.library import load_factors

    cfg = _load(config)
    prepared = pipeline.prepare(cfg)
    library = load_factors(pipeline.artifacts_dir(cfg) / "factors.json")
    report = pipeline.evaluate_factors(cfg, prepared, library)
    path = pipeline.artifacts_dir(cfg) / "factor_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command()
def train(config: Path | None = ConfigOpt) -> None:
    """Walk-forward train the combiner over the mined factors."""
    from .gp.library import load_factors

    cfg = _load(config)
    prepared = pipeline.prepare(cfg)
    library = load_factors(pipeline.artifacts_dir(cfg) / "factors.json")
    _, results = pipeline.train(cfg, prepared, library)
    for r in results:
        typer.echo(f"fold {r.fold}: valid gini {r.valid_gini:.4f} | test gini {r.test_gini:.4f}")


@app.command()
def backtest(config: Path | None = ConfigOpt) -> None:
    """Backtest the stitched out-of-sample predictions."""
    import numpy as np

    cfg = _load(config)
    prepared = pipeline.prepare(cfg)
    path = pipeline.artifacts_dir(cfg) / "predictions.npz"
    if not path.exists():
        raise typer.BadParameter("no predictions.npz; run `helix train` first")
    with np.load(path, allow_pickle=False) as z:
        predictions = z["predictions"]
    summary = pipeline.backtest(cfg, prepared, predictions)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def score(
    config: Path | None = ConfigOpt,
    date: str = typer.Option("", "--date", help="Trade date YYYYMMDD to score (default: latest)."),
    top: int = typer.Option(20, "--top", help="How many candidates to print."),
) -> None:
    """Rank today's tradable universe with the most recently trained fold."""
    from .gp.library import load_factors

    cfg = _load(config)
    prepared = pipeline.prepare(cfg)
    library = load_factors(pipeline.artifacts_dir(cfg) / "factors.json")
    frame = pipeline.score(cfg, prepared, library, date=date)
    typer.echo(frame.head(top).to_string(index=False))


@app.command()
def run(config: Path | None = ConfigOpt, rebuild: bool = RebuildOpt) -> None:
    """prepare -> mine -> train -> backtest."""
    cfg = _load(config)
    summary = pipeline.run_all(cfg, rebuild=rebuild)
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
