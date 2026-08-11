"""Event-table mining flow: load -> select features -> evolve -> score IC -> export.

The deliverable of this pipeline is **factor columns**, not predictions. Everything is
oriented around producing expressions that can be appended to the source training table
and evaluated with IC / IC_IR.

Date discipline is the same as the panel pipeline and matters just as much here: the
search window is the oldest block of dates, feature screening happens inside it, and the
IC that gets reported is measured on the dates after it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data.event_table import EventPanel, assert_no_label_columns, load_event_panel
from .eval.ic import daily_ic, summarize_ic
from .gp.engine import run_search
from .gp.event_primitives import build_event_pset
from .gp.export import write_apply_script
from .gp.feature_select import select_features
from .gp.library import FactorLibrary, compute_factors, save_factors
from .logging_setup import get_logger

log = get_logger(__name__)

#: Continuous target is primary -- it uses the full magnitude of the D+2 excursion
#: instead of collapsing it to a yes/no, so IC is far less noisy per date.
PRIMARY_TARGET = "label_d2_peak_return"
BINARY_TARGET = "label_d2_hit_8pct"
DEFAULT_LABELS = (
    PRIMARY_TARGET, BINARY_TARGET, "label_d2_return",
    "label_px_d1_open", "label_px_d2_high", "label_px_d2_close",
)


@dataclass
class EventRun:
    panel: EventPanel
    library: FactorLibrary
    selected_features: list[str]
    search_rows: slice
    report: dict


def _search_rows(n_dates: int, fraction: float) -> slice:
    return slice(0, max(int(n_dates * fraction), 1))


def load(path: Path, labels: tuple[str, ...] = DEFAULT_LABELS) -> EventPanel:
    panel = load_event_panel(Path(path), label_columns=[c for c in labels])
    log.info("loaded %d features, %d rows", len(panel.fields), panel.n_rows)
    return panel


def mine_events(
    panel: EventPanel,
    cfg: Config,
    search_fraction: float = 0.6,
    n_features: int = 80,
    target: str = PRIMARY_TARGET,
    feature_max_abs_corr: float = 0.85,
) -> EventRun:
    """Screen features and evolve factors, both confined to the search window."""
    rows = _search_rows(len(panel.dates), search_fraction)
    log.info(
        "search window: %s ~ %s (%d/%d dates); evaluation uses everything after",
        panel.dates[rows][0], panel.dates[rows][-1], len(panel.dates[rows]), len(panel.dates),
    )

    mask = panel.occupied[rows]
    y_target = panel.f64(target)[rows]

    selected, scores = select_features(
        fields={k: v[rows] for k, v in panel.fields.items()},
        target=y_target,
        mask=mask,
        n_keep=n_features,
        max_abs_corr=feature_max_abs_corr,
        min_samples=cfg.gp.min_daily_samples,
    )
    assert_no_label_columns(selected)
    for s in scores[:15]:
        log.info("  feature %-34s IC %+.5f  ICIR %+.3f", s.name, s.ic_mean, s.icir)

    # The GP fitness is a ranking metric on the binary hit label, which is the tradable
    # event; the continuous target drives feature screening because it is less noisy.
    result = run_search(
        fields={k: np.asarray(panel.fields[k][rows], dtype=np.float64) for k in selected},
        field_names=selected,
        y=panel.f64(BINARY_TARGET)[rows],
        mask=mask,
        cfg=cfg.gp,
        embargo_days=cfg.split.embargo_days,
        pset=build_event_pset(selected),
        kind="event",
    )
    return EventRun(
        panel=panel,
        library=result.library,
        selected_features=selected,
        search_rows=rows,
        report={},
    )


def evaluate_ic(run: EventRun, min_samples: int = 30) -> dict:
    """IC / IC_IR of every kept factor, split into search rows and everything after."""
    panel = run.panel
    if not run.library.factors:
        log.warning("no factors to evaluate")
        return {}

    names, values = compute_factors(run.library, panel.fields)
    after = slice(run.search_rows.stop, len(panel.dates))
    targets = {
        name: panel.f64(name)
        for name in (PRIMARY_TARGET, BINARY_TARGET)
        if name in panel.labels
    }

    report: dict[str, dict] = {}
    for k, name in enumerate(names):
        factor = values[:, :, k].astype(np.float64)
        entry: dict[str, dict] = {
            "expression": run.library.factors[k].expression,
            "sign": run.library.factors[k].sign,
        }
        for target_name, target in targets.items():
            entry[target_name] = {
                "in_sample": summarize_ic(
                    daily_ic(factor[run.search_rows], target[run.search_rows],
                             panel.occupied[run.search_rows], min_samples)
                ),
                "out_of_sample": summarize_ic(
                    daily_ic(factor[after], target[after], panel.occupied[after], min_samples)
                ),
            }
        report[name] = entry
        primary = entry[PRIMARY_TARGET]
        log.info(
            "%s | IS  IC %+.5f ICIR %+.3f | OOS IC %+.5f ICIR %+.3f (ann %+.2f, pos %.0f%%)",
            name,
            primary["in_sample"]["ic_mean"], primary["in_sample"]["icir"],
            primary["out_of_sample"]["ic_mean"], primary["out_of_sample"]["icir"],
            primary["out_of_sample"]["icir_ann"], 100 * primary["out_of_sample"]["positive_rate"],
        )
    run.report = report
    return report


def save(run: EventRun, out_dir: Path) -> dict[str, Path]:
    """Persist the library, the IC report and the standalone apply script."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "factors": out_dir / "event_factors.json",
        "report": out_dir / "event_ic_report.json",
        "script": out_dir / "apply_factors.py",
        "features": out_dir / "selected_features.json",
    }
    save_factors(paths["factors"], run.library)
    paths["report"].write_text(
        json.dumps(run.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["features"].write_text(
        json.dumps(run.selected_features, indent=2), encoding="utf-8"
    )
    write_apply_script(
        paths["script"],
        run.library,
        [PRIMARY_TARGET, BINARY_TARGET],
        search_end=str(run.panel.dates[run.search_rows][-1]),
    )
    return paths


def factor_frame(run: EventRun) -> pd.DataFrame:
    """Long frame of ``(trade_date, stock_code, <factor columns>)`` for local inspection."""
    names, values = compute_factors(run.library, run.panel.fields)
    return run.panel.to_long({name: values[:, :, k] for k, name in enumerate(names)})
