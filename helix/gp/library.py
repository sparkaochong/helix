"""Persistence and replay of discovered factors.

A factor is stored as its printed expression plus the field ordering it was built
against. That makes the library human-readable (you can see what was found) and
reproducible: :func:`compute_factors` rebuilds the tree and re-evaluates it on any
panel, so training, backtesting and live scoring all run the exact same formula.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from deap import gp

from ..logging_setup import get_logger
from .primitives import build_pset, parse_expression

log = get_logger(__name__)


@dataclass
class FactorSpec:
    name: str
    expression: str
    sign: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FactorLibrary:
    factors: list[FactorSpec]
    field_names: list[str]
    windows: list[int]
    #: ``"panel"`` = true (dates x stocks) panel, windowed operators allowed.
    #: ``"event"`` = slot panel from a long event table, cross-sectional operators only.
    kind: str = "panel"

    def build_pset(self):
        if self.kind == "event":
            from .event_primitives import build_event_pset

            return build_event_pset(self.field_names)
        return build_pset(self.field_names, self.windows)


def save_factors(path: Path, library: FactorLibrary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": library.kind,
        "field_names": library.field_names,
        "windows": library.windows,
        "factors": [asdict(f) for f in library.factors],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("saved %d %s factors to %s", len(library.factors), library.kind, path)


def load_factors(path: Path) -> FactorLibrary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FactorLibrary(
        factors=[FactorSpec(**f) for f in payload["factors"]],
        field_names=list(payload["field_names"]),
        windows=list(payload["windows"]),
        kind=payload.get("kind", "panel"),
    )


def compute_factors(
    library: FactorLibrary, fields: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    """Evaluate every factor on ``fields``.

    Returns ``(names, values)`` where ``values`` has shape ``(T, N, K)``. The stored
    sign is applied, so every returned factor is oriented so that *higher is more
    likely to touch the target*.
    """
    missing = [n for n in library.field_names if n not in fields]
    if missing:
        raise ValueError(f"panel is missing fields required by the library: {missing}")

    pset = library.build_pset()
    args = [np.asarray(fields[n], dtype=np.float64) for n in library.field_names]
    shape = args[0].shape

    names: list[str] = []
    columns: list[np.ndarray] = []
    for spec in library.factors:
        tree = parse_expression(spec.expression, pset)
        func = gp.compile(tree, pset)
        with np.errstate(all="ignore"):
            values = func(*args)
        if not isinstance(values, np.ndarray) or values.shape != shape:
            log.warning("factor %s did not return a panel; skipping", spec.name)
            continue
        values = np.where(np.isfinite(values), values, np.nan) * spec.sign
        names.append(spec.name)
        columns.append(values.astype(np.float32))

    if not columns:
        raise RuntimeError("no factor in the library evaluated successfully")
    return names, np.stack(columns, axis=-1)
