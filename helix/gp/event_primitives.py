"""Primitive set for slot-panel event tables.

The single difference from :mod:`helix.gp.primitives` is what is *absent*: no windowed
operator appears here, and none can be added. In an :class:`~helix.data.event_table.EventPanel`
column ``j`` holds a different company on each date, so ``ts_mean(x, 20)`` would average
twenty unrelated stocks and return a number that looks like a factor and means nothing.

That is not a hypothetical -- it is the obvious way to break this pipeline, so the guard
is a hard assertion rather than a comment.
"""

from __future__ import annotations

import numpy as np
from deap import gp

from ..data.event_table import assert_no_label_columns
from .primitives import BINARY, UNARY, WINDOWED, WINDOWED_BINARY, Window

#: Names that must never reach an event-table primitive set.
FORBIDDEN = frozenset(name for name, _ in (*WINDOWED, *WINDOWED_BINARY))


def build_event_pset(field_names: list[str]) -> gp.PrimitiveSetTyped:
    """Cross-sectional and arithmetic operators only, over the given feature columns.

    Time-series structure is already baked into the source features (rolling means,
    multi-day ratios, EMA deviations), so the search does not lose expressiveness by
    dropping windowed operators -- it only loses a way to be wrong.
    """
    if not field_names:
        raise ValueError("an event primitive set needs at least one feature column")
    # Last line of defence: an outcome column reaching the terminal set produces
    # factors with IC > 0.5 that are pure leakage.
    assert_no_label_columns(field_names)

    pset = gp.PrimitiveSetTyped("EVENT", [np.ndarray] * len(field_names), np.ndarray)
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(field_names)})

    for name, fn in UNARY:
        pset.addPrimitive(fn, [np.ndarray], np.ndarray, name=name)
    for name, fn in BINARY:
        pset.addPrimitive(fn, [np.ndarray, np.ndarray], np.ndarray, name=name)

    assert_no_time_series(pset)
    return pset


def assert_no_time_series(pset: gp.PrimitiveSetTyped) -> None:
    """Fail loudly if any windowed operator or window terminal slipped in."""
    present = {p.name for prims in pset.primitives.values() for p in prims}
    leaked = present & FORBIDDEN
    if leaked:
        raise AssertionError(
            f"time-series operators are invalid on a slot panel (slot j is a different "
            f"stock each date), but the primitive set contains: {sorted(leaked)}"
        )
    if Window in pset.terminals and pset.terminals[Window]:
        raise AssertionError("window terminals have no meaning on a slot panel")
