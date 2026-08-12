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

#: Operators withheld from the *search*, which is a weaker statement than :data:`FORBIDDEN`.
#: A windowed operator is meaningless on a slot panel and is banned everywhere; ``sign`` is
#: merely a bad use of the search budget, so a saved expression containing it must still
#: replay. See ``exclude`` in :func:`build_event_pset`.
#:
#: ``sign`` crushes a continuous column onto three levels. On a skewed column that buys
#: respectable rank-gini cheaply, and the typical winner looked like ``sub(sign(A), B)`` --
#: "B shifted by the sign of A", which a row-wise tree reproduces with one split on A and
#: one on B. It took 27 of the last 30 factors and left the cross-sectional structure, the
#: one thing such a tree cannot compute, untouched.
SEARCH_EXCLUDED = frozenset({"sign"})


def build_event_pset(
    field_names: list[str], exclude: frozenset[str] = SEARCH_EXCLUDED
) -> gp.PrimitiveSetTyped:
    """Cross-sectional and arithmetic operators only, over the given feature columns.

    Time-series structure is already baked into the source features (rolling means,
    multi-day ratios, EMA deviations), so the search does not lose expressiveness by
    dropping windowed operators -- it only loses a way to be wrong.

    ``exclude`` defaults to :data:`SEARCH_EXCLUDED` and is filtered out of the unary
    operators. Pass an empty set to rebuild a pset that can parse any expression ever
    saved -- :meth:`helix.gp.library.FactorLibrary.build_pset` does exactly that, since a
    stored factor has to keep evaluating after the search space narrows underneath it.

    Filtering happens here rather than in :data:`helix.gp.primitives.UNARY`, which the
    panel path shares: on a real ``(dates x stocks)`` panel none of this reasoning applies.
    """
    if not field_names:
        raise ValueError("an event primitive set needs at least one feature column")
    # Last line of defence: an outcome column reaching the terminal set produces
    # factors with IC > 0.5 that are pure leakage.
    assert_no_label_columns(field_names)

    pset = gp.PrimitiveSetTyped("EVENT", [np.ndarray] * len(field_names), np.ndarray)
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(field_names)})

    # A typo in `exclude` would otherwise remove nothing and report nothing, leaving the
    # operator in the search while the caller believes it is gone.
    exclude = frozenset(exclude)
    unknown = exclude - {name for name, _ in UNARY}
    if unknown:
        raise ValueError(f"nothing to exclude named: {sorted(unknown)}")
    for name, fn in UNARY:
        if name in exclude:
            continue
        pset.addPrimitive(fn, [np.ndarray], np.ndarray, name=name)
    for name, fn in BINARY:
        pset.addPrimitive(fn, [np.ndarray, np.ndarray], np.ndarray, name=name)

    assert_no_time_series(pset)
    assert_excluded_absent(pset, exclude)
    return pset


def _primitive_names(pset: gp.PrimitiveSetTyped) -> set[str]:
    return {p.name for prims in pset.primitives.values() for p in prims}


def assert_excluded_absent(
    pset: gp.PrimitiveSetTyped, excluded: frozenset[str] = SEARCH_EXCLUDED
) -> None:
    """Fail loudly if a search-excluded operator is in the set.

    Unlike :func:`assert_no_time_series` this is parameterised, because the exclusion is a
    property of how the pset was requested rather than of the slot panel itself.
    """
    leaked = _primitive_names(pset) & frozenset(excluded)
    if leaked:
        raise AssertionError(
            f"these operators are withheld from the event-table search but reached the "
            f"primitive set: {sorted(leaked)}"
        )


def assert_no_time_series(pset: gp.PrimitiveSetTyped) -> None:
    """Fail loudly if any windowed operator or window terminal slipped in."""
    leaked = _primitive_names(pset) & FORBIDDEN
    if leaked:
        raise AssertionError(
            f"time-series operators are invalid on a slot panel (slot j is a different "
            f"stock each date), but the primitive set contains: {sorted(leaked)}"
        )
    if Window in pset.terminals and pset.terminals[Window]:
        raise AssertionError("window terminals have no meaning on a slot panel")
