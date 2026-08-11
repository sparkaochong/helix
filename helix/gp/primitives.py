"""The typed primitive set the genetic program searches over.

Typing matters here. An untyped set lets GP write ``ts_mean(x, close)`` -- nonsense
that still evaluates -- and wastes most of the population on such trees. Windows get
their own type, so a rolling window can only ever be one of the configured integers.

No primitive in this set can see the future: ``ts_*`` operators are backward-looking
by construction and :func:`helix.features.operators.lead` is deliberately absent.
"""

from __future__ import annotations

import numpy as np
from deap import gp

from ..features import operators as ops


class Window(int):
    """Rolling-window length. A distinct type so GP cannot pass a price as a window."""

    __slots__ = ()

    def __repr__(self) -> str:  # keeps printed expressions readable
        return str(int(self))


UNARY = (
    ("neg", ops.neg),
    ("abs", ops.abs_),
    ("sign", ops.sign),
    ("log", ops.log_abs),
    ("sqrt", ops.sqrt_abs),
    ("cs_rank", ops.cs_rank),
    ("cs_zscore", ops.cs_zscore),
    ("cs_demean", ops.cs_demean),
)

BINARY = (
    ("add", ops.add),
    ("sub", ops.sub),
    ("mul", ops.mul),
    ("div", ops.div),
)

WINDOWED = (
    ("ts_mean", ops.ts_mean),
    ("ts_std", ops.ts_std),
    ("ts_max", ops.ts_max),
    ("ts_min", ops.ts_min),
    ("ts_sum", ops.ts_sum),
    ("ts_rank", ops.ts_rank),
    ("ts_delta", ops.ts_delta),
    ("ts_delay", ops.delay),
    ("ts_zscore", ops.ts_zscore),
    ("ts_pct", ops.ts_pct_change),
    ("ts_argmax", ops.ts_argmax),
    ("ts_argmin", ops.ts_argmin),
    ("ts_decay", ops.ts_decay_linear),
)

WINDOWED_BINARY = (
    ("ts_corr", ops.ts_corr),
    ("ts_cov", ops.ts_cov),
)


def build_pset(field_names: list[str], windows: list[int]) -> gp.PrimitiveSetTyped:
    """Primitive set whose arguments are the base fields, in the given order."""
    pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray] * len(field_names), np.ndarray)
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(field_names)})

    for name, fn in UNARY:
        pset.addPrimitive(fn, [np.ndarray], np.ndarray, name=name)
    for name, fn in BINARY:
        pset.addPrimitive(fn, [np.ndarray, np.ndarray], np.ndarray, name=name)
    for name, fn in WINDOWED:
        pset.addPrimitive(fn, [np.ndarray, Window], np.ndarray, name=name)
    for name, fn in WINDOWED_BINARY:
        pset.addPrimitive(fn, [np.ndarray, np.ndarray, Window], np.ndarray, name=name)

    for w in windows:
        pset.addTerminal(Window(w), Window, name=f"w{w}")
    return pset


def parse_expression(expression: str, pset: gp.PrimitiveSetTyped) -> gp.PrimitiveTree:
    """Rebuild a tree from its printed form, so saved factors survive a restart."""
    return gp.PrimitiveTree.from_string(expression, pset)
