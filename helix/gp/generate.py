"""Tree generation that tolerates terminal-only types.

DEAP's :func:`deap.gp.generate` assumes every type in a typed primitive set has at
least one primitive producing it. Helix's ``Window`` type deliberately has none -- a
rolling window is always one of the configured integers, never something computed --
so DEAP raises ``IndexError`` as soon as it wants a ``Window`` above the leaf level.

These generators mirror DEAP's semantics exactly but fall back to the other kind of
node when a type offers no choice, which is the behaviour a terminal-only type needs.
"""

from __future__ import annotations

import random
from inspect import isclass

from deap import gp


def generate(pset: gp.PrimitiveSetTyped, min_: int, max_: int, condition, type_=None) -> list:
    if type_ is None:
        type_ = pset.ret
    expr: list = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]

    while stack:
        depth, node_type = stack.pop()
        want_terminal = condition(height, depth)
        terminals = pset.terminals[node_type]
        primitives = pset.primitives[node_type]

        if not terminals and not primitives:
            raise TypeError(f"no primitive or terminal produces type {node_type}")
        if want_terminal and not terminals:
            want_terminal = False
        elif not want_terminal and not primitives:
            want_terminal = True

        if want_terminal:
            term = random.choice(terminals)
            expr.append(term() if isclass(term) else term)
        else:
            prim = random.choice(primitives)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def gen_full(pset, min_, max_, type_=None) -> list:
    return generate(pset, min_, max_, lambda height, depth: depth == height, type_)


def gen_grow(pset, min_, max_, type_=None) -> list:
    def condition(height: int, depth: int) -> bool:
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def gen_half_and_half(pset, min_, max_, type_=None) -> list:
    method = random.choice((gen_grow, gen_full))
    return method(pset, min_, max_, type_)
