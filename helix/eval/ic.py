"""Information Coefficient and IC_IR.

IC here is the **per-date Spearman rank correlation** between a factor and a forward
target, averaged over dates -- never a single correlation pooled across the whole table.
Pooling rewards a factor for knowing which days were strong market-wide, which is worth
nothing to a strategy that must choose among names available on the same morning.

``IC_IR = mean(IC) / std(IC)``. It is the number that decides whether a factor is
tradable: a factor with IC 0.05 that is positive four days in five beats one with IC
0.08 that alternates sign.
"""

from __future__ import annotations

import numpy as np

from ..features.operators import cs_rank

TRADING_DAYS = 252


def _row_pearson(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Correlation per row over the cells where ``valid`` holds. NaN where degenerate."""
    w = valid.astype(np.float64)
    xs = np.where(valid, x, 0.0)
    ys = np.where(valid, y, 0.0)

    n = w.sum(axis=1)
    sx, sy = xs.sum(axis=1), ys.sum(axis=1)
    sxx, syy = (xs * xs).sum(axis=1), (ys * ys).sum(axis=1)
    sxy = (xs * ys).sum(axis=1)

    cov = n * sxy - sx * sy
    var_x = n * sxx - sx * sx
    var_y = n * syy - sy * sy
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), np.nan)
    return out


def daily_ic(
    factor: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    min_samples: int = 30,
    rank: bool = True,
) -> np.ndarray:
    """Per-date IC. ``rank=True`` gives Spearman, ``False`` gives Pearson."""
    valid = mask & np.isfinite(factor) & np.isfinite(target)
    x = np.where(valid, factor, np.nan)
    y = np.where(valid, target, np.nan)
    if rank:
        x, y = cs_rank(x), cs_rank(y)
    ic = _row_pearson(np.nan_to_num(x), np.nan_to_num(y), valid)
    return np.where(valid.sum(axis=1) >= min_samples, ic, np.nan)


def summarize_ic(ic: np.ndarray, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """Mean / std / IR / t-stat / sign stability of a per-date IC series."""
    finite = ic[np.isfinite(ic)]
    n = finite.size
    if n == 0:
        return {
            "ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"),
            "icir_ann": float("nan"), "t_stat": float("nan"),
            "positive_rate": float("nan"), "n_days": 0.0, "coverage": 0.0,
        }
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if std and std > 0 else float("nan")
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "icir_ann": icir * np.sqrt(periods_per_year) if np.isfinite(icir) else float("nan"),
        "t_stat": icir * np.sqrt(n) if np.isfinite(icir) else float("nan"),
        "positive_rate": float((finite > 0).mean()),
        "n_days": float(n),
        "coverage": float(n / ic.size),
    }


def ic_report(
    factor: np.ndarray,
    targets: dict[str, np.ndarray],
    mask: np.ndarray,
    min_samples: int = 30,
) -> dict[str, dict[str, float]]:
    """IC summary of one factor against several targets (e.g. binary hit and peak return)."""
    return {
        name: summarize_ic(daily_ic(factor, target, mask, min_samples))
        for name, target in targets.items()
    }
