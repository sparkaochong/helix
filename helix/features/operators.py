"""Operators over ``(T, N)`` panels: axis 0 is trade date (ascending), axis 1 is stock.

Two hard invariants hold for everything in this module:

* every ``ts_*`` operator looks **strictly backward** -- row ``t`` of the result is a
  function of rows ``<= t`` only, so no operator can introduce look-ahead;
* NaN propagates rather than being silently filled, so suspended days and
  not-yet-listed stocks stay excluded instead of contributing fabricated values.

The one forward-looking helper, :func:`lead`, lives here too but is used *only* by
the label module -- never expose it to the GP primitive set.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1e-9


# --------------------------------------------------------------------- shifts --
def delay(x: np.ndarray, d: int) -> np.ndarray:
    """Value from ``d`` rows ago. Backward-looking."""
    if d <= 0:
        raise ValueError("delay requires d >= 1; use the raw array for d == 0")
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[d:] = x[:-d]
    return out


def lead(x: np.ndarray, d: int) -> np.ndarray:
    """Value from ``d`` rows ahead. FUTURE-LOOKING -- label construction only."""
    if d <= 0:
        raise ValueError("lead requires d >= 1")
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[:-d] = x[d:]
    return out


# ---------------------------------------------------------------- arithmetic --
def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y


def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Protected division: a near-zero denominator yields NaN, never inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(y) < EPS, np.nan, x / np.where(np.abs(y) < EPS, 1.0, y))
    return out


def neg(x: np.ndarray) -> np.ndarray:
    return -x


def abs_(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def log_abs(x: np.ndarray) -> np.ndarray:
    """``log(1 + |x|)`` keeping the sign -- a scale compressor that tolerates zeros."""
    return np.sign(x) * np.log1p(np.abs(x))


def sqrt_abs(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


# ------------------------------------------------------------ cross-section --
def cs_rank(x: np.ndarray) -> np.ndarray:
    """Per-row rank scaled to ``(0, 1]``; NaN stays NaN and is excluded from the count.

    Ties get ordinal (not averaged) ranks. Exact ties are rare for real factor
    values, and a fully constant row maps to a uniform ramp whose correlation with
    any label is ~0 -- which is the desired outcome for a degenerate factor.
    """
    filled = np.where(np.isnan(x), np.inf, x)
    order = np.argsort(filled, axis=1, kind="stable")
    positions = np.broadcast_to(
        np.arange(1, x.shape[1] + 1, dtype=np.float64), x.shape
    )
    ranks = np.empty(x.shape, dtype=np.float64)
    np.put_along_axis(ranks, order, positions, axis=1)
    valid = ~np.isnan(x)
    counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    return np.where(valid, ranks / counts, np.nan)


def cs_rank_ordinal(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer per-row ranks ``1..n_valid`` (0 where invalid) plus the valid mask."""
    valid = ~np.isnan(x)
    filled = np.where(valid, x, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    positions = np.broadcast_to(np.arange(1, x.shape[1] + 1, dtype=np.float64), x.shape)
    ranks = np.empty(x.shape, dtype=np.float64)
    np.put_along_axis(ranks, order, positions, axis=1)
    return np.where(valid, ranks, 0.0), valid


def cs_zscore(x: np.ndarray) -> np.ndarray:
    mean, std = _nan_mean_std(x)
    return np.where(std < EPS, np.nan, (x - mean) / np.where(std < EPS, 1.0, std))


def cs_demean(x: np.ndarray) -> np.ndarray:
    mean, _ = _nan_mean_std(x)
    return x - mean


def _nan_mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row mean/std ignoring NaN. All-NaN rows yield NaN instead of a RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            np.nanmean(x, axis=1, keepdims=True),
            np.nanstd(x, axis=1, keepdims=True),
        )


# ------------------------------------------------------------- time series --
def _roll(x: np.ndarray, d: int):
    return pd.DataFrame(x).rolling(window=d, min_periods=d)


def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    return _roll(x, d).mean().to_numpy()


def ts_std(x: np.ndarray, d: int) -> np.ndarray:
    return _roll(x, d).std(ddof=1).to_numpy()


def ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    return _roll(x, d).sum().to_numpy()


def ts_max(x: np.ndarray, d: int) -> np.ndarray:
    return _roll(x, d).max().to_numpy()


def ts_min(x: np.ndarray, d: int) -> np.ndarray:
    return _roll(x, d).min().to_numpy()


def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """Percentile of the current value inside its own trailing ``d``-day window."""
    return _roll(x, d).rank(pct=True).to_numpy()


def ts_argmax(x: np.ndarray, d: int) -> np.ndarray:
    """Days since the window maximum, scaled to ``[0, 1]``."""
    idx = _roll(x, d).apply(np.nanargmax, raw=True).to_numpy()
    return (d - 1 - idx) / max(d - 1, 1)


def ts_argmin(x: np.ndarray, d: int) -> np.ndarray:
    idx = _roll(x, d).apply(np.nanargmin, raw=True).to_numpy()
    return (d - 1 - idx) / max(d - 1, 1)


def ts_delta(x: np.ndarray, d: int) -> np.ndarray:
    return x - delay(x, d)


def ts_pct_change(x: np.ndarray, d: int) -> np.ndarray:
    prev = delay(x, d)
    return div(x - prev, np.abs(prev))


def ts_zscore(x: np.ndarray, d: int) -> np.ndarray:
    mean = ts_mean(x, d)
    std = ts_std(x, d)
    return div(x - mean, std)


def ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    dfx, dfy = pd.DataFrame(x), pd.DataFrame(y)
    out = dfx.rolling(window=d, min_periods=d).corr(dfy).to_numpy()
    return np.where(np.isfinite(out), out, np.nan)


def ts_cov(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    dfx, dfy = pd.DataFrame(x), pd.DataFrame(y)
    return dfx.rolling(window=d, min_periods=d).cov(dfy).to_numpy()


def ts_decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """Linearly weighted mean, heaviest on the most recent day."""
    weights = np.arange(1, d + 1, dtype=np.float64)
    weights /= weights.sum()
    return _roll(x, d).apply(lambda w: float(np.dot(w, weights)), raw=True).to_numpy()


def clip_sigma(x: np.ndarray, n_sigma: float) -> np.ndarray:
    """Winsorise cross-sectionally at +/- ``n_sigma`` standard deviations."""
    mean, std = _nan_mean_std(x)
    return np.clip(x, mean - n_sigma * std, mean + n_sigma * std)
