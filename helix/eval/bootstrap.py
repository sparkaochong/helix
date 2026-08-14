"""Pure NumPy helpers for seeded circular moving-block bootstrap inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

PERFORMANCE_METRICS = (
    "cagr",
    "sharpe",
    "day_win_rate",
    "mean_trade_return_net",
)


def validate_bootstrap_seeds(
    seeds: Sequence[int], *, minimum: int = 2
) -> tuple[int, ...]:
    """Return integer seeds after enforcing an independent-replicate contract."""
    values = tuple(int(seed) for seed in seeds)
    if len(values) < minimum or len(set(values)) != len(values):
        raise ValueError(f"bootstrap requires at least {minimum} unique seeds")
    return values


def circular_block_bootstrap_indices(
    n_dates: int,
    block_length: int,
    seeds: Sequence[int],
) -> np.ndarray:
    """Generate one full-length circular moving-block date sample per seed."""
    if n_dates <= 0 or block_length <= 0:
        raise ValueError("n_dates and block_length must be positive")
    seed_values = validate_bootstrap_seeds(seeds)
    block_count = int(np.ceil(n_dates / block_length))
    offsets = np.arange(block_length, dtype=np.intp)
    rows = []
    for seed in seed_values:
        starts = np.random.default_rng(seed).integers(0, n_dates, size=block_count)
        rows.append(
            ((starts[:, None] + offsets[None, :]) % n_dates).reshape(-1)[:n_dates]
        )
    return np.stack(rows)


def bootstrap_performance_metrics(
    daily_returns: np.ndarray,
    trade_return_sum: np.ndarray,
    trade_count: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate strategy metrics over a seed-by-date resampling index matrix."""
    daily = np.asarray(daily_returns, dtype=np.float64)
    trade_sum = np.asarray(trade_return_sum, dtype=np.float64)
    counts = np.asarray(trade_count, dtype=np.float64)
    index = np.asarray(indices)
    if daily.ndim != 1 or trade_sum.ndim != 1 or counts.ndim != 1:
        raise ValueError("performance inputs must be one-dimensional")
    if len({daily.size, trade_sum.size, counts.size}) != 1:
        raise ValueError("daily returns and per-date trade inputs must be aligned")
    if daily.size < 2:
        raise ValueError("performance bootstrap requires at least two dates")
    if not np.isfinite(daily).all() or not np.isfinite(trade_sum).all():
        raise ValueError("return inputs must be finite")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("trade counts must be finite and non-negative")
    if index.ndim != 2:
        raise ValueError("bootstrap indices must be two-dimensional")
    if index.shape[1] != daily.size:
        raise ValueError("each bootstrap replicate must preserve the full date count")
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError("bootstrap indices must be integers")
    if index.size == 0 or index.min() < 0 or index.max() >= daily.size:
        raise ValueError("bootstrap indices are out of bounds")

    sampled = daily[index]
    sampled_trade_count = counts[index].sum(axis=1)
    if np.any(sampled_trade_count <= 0):
        raise ValueError("each replicate must contain at least one resolved trade")
    final_equity = np.prod(1.0 + sampled, axis=1)
    volatility = sampled.std(axis=1, ddof=1)
    sharpe = np.divide(
        sampled.mean(axis=1) * np.sqrt(252.0),
        volatility,
        out=np.full(len(index), np.nan, dtype=np.float64),
        where=volatility > 0,
    )
    return {
        "cagr": np.where(
            final_equity > 0,
            final_equity ** (252.0 / sampled.shape[1]) - 1.0,
            -1.0,
        ),
        "sharpe": sharpe,
        "day_win_rate": (sampled > 0).mean(axis=1),
        "mean_trade_return_net": trade_sum[index].sum(axis=1) / sampled_trade_count,
    }


def summarize_bootstrap_distribution(
    values: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | list[float]]]:
    """Summarize finite bootstrap replicates with sample dispersion and percentile CI."""
    output: dict[str, dict[str, float | list[float]]] = {}
    for metric, raw_samples in values.items():
        samples = np.asarray(raw_samples, dtype=np.float64)
        if samples.ndim != 1 or samples.size < 2:
            raise ValueError(f"{metric} bootstrap distribution needs at least two values")
        if not np.isfinite(samples).all():
            raise ValueError(f"{metric} bootstrap distribution must be finite")
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975], method="linear")
        output[metric] = {
            "mean": float(samples.mean()),
            "std": float(samples.std(ddof=1)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "values": samples.tolist(),
        }
    return output


def summarize_metric_runs(
    runs: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float | list[float]]]:
    """Aggregate generic seeded metric mappings while retaining non-finite values."""
    keys = tuple(dict.fromkeys(key for run in runs for key in run))
    output: dict[str, dict[str, float | list[float]]] = {}
    for key in keys:
        values = np.asarray([run.get(key, np.nan) for run in runs], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "values": values.tolist(),
        }
    return output
