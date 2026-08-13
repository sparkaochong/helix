"""Vectorized placebo calibration for cross-sectional factor metrics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral

import numpy as np
import pandas as pd

from ..config import PlaceboThresholdConfig
from ..features.operators import cs_rank, cs_rank_ordinal
from .ic import daily_ic, summarize_ic
from .metrics import daily_gini, summarize_daily

QUANTILES = {"p95": 0.95, "p99": 0.99, "p999": 0.999}
METRICS = ("ic_mean", "icir", "gini")

# Each float64 ranked-label input is capped near 16 MiB when P * N permits it.
_MAX_RANKED_LABEL_ELEMENTS = 2_000_000
_LEVEL_LABELS = np.array(
    ["低于随机水平", "超 p95", "超 p99", "超 p99.9"], dtype=object
)


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _binary_array(values: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite binary values") from error
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only finite binary values")
    return array


def _binary_or_nan_array(values: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only binary values or NaN") from error
    if not (np.isnan(array) | np.isin(array, (0.0, 1.0))).all():
        raise ValueError(f"{name} must contain only binary values or NaN")
    return array


def permute_binary_labels(
    labels: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate binary label permutations while preserving the exact class count."""
    count = _positive_integer(n_permutations, "n_permutations")
    binary = _binary_array(labels, "labels")
    if binary.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    n_positive = int(binary.sum())
    if n_positive == 0 or n_positive == binary.size:
        raise ValueError("labels must contain both binary classes")

    random_keys = rng.random((count, binary.size))
    positive_positions = np.argpartition(
        random_keys, kth=n_positive - 1, axis=1
    )[:, :n_positive]
    permutations = np.zeros((count, binary.size), dtype=bool)
    np.put_along_axis(permutations, positive_positions, True, axis=1)
    return permutations


def iter_cross_sectional_permutations(
    labels: np.ndarray,
    mask: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield within-date permutations over observable binary labels."""
    label_array = _binary_or_nan_array(labels, "labels")
    mask_array = np.asarray(mask, dtype=bool)
    if label_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("labels and mask must be two-dimensional")
    if label_array.shape != mask_array.shape:
        raise ValueError("labels and mask must have the same shape")
    count = _positive_integer(n_permutations, "n_permutations")

    for date_index in range(label_array.shape[0]):
        valid_positions = np.flatnonzero(
            mask_array[date_index] & np.isfinite(label_array[date_index])
        )
        permutations = permute_binary_labels(
            label_array[date_index, valid_positions], count, rng
        )
        yield date_index, valid_positions, permutations


def placebo_daily_metrics(
    factors: np.ndarray,
    permuted_labels: np.ndarray,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate daily IC and gini for every permutation and factor at once."""
    factor_array = np.asarray(factors, dtype=np.float64)
    label_array = _binary_array(permuted_labels, "permuted_labels")
    minimum = _positive_integer(min_samples, "min_samples")
    if factor_array.ndim != 2 or label_array.ndim != 2:
        raise ValueError("factors and permuted_labels must be two-dimensional")
    if factor_array.shape[0] != label_array.shape[1]:
        raise ValueError("factors and permuted_labels must share the sample dimension")

    n_permutations, n_samples = label_array.shape
    n_factors = factor_array.shape[1]
    if n_permutations == 0:
        raise ValueError("permuted_labels must contain at least one permutation")
    if n_factors == 0:
        return (
            np.empty((n_permutations, 0), dtype=np.float64),
            np.empty((n_permutations, 0), dtype=np.float64),
        )

    finite = np.isfinite(factor_array).T
    ranked_factors = cs_rank(np.where(finite, factor_array.T, np.nan))
    ordinal_ranks, _ = cs_rank_ordinal(np.where(finite, factor_array.T, np.nan))

    labels_float = label_array.astype(np.float64, copy=False)
    sample_counts = finite.sum(axis=1).astype(np.float64)
    positive_counts = labels_float @ finite.T
    negative_counts = sample_counts[None, :] - positive_counts
    gini_usable = (
        (sample_counts[None, :] >= minimum)
        & (positive_counts > 0)
        & (negative_counts > 0)
    )

    ranked_factor_values = np.nan_to_num(ranked_factors)
    factor_sums = ranked_factor_values.sum(axis=1)
    factor_squares = (ranked_factor_values * ranked_factor_values).sum(axis=1)
    factor_variance = sample_counts * factor_squares - factor_sums * factor_sums

    ic = np.full((n_permutations, n_factors), np.nan, dtype=np.float64)
    ranked_label_elements_per_factor = max(n_permutations * n_samples, 1)
    factors_per_chunk = max(
        _MAX_RANKED_LABEL_ELEMENTS // ranked_label_elements_per_factor, 1
    )
    for factor_start in range(0, n_factors, factors_per_chunk):
        factor_stop = min(factor_start + factors_per_chunk, n_factors)
        factor_slice = slice(factor_start, factor_stop)
        chunk_size = factor_stop - factor_start
        chunk_finite = finite[factor_slice]
        ranked_labels = cs_rank(
            np.where(
                chunk_finite[None, :, :], label_array[:, None, :], np.nan
            ).reshape(n_permutations * chunk_size, n_samples)
        ).reshape(n_permutations, chunk_size, n_samples)
        ranked_label_values = np.nan_to_num(ranked_labels, copy=False)

        label_sums = ranked_label_values.sum(axis=2)
        label_squares = (ranked_label_values * ranked_label_values).sum(axis=2)
        cross_products = (
            ranked_label_values
            * ranked_factor_values[None, factor_slice, :]
        ).sum(axis=2)
        covariance = (
            sample_counts[None, factor_slice] * cross_products
            - factor_sums[None, factor_slice] * label_sums
        )
        label_variance = (
            sample_counts[None, factor_slice] * label_squares
            - label_sums * label_sums
        )
        denominator = np.sqrt(
            np.maximum(factor_variance[None, factor_slice], 0.0)
            * np.maximum(label_variance, 0.0)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            chunk_ic = np.where(denominator > 0, covariance / denominator, np.nan)
        ic[:, factor_slice] = np.where(
            sample_counts[None, factor_slice] >= minimum,
            chunk_ic,
            np.nan,
        )

    positive_rank_sums = labels_float @ ordinal_ranks.T
    auc_denominator = positive_counts * negative_counts
    auc = (
        positive_rank_sums - positive_counts * (positive_counts + 1.0) / 2.0
    ) / np.where(auc_denominator > 0, auc_denominator, 1.0)
    gini = np.where(gini_usable, 2.0 * auc - 1.0, np.nan)
    return ic, gini


def placebo_distribution(
    factors: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 20260813,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Build the formal-factor null distribution without storing all dates."""
    factor_array = np.asarray(factors, dtype=np.float64)
    label_array = np.asarray(labels)
    mask_array = np.asarray(mask, dtype=bool)
    count = _positive_integer(n_permutations, "n_permutations")
    minimum = _positive_integer(min_samples, "min_samples")
    if factor_array.ndim != 3:
        raise ValueError("factors must have shape (dates, samples, factors)")
    if label_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("labels and mask must be two-dimensional")
    if label_array.shape != mask_array.shape or factor_array.shape[:2] != label_array.shape:
        raise ValueError("factors, labels, and mask have incompatible shapes")
    if factor_array.shape[2] == 0:
        raise ValueError("at least one formal factor is required")

    accumulator_shape = (count, factor_array.shape[2])
    ic_sums = np.zeros(accumulator_shape, dtype=np.float64)
    ic_square_sums = np.zeros(accumulator_shape, dtype=np.float64)
    ic_counts = np.zeros(accumulator_shape, dtype=np.int64)
    gini_sums = np.zeros(accumulator_shape, dtype=np.float64)
    gini_counts = np.zeros(accumulator_shape, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for date_index, valid_positions, daily_labels in iter_cross_sectional_permutations(
        label_array, mask_array, count, rng
    ):
        daily_ic_values, daily_gini_values = placebo_daily_metrics(
            factor_array[date_index, valid_positions, :],
            daily_labels,
            minimum,
        )
        finite_ic = np.isfinite(daily_ic_values)
        finite_gini = np.isfinite(daily_gini_values)
        ic_sums += np.where(finite_ic, daily_ic_values, 0.0)
        ic_square_sums += np.where(finite_ic, daily_ic_values * daily_ic_values, 0.0)
        ic_counts += finite_ic
        gini_sums += np.where(finite_gini, daily_gini_values, 0.0)
        gini_counts += finite_gini

    with np.errstate(invalid="ignore", divide="ignore"):
        ic_means = ic_sums / ic_counts
        ic_variances = (
            ic_square_sums - ic_sums * ic_sums / ic_counts
        ) / (ic_counts - 1)
        ic_stds = np.sqrt(np.maximum(ic_variances, 0.0))
        icirs = ic_means / ic_stds
        gini_means = gini_sums / gini_counts

    final_metrics = np.column_stack(
        (
            np.max(np.abs(ic_means), axis=1),
            np.max(np.abs(icirs), axis=1),
            np.max(np.abs(gini_means), axis=1),
        )
    )
    if not np.isfinite(final_metrics).all():
        raise ValueError("placebo distribution contains non-finite final metrics")

    return pd.DataFrame(
        {
            "permutation_id": np.arange(count),
            "ic_mean": final_metrics[:, 0],
            "icir": final_metrics[:, 1],
            "gini": final_metrics[:, 2],
        }
    )


def factor_metrics(
    names: Sequence[str],
    factors: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Summarize real-label metrics with the canonical evaluation functions."""
    factor_array = np.asarray(factors, dtype=np.float64)
    label_array = np.asarray(labels)
    mask_array = np.asarray(mask, dtype=bool)
    minimum = _positive_integer(min_samples, "min_samples")
    if factor_array.ndim != 3:
        raise ValueError("factors must have shape (dates, samples, factors)")
    if label_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("labels and mask must be two-dimensional")
    if label_array.shape != mask_array.shape or factor_array.shape[:2] != label_array.shape:
        raise ValueError("factors, labels, and mask have incompatible shapes")
    if len(names) != factor_array.shape[2]:
        raise ValueError("names must contain one entry per factor")

    rows = []
    for factor_index, factor_name in enumerate(names):
        ic_summary = summarize_ic(
            daily_ic(
                factor_array[:, :, factor_index],
                label_array,
                mask_array,
                min_samples=minimum,
            )
        )
        gini_summary = summarize_daily(
            daily_gini(
                factor_array[:, :, factor_index],
                label_array,
                mask_array,
                min_samples=minimum,
            )
        )
        rows.append(
            {
                "factor_name": factor_name,
                "ic_mean_signed": ic_summary["ic_mean"],
                "ic_mean": abs(ic_summary["ic_mean"]),
                "icir_signed": ic_summary["icir"],
                "icir": abs(ic_summary["icir"]),
                "gini_signed": gini_summary["mean"],
                "gini": abs(gini_summary["mean"]),
            }
        )
    return pd.DataFrame(rows)


def metric_quantiles(distribution: pd.DataFrame) -> pd.DataFrame:
    """Calculate the configured empirical quantiles with linear interpolation."""
    return pd.DataFrame(
        [
            {
                "metric": metric,
                **{
                    name: np.quantile(distribution[metric], value, method="linear")
                    for name, value in QUANTILES.items()
                },
            }
            for metric in METRICS
        ]
    )


def passes_placebo_threshold(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float] | PlaceboThresholdConfig,
) -> bool:
    """Return whether every finite metric strictly exceeds its finite threshold."""
    try:
        values = np.array([metrics[metric] for metric in METRICS], dtype=np.float64)
        if isinstance(thresholds, PlaceboThresholdConfig):
            limits = np.array(
                [getattr(thresholds, metric) for metric in METRICS],
                dtype=np.float64,
            )
        else:
            limits = np.array(
                [thresholds[metric] for metric in METRICS], dtype=np.float64
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all() and np.isfinite(limits).all() and np.all(values > limits))


def screen_factor_metrics(
    metrics: pd.DataFrame,
    quantiles: pd.DataFrame,
    scope: str,
    library_path: str,
) -> pd.DataFrame:
    """Grade factor metrics against formal placebo quantiles."""
    thresholds = quantiles.set_index("metric")
    result = metrics.copy()
    result.insert(0, "scope", scope)
    result.insert(1, "library_path", library_path)

    level_codes = []
    for metric in METRICS:
        values = pd.to_numeric(result[metric], errors="coerce").to_numpy()
        metric_thresholds = thresholds.loc[metric]
        codes = np.select(
            (
                values > metric_thresholds["p999"],
                values > metric_thresholds["p99"],
                values > metric_thresholds["p95"],
            ),
            (3, 2, 1),
            default=0,
        )
        level_codes.append(codes)
        level_name = "ic" if metric == "ic_mean" else metric
        result[f"{level_name}_level"] = _LEVEL_LABELS[codes]

    weakest = np.minimum.reduce(level_codes)
    result["overall_level"] = _LEVEL_LABELS[weakest]
    p99 = thresholds.loc[list(METRICS), "p99"].to_dict()
    eligible = np.fromiter(
        (
            passes_placebo_threshold(row, p99)
            for row in result.loc[:, list(METRICS)].to_dict(orient="records")
        ),
        dtype=bool,
        count=len(result),
    )
    result["candidate_eligible"] = eligible if scope == "formal" else False
    if scope == "formal":
        result["suggest_evict"] = ~eligible
    else:
        result["suggest_evict"] = pd.array([pd.NA] * len(result), dtype="boolean")
    return result
