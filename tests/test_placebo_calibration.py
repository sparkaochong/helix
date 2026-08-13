"""Behavioral contract for training-only placebo IC/Gini calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _threshold_kwargs() -> dict[str, object]:
    return {
        "ic_mean": 0.04,
        "icir": 0.25,
        "gini": 0.08,
        "quantile": 0.99,
        "train_start": "2022-01-04",
        "train_end": "2024-09-04",
    }


def _screening_quantiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["ic_mean", "icir", "gini"],
            "p95": [0.20, 0.20, 0.20],
            "p99": [0.30, 0.30, 0.30],
            "p999": [0.40, 0.40, 0.40],
        }
    )


def _factor_metric_row(factor_name: str, value: float) -> dict[str, object]:
    return {
        "factor_name": factor_name,
        "ic_mean_signed": value,
        "ic_mean": abs(value),
        "icir_signed": value,
        "icir": abs(value),
        "gini_signed": value,
        "gini": abs(value),
    }


def _two_date_factor_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, np.nan],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    factors = np.array(
        [
            [
                [1.0, 2.0],
                [2.0, 5.0],
                [3.0, 1.0],
                [4.0, 4.0],
                [5.0, 3.0],
                [np.nan, np.nan],
            ],
            [
                [6.0, 1.0],
                [1.0, 6.0],
                [5.0, 2.0],
                [2.0, 5.0],
                [4.0, 3.0],
                [3.0, 4.0],
            ],
        ]
    )
    return factors, labels, np.isfinite(labels)


def _small_calibration_case(tmp_path, monkeypatch):
    import hashlib

    import scripts.calibrate_placebo as calibration
    from helix.gp.library import FactorLibrary, FactorSpec, save_factors

    dates = ["2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04"]
    signals = [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 1.0, 4.0, 2.0, 3.0, 0.0],
        [2.0, 5.0, 0.0, 4.0, 1.0, 3.0],
        [4.0, 0.0, 5.0, 1.0, 3.0, 2.0],
    ]
    labels = [
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    ]
    rows = []
    for trade_date, daily_signals, daily_labels in zip(dates, signals, labels, strict=True):
        for index, (signal, label) in enumerate(
            zip(daily_signals, daily_labels, strict=True)
        ):
            rows.append(
                {
                    "trade_date": trade_date,
                    "stock_code": f"{index:06d}",
                    "label_d2_hit_8pct": label,
                    "signal": signal,
                }
            )
    input_path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    library_paths = []
    for scope in ("formal", "n40", "multi"):
        library_path = tmp_path / f"{scope}.json"
        save_factors(
            library_path,
            FactorLibrary(
                factors=[FactorSpec(name="gp_000", expression="signal", sign=1.0)],
                field_names=["signal"],
                windows=[],
                kind="event",
            ),
        )
        library_paths.append(library_path)

    monkeypatch.setattr(calibration, "FORMAL_TRAIN_START", dates[0])
    monkeypatch.setattr(calibration, "FORMAL_TRAIN_END", dates[-1])
    monkeypatch.setattr(calibration, "FORMAL_TRAIN_DATES", len(dates))
    monkeypatch.setattr(
        calibration,
        "FORMAL_TRAIN_DATE_DIGEST",
        hashlib.sha256("\n".join(dates).encode()).hexdigest(),
    )

    distribution_path = tmp_path / "distribution.parquet"
    screening_path = tmp_path / "screening.parquet"
    report_path = tmp_path / "report.md"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("gp:\n  population: 321\n", encoding="utf-8")
    kwargs = {
        "input_path": input_path,
        "formal_library_path": library_paths[0],
        "n40_library_path": library_paths[1],
        "multi_library_path": library_paths[2],
        "train_end": dates[-1],
        "seed": 7,
        "n_permutations": 5,
        "min_samples": 1,
        "distribution_path": distribution_path,
        "screening_path": screening_path,
        "report_path": report_path,
        "write_config_path": config_path,
    }
    return calibration, dates, kwargs, [
        distribution_path,
        screening_path,
        report_path,
        config_path,
    ]


def test_cross_sectional_permutations_preserve_daily_class_counts():
    from helix.eval.placebo import permute_binary_labels

    labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0])

    permutations = permute_binary_labels(labels, 1000, np.random.default_rng(7))

    assert permutations.shape == (1000, 5)
    np.testing.assert_array_equal(permutations.sum(axis=1), np.full(1000, 2))
    np.testing.assert_array_equal(
        (~permutations.astype(bool)).sum(axis=1), np.full(1000, 3)
    )


def test_placebo_generation_never_moves_labels_between_dates():
    from helix.eval.placebo import iter_cross_sectional_permutations

    labels = np.array(
        [
            [1.0, 0.0, 0.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, 0.0, np.nan],
        ]
    )
    mask = np.isfinite(labels)

    blocks = list(
        iter_cross_sectional_permutations(
            labels, mask, n_permutations=1000, rng=np.random.default_rng(8)
        )
    )

    assert [date_index for date_index, _, _ in blocks] == [0, 1]
    np.testing.assert_array_equal(blocks[0][1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(blocks[1][1], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(blocks[0][2].sum(axis=1), np.full(1000, 1))
    np.testing.assert_array_equal(blocks[1][2].sum(axis=1), np.full(1000, 3))


def test_permutations_are_reproducible_by_seed():
    from helix.eval.placebo import permute_binary_labels

    labels = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    first = permute_binary_labels(labels, 50, np.random.default_rng(11))
    repeated = permute_binary_labels(labels, 50, np.random.default_rng(11))
    changed = permute_binary_labels(labels, 50, np.random.default_rng(12))

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)


def test_vectorized_placebo_daily_metrics_match_existing_daily_metrics():
    from helix.eval.ic import daily_ic
    from helix.eval.metrics import daily_gini
    from helix.eval.placebo import permute_binary_labels, placebo_daily_metrics

    factors = np.array(
        [
            [1.0, 9.0],
            [1.0, 7.0],
            [2.0, 7.0],
            [3.0, 4.0],
            [5.0, 2.0],
            [8.0, 0.0],
        ]
    )
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    permutations = permute_binary_labels(labels, 7, np.random.default_rng(19))

    actual_ic, actual_gini = placebo_daily_metrics(
        factors, permutations, min_samples=1
    )
    expected_ic = np.empty_like(actual_ic)
    expected_gini = np.empty_like(actual_gini)
    daily_mask = np.ones((1, labels.size), dtype=bool)
    for permutation_index, permuted_labels in enumerate(permutations):
        target = permuted_labels.astype(float)[None, :]
        for factor_index in range(factors.shape[1]):
            factor = factors[:, factor_index][None, :]
            expected_ic[permutation_index, factor_index] = daily_ic(
                factor, target, daily_mask, min_samples=1
            )[0]
            expected_gini[permutation_index, factor_index] = daily_gini(
                factor, target, daily_mask, min_samples=1
            )[0]

    assert actual_ic.shape == (7, 2)
    assert actual_gini.shape == (7, 2)
    np.testing.assert_allclose(actual_ic, expected_ic, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_gini, expected_gini, rtol=0.0, atol=1e-12)


def test_placebo_daily_metrics_match_canonical_metrics_with_factor_missingness():
    from helix.eval.ic import daily_ic
    from helix.eval.metrics import daily_gini
    from helix.eval.placebo import placebo_daily_metrics

    factors = np.array(
        [
            [4.0, np.nan],
            [1.0, 4.0],
            [3.0, 1.0],
            [2.0, 3.0],
            [np.nan, 2.0],
        ]
    )
    permutations = np.array(
        [
            [1, 1, 1, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=bool,
    )

    actual_ic, actual_gini = placebo_daily_metrics(
        factors, permutations, min_samples=1
    )
    expected_ic = np.empty_like(actual_ic)
    expected_gini = np.empty_like(actual_gini)
    daily_mask = np.ones((1, factors.shape[0]), dtype=bool)
    for permutation_index, permuted_labels in enumerate(permutations):
        target = permuted_labels.astype(float)[None, :]
        for factor_index in range(factors.shape[1]):
            factor = factors[:, factor_index][None, :]
            expected_ic[permutation_index, factor_index] = daily_ic(
                factor, target, daily_mask, min_samples=1
            )[0]
            expected_gini[permutation_index, factor_index] = daily_gini(
                factor, target, daily_mask, min_samples=1
            )[0]

    assert np.isfinite(expected_ic[0, 0])
    assert np.isnan(expected_gini[0, 0])
    np.testing.assert_allclose(
        actual_ic, expected_ic, rtol=0.0, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        actual_gini, expected_gini, rtol=0.0, atol=1e-12, equal_nan=True
    )


def test_placebo_daily_metrics_bounds_ranked_label_workspaces(monkeypatch):
    import helix.eval.placebo as placebo
    from helix.eval.ic import daily_ic
    from helix.eval.metrics import daily_gini

    n_permutations, n_samples, n_factors = 5, 8, 9
    rng = np.random.default_rng(31)
    factors = rng.integers(-4, 5, size=(n_samples, n_factors)).astype(float)
    base_labels = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    permutations = np.vstack(
        [np.roll(base_labels, shift) for shift in range(n_permutations)]
    )
    rank_element_budget = 80
    rank_input_sizes = []
    canonical_cs_rank = placebo.cs_rank

    def recording_cs_rank(values):
        rank_input_sizes.append(values.size)
        return canonical_cs_rank(values)

    monkeypatch.setattr(
        placebo, "_MAX_RANKED_LABEL_ELEMENTS", rank_element_budget, raising=False
    )
    monkeypatch.setattr(placebo, "cs_rank", recording_cs_rank)

    actual_ic, actual_gini = placebo.placebo_daily_metrics(
        factors, permutations, min_samples=1
    )
    expected_ic = np.empty_like(actual_ic)
    expected_gini = np.empty_like(actual_gini)
    daily_mask = np.ones((1, n_samples), dtype=bool)
    for permutation_index, permuted_labels in enumerate(permutations):
        target = permuted_labels.astype(float)[None, :]
        for factor_index in range(n_factors):
            factor = factors[:, factor_index][None, :]
            expected_ic[permutation_index, factor_index] = daily_ic(
                factor, target, daily_mask, min_samples=1
            )[0]
            expected_gini[permutation_index, factor_index] = daily_gini(
                factor, target, daily_mask, min_samples=1
            )[0]

    assert actual_ic.shape == (n_permutations, n_factors)
    assert actual_gini.shape == (n_permutations, n_factors)
    assert max(rank_input_sizes) <= rank_element_budget
    np.testing.assert_allclose(actual_ic, expected_ic, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_gini, expected_gini, rtol=0.0, atol=1e-12)


def test_placebo_distribution_matches_full_cycle_metrics_and_formal_factor_maximum():
    from helix.eval.ic import daily_ic
    from helix.eval.metrics import daily_gini
    from helix.eval.placebo import (
        iter_cross_sectional_permutations,
        placebo_distribution,
    )

    factors, labels, mask = _two_date_factor_inputs()
    n_permutations = 6
    seed = 1
    distribution = placebo_distribution(
        factors,
        labels,
        mask,
        n_permutations=n_permutations,
        seed=seed,
        min_samples=1,
    )

    permuted = np.full((n_permutations, *labels.shape), np.nan)
    for date_index, valid_positions, daily_permutations in (
        iter_cross_sectional_permutations(
            labels,
            mask,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
        )
    ):
        permuted[:, date_index, valid_positions] = daily_permutations

    per_factor = np.empty((n_permutations, factors.shape[2], 3))
    for permutation_index in range(n_permutations):
        for factor_index in range(factors.shape[2]):
            daily_ics = daily_ic(
                factors[:, :, factor_index],
                permuted[permutation_index],
                mask,
                min_samples=1,
            )
            daily_ginis = daily_gini(
                factors[:, :, factor_index],
                permuted[permutation_index],
                mask,
                min_samples=1,
            )
            ic_mean = daily_ics.mean()
            per_factor[permutation_index, factor_index] = (
                ic_mean,
                ic_mean / daily_ics.std(ddof=1),
                daily_ginis.mean(),
            )

    expected = np.max(np.abs(per_factor), axis=1)
    assert set(np.argmax(np.abs(per_factor), axis=1).ravel()) == {0, 1}
    metric_columns = ["ic_mean", "icir", "gini"]
    np.testing.assert_array_equal(
        distribution["permutation_id"], np.arange(n_permutations)
    )
    np.testing.assert_allclose(
        distribution[metric_columns].to_numpy(), expected, rtol=0.0, atol=1e-12
    )


def test_factor_metrics_match_existing_signed_and_absolute_metric_summaries():
    from helix.eval.ic import daily_ic, summarize_ic
    from helix.eval.metrics import daily_gini, summarize_daily
    from helix.eval.placebo import factor_metrics

    factors, labels, mask = _two_date_factor_inputs()
    factor_names = ["ascending", "mixed"]
    actual = factor_metrics(
        factor_names, factors, labels, mask, min_samples=1
    ).set_index("factor_name")

    expected = []
    for factor_index in range(factors.shape[2]):
        ic_summary = summarize_ic(
            daily_ic(
                factors[:, :, factor_index], labels, mask, min_samples=1
            )
        )
        gini_summary = summarize_daily(
            daily_gini(
                factors[:, :, factor_index], labels, mask, min_samples=1
            )
        )
        expected.append(
            [
                ic_summary["ic_mean"],
                abs(ic_summary["ic_mean"]),
                ic_summary["icir"],
                abs(ic_summary["icir"]),
                gini_summary["mean"],
                abs(gini_summary["mean"]),
            ]
        )

    metric_columns = [
        "ic_mean_signed",
        "ic_mean",
        "icir_signed",
        "icir",
        "gini_signed",
        "gini",
    ]
    np.testing.assert_allclose(
        actual.loc[factor_names, metric_columns].to_numpy(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_quantiles_use_numpy_linear_method():
    from helix.eval.placebo import metric_quantiles

    distribution = pd.DataFrame(
        {
            "permutation_id": np.arange(5),
            "ic_mean": np.arange(5, dtype=float),
            "icir": np.arange(5, dtype=float) * 2,
            "gini": np.arange(5, dtype=float) * 3,
        }
    )

    quantiles = metric_quantiles(distribution).set_index("metric")

    assert quantiles.loc["ic_mean", "p95"] == pytest.approx(3.8)
    assert quantiles.loc["ic_mean", "p99"] == pytest.approx(3.96)
    assert quantiles.loc["ic_mean", "p999"] == pytest.approx(3.996)
    assert quantiles.loc["icir", "p95"] == pytest.approx(7.6)
    assert quantiles.loc["icir", "p99"] == pytest.approx(7.92)
    assert quantiles.loc["icir", "p999"] == pytest.approx(7.992)
    assert quantiles.loc["gini", "p95"] == pytest.approx(11.4)
    assert quantiles.loc["gini", "p99"] == pytest.approx(11.88)
    assert quantiles.loc["gini", "p999"] == pytest.approx(11.988)


def test_admission_requires_every_metric_to_strictly_exceed_p99():
    from helix.eval.placebo import passes_placebo_threshold

    thresholds = {"ic_mean": 0.04, "icir": 0.25, "gini": 0.08}
    above = {name: value + 0.001 for name, value in thresholds.items()}
    assert passes_placebo_threshold(above, thresholds)

    for metric_name in thresholds:
        metrics = above.copy()
        metrics[metric_name] = thresholds[metric_name]
        assert not passes_placebo_threshold(metrics, thresholds)


def test_admission_accepts_typed_placebo_threshold_config():
    from helix.config import PlaceboThresholdConfig
    from helix.eval.placebo import passes_placebo_threshold

    config = PlaceboThresholdConfig(**_threshold_kwargs())
    metrics = {
        "ic_mean": config.ic_mean + 0.001,
        "icir": config.icir + 0.001,
        "gini": config.gini + 0.001,
    }

    assert passes_placebo_threshold(metrics, config)


def test_training_loader_excludes_post_cutoff_extremes_from_frame_and_result(
    tmp_path, monkeypatch
):
    from scripts.calibrate_placebo import load_training_frame

    train_end = "2024-09-04"
    training_rows = pd.DataFrame(
        {
            "trade_date": ["2024-09-03", "2024-09-03", train_end, train_end],
            "stock_code": ["000001", "000002", "000001", "000002"],
            "label_d2_hit_8pct": [0.0, 1.0, 1.0, 0.0],
            "signal": [1.0, 2.0, 3.0, 4.0],
            "unused": [10.0, 20.0, 30.0, 40.0],
        }
    )
    post_cutoff = pd.DataFrame(
        {
            "trade_date": ["2024-09-05", "2024-09-05"],
            "stock_code": ["000001", "000002"],
            "label_d2_hit_8pct": [1.0, 1.0],
            "signal": [1e100, 1e100],
            "unused": [1e100, 1e100],
        }
    )
    training_path = tmp_path / "training_only.parquet"
    contaminated_path = tmp_path / "with_post_cutoff.parquet"
    training_rows.to_parquet(training_path, index=False)
    pd.concat([training_rows, post_cutoff], ignore_index=True).to_parquet(
        contaminated_path, index=False
    )

    real_read_parquet = pd.read_parquet
    read_calls = []

    def recording_read_parquet(path, *args, **kwargs):
        read_calls.append({"path": path, **kwargs})
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)

    expected = load_training_frame(training_path, ["signal"], train_end)
    actual = load_training_frame(contaminated_path, ["signal"], train_end)

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert actual["trade_date"].max() == train_end
    assert actual["signal"].mean() == pytest.approx(2.5)
    assert "unused" not in actual.columns
    assert len(read_calls) == 2
    for call in read_calls:
        assert call["filters"] == [("trade_date", "<=", train_end)]
        assert set(call["columns"]) == {
            "trade_date",
            "stock_code",
            "label_d2_hit_8pct",
            "signal",
        }
        assert "unused" not in call["columns"]


@pytest.mark.parametrize("train_end", [None, "", "2024-09-03", "2024-09-05"])
def test_training_loader_rejects_every_nonformal_cutoff(train_end):
    from scripts.calibrate_placebo import validate_train_end

    with pytest.raises(ValueError):
        validate_train_end(train_end)


def test_training_loader_accepts_only_the_formal_cutoff():
    from scripts.calibrate_placebo import validate_train_end

    assert validate_train_end("2024-09-04") == "2024-09-04"


def test_training_loader_rejects_input_without_the_exact_formal_cutoff_date(tmp_path):
    from scripts.calibrate_placebo import load_training_frame

    path = tmp_path / "cutoff_absent.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2024-09-03", "2024-09-03"],
            "stock_code": ["000001", "000002"],
            "label_d2_hit_8pct": [0.0, 1.0],
            "signal": [1.0, 2.0],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError):
        load_training_frame(path, ["signal"], "2024-09-04")


def test_training_loader_rejects_a_missing_requested_column(tmp_path):
    from scripts.calibrate_placebo import load_training_frame

    path = tmp_path / "missing_feature.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2024-09-04", "2024-09-04"],
            "stock_code": ["000001", "000002"],
            "label_d2_hit_8pct": [0.0, 1.0],
        }
    ).to_parquet(path, index=False)

    with pytest.raises((KeyError, ValueError), match="missing"):
        load_training_frame(path, ["missing"], "2024-09-04")


def test_calibrate_scopes_isolates_formal_null_from_both_supplemental_scopes():
    from helix.eval.placebo import metric_quantiles, placebo_distribution
    from scripts.calibrate_placebo import calibrate_scopes

    formal_values, labels, mask = _two_date_factor_inputs()
    formal_names = ["formal_a", "formal_b"]
    first_supplemental = {
        "argus_n40": (
            ["n40_a"],
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ]
            )[:, :, None],
            "data/artifacts/argus_n40/event_factors.json",
        ),
        "argus_multi": (
            ["multi_a"],
            np.array(
                [
                    [2.0, 5.0, 1.0, 4.0, 3.0, 6.0],
                    [1.0, 6.0, 2.0, 5.0, 3.0, 4.0],
                ]
            )[:, :, None],
            "data/artifacts/argus_multi/event_factors.json",
        ),
    }
    changed_supplemental = {
        "argus_n40": (
            ["n40_a"],
            np.array(
                [
                    [5.0, 1.0, 4.0, 2.0, 3.0, 6.0],
                    [6.0, 2.0, 5.0, 1.0, 4.0, 3.0],
                ]
            )[:, :, None],
            "data/artifacts/argus_n40/event_factors.json",
        ),
        "argus_multi": (
            ["multi_a"],
            np.array(
                [
                    [3.0, 1.0, 5.0, 2.0, 4.0, 6.0],
                    [4.0, 1.0, 6.0, 3.0, 2.0, 5.0],
                ]
            )[:, :, None],
            "data/artifacts/argus_multi/event_factors.json",
        ),
    }
    calibration_kwargs = {
        "labels": labels,
        "mask": mask,
        "n_permutations": 6,
        "seed": 1,
        "min_samples": 1,
    }

    first_distribution, first_quantiles, first_screening = calibrate_scopes(
        formal_names,
        formal_values,
        first_supplemental,
        **calibration_kwargs,
    )
    changed_distribution, changed_quantiles, changed_screening = calibrate_scopes(
        formal_names,
        formal_values,
        changed_supplemental,
        **calibration_kwargs,
    )
    expected_distribution = placebo_distribution(
        formal_values, **calibration_kwargs
    )
    expected_quantiles = metric_quantiles(expected_distribution)
    contaminated_distribution = placebo_distribution(
        np.concatenate(
            (
                formal_values,
                first_supplemental["argus_n40"][1],
                first_supplemental["argus_multi"][1],
            ),
            axis=2,
        ),
        **calibration_kwargs,
    )

    pd.testing.assert_frame_equal(
        first_distribution, changed_distribution, check_exact=True
    )
    pd.testing.assert_frame_equal(
        first_quantiles, changed_quantiles, check_exact=True
    )
    pd.testing.assert_frame_equal(
        first_distribution, expected_distribution, check_exact=True
    )
    pd.testing.assert_frame_equal(first_quantiles, expected_quantiles, check_exact=True)
    assert not contaminated_distribution.equals(expected_distribution)
    for screening in (first_screening, changed_screening):
        formal = screening.query("scope == 'formal'")
        supplemental = screening.query("scope != 'formal'")
        assert set(formal["factor_name"]) == set(formal_names)
        assert set(supplemental["scope"]) == {"argus_n40", "argus_multi"}
        assert len(supplemental) == 2
        assert not supplemental["candidate_eligible"].astype(bool).any()
        assert supplemental["suggest_evict"].isna().all()


def test_report_documents_training_only_thresholds_and_separate_scopes():
    from scripts.calibrate_placebo import render_report

    distribution = pd.DataFrame(
        {
            "permutation_id": [0, 1, 2],
            "ic_mean": [0.01, 0.02, 0.03],
            "icir": [0.10, 0.20, 0.30],
            "gini": [0.04, 0.05, 0.06],
        }
    )
    quantiles = pd.DataFrame(
        {
            "metric": ["ic_mean", "icir", "gini"],
            "p95": [0.028, 0.28, 0.058],
            "p99": [0.0296, 0.296, 0.0596],
            "p999": [0.02996, 0.2996, 0.05996],
        }
    )
    screening = pd.DataFrame(
        [
            {
                **_factor_metric_row("formal_a", 0.31),
                "scope": "formal",
                "library_path": "data/artifacts/argus/event_factors.json",
                "ic_level": "超 p99.9",
                "icir_level": "超 p99.9",
                "gini_level": "超 p99.9",
                "overall_level": "超 p99.9",
                "candidate_eligible": True,
                "suggest_evict": False,
            },
            {
                **_factor_metric_row("gp_000", 0.21),
                "scope": "argus_n40",
                "library_path": "data/artifacts/argus_n40/event_factors.json",
                "ic_level": "超 p95",
                "icir_level": "超 p95",
                "gini_level": "超 p95",
                "overall_level": "超 p95",
                "candidate_eligible": False,
                "suggest_evict": pd.NA,
            },
            {
                **_factor_metric_row("gp_000", 0.10),
                "scope": "argus_multi",
                "library_path": "data/artifacts/argus_multi/event_factors.json",
                "ic_level": "低于随机水平",
                "icir_level": "低于随机水平",
                "gini_level": "低于随机水平",
                "overall_level": "低于随机水平",
                "candidate_eligible": False,
                "suggest_evict": pd.NA,
            },
        ]
    )
    metadata = {
        "input": "data/raw/argus_quant_working.parquet",
        "target": "label_d2_hit_8pct",
        "seed": 20260813,
        "n_permutations": 1000,
        "min_samples": 50,
        "train_start": "2022-01-04",
        "train_end": "2024-09-04",
        "n_train_dates": 649,
        "formal_factor_count": 1,
        "supplemental_factor_count": 2,
    }

    report = render_report(distribution, quantiles, screening, metadata)

    for value in (
        metadata["input"],
        metadata["target"],
        str(metadata["seed"]),
        str(metadata["n_permutations"]),
        str(metadata["min_samples"]),
        metadata["train_start"],
        metadata["train_end"],
        str(metadata["n_train_dates"]),
    ):
        assert str(value) in report
    assert "Null Distribution Summary" in report
    assert all(statistic in report for statistic in ("mean", "std", "min", "median", "max"))
    assert "Core Thresholds" in report
    assert all(level in report for level in ("p95", "p99", "p99.9"))
    assert "Formal Library" in report
    assert "formal_a" in report
    assert "data/artifacts/argus/event_factors.json" in report
    assert "Formal level counts" in report
    assert "Supplemental: argus_n40" in report
    assert "Supplemental: argus_multi" in report
    assert "data/artifacts/argus_n40/event_factors.json" in report
    assert "data/artifacts/argus_multi/event_factors.json" in report
    assert report.count("gp_000") == 2
    assert "argus_n40 level counts" in report
    assert "argus_multi level counts" in report
    assert "Only the formal library and training dates generated these thresholds" in report
    assert "Supplemental scopes never participated in threshold generation" in report
    assert "never eligible for formal admission" in report
    assert "Passing G1 does not override G3" in report


def _render_minimal_report(input_path: str | Path, library_path: str | Path) -> str:
    from scripts.calibrate_placebo import render_report

    distribution = pd.DataFrame(
        {
            "permutation_id": [0],
            "ic_mean": [0.01],
            "icir": [0.10],
            "gini": [0.04],
        }
    )
    quantiles = pd.DataFrame(
        {
            "metric": ["ic_mean", "icir", "gini"],
            "p95": [0.01, 0.10, 0.04],
            "p99": [0.01, 0.10, 0.04],
            "p999": [0.01, 0.10, 0.04],
        }
    )
    screening = pd.DataFrame(
        [
            {
                **_factor_metric_row("formal_a", 0.31),
                "scope": "formal",
                "library_path": str(library_path),
                "ic_level": "超 p99.9",
                "icir_level": "超 p99.9",
                "gini_level": "超 p99.9",
                "overall_level": "超 p99.9",
                "candidate_eligible": True,
                "suggest_evict": False,
            }
        ]
    )
    metadata = {
        "input": str(input_path),
        "target": "label_d2_hit_8pct",
        "seed": 20260813,
        "n_permutations": 1,
        "min_samples": 1,
        "train_start": "2022-01-04",
        "train_end": "2024-09-04",
        "n_train_dates": 649,
        "formal_factor_count": 1,
        "supplemental_factor_count": 0,
    }
    return render_report(distribution, quantiles, screening, metadata)


def test_report_renders_repository_paths_relative_and_preserves_external_paths(
    tmp_path,
):
    repository_root = Path(__file__).resolve().parents[1]
    report = _render_minimal_report(
        repository_root / "data/raw/argus_quant_working.parquet",
        repository_root / "data/artifacts/argus/event_factors.json",
    )

    assert "`data/raw/argus_quant_working.parquet`" in report
    assert "data/artifacts/argus/event_factors.json" in report
    assert str(repository_root) not in report

    external_input = tmp_path / "external.parquet"
    external_library = tmp_path / "external.json"
    external_report = _render_minimal_report(external_input, external_library)

    assert str(external_input) in external_report
    assert str(external_library) in external_report


def test_report_documents_ordinal_binary_ic_calibration_scope():
    report = _render_minimal_report(
        "data/raw/argus_quant_working.parquet",
        "data/artifacts/argus/event_factors.json",
    )

    for caveat in (
        "canonical `daily_ic`",
        "ordinal ranks",
        "binary label",
        "event-slot layout",
        "missingness structure",
        "formal factor family",
        "G1 baseline",
        "current event-table formal library",
        "requires recalibration",
    ):
        assert caveat in report


def test_threshold_config_appends_generated_block_and_round_trips_exact_values(tmp_path):
    from helix.config import Config
    from scripts.calibrate_placebo import write_threshold_config

    path = tmp_path / "config.yaml"
    original = '# hand-written content\ngp:\n  population: 321\n'
    path.write_text(original, encoding="utf-8")
    quantiles = _screening_quantiles()
    exact_p99 = {
        "ic_mean": float.fromhex("0x1.23456789abcdep-5"),
        "icir": float.fromhex("0x1.3456789abcdefp-2"),
        "gini": float.fromhex("0x1.456789abcdef0p-4"),
    }
    for metric, value in exact_p99.items():
        quantiles.loc[quantiles["metric"] == metric, "p99"] = value

    write_threshold_config(
        path,
        quantiles,
        train_start="2022-01-04",
        train_end="2024-09-04",
    )

    rendered = path.read_text(encoding="utf-8")
    assert rendered.startswith(original)
    assert rendered.count("# BEGIN PLACEBO CALIBRATION") == 1
    assert rendered.count("# END PLACEBO CALIBRATION") == 1
    for metric, value in exact_p99.items():
        assert f"    {metric}: {repr(value)}\n" in rendered
    assert '    train_start: "2022-01-04"\n' in rendered
    assert '    train_end: "2024-09-04"\n' in rendered

    config = Config.load(path)
    threshold = config.factor_admission.placebo_threshold
    assert threshold is not None
    assert threshold.ic_mean == exact_p99["ic_mean"]
    assert threshold.icir == exact_p99["icir"]
    assert threshold.gini == exact_p99["gini"]
    assert threshold.quantile == 0.99
    assert threshold.train_start == "2022-01-04"
    assert threshold.train_end == "2024-09-04"


def test_threshold_config_replaces_only_its_delimited_block(tmp_path):
    from scripts.calibrate_placebo import write_threshold_config

    path = tmp_path / "config.yaml"
    prefix = '# keep this comment\ndata:\n  start_date: "20200101"\n'
    old_block = """# BEGIN PLACEBO CALIBRATION
factor_admission:
  placebo_threshold:
    ic_mean: 9.0
# END PLACEBO CALIBRATION
"""
    suffix = "backtest:\n  top_k: 7\n# keep this footer\n"
    path.write_text(prefix + old_block + suffix, encoding="utf-8")
    quantiles = _screening_quantiles()

    write_threshold_config(
        path,
        quantiles,
        train_start="2022-01-04",
        train_end="2024-09-04",
    )

    rendered = path.read_text(encoding="utf-8")
    assert rendered.startswith(prefix)
    assert rendered.endswith(suffix)
    assert old_block not in rendered
    assert rendered.count("# BEGIN PLACEBO CALIBRATION") == 1
    assert rendered.count("# END PLACEBO CALIBRATION") == 1
    assert "    ic_mean: 0.3\n" in rendered
    assert "    icir: 0.3\n" in rendered
    assert "    gini: 0.3\n" in rendered


def test_cli_requires_an_explicit_train_end():
    from scripts.calibrate_placebo import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_calibration_rejects_a_shortened_formal_training_window():
    from scripts.calibrate_placebo import validate_training_dates

    shortened = np.array(["2022-01-05", "2024-09-04"])

    with pytest.raises(ValueError, match="formal training window"):
        validate_training_dates(shortened, "2024-09-04")


def test_calibration_rejects_a_changed_interior_training_date(monkeypatch):
    import hashlib

    import scripts.calibrate_placebo as calibration

    formal_dates = ["2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04"]
    monkeypatch.setattr(calibration, "FORMAL_TRAIN_START", formal_dates[0])
    monkeypatch.setattr(calibration, "FORMAL_TRAIN_END", formal_dates[-1])
    monkeypatch.setattr(calibration, "FORMAL_TRAIN_DATES", len(formal_dates))
    monkeypatch.setattr(
        calibration,
        "FORMAL_TRAIN_DATE_DIGEST",
        hashlib.sha256("\n".join(formal_dates).encode()).hexdigest(),
        raising=False,
    )
    changed_dates = [formal_dates[0], "2024-09-02x", formal_dates[2], formal_dates[-1]]

    with pytest.raises(ValueError, match="formal training calendar"):
        calibration.validate_training_dates(changed_dates, formal_dates[-1])


def test_calibration_rejects_labels_that_panel_packing_would_coerce_to_nan():
    from scripts.calibrate_placebo import validate_training_labels

    with pytest.raises(ValueError, match="binary"):
        validate_training_labels(pd.Series([0.0, 1.0, np.nan, "bad"]))


def test_calibration_rejects_library_fields_that_are_outcomes():
    from scripts.calibrate_placebo import validate_library_fields

    with pytest.raises(ValueError, match="label"):
        validate_library_fields(["volume", "label_d2_hit_8pct"])


def test_calibration_rejects_a_partially_replayed_library(monkeypatch):
    import scripts.calibrate_placebo as calibration
    from helix.gp.library import FactorLibrary, FactorSpec

    library = FactorLibrary(
        factors=[
            FactorSpec(name="first", expression="signal", sign=1.0),
            FactorSpec(name="skipped", expression="signal", sign=1.0),
        ],
        field_names=["signal"],
        windows=[],
        kind="event",
    )
    monkeypatch.setattr(
        calibration,
        "compute_factors",
        lambda _library, _fields: (
            ["first"],
            np.ones((2, 3, 1), dtype=np.float32),
        ),
    )

    with pytest.raises(RuntimeError, match="replay every saved factor"):
        calibration.compute_complete_library(
            library,
            {"signal": np.ones((2, 3))},
            scope="formal",
        )


def test_threshold_config_rejects_an_undelimited_factor_admission_section(tmp_path):
    from scripts.calibrate_placebo import write_threshold_config

    path = tmp_path / "config.yaml"
    original = "factor_admission:\n  placebo_threshold: null\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="outside the delimited block"):
        write_threshold_config(
            path,
            _screening_quantiles(),
            train_start="2022-01-04",
            train_end="2024-09-04",
        )

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("train_start", "train_end"),
    [
        ("2022-01-05", "2024-09-04"),
        ("2022-01-04", "2024-09-03"),
        ("2022-01-04", "2024-09-05"),
    ],
)
def test_threshold_config_rejects_any_nonformal_training_window(
    tmp_path, train_start, train_end
):
    from scripts.calibrate_placebo import write_threshold_config

    path = tmp_path / "config.yaml"
    original = "gp:\n  population: 321\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        write_threshold_config(
            path,
            _screening_quantiles(),
            train_start=train_start,
            train_end=train_end,
        )

    assert path.read_text(encoding="utf-8") == original


def test_run_calibration_replays_small_event_libraries_and_writes_idempotent_outputs(
    tmp_path, monkeypatch
):
    from helix.config import Config
    from helix.eval.placebo import metric_quantiles
    calibration, dates, kwargs, outputs = _small_calibration_case(tmp_path, monkeypatch)

    calibration.run_calibration(**kwargs)

    distribution_path, screening_path, _, config_path = outputs
    assert all(path.exists() for path in outputs)
    distribution = pd.read_parquet(distribution_path)
    screening = pd.read_parquet(screening_path)
    assert len(distribution) == 5
    assert distribution["seed"].eq(7).all()
    assert distribution["train_start"].eq(dates[0]).all()
    assert distribution["train_end"].eq(dates[-1]).all()
    assert distribution["n_train_dates"].eq(len(dates)).all()
    assert distribution["formal_factor_count"].eq(1).all()
    assert screening["scope"].value_counts().to_dict() == {
        "formal": 1,
        "argus_n40": 1,
        "argus_multi": 1,
    }
    expected = metric_quantiles(distribution).set_index("metric")["p99"]
    threshold = Config.load(config_path).factor_admission.placebo_threshold
    assert threshold is not None
    assert threshold.ic_mean == expected["ic_mean"]
    assert threshold.icir == expected["icir"]
    assert threshold.gini == expected["gini"]

    first_contents = {path: path.read_bytes() for path in outputs}
    calibration.run_calibration(**kwargs)
    assert {path: path.read_bytes() for path in outputs} == first_contents


def test_run_calibration_staging_failure_leaves_all_final_targets_unchanged(
    tmp_path, monkeypatch
):
    calibration, _, kwargs, outputs = _small_calibration_case(tmp_path, monkeypatch)
    before_contents = {
        path: path.read_bytes() if path.exists() else None for path in outputs
    }
    before_names = set(tmp_path.iterdir())
    real_to_parquet = pd.DataFrame.to_parquet
    calls = 0

    def fail_second_staging_write(self, path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected staging failure")
        return real_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second_staging_write)

    with pytest.raises(RuntimeError, match="injected staging failure"):
        calibration.run_calibration(**kwargs)

    assert {
        path: path.read_bytes() if path.exists() else None for path in outputs
    } == before_contents
    assert set(tmp_path.iterdir()) == before_names


def test_run_calibration_publish_failure_rolls_back_every_final_target(
    tmp_path, monkeypatch
):
    import os
    from pathlib import Path

    calibration, _, kwargs, outputs = _small_calibration_case(tmp_path, monkeypatch)
    for index, path in enumerate(outputs[1:-1], start=1):
        path.write_bytes(f"old-{index}".encode())
    before_contents = {
        path: path.read_bytes() if path.exists() else None for path in outputs
    }
    before_names = set(tmp_path.iterdir())
    real_replace = os.replace
    final_targets = set(outputs)
    publish_calls = 0
    injected = False

    def fail_second_final_replace(source, destination):
        nonlocal publish_calls, injected
        if Path(destination) in final_targets and not injected:
            publish_calls += 1
            if publish_calls == 2:
                injected = True
                raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_final_replace)

    with pytest.raises(OSError, match="injected publish failure"):
        calibration.run_calibration(**kwargs)

    assert {
        path: path.read_bytes() if path.exists() else None for path in outputs
    } == before_contents
    assert set(tmp_path.iterdir()) == before_names


def test_threshold_config_uses_same_directory_atomic_replace(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    import scripts.calibrate_placebo as calibration

    path = tmp_path / "config.yaml"
    path.write_text("gp:\n  population: 321\n", encoding="utf-8")
    replacements = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", recording_replace)

    calibration.write_threshold_config(
        path,
        _screening_quantiles(),
        train_start="2022-01-04",
        train_end="2024-09-04",
    )

    assert len(replacements) == 1
    staged, destination = replacements[0]
    assert staged != destination
    assert staged.parent == destination.parent == tmp_path
    assert destination == path
    assert not staged.exists()


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("distribution_path", "screening_path"),
        ("distribution_path", "report_path"),
        ("screening_path", "report_path"),
        ("distribution_path", "input_path"),
        ("screening_path", "formal_library_path"),
        ("report_path", "multi_library_path"),
        ("write_config_path", "input_path"),
        ("write_config_path", "n40_library_path"),
        ("write_config_path", "distribution_path"),
        ("write_config_path", "report_path"),
    ],
)
def test_run_calibration_rejects_path_collisions_before_loading_libraries(
    tmp_path, monkeypatch, left, right
):
    import scripts.calibrate_placebo as calibration

    paths = {
        "input_path": tmp_path / "input.parquet",
        "formal_library_path": tmp_path / "formal.json",
        "n40_library_path": tmp_path / "n40.json",
        "multi_library_path": tmp_path / "multi.json",
        "distribution_path": tmp_path / "distribution.parquet",
        "screening_path": tmp_path / "screening.parquet",
        "report_path": tmp_path / "report.md",
        "write_config_path": tmp_path / "config.yaml",
    }
    paths[left] = paths[right]
    monkeypatch.setattr(
        calibration,
        "_load_library",
        lambda *_args, **_kwargs: pytest.fail("library load happened before preflight"),
    )

    with pytest.raises(ValueError, match="path collision"):
        calibration.run_calibration(
            **paths,
            train_end="2024-09-04",
            n_permutations=2,
            min_samples=1,
        )


@pytest.mark.parametrize(
    ("argument", "invalid"),
    [
        ("n_permutations", 0),
        ("n_permutations", -1),
        ("n_permutations", True),
        ("n_permutations", 1.5),
        ("min_samples", 0),
        ("min_samples", -1),
        ("min_samples", True),
        ("min_samples", 1.5),
    ],
)
def test_run_calibration_rejects_invalid_counts_before_loading_libraries(
    tmp_path, monkeypatch, argument, invalid
):
    import scripts.calibrate_placebo as calibration

    monkeypatch.setattr(
        calibration,
        "_load_library",
        lambda *_args, **_kwargs: pytest.fail("library load happened before validation"),
    )
    kwargs = {
        "input_path": tmp_path / "input.parquet",
        "formal_library_path": tmp_path / "formal.json",
        "n40_library_path": tmp_path / "n40.json",
        "multi_library_path": tmp_path / "multi.json",
        "train_end": "2024-09-04",
        "n_permutations": 2,
        "min_samples": 1,
        "distribution_path": tmp_path / "distribution.parquet",
        "screening_path": tmp_path / "screening.parquet",
        "report_path": tmp_path / "report.md",
    }
    kwargs[argument] = invalid

    with pytest.raises(ValueError, match="positive integer"):
        calibration.run_calibration(**kwargs)


def test_formal_screening_uses_joint_admission_and_eviction_advice():
    from helix.eval.placebo import screen_factor_metrics

    metrics = pd.DataFrame(
        [
            _factor_metric_row("eligible", 0.31),
            _factor_metric_row("at_p99", 0.30),
        ]
    )

    screened = screen_factor_metrics(
        metrics,
        _screening_quantiles(),
        scope="formal",
        library_path="data/artifacts/argus/event_factors.json",
    ).set_index("factor_name")

    assert bool(screened.loc["eligible", "candidate_eligible"])
    assert not bool(screened.loc["eligible", "suggest_evict"])
    assert not bool(screened.loc["at_p99", "candidate_eligible"])
    assert bool(screened.loc["at_p99", "suggest_evict"])


def test_formal_screening_uses_shared_admission_entrypoint(monkeypatch):
    import helix.eval.placebo as placebo

    metrics = pd.DataFrame(
        [
            _factor_metric_row("first", 0.31),
            _factor_metric_row("second", 0.31),
        ]
    )
    calls = []

    def fake_admission(row, thresholds):
        calls.append((dict(row), dict(thresholds)))
        return len(calls) == 1

    monkeypatch.setattr(placebo, "passes_placebo_threshold", fake_admission)

    screened = placebo.screen_factor_metrics(
        metrics,
        _screening_quantiles(),
        scope="formal",
        library_path="data/artifacts/argus/event_factors.json",
    )

    assert screened["candidate_eligible"].tolist() == [True, False]
    assert len(calls) == len(metrics)
    assert calls[0][1] == {"ic_mean": 0.3, "icir": 0.3, "gini": 0.3}


def test_screening_levels_each_metric_and_uses_the_weakest_overall_level():
    from helix.eval.placebo import screen_factor_metrics

    metrics = pd.DataFrame(
        [
            {
                "factor_name": "mixed_levels",
                "ic_mean_signed": -0.41,
                "ic_mean": 0.41,
                "icir_signed": 0.31,
                "icir": 0.31,
                "gini_signed": -0.21,
                "gini": 0.21,
            }
        ]
    )

    screened = screen_factor_metrics(
        metrics,
        _screening_quantiles(),
        scope="formal",
        library_path="data/artifacts/argus/event_factors.json",
    ).iloc[0]

    assert screened["ic_level"] == "超 p99.9"
    assert screened["icir_level"] == "超 p99"
    assert screened["gini_level"] == "超 p95"
    assert screened["overall_level"] == "超 p95"
    assert not bool(screened["candidate_eligible"])
    assert bool(screened["suggest_evict"])


def test_placebo_threshold_config_accepts_valid_contract():
    from helix.config import PlaceboThresholdConfig

    config = PlaceboThresholdConfig(**_threshold_kwargs())

    assert config.quantile == pytest.approx(0.99)
    assert config.train_start == "2022-01-04"
    assert config.train_end == "2024-09-04"


def test_root_config_retains_typed_factor_admission_thresholds():
    from helix.config import (
        Config,
        FactorAdmissionConfig,
        PlaceboThresholdConfig,
    )

    config = Config.model_validate(
        {"factor_admission": {"placebo_threshold": _threshold_kwargs()}}
    )

    assert isinstance(config.factor_admission, FactorAdmissionConfig)
    assert isinstance(
        config.factor_admission.placebo_threshold, PlaceboThresholdConfig
    )
    assert config.factor_admission.placebo_threshold.ic_mean == pytest.approx(0.04)
    assert config.factor_admission.placebo_threshold.train_end == "2024-09-04"


def test_new_factor_admission_models_reject_unknown_fields_through_root_config():
    from helix.config import Config, FactorAdmissionConfig

    with pytest.raises(ValueError):
        FactorAdmissionConfig.model_validate({"unexpected": True})

    with pytest.raises(ValueError):
        Config.model_validate({"factor_admission": {"unexpected": True}})

    threshold = _threshold_kwargs()
    threshold["unexpected"] = True
    with pytest.raises(ValueError):
        Config.model_validate(
            {"factor_admission": {"placebo_threshold": threshold}}
        )


def test_placebo_threshold_config_accepts_zero_thresholds():
    from helix.config import PlaceboThresholdConfig

    values = _threshold_kwargs()
    values.update(ic_mean=0.0, icir=0.0, gini=0.0)

    config = PlaceboThresholdConfig(**values)

    assert config.ic_mean == pytest.approx(0.0)
    assert config.icir == pytest.approx(0.0)
    assert config.gini == pytest.approx(0.0)


def test_placebo_threshold_config_rejects_unknown_fields():
    from helix.config import PlaceboThresholdConfig

    values = _threshold_kwargs()
    values["unexpected"] = "silently accepting this would hide stale configuration"

    with pytest.raises(ValueError):
        PlaceboThresholdConfig(**values)


@pytest.mark.parametrize("metric", ["ic_mean", "icir", "gini"])
@pytest.mark.parametrize("invalid", [-0.001, np.nan, np.inf, -np.inf])
def test_placebo_threshold_config_rejects_negative_or_nonfinite_metrics(metric, invalid):
    from helix.config import PlaceboThresholdConfig

    values = _threshold_kwargs()
    values[metric] = invalid

    with pytest.raises(ValueError):
        PlaceboThresholdConfig(**values)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.01, 1.01, np.nan])
def test_placebo_threshold_config_rejects_invalid_quantiles(quantile):
    from helix.config import PlaceboThresholdConfig

    values = _threshold_kwargs()
    values["quantile"] = quantile

    with pytest.raises(ValueError):
        PlaceboThresholdConfig(**values)


@pytest.mark.parametrize(
    ("train_start", "train_end"),
    [
        ("2024-09-05", "2024-09-04"),
        ("", "2024-09-04"),
        ("2022-01-04", ""),
    ],
)
def test_placebo_threshold_config_rejects_invalid_training_dates(train_start, train_end):
    from helix.config import PlaceboThresholdConfig

    values = _threshold_kwargs()
    values.update(train_start=train_start, train_end=train_end)

    with pytest.raises(ValueError):
        PlaceboThresholdConfig(**values)


def test_permutation_rejects_nonbinary_labels():
    from helix.eval.placebo import permute_binary_labels

    with pytest.raises(ValueError):
        permute_binary_labels(
            np.array([0.0, 0.5, 1.0]), 10, np.random.default_rng(1)
        )


@pytest.mark.parametrize("n_permutations", [0, -1])
def test_permutation_rejects_nonpositive_permutation_counts(n_permutations):
    from helix.eval.placebo import permute_binary_labels

    with pytest.raises(ValueError):
        permute_binary_labels(
            np.array([0.0, 1.0]),
            n_permutations,
            np.random.default_rng(1),
        )


def test_cross_sectional_permutations_reject_a_single_class_date():
    from helix.eval.placebo import iter_cross_sectional_permutations

    labels = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])

    with pytest.raises(ValueError):
        list(
            iter_cross_sectional_permutations(
                labels,
                np.ones_like(labels, dtype=bool),
                n_permutations=10,
                rng=np.random.default_rng(2),
            )
        )


@pytest.mark.parametrize("invalid_label", [np.inf, -np.inf, 0.5])
def test_cross_sectional_permutations_reject_invalid_masked_labels(invalid_label):
    from helix.eval.placebo import iter_cross_sectional_permutations

    labels = np.array([[0.0, 1.0, invalid_label]])
    mask = np.array([[True, True, False]])

    with pytest.raises(ValueError):
        list(
            iter_cross_sectional_permutations(
                labels,
                mask,
                n_permutations=10,
                rng=np.random.default_rng(2),
            )
        )


def test_placebo_distribution_rejects_an_empty_formal_factor_array():
    from helix.eval.placebo import placebo_distribution

    labels = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    factors = np.empty((*labels.shape, 0), dtype=float)

    with pytest.raises(ValueError):
        placebo_distribution(
            factors,
            labels,
            np.ones_like(labels, dtype=bool),
            n_permutations=10,
            seed=3,
            min_samples=1,
        )


def test_placebo_distribution_rejects_nonfinite_final_metrics():
    from helix.eval.placebo import placebo_distribution

    labels = np.array([[0.0, 1.0, 0.0, 1.0]])
    factors = np.array([[[1.0], [2.0], [3.0], [4.0]]])

    with pytest.raises(ValueError):
        placebo_distribution(
            factors,
            labels,
            np.ones_like(labels, dtype=bool),
            n_permutations=10,
            seed=4,
            min_samples=1,
        )
