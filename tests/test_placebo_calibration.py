"""Behavioral contract for training-only placebo IC/Gini calibration."""

from __future__ import annotations

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
    from scripts.calibrate_placebo import calibrate_scopes

    from helix.eval.placebo import metric_quantiles, placebo_distribution

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
