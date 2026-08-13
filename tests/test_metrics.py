"""Metric correctness against hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest

from helix.eval.metrics import (
    daily_auc,
    daily_gini,
    lift_at_k,
    pairwise_max_abs_corr,
    precision_at_k,
    summarize_daily,
)


def test_perfect_and_inverted_rankings():
    factor = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([[0.0, 0.0, 1.0, 1.0]])
    mask = np.ones((1, 4), dtype=bool)

    assert daily_auc(factor, y, mask, min_samples=1)[0] == pytest.approx(1.0)
    assert daily_auc(-factor, y, mask, min_samples=1)[0] == pytest.approx(0.0)
    assert daily_gini(factor, y, mask, min_samples=1)[0] == pytest.approx(1.0)


def test_auc_of_an_interleaved_ranking():
    # ranks 1..4 -> labels 0,1,0,1: positives sit at ranks 2 and 4
    factor = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([[0.0, 1.0, 0.0, 1.0]])
    mask = np.ones((1, 4), dtype=bool)
    assert daily_auc(factor, y, mask, min_samples=1)[0] == pytest.approx(0.75)


def test_dates_below_the_sample_floor_are_nan():
    factor = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[0.0, 1.0, 1.0]])
    mask = np.ones((1, 3), dtype=bool)
    assert np.isnan(daily_auc(factor, y, mask, min_samples=50)[0])


def test_single_class_dates_are_nan():
    factor = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[0.0, 0.0, 0.0]])
    mask = np.ones((1, 3), dtype=bool)
    assert np.isnan(daily_auc(factor, y, mask, min_samples=1)[0])


def test_masked_out_names_do_not_participate():
    factor = np.array([[100.0, 1.0, 2.0, 3.0, 4.0]])
    y = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    mask = np.array([[False, True, True, True, True]])

    # Inside the mask the ranking is perfect.
    assert daily_auc(factor, y, mask, min_samples=1)[0] == pytest.approx(1.0)
    # The excluded name is a top-ranked negative, so ignoring the mask hurts the score.
    unmasked = daily_auc(factor, y, np.ones((1, 5), dtype=bool), min_samples=1)[0]
    assert unmasked == pytest.approx(2.0 / 3.0)


def test_summarize_reports_coverage_over_all_dates():
    values = np.array([0.1, np.nan, 0.3, np.nan])
    stats = summarize_daily(values)
    assert stats["coverage"] == pytest.approx(0.5)
    assert stats["mean"] == pytest.approx(0.2)


def test_precision_at_k_versus_base_rate():
    score = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    y = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    mask = np.ones((1, 5), dtype=bool)
    precision, base = precision_at_k(score, y, mask, k=2)
    assert precision[0] == pytest.approx(1.0)
    assert base[0] == pytest.approx(0.4)
    assert lift_at_k(score, y, mask, k=2) == pytest.approx(2.5)


def test_precision_is_nan_when_fewer_than_k_names_are_available():
    score = np.array([[5.0, 4.0]])
    y = np.array([[1.0, 0.0]])
    mask = np.ones((1, 2), dtype=bool)
    precision, _ = precision_at_k(score, y, mask, k=5)
    assert np.isnan(precision[0])


def test_precision_does_not_replace_a_selected_name_with_an_unobservable_outcome():
    score = np.array([[5.0, 4.0]])
    y = np.array([[np.nan, 0.0]])
    d0_candidates = np.ones((1, 2), dtype=bool)

    precision, base = precision_at_k(score, y, d0_candidates, k=1)

    assert np.isnan(precision[0])
    assert base[0] == pytest.approx(0.0)


def test_correlation_dedup_detects_a_duplicate_factor():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 20))
    assert pairwise_max_abs_corr(a, []) == pytest.approx(0.0)
    assert pairwise_max_abs_corr(a, [a.copy()]) == pytest.approx(1.0, abs=1e-9)
    assert pairwise_max_abs_corr(a, [-a]) == pytest.approx(1.0, abs=1e-9)
    assert pairwise_max_abs_corr(a, [rng.normal(size=(50, 20))]) < 0.3
