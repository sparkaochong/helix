"""Sequence construction and class weighting for the DL stage."""

from __future__ import annotations

import numpy as np
import pytest

from helix.dl.dataset import SequenceDataset, normalize_factors, positive_weight, sample_index


def test_normalization_is_per_date_and_per_factor():
    values = np.zeros((2, 4, 2))
    values[0, :, 0] = [1.0, 2.0, 3.0, 4.0]
    values[1, :, 0] = [100.0, 200.0, 300.0, 400.0]
    values[:, :, 1] = 7.0
    mask = np.ones((2, 4), dtype=bool)

    out = normalize_factors(values, mask, n_sigma=4.0)
    # Both dates share the same shape, so the same z-scores despite the scale jump.
    np.testing.assert_allclose(out[0, :, 0], out[1, :, 0], atol=1e-5)
    assert out[0, :, 0].mean() == pytest.approx(0.0, abs=1e-5)
    # A constant factor has no cross-sectional information.
    assert np.isnan(out[:, :, 1]).all()


def test_normalization_ignores_names_outside_the_mask():
    values = np.zeros((1, 4, 1))
    values[0, :, 0] = [1.0, 2.0, 3.0, 1000.0]
    mask = np.array([[True, True, True, False]])
    out = normalize_factors(values, mask, n_sigma=10.0)
    assert np.isnan(out[0, 3, 0])
    assert out[0, 1, 0] == pytest.approx(0.0, abs=1e-5)


def test_sample_index_drops_rows_without_a_full_lookback():
    mask = np.ones((10, 3), dtype=bool)
    idx = sample_index(mask, slice(0, 10), seq_len=4)
    assert idx[:, 0].min() == 3
    assert len(idx) == (10 - 3) * 3


def test_sample_index_respects_the_row_slice():
    mask = np.ones((10, 2), dtype=bool)
    idx = sample_index(mask, slice(6, 9), seq_len=3)
    assert set(idx[:, 0].tolist()) == {6, 7, 8}


def test_sample_index_is_empty_when_the_window_never_fits():
    mask = np.ones((10, 2), dtype=bool)
    assert len(sample_index(mask, slice(0, 3), seq_len=8)) == 0


def test_dataset_returns_the_trailing_window_plus_an_observed_channel():
    values = np.arange(30.0, dtype=np.float32).reshape(10, 3, 1)
    traded = np.ones((10, 3))
    traded[8, 1] = 0.0
    y = np.zeros((10, 3))
    y[9, 1] = 1.0

    ds = SequenceDataset(values, traded, y, np.array([[9, 1]]), seq_len=3)
    x, target = ds[0]

    assert x.shape == (3, 2)
    assert ds.n_features == 2
    np.testing.assert_allclose(x[:, 0].numpy(), values[7:10, 1, 0])
    np.testing.assert_allclose(x[:, 1].numpy(), [1.0, 0.0, 1.0])
    assert float(target) == 1.0


def test_dataset_fills_nan_with_zero():
    values = np.full((5, 1, 1), np.nan, dtype=np.float32)
    ds = SequenceDataset(values, np.ones((5, 1)), np.zeros((5, 1)), np.array([[4, 0]]), seq_len=2)
    x, _ = ds[0]
    assert np.isfinite(x.numpy()).all()


def test_positive_weight_is_capped():
    y = np.zeros((100, 10))
    y[0, 0] = 1.0  # 1 positive out of 1000
    index = np.array([[t, n] for t in range(100) for n in range(10)])
    assert positive_weight(y, index, cap=20.0) == pytest.approx(20.0)


def test_positive_weight_of_a_balanced_sample():
    y = np.array([[1.0, 1.0, 0.0, 0.0]])
    index = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    assert positive_weight(y, index, cap=20.0) == pytest.approx(1.0)
