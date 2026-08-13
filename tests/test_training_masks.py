"""Training targets and OOS scoring candidates must use different masks."""

from __future__ import annotations

import numpy as np

from helix.config import DLConfig
from helix.dl.train import train_fold
from helix.splits import Fold


def test_test_scoring_does_not_require_a_future_observable_label():
    shape = (8, 2)
    values = np.arange(16.0, dtype=np.float32).reshape(8, 2, 1)
    traded = np.ones(shape, dtype=np.float32)
    y = np.tile([[0.0, 1.0]], (8, 1))
    label_mask = np.ones(shape, dtype=bool)
    label_mask[7, 0] = False
    y[7, 0] = np.nan
    prediction_mask = np.ones(shape, dtype=bool)
    fold = Fold(index=0, train=slice(0, 4), valid=slice(4, 6), test=slice(6, 8))

    grid, result = train_fold(
        fold=fold,
        values=values,
        traded=traded,
        y=y,
        label_mask=label_mask,
        prediction_mask=prediction_mask,
        cfg=DLConfig(
            seq_len=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=8,
            early_stopping_patience=1,
        ),
    )

    assert result.n_test == 4
    assert np.isfinite(grid[7, 0]), "D0 candidate disappeared because its D+2 label is invalid"
