"""Turning the factor panel into training sequences.

Normalisation is **cross-sectional per date**: each factor is z-scored across the
stocks trading that day. That is leak-free by construction (it only ever touches
same-day information) and it is also what the model needs -- the decision is "which
of today's names", not "is today a good day".

Suspended days inside a lookback window become zeros, which is indistinguishable from
a genuinely average value. An extra ``traded`` channel is appended so the network can
tell the two apart instead of learning from fabricated flat stretches.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.operators import clip_sigma
from ..logging_setup import get_logger

log = get_logger(__name__)


def normalize_factors(values: np.ndarray, mask: np.ndarray, n_sigma: float) -> np.ndarray:
    """Per-date, per-factor cross-sectional z-score, winsorised at +/- ``n_sigma``.

    ``values`` is ``(T, N, K)``; the mask restricts the statistics to the tradable
    universe so a tail of untradable names cannot shift the distribution.
    """
    out = np.empty_like(values, dtype=np.float32)
    masked = np.where(mask[:, :, None], values, np.nan)
    with warnings.catch_warnings():
        # Dates with no tradable names produce all-NaN slices; NaN is the right answer.
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(values.shape[2]):
            layer = masked[:, :, k].astype(np.float64)
            mean = np.nanmean(layer, axis=1, keepdims=True)
            std = np.nanstd(layer, axis=1, keepdims=True)
            z = np.where(std > 1e-9, (layer - mean) / np.where(std > 1e-9, std, 1.0), np.nan)
            out[:, :, k] = clip_sigma(z, n_sigma).astype(np.float32)
    return out


def sample_index(mask: np.ndarray, rows: slice, seq_len: int) -> np.ndarray:
    """``(M, 2)`` array of ``(date_idx, stock_idx)`` samples inside ``rows``.

    Samples whose lookback window would run off the start of the panel are dropped.
    """
    start = rows.start or 0
    stop = rows.stop if rows.stop is not None else mask.shape[0]
    first = max(start, seq_len - 1)
    if first >= stop:
        return np.empty((0, 2), dtype=np.int64)
    local = np.argwhere(mask[first:stop])
    local[:, 0] += first
    return local.astype(np.int64)


class SequenceDataset(Dataset):
    """Lookback windows of normalised factors, one sample per ``(date, stock)``."""

    def __init__(
        self,
        values: np.ndarray,   # (T, N, K) normalised
        traded: np.ndarray,   # (T, N) 1.0 if the stock traded that day
        y: np.ndarray,        # (T, N) binary label
        index: np.ndarray,    # (M, 2)
        seq_len: int,
    ):
        self.values = values
        self.traded = traded.astype(np.float32)
        self.y = y.astype(np.float32)
        self.index = index
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        t, n = self.index[i]
        lo = t - self.seq_len + 1
        seq = self.values[lo : t + 1, n, :]
        obs = self.traded[lo : t + 1, n][:, None]
        x = np.concatenate([np.nan_to_num(seq, nan=0.0), obs], axis=1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.y[t, n])

    @property
    def n_features(self) -> int:
        return self.values.shape[2] + 1


def positive_weight(y: np.ndarray, index: np.ndarray, cap: float) -> float:
    """``n_negative / n_positive`` over the training samples, capped.

    Uncapped, a 2% base rate produces a weight of 49 and the model chases outliers.
    """
    if len(index) == 0:
        return 1.0
    labels = y[index[:, 0], index[:, 1]]
    n_pos = float(np.nansum(labels == 1.0))
    n_neg = float(np.nansum(labels == 0.0))
    if n_pos <= 0:
        return 1.0
    return float(min(n_neg / n_pos, cap))
