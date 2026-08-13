"""Walk-forward training of the factor-combination network.

One model per fold, trained on that fold's train rows, early-stopped on its validation
rows, and used to score only its test rows. Concatenating the per-fold test scores
gives a prediction series in which every point was produced by a model that never saw
that date -- the only version of these numbers worth reading.

Early stopping watches **mean daily gini**, not pooled loss. Loss improves when the
model learns which weeks were easy; daily gini only improves when it ranks names
correctly within a day, which is what the strategy trades.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import DLConfig
from ..eval.metrics import daily_gini, summarize_daily
from ..logging_setup import get_logger
from ..splits import Fold
from .checkpoint import CHECKPOINT_SUFFIX, Checkpoint, save_checkpoint
from .dataset import SequenceDataset, positive_weight, sample_index
from .models import GRUCombiner, pick_device

log = get_logger(__name__)


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_valid: int
    n_test: int
    best_epoch: int
    valid_gini: float
    test_gini: float


def _loader(dataset: SequenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    # num_workers=0 on purpose: the panel is already in memory and worker processes
    # would copy it per worker on spawn-based platforms.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict_scores(model: nn.Module, dataset: SequenceDataset, device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for x, _ in _loader(dataset, batch_size, shuffle=False):
        logits = model(x.to(device))
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _scatter(scores: np.ndarray, index: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, np.nan, dtype=np.float32)
    if len(index):
        grid[index[:, 0], index[:, 1]] = scores
    return grid


def _grid_gini(grid, y, mask, rows: slice, min_samples: int = 50) -> float:
    stats = summarize_daily(daily_gini(grid[rows], y[rows], mask[rows], min_samples))
    return stats["mean"]


def train_fold(
    fold: Fold,
    values: np.ndarray,
    traded: np.ndarray,
    y: np.ndarray,
    label_mask: np.ndarray,
    prediction_mask: np.ndarray,
    cfg: DLConfig,
    checkpoint_dir: Path | None = None,
    factor_names: list[str] | None = None,
    dates: np.ndarray | None = None,
) -> tuple[np.ndarray, FoldResult]:
    torch.manual_seed(cfg.seed + fold.index)
    np.random.seed(cfg.seed + fold.index)
    device = pick_device()

    idx_train = sample_index(label_mask, fold.train, cfg.seq_len)
    idx_valid = sample_index(label_mask, fold.valid, cfg.seq_len)
    idx_test = sample_index(prediction_mask, fold.test, cfg.seq_len)
    if len(idx_train) == 0 or len(idx_valid) == 0:
        raise ValueError(f"fold {fold.index} has no usable train/valid samples")

    ds_train = SequenceDataset(values, traded, y, idx_train, cfg.seq_len)
    ds_valid = SequenceDataset(values, traded, y, idx_valid, cfg.seq_len)
    ds_test = SequenceDataset(values, traded, y, idx_test, cfg.seq_len)

    model = GRUCombiner(
        n_features=ds_train.n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = positive_weight(y, idx_train, cfg.pos_weight_cap)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    log.info(
        "fold %d | train %d / valid %d / test %d samples | pos_weight %.1f | device %s",
        fold.index, len(idx_train), len(idx_valid), len(idx_test), pos_weight, device.type,
    )

    best_gini, best_epoch, best_state, stale = -np.inf, -1, None, 0
    train_loader = _loader(ds_train, cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        total, seen = 0.0, 0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)

        valid_grid = _scatter(predict_scores(model, ds_valid, device, cfg.batch_size), idx_valid, y.shape)
        valid_gini = _grid_gini(valid_grid, y, label_mask, fold.valid)
        log.info(
            "  fold %d epoch %02d | loss %.4f | valid gini %.4f",
            fold.index, epoch, total / max(seen, 1), valid_gini,
        )

        if np.isfinite(valid_gini) and valid_gini > best_gini:
            best_gini, best_epoch, stale = valid_gini, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.early_stopping_patience:
                log.info("  fold %d early stop at epoch %d", fold.index, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_dir is not None:
        save_checkpoint(
            Path(checkpoint_dir) / f"fold_{fold.index:03d}{CHECKPOINT_SUFFIX}",
            Checkpoint(
                state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                n_features=ds_train.n_features,
                seq_len=cfg.seq_len,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                fold=fold.index,
                factor_names=list(factor_names or []),
                train_end=str(dates[fold.train][-1]) if dates is not None else "",
                test_start=str(dates[fold.test][0]) if dates is not None else "",
                test_end=str(dates[fold.test][-1]) if dates is not None else "",
            ),
        )

    test_grid = _scatter(predict_scores(model, ds_test, device, cfg.batch_size), idx_test, y.shape)
    result = FoldResult(
        fold=fold.index,
        n_train=len(idx_train),
        n_valid=len(idx_valid),
        n_test=len(idx_test),
        best_epoch=best_epoch,
        valid_gini=float(best_gini),
        test_gini=float(_grid_gini(test_grid, y, label_mask, fold.test)),
    )
    log.info(
        "fold %d done | best epoch %d | valid gini %.4f | test gini %.4f",
        fold.index, best_epoch, result.valid_gini, result.test_gini,
    )
    return test_grid, result


def train_walk_forward(
    folds: list[Fold],
    values: np.ndarray,
    traded: np.ndarray,
    y: np.ndarray,
    label_mask: np.ndarray,
    prediction_mask: np.ndarray,
    cfg: DLConfig,
    checkpoint_dir: Path | None = None,
    factor_names: list[str] | None = None,
    dates: np.ndarray | None = None,
) -> tuple[np.ndarray, list[FoldResult]]:
    """Returns the stitched out-of-sample prediction grid and per-fold diagnostics."""
    predictions = np.full(y.shape, np.nan, dtype=np.float32)
    results: list[FoldResult] = []
    for fold in folds:
        grid, result = train_fold(
            fold, values, traded, y, label_mask, prediction_mask, cfg,
            checkpoint_dir=checkpoint_dir, factor_names=factor_names, dates=dates,
        )
        rows = fold.test
        predictions[rows] = grid[rows]
        results.append(result)
    return predictions, results
