"""Model persistence, so a trained fold can score dates it has never seen.

A checkpoint carries the architecture and the exact factor list it was trained on.
Loading refuses to proceed if the factor library has changed underneath it -- silently
feeding a re-mined set of factors into an old network produces plausible-looking
probabilities that mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..logging_setup import get_logger
from .models import GRUCombiner

log = get_logger(__name__)

CHECKPOINT_SUFFIX = ".pt"


@dataclass
class Checkpoint:
    state_dict: dict
    n_features: int
    seq_len: int
    hidden_size: int
    num_layers: int
    dropout: float
    fold: int
    factor_names: list[str] = field(default_factory=list)
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""

    def build(self, device) -> GRUCombiner:
        model = GRUCombiner(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in checkpoint.__dict__.items()}
    torch.save(payload, path)
    log.info("saved fold %d checkpoint to %s", checkpoint.fold, path.name)


def load_checkpoint(path: Path) -> Checkpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return Checkpoint(**payload)


def latest_checkpoint(directory: Path) -> Path:
    """The checkpoint whose test window ends last -- the most recently trained model."""
    candidates = sorted(Path(directory).glob(f"fold_*{CHECKPOINT_SUFFIX}"))
    if not candidates:
        raise FileNotFoundError(
            f"no model checkpoints in {directory}; run `helix train` first"
        )
    return candidates[-1]


def require_matching_factors(checkpoint: Checkpoint, factor_names: list[str]) -> None:
    if checkpoint.factor_names and checkpoint.factor_names != factor_names:
        raise ValueError(
            "the factor library no longer matches this checkpoint.\n"
            f"  model was trained on: {checkpoint.factor_names}\n"
            f"  library now provides: {factor_names}\n"
            "Re-run `helix train` after re-mining."
        )
