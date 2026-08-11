"""The factor-combination network.

A GRU over the lookback window, not a Transformer: with ~20 timesteps, a couple of
dozen inputs and a label whose positive rate is a few percent, attention has nothing
to attend to and simply overfits faster. The recurrent path captures "this factor has
been building for a week", which is the shape of information that actually matters
for a two-day touch.
"""

from __future__ import annotations

import torch
from torch import nn


class GRUCombiner(nn.Module):
    """``(B, L, F) -> (B,)`` logits for "the D+2 high reaches the target"."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.input_norm(x))
        return self.head(out[:, -1, :]).squeeze(-1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
