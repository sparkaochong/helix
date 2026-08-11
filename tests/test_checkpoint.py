"""Model persistence and the guard that keeps a stale model from scoring new factors."""

from __future__ import annotations

import pytest
import torch

from helix.dl.checkpoint import (
    Checkpoint,
    latest_checkpoint,
    load_checkpoint,
    require_matching_factors,
    save_checkpoint,
)
from helix.dl.models import GRUCombiner


def make_checkpoint(fold: int = 0, factors: list[str] | None = None) -> Checkpoint:
    model = GRUCombiner(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
    return Checkpoint(
        state_dict=model.state_dict(),
        n_features=4,
        seq_len=5,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        fold=fold,
        factor_names=factors if factors is not None else ["gp_000", "gp_001", "gp_002"],
        train_end="20240101",
        test_start="20240110",
        test_end="20240131",
    )


def test_round_trip_preserves_weights_and_metadata(tmp_path):
    original = make_checkpoint()
    path = tmp_path / "fold_000.pt"
    save_checkpoint(path, original)

    loaded = load_checkpoint(path)
    assert loaded.seq_len == 5
    assert loaded.factor_names == ["gp_000", "gp_001", "gp_002"]
    assert loaded.test_end == "20240131"
    for key, tensor in original.state_dict.items():
        assert torch.allclose(tensor, loaded.state_dict[key])


def test_rebuilt_model_reproduces_the_original_output(tmp_path):
    original = make_checkpoint()
    path = tmp_path / "fold_000.pt"
    save_checkpoint(path, original)

    reference = GRUCombiner(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
    reference.load_state_dict(original.state_dict)
    reference.eval()

    restored = load_checkpoint(path).build(torch.device("cpu"))
    x = torch.randn(3, 5, 4)
    with torch.no_grad():
        assert torch.allclose(reference(x), restored(x), atol=1e-6)


def test_latest_checkpoint_picks_the_highest_fold(tmp_path):
    for fold in (0, 1, 2):
        save_checkpoint(tmp_path / f"fold_{fold:03d}.pt", make_checkpoint(fold=fold))
    assert load_checkpoint(latest_checkpoint(tmp_path)).fold == 2


def test_missing_checkpoint_directory_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="helix train"):
        latest_checkpoint(tmp_path)


def test_a_remined_factor_library_is_rejected():
    checkpoint = make_checkpoint(factors=["gp_000", "gp_001"])
    require_matching_factors(checkpoint, ["gp_000", "gp_001"])  # unchanged: fine
    with pytest.raises(ValueError, match="no longer matches"):
        require_matching_factors(checkpoint, ["gp_000", "gp_999"])


def test_checkpoints_without_recorded_factors_are_not_blocked():
    require_matching_factors(make_checkpoint(factors=[]), ["anything"])
