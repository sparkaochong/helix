"""Walk-forward geometry: the embargo gaps are what make the test numbers meaningful."""

from __future__ import annotations

import pytest

from helix.config import Config, SplitConfig
from helix.splits import search_window, walk_forward


@pytest.fixture
def cfg() -> SplitConfig:
    return SplitConfig(train_days=100, valid_days=20, test_days=20, step_days=20, embargo_days=5)


def test_windows_are_ordered_and_separated_by_the_embargo(cfg):
    folds = walk_forward(300, cfg)
    assert folds
    for fold in folds:
        assert fold.train.stop + cfg.embargo_days == fold.valid.start
        assert fold.valid.stop + cfg.embargo_days == fold.test.start
        assert fold.train.stop - fold.train.start == cfg.train_days
        assert fold.test.stop - fold.test.start == cfg.test_days


def test_folds_roll_forward_by_step_days(cfg):
    folds = walk_forward(300, cfg)
    starts = [f.train.start for f in folds]
    assert starts == list(range(0, starts[-1] + 1, cfg.step_days))


def test_no_fold_runs_past_the_end_of_the_panel(cfg):
    n = 300
    for fold in walk_forward(n, cfg):
        assert fold.test.stop <= n


def test_too_little_history_is_a_loud_error(cfg):
    with pytest.raises(ValueError, match="at least"):
        walk_forward(50, cfg)


def test_search_window_is_the_first_training_block(cfg):
    assert search_window(1000, cfg) == slice(0, cfg.train_days)
    assert search_window(40, cfg) == slice(0, 40)


def test_config_rejects_an_embargo_shorter_than_the_label_horizon(tmp_path):
    """touch_offset=2 means a D0 sample resolves on D+2, so the seam needs >= 3 days."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "label:\n  entry_offset: 1\n  touch_offset: 2\nsplit:\n  embargo_days: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="embargo_days"):
        Config.load(path)
