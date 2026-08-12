"""Pin the cross-sectional rank generator, whose failure modes are all silent.

Three ways this script can be wrong without anything complaining:

1. **Ranking across dates instead of within one.** A date split over two read batches that
   gets ranked twice, or an unsorted table, produces ranks computed against a fragment.
   The output still looks like a rank -- values in [0, 1], right shape -- and the ablation
   downstream reports a number rather than an error.
2. **A placebo that is not a control.** The whole point of the placebo arm is to separate
   "cross-sectional information is worthless" from "402 extra columns diluted the feature
   sampling". That only works if the placebo matches the real arm in every respect but
   alignment: same values, same per-date distributions, same column count.
3. **Column selection drifting from the real definition.** The selection logic is
   duplicated here rather than imported from helix so the script runs on a host without
   helix installed. Duplication that nothing checks is duplication that diverges, so the
   agreement is asserted against the actual table when it is present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_cs_columns import (  # noqa: E402
    PLACEBO_SUFFIX,
    REAL_SUFFIX,
    cs_rank_block,
    iter_date_blocks,
    main,
    numeric_feature_names,
    placebo_block,
    select_columns,
)

ARGUS = Path(__file__).resolve().parents[1] / "data" / "raw" / "argus_quant_working.parquet"


def _frame(dates: list[str], values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": dates, "stock_code": [f"{i:06d}" for i in range(len(dates))],
                         **values})


def _write(tmp_path: Path, frame: pd.DataFrame, row_group_size: int = 1 << 20) -> Path:
    path = tmp_path / "src.parquet"
    frame.to_parquet(path, index=False, row_group_size=row_group_size)
    return path


# --------------------------------------------------------------------- selection --

def test_a_column_that_already_ends_in_a_rank_suffix_is_left_alone():
    """`_pool_rank` and friends are ranked inside the same daily pool this script would
    rank over, so re-ranking them produces a copy of a column already in the table."""
    assert select_columns(["breadth_pctl", "hp_pct_div_rank", "flow"]) == ["flow"]


def test_a_column_with_a_sibling_rank_column_is_left_alone():
    """`from_high_2y_pct` and `from_high_2y_pct_rank` both exist in the real table; ranking
    the raw one again yields a near-duplicate of a column the model already has."""
    features = ["from_high_2y_pct", "from_high_2y_pct_rank", "atr_pct"]
    assert select_columns(features) == ["atr_pct"]


def test_everything_else_is_selected():
    assert select_columns(["a", "b", "c"]) == ["a", "b", "c"]


def test_outcome_columns_never_reach_the_feature_set(tmp_path):
    frame = _frame(["d1"], {"feat": [1.0], "label_d2_hit_8pct": [1.0], "future_ret": [0.1]})
    schema = pq.ParquetFile(_write(tmp_path, frame)).schema_arrow
    assert numeric_feature_names(schema) == ["feat"]


# ------------------------------------------------------------------------ ranks --

def test_ranks_are_percentiles_within_the_block():
    block = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    assert cs_rank_block(block, ["x"]).ravel() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_a_missing_value_stays_missing_rather_than_becoming_a_middling_rank():
    """Imputing here would be invisible: gradient-boosting libraries route NaN down their
    own branch, and a filled rank silently removes that signal."""
    got = cs_rank_block(pd.DataFrame({"x": [1.0, np.nan, 3.0]}), ["x"]).ravel()
    assert np.isnan(got[1])
    assert got[[0, 2]] == pytest.approx([0.5, 1.0])


def test_two_dates_on_wildly_different_scales_produce_the_same_ranks(tmp_path):
    """The mechanism the whole experiment rests on: the pool swings between 174 and 2656
    names, so an absolute threshold does not transfer between days but a rank does."""
    frame = _frame(["d1"] * 3 + ["d2"] * 3, {"x": [1.0, 2.0, 3.0, 1000.0, 2000.0, 3000.0]})
    blocks = dict(iter_date_blocks(_write(tmp_path, frame), batch_size=1 << 20))
    first = cs_rank_block(blocks["d1"], ["x"])
    second = cs_rank_block(blocks["d2"], ["x"])
    assert first == pytest.approx(second)


# ---------------------------------------------------------------------- placebo --

def test_the_placebo_holds_exactly_the_same_values_as_the_real_arm():
    real = np.arange(12, dtype=np.float32).reshape(4, 3)
    fake = placebo_block(real, np.random.default_rng(0))
    for column in range(real.shape[1]):
        assert sorted(fake[:, column]) == sorted(real[:, column])


def test_every_placebo_row_is_some_real_row_rather_than_a_column_wise_shuffle():
    """Permuting each column independently would also destroy the inter-column
    correlations, making the placebo a weaker control than the arm it controls for. One
    permutation for all columns leaves every row an internally coherent vector."""
    real = np.random.default_rng(1).normal(size=(50, 8)).astype(np.float32)
    fake = placebo_block(real, np.random.default_rng(2))
    rows = {tuple(row) for row in real}
    assert all(tuple(row) in rows for row in fake)


def test_the_placebo_is_actually_shuffled():
    real = np.arange(200, dtype=np.float32).reshape(100, 2)
    fake = placebo_block(real, np.random.default_rng(3))
    assert not np.array_equal(real, fake)


# ------------------------------------------------------------- date reassembly --

def test_a_date_split_across_read_batches_is_ranked_as_one_cross_section(tmp_path):
    """The table is a single row group of 617k rows, so reads are batched and a date lands
    on both sides of a boundary. Ranking the halves separately gives every fragment its
    own [0, 1] range -- output that looks entirely well-formed and is wrong."""
    frame = _frame(["d1"] * 10, {"x": list(range(10, 0, -1))})
    path = _write(tmp_path, frame)
    whole = dict(iter_date_blocks(path, batch_size=1 << 20))
    split = dict(iter_date_blocks(path, batch_size=3))
    assert list(split) == ["d1"]
    assert len(split["d1"]) == 10
    assert cs_rank_block(split["d1"], ["x"]) == pytest.approx(cs_rank_block(whole["d1"], ["x"]))


def test_dates_arrive_once_and_in_order(tmp_path):
    frame = _frame(["d1"] * 4 + ["d2"] * 4 + ["d3"] * 2, {"x": list(range(10))})
    got = [date for date, _ in iter_date_blocks(_write(tmp_path, frame), batch_size=3)]
    assert got == ["d1", "d2", "d3"]


def test_a_date_that_resumes_after_a_later_one_is_refused(tmp_path):
    """The hazard is not disorder as such -- rows of one date scattered inside a single
    batch are still gathered by the groupby. It is a date whose rows resume after a later
    date has already been closed out: that date gets emitted twice, each half ranked
    against itself, and both halves span a full [0, 1]."""
    frame = _frame(["d2", "d2", "d1", "d1"], {"x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(SystemExit, match="not sorted"):
        list(iter_date_blocks(_write(tmp_path, frame), batch_size=2))


def test_rows_scattered_within_one_batch_are_still_gathered_into_one_cross_section(tmp_path):
    frame = _frame(["d1", "d2", "d1", "d2"], {"x": [1.0, 2.0, 3.0, 4.0]})
    blocks = dict(iter_date_blocks(_write(tmp_path, frame), batch_size=1 << 20))
    assert sorted(blocks["d1"]["x"]) == [1.0, 3.0]
    assert sorted(blocks["d2"]["x"]) == [2.0, 4.0]


# ------------------------------------------------------------------ end to end --

def test_the_generated_table_carries_both_arms_and_the_column_lists(tmp_path, monkeypatch):
    frame = _frame(
        ["d1"] * 4 + ["d2"] * 4,
        {
            "momentum": [3.0, 1.0, 4.0, 2.0, 40.0, 10.0, 30.0, 20.0],
            "momentum_rank": [0.75, 0.25, 1.0, 0.5, 1.0, 0.25, 0.75, 0.5],  # already ranked
            "flow": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            "label_d2_hit_8pct": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        },
    )
    src = _write(tmp_path, frame)
    out_dir = tmp_path / "cs"
    monkeypatch.setattr(sys, "argv", ["make_cs_columns.py", "--input", str(src),
                                      "--out-dir", str(out_dir), "--row-group-rows", "4"])
    main()

    base = json.loads((out_dir / "base_features.json").read_text())
    real = json.loads((out_dir / "cs_real.json").read_text())
    placebo = json.loads((out_dir / "cs_placebo.json").read_text())
    assert base == ["momentum", "momentum_rank", "flow"]      # the label is excluded
    assert real == [f"flow{REAL_SUFFIX}"]                     # momentum has a sibling rank
    assert placebo == [f"flow{PLACEBO_SUFFIX}"]

    out = pd.read_parquet(out_dir / "src_cs.parquet")
    assert len(out) == len(frame)
    assert list(out["trade_date"]) == list(frame["trade_date"])
    assert set(frame.columns) <= set(out.columns)             # a drop-in replacement
    assert out[f"flow{REAL_SUFFIX}"].tolist() == pytest.approx(
        [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25]
    )
    # Same values within each date, reordered -- that is the whole contract of the placebo.
    for date in ("d1", "d2"):
        rows = out[out["trade_date"] == date]
        assert sorted(rows[f"flow{PLACEBO_SUFFIX}"]) == pytest.approx(
            sorted(rows[f"flow{REAL_SUFFIX}"])
        )


def test_the_run_is_reproducible_given_a_seed(tmp_path, monkeypatch):
    frame = _frame(["d1"] * 20, {"x": list(range(20)), "y": list(range(20, 0, -1))})
    src = _write(tmp_path, frame)
    outs = []
    for run in ("a", "b"):
        out_dir = tmp_path / run
        monkeypatch.setattr(sys, "argv", ["make_cs_columns.py", "--input", str(src),
                                          "--out-dir", str(out_dir), "--seed", "11"])
        main()
        outs.append(pd.read_parquet(out_dir / "src_cs.parquet"))
    pd.testing.assert_frame_equal(outs[0], outs[1])


# ------------------------------------------------- agreement with the real table --

@pytest.mark.skipif(not ARGUS.exists(), reason="argus_quant table not present")
def test_the_standalone_selection_matches_helix_on_the_real_schema():
    """This script duplicates the feature-column rule so it can run without helix
    installed. Nothing else would notice the two drifting apart."""
    from helix.data.event_table import numeric_feature_columns
    from helix.pipeline_events import DEFAULT_LABELS

    theirs = numeric_feature_columns(ARGUS, list(DEFAULT_LABELS))
    mine = numeric_feature_names(pq.ParquetFile(ARGUS).schema_arrow)
    assert mine == theirs
    assert len(mine) == 459
    assert len(select_columns(mine)) == 402
