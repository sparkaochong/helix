"""Pin the categorical encoding, because its failure mode is silent.

A string column is invisible to a numeric feature list -- `feature_columns()` keeps only
int/float types -- so a categorical that carries real signal can sit unread in the table
indefinitely and nothing reports it missing. That is exactly what happened to
`strategy_name`, whose six levels span an 8.2%-to-21.7% base rate.

The encoding itself has one way to go quietly wrong: doing it per split. That produces
columns one half has and the other does not, and the arm is then trained and scored on
different feature spaces -- which looks like a modelling result rather than a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ablate_factors import one_hot_columns  # noqa: E402


def _frame(levels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"strategy_name": levels, "label": [0] * len(levels)})


def test_every_level_becomes_its_own_indicator_column():
    got = one_hot_columns(_frame(["a", "b", "a", "c"]), "strategy_name")
    assert list(got.columns) == ["strategy_name==a", "strategy_name==b", "strategy_name==c"]
    assert got.to_numpy().sum(axis=1) == pytest.approx(np.ones(4))


def test_encoding_the_whole_frame_gives_both_halves_the_same_columns():
    """The bug this guards: encode train and test separately and a level missing from one
    side yields a column the other side lacks. Fitting then scoring across that mismatch
    is not a comparison of anything."""
    df = _frame(["a", "b", "a", "c"])          # 'c' appears only in the last row
    encoded = one_hot_columns(df, "strategy_name")
    train, test = encoded.iloc[:3], encoded.iloc[3:]
    assert list(train.columns) == list(test.columns)
    # The level absent from train is a legitimate all-zero column there, not a missing one.
    assert float(train["strategy_name==c"].sum()) == 0.0
    assert float(test["strategy_name==c"].sum()) == 1.0


def test_the_index_survives_so_the_columns_can_be_concatenated_back():
    """The frame has already been through a dropna, so its index has holes. Encoding must
    keep them or the concat silently misaligns every row against a different label."""
    df = _frame(["a", "b", "a", "c"]).drop(index=[1])
    got = one_hot_columns(df, "strategy_name")
    assert list(got.index) == [0, 2, 3]


def test_a_single_level_is_refused_rather_than_encoded_as_a_constant():
    """A constant column costs a fit and cannot change any split, so an arm built on one
    would report 'no incremental value' for a reason that has nothing to do with the data."""
    with pytest.raises(SystemExit):
        one_hot_columns(_frame(["a", "a", "a"]), "strategy_name")


def test_a_high_cardinality_column_is_refused_rather_than_exploded():
    with pytest.raises(SystemExit):
        one_hot_columns(_frame([f"s{i}" for i in range(40)]), "strategy_name")


def test_the_cap_is_inclusive_so_a_column_exactly_at_the_limit_encodes():
    got = one_hot_columns(_frame(["a", "b", "c"]), "strategy_name", max_levels=3)
    assert got.shape == (3, 3)


def test_values_are_numeric_because_the_model_is_fed_a_float_array():
    """`.to_numpy(dtype=np.float32)` on a bool column works, but only by accident of
    pandas; pinning the dtype here keeps the arm from depending on that."""
    got = one_hot_columns(_frame(["a", "b"]), "strategy_name")
    assert got.dtypes.unique().tolist() == [np.dtype("float32")]
