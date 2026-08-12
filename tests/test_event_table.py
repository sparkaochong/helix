"""Slot packing, the no-time-series guard, and IC arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.data.event_table import (
    assert_no_label_columns,
    build_event_panel,
    is_label_column,
    numeric_feature_columns,
)
from helix.eval.ic import daily_ic, summarize_ic
from helix.gp.event_primitives import (
    FORBIDDEN,
    SEARCH_EXCLUDED,
    assert_excluded_absent,
    assert_no_time_series,
    build_event_pset,
)
from helix.gp.library import FactorLibrary, FactorSpec, compute_factors
from helix.gp.primitives import build_pset


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102", "20240102", "20240103", "20240103"],
            "stock_code": ["000001.SZ", "000002.SZ", "600000.SH", "000001.SZ", "600519.SH"],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [0.5, np.nan, 1.5, 2.5, 3.5],
            "label_hit": [1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )


def test_ragged_days_pack_into_slots(frame):
    panel = build_event_panel(frame, ["feat_a", "feat_b"], ["label_hit"])
    assert panel.shape == (2, 3)          # 2 dates, widest day has 3 names
    assert panel.n_rows == 5
    assert panel.occupied[0].tolist() == [True, True, True]
    assert panel.occupied[1].tolist() == [True, True, False]
    assert np.isnan(panel["feat_a"][1, 2])


def test_slots_are_filled_in_code_order_within_each_date(frame):
    panel = build_event_panel(frame, ["feat_a"], ["label_hit"])
    assert panel.codes[0].tolist() == ["000001.SZ", "000002.SZ", "600000.SH"]
    assert panel.codes[1, :2].tolist() == ["000001.SZ", "600519.SH"]


def test_a_slot_holds_different_stocks_on_different_dates(frame):
    """This is exactly why time-series operators are banned on an event panel."""
    panel = build_event_panel(frame, ["feat_a"], ["label_hit"])
    assert panel.codes[0, 1] != panel.codes[1, 1]


def test_duplicate_date_code_rows_are_dropped(frame):
    doubled = pd.concat([frame, frame.iloc[[0]].assign(feat_a=99.0)], ignore_index=True)
    panel = build_event_panel(doubled, ["feat_a"], ["label_hit"])
    assert panel.n_rows == 5
    assert panel["feat_a"][0, 0] == 99.0  # keep="last"


def test_missing_columns_raise(frame):
    with pytest.raises(KeyError, match="absent"):
        build_event_panel(frame, ["feat_a", "nope"], ["label_hit"])


def test_round_trip_back_to_long_preserves_values(frame):
    panel = build_event_panel(frame, ["feat_a", "feat_b"], ["label_hit"])
    out = panel.to_long({"feat_a": panel["feat_a"]})
    merged = frame.merge(out, on=["trade_date", "stock_code"], suffixes=("", "_rt"))
    assert len(merged) == 5
    np.testing.assert_allclose(merged["feat_a"], merged["feat_a_rt"])


def test_event_pset_excludes_every_windowed_operator():
    pset = build_event_pset(["feat_a", "feat_b"])
    present = {p.name for prims in pset.primitives.values() for p in prims}
    assert not (present & FORBIDDEN)
    assert "cs_rank" in present and "div" in present


def test_the_guard_catches_a_panel_pset():
    """A normal panel pset must be rejected outright by the event guard."""
    with pytest.raises(AssertionError, match="time-series operators are invalid"):
        assert_no_time_series(build_pset(["feat_a"], [5, 10]))


def test_event_pset_needs_at_least_one_feature():
    with pytest.raises(ValueError, match="at least one feature"):
        build_event_pset([])


# ------------------------------------------------- sign withheld from the search ----
def test_sign_is_withheld_from_the_search_but_the_rest_of_unary_survives():
    """`sign` took 27 of the last 30 factors. It flattens a column onto three levels,
    which is cheap rank-gini on a skewed column and two splits for a row-wise tree, so it
    consumed the budget without reaching anything such a tree cannot already compute."""
    present = {p.name for prims in build_event_pset(["feat_a"]).primitives.values()
               for p in prims}
    assert "sign" not in present
    assert {"neg", "abs", "log", "sqrt", "cs_rank", "cs_zscore", "cs_demean"} <= present


def test_the_exclusion_is_a_search_restriction_not_a_ban():
    """Weaker than FORBIDDEN on purpose: `sign` is a poor use of the budget, not invalid
    on a slot panel, so a pset asked for without the restriction must still offer it."""
    present = {p.name for prims in build_event_pset(["feat_a"], exclude=frozenset())
               .primitives.values() for p in prims}
    assert "sign" in present


def test_time_series_operators_stay_banned_even_with_the_restriction_lifted():
    """The asymmetry, pinned: lifting the search restriction must not lift the guard that
    exists because slot j is a different company on every date."""
    present = {p.name for prims in build_event_pset(["feat_a"], exclude=frozenset())
               .primitives.values() for p in prims}
    assert not (present & FORBIDDEN)


def test_a_saved_factor_using_sign_still_replays():
    """The regression this guards: `compute_factors` rebuilds the pset to parse stored
    expressions, so narrowing the search set would have made every previously mined
    `sign(...)` factor fail to load -- 27 of the 30 currently on disk."""
    library = FactorLibrary(
        factors=[FactorSpec(name="gp_000", expression="sub(sign(feat_a), feat_b)", sign=1.0)],
        field_names=["feat_a", "feat_b"], windows=[], kind="event",
    )
    fields = {"feat_a": np.array([[-2.0, 0.0, 3.0]]), "feat_b": np.array([[1.0, 1.0, 1.0]])}
    names, values = compute_factors(library, fields)
    assert names == ["gp_000"]
    np.testing.assert_allclose(values[..., 0], [[-2.0, -1.0, 0.0]])


def test_an_unknown_exclusion_name_is_refused_rather_than_silently_ignored():
    """A typo would otherwise remove nothing and report nothing, leaving the operator in
    the search while the caller believes it is gone."""
    with pytest.raises(ValueError, match="nothing to exclude named"):
        build_event_pset(["feat_a"], exclude=frozenset({"sgn"}))


def test_the_exclusion_guard_is_an_assertion_not_a_convention():
    assert_excluded_absent(build_event_pset(["feat_a"]))          # clean set passes
    with pytest.raises(AssertionError, match="withheld from the event-table search"):
        assert_excluded_absent(build_event_pset(["feat_a"], exclude=frozenset()))


def test_the_default_exclusion_is_exactly_sign():
    """If this set grows, the replay path above needs another look."""
    assert set(SEARCH_EXCLUDED) == {"sign"}


# ------------------------------------------------------- label leak guard ----
def test_label_columns_are_detected_by_prefix_not_by_a_list():
    """Regression: `label_d2_hit_5pct` once slipped into the terminal set.

    It was not in the enumerated label list, but it answers "did D+2 reach +5%",
    which predicts the +8% target almost perfectly. The mined factors scored
    IC 0.63 / ICIR 5.4 and were worthless.
    """
    assert is_label_column("label_d2_hit_5pct")
    assert is_label_column("label_d2_hit_3pct")
    assert is_label_column("LABEL_px_d1_open")
    assert is_label_column("target_return")
    assert is_label_column("y_hit")
    assert is_label_column("fwd_ret_5d")

    assert not is_label_column("stock_intra_amp_d0")
    assert not is_label_column("relabeled_score")  # only a prefix match counts
    assert not is_label_column("boll_width")


def test_assert_rejects_a_feature_set_containing_an_outcome():
    with pytest.raises(AssertionError, match="outcome columns reached the feature set"):
        assert_no_label_columns(["stock_intra_amp_d0", "label_d2_hit_5pct"])
    assert_no_label_columns(["stock_intra_amp_d0", "boll_width"])  # clean set passes


def test_event_pset_refuses_to_build_on_an_outcome_column():
    with pytest.raises(AssertionError, match="outcome columns reached the feature set"):
        build_event_pset(["feat_a", "label_d2_hit_5pct"])


def test_numeric_feature_columns_excludes_every_label(tmp_path):
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 3,
            "stock_code": ["a", "b", "c"],
            "stock_intra_amp_d0": [1.0, 2.0, 3.0],
            "label_d2_hit_8pct": [1.0, 0.0, 1.0],
            "label_d2_hit_5pct": [1.0, 1.0, 1.0],
            "label_d2_hit_3pct": [1.0, 1.0, 1.0],
            "label_px_d1_open": [10.0, 11.0, 12.0],
        }
    )
    path = tmp_path / "t.parquet"
    frame.to_parquet(path, index=False)

    # Only the 8% target is named explicitly; the other labels must still be excluded.
    columns = numeric_feature_columns(path, ["label_d2_hit_8pct"])
    assert columns == ["stock_intra_amp_d0"]


# ------------------------------------------------------------------------- IC ----
def test_ic_of_a_perfect_ranking_is_one():
    factor = np.array([[1.0, 2.0, 3.0, 4.0]])
    target = np.array([[10.0, 20.0, 30.0, 40.0]])
    mask = np.ones((1, 4), dtype=bool)
    assert daily_ic(factor, target, mask, min_samples=1)[0] == pytest.approx(1.0)
    assert daily_ic(-factor, target, mask, min_samples=1)[0] == pytest.approx(-1.0)


def test_ic_is_rank_based_so_monotone_rescaling_does_not_change_it():
    rng = np.random.default_rng(0)
    factor = rng.normal(size=(5, 40))
    target = rng.normal(size=(5, 40))
    mask = np.ones((5, 40), dtype=bool)
    base = daily_ic(factor, target, mask, min_samples=5)
    squashed = daily_ic(np.exp(factor), target, mask, min_samples=5)
    np.testing.assert_allclose(base, squashed, atol=1e-9)


def test_ic_ignores_unoccupied_slots():
    factor = np.array([[1.0, 2.0, 3.0, 999.0]])
    target = np.array([[10.0, 20.0, 30.0, -999.0]])
    mask = np.array([[True, True, True, False]])
    assert daily_ic(factor, target, mask, min_samples=1)[0] == pytest.approx(1.0)


def test_thin_dates_are_dropped():
    factor = np.array([[1.0, 2.0]])
    target = np.array([[1.0, 2.0]])
    mask = np.ones((1, 2), dtype=bool)
    assert np.isnan(daily_ic(factor, target, mask, min_samples=30)[0])


def test_icir_summary_arithmetic():
    ic = np.array([0.02, 0.04, np.nan, 0.06])
    stats = summarize_ic(ic)
    assert stats["ic_mean"] == pytest.approx(0.04)
    assert stats["ic_std"] == pytest.approx(0.02)
    assert stats["icir"] == pytest.approx(2.0)
    assert stats["icir_ann"] == pytest.approx(2.0 * np.sqrt(252))
    assert stats["positive_rate"] == pytest.approx(1.0)
    assert stats["n_days"] == 3
    assert stats["coverage"] == pytest.approx(0.75)


def test_all_nan_ic_series_is_reported_not_crashed():
    stats = summarize_ic(np.array([np.nan, np.nan]))
    assert stats["n_days"] == 0
    assert np.isnan(stats["ic_mean"])
