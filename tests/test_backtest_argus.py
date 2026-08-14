"""Pin the accounting, because a costing bug reads as a strategy result.

The model needs xgboost and a GPU box; none of what is tested here does. These are the
parts that decide the *sign* of the answer while looking like arithmetic nobody needs to
check: how costs compose, what each exit rule pays out, whether an unfillable pick is
dropped or quietly replaced, and whether the regression target has had the market's daily
move taken out of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backtest_argus  # noqa: E402
import mine_argus  # noqa: E402
from backtest_argus import (  # noqa: E402
    COMMISSION_BPS,
    ENTRY_HFQ,
    EXIT_HFQ,
    HIGH_HFQ,
    STAMP_SELL_BPS,
    TRANSFER_BPS,
    complete_training_mask,
    cost_rates,
    cross_sectional_z,
    gross_returns,
    load_backtest_frame,
    net_return,
    run_book,
    target_hit,
)
from fillability import unfillable_mask  # noqa: E402

from helix.data.event_lineage import EventLineageError  # noqa: E402

VERSION = "raw-times-same-day-adj-v1:" + "a" * 64


def test_stamp_duty_is_charged_on_the_sell_side_only():
    buy, sell = cost_rates(0.0)
    assert buy == pytest.approx((COMMISSION_BPS + TRANSFER_BPS) / 1e4)
    assert sell - buy == pytest.approx(STAMP_SELL_BPS / 1e4)


def test_slippage_is_charged_on_both_sides():
    buy, sell = cost_rates(10.0)
    base_buy, base_sell = cost_rates(0.0)
    assert buy - base_buy == pytest.approx(10.0 / 1e4)
    assert sell - base_sell == pytest.approx(10.0 / 1e4)


def test_a_flat_trade_still_loses_the_round_trip():
    """Round-tripping at the entry price costs 2.6bp in, 7.6bp out -- 10.2bp all in."""
    assert float(net_return(np.array([0.0]), 0.0)[0]) == pytest.approx(-1.0197e-3, rel=1e-3)


def test_costs_scale_with_notional_so_this_is_not_a_subtraction():
    """The invariant a `gross - c` implementation would violate.

    Sell-side cost is charged on the exit value, so a winner pays more of it than a
    loser. Subtracting a constant would make the drag identical at every gross return and
    would flatter exactly the trades that matter most.
    """
    drag = lambda g: float(net_return(np.array([g]), 0.0)[0]) - g  # noqa: E731
    assert drag(0.20) < drag(0.0) < drag(-0.20)


def _picks(hit: tuple[int, ...], to_close: tuple[float, ...],
           peak: tuple[float, ...] | None = None) -> pd.DataFrame:
    """`hit` is shorthand for a D+2 high that clears 8%; `peak` sets it explicitly.

    The high is floored at the close because a bar whose high is below its close does not
    exist, and a fixture that is impossible would let a ratio bug pass.
    """
    open_d1 = 10.0
    peaks = peak if peak is not None else tuple(
        max(t, 0.08 if h else 0.0) for h, t in zip(hit, to_close, strict=True))
    return pd.DataFrame({
        "label_px_d1_open": [open_d1] * len(to_close),
        ENTRY_HFQ: [open_d1] * len(to_close),
        HIGH_HFQ: [open_d1 * (1 + p) for p in peaks],
        EXIT_HFQ: [open_d1 * (1 + r) for r in to_close],
    })


def test_close_exit_ignores_the_take_profit_entirely():
    frame = _picks(hit=(1, 0), to_close=(0.10, -0.05))
    assert gross_returns(frame, 1.08, "close") == pytest.approx([0.10, -0.05])


def test_target_exit_caps_the_winner_and_lets_the_loser_run():
    """The whole finding in one assertion: +10% becomes +8%, -5% stays -5%."""
    frame = _picks(hit=(1, 0), to_close=(0.10, -0.05))
    assert gross_returns(frame, 1.08, "target") == pytest.approx([0.08, -0.05])


def test_the_take_profit_level_decides_which_trades_it_applies_to():
    """The bug this guards: reading the hit off the 8% label while paying out 10% would
    hand every 8%-toucher a 10% exit, i.e. a strategy that cannot be traded.

    Peak +15% clears both levels; peak +9% clears 8% and not 10%, so raising the target
    must move it from a capped +8% winner to an uncapped -2% close.
    """
    frame = _picks(hit=(), to_close=(0.04, -0.02), peak=(0.15, 0.09))
    assert gross_returns(frame, 1.08, "target") == pytest.approx([0.08, 0.08])
    assert gross_returns(frame, 1.10, "target") == pytest.approx([0.10, -0.02])
    assert target_hit(frame, 1.10) == pytest.approx([True, False])


def test_raw_open_cannot_change_hits_or_returns_but_hfq_entry_can():
    frame = _picks(hit=(1, 0), to_close=(0.10, -0.05))
    expected_hit = target_hit(frame, 1.08)
    expected_return = gross_returns(frame, 1.08, "close")

    changed_raw = frame.assign(label_px_d1_open=[1.0, 1000.0])
    np.testing.assert_array_equal(target_hit(changed_raw, 1.08), expected_hit)
    np.testing.assert_allclose(gross_returns(changed_raw, 1.08, "close"), expected_return)

    changed_hfq = frame.copy()
    changed_hfq[ENTRY_HFQ] = [9.0, 9.0]
    assert target_hit(changed_hfq, 1.08).tolist() != expected_hit.tolist()
    assert gross_returns(changed_hfq, 1.08, "close") != pytest.approx(expected_return)


def test_fillability_still_uses_raw_open_not_hfq_entry():
    frame = _picks(hit=(1,), to_close=(0.0,)).assign(
        stock_code="000001.SZ", label_open_gap=0.1, label_px_d1_open=11.0
    )
    assert unfillable_mask(frame).tolist() == [True]

    frame[ENTRY_HFQ] = 1000.0
    assert unfillable_mask(frame).tolist() == [True]
    frame["label_px_d1_open"] = np.nan
    assert unfillable_mask(frame).tolist() == [False]


def test_training_rows_require_complete_d2_by_training_boundary():
    frame = pd.DataFrame({"trade_date": ["20240102", "20240103", "20240104"]})
    calendar = ("20240102", "20240103", "20240104", "20240105", "20240108")

    mask = complete_training_mask(frame, calendar, train_end="20240104", horizon=2)

    assert mask.tolist() == [True, False, False]


def test_backtest_cli_rejects_omitted_lineage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["backtest_argus.py", "--input", "events.parquet"])
    with pytest.raises(SystemExit, match="2"):
        backtest_argus.main()
    assert "--lineage" in capsys.readouterr().err


def test_mine_cli_rejects_omitted_lineage(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["mine_argus.py", "--input", "events.parquet", "--calendar", "calendar.parquet"],
    )
    with pytest.raises(SystemExit, match="2"):
        mine_argus.main()
    assert "--lineage" in capsys.readouterr().err


def test_backtest_rejects_event_file_as_its_own_calendar(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_argus.py",
            "--input",
            "same.parquet",
            "--calendar",
            "same.parquet",
            "--lineage",
            "missing.json",
        ],
    )
    with pytest.raises(EventLineageError, match="calendar.*independently"):
        backtest_argus.main()


def test_backtest_final_read_excludes_all_numeric_audit_groups(tmp_path, monkeypatch):
    group_count = 459
    data: dict[str, list] = {
        "trade_date": ["20240102"],
        "stock_code": ["000001.SZ"],
        "label_d2_hit_8pct_hfq": [1.0],
        "label_d2_peak_return_hfq": [0.1],
        "label_d2_return_hfq": [0.05],
        ENTRY_HFQ: [10.0],
        HIGH_HFQ: [11.0],
        EXIT_HFQ: [10.5],
        "label_px_d1_open": [10.0],
        "label_open_gap": [0.0],
    }
    fields: dict[str, dict] = {}
    audits: set[str] = set()
    feature_names: list[str] = []
    for group in range(group_count):
        feature = f"feat_{group:03d}"
        feature_names.append(feature)
        data[feature] = [float(group)]
        audit = {
            "source_date": f"source_{group}",
            "as_of_time": f"asof_{group}",
            "price_basis": f"basis_{group}",
            "adj_factor_version": f"version_{group}",
            "horizon": 0,
        }
        fields[feature] = audit
        audits.update(value for key, value in audit.items() if key != "horizon")
        data[audit["source_date"]] = ["20240102"]
        data[audit["as_of_time"]] = ["2024-01-02T15:00:00+08:00"]
        data[audit["price_basis"]] = ["hfq"]
        data[audit["adj_factor_version"]] = [VERSION]
    for outcome in (
        "label_d2_hit_8pct_hfq",
        "label_d2_peak_return_hfq",
        "label_d2_return_hfq",
        ENTRY_HFQ,
        HIGH_HFQ,
        EXIT_HFQ,
    ):
        fields[outcome] = fields["feat_000"].copy()
    event_path = tmp_path / "wide-events.parquet"
    pd.DataFrame(data).to_parquet(event_path, index=False)
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(
        json.dumps({"schema_version": 1, "fields": fields}), encoding="utf-8"
    )
    calendar_path = tmp_path / "calendar.parquet"
    pd.DataFrame({"cal_date": ["20240102"], "is_open": [1]}).to_parquet(
        calendar_path, index=False
    )

    validated: list[str] = []

    def fake_validate(path, manifest, governed, **kwargs):
        validated.extend(governed)

    monkeypatch.setattr(backtest_argus, "validate_event_parquet_fields", fake_validate)
    original_read = pd.read_parquet
    event_projections: list[set[str]] = []

    def recording_read(path_arg, *args, **kwargs):
        if Path(path_arg) == event_path:
            event_projections.append(set(kwargs["columns"]))
        return original_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)

    frame, features, *_ = load_backtest_frame(
        event_path,
        lineage_path,
        calendar_path,
        "label_d2_hit_8pct_hfq",
        "label_d2_peak_return_hfq",
        "label_d2_return_hfq",
    )

    assert len(validated) == group_count + 6
    assert features == feature_names
    assert event_projections
    assert all(not (projection & audits) for projection in event_projections)
    assert frame["unfillable"].tolist() == [False]


def _book(scores, unfillable, hit, to_close) -> pd.DataFrame:
    frame = _picks(hit, to_close)
    frame["trade_date"] = "20250101"
    frame["score"] = scores
    frame["unfillable"] = unfillable
    return frame


def _run(book: pd.DataFrame, hold_k: int, signal_k: int, exit_rule: str = "close",
         min_score: float | None = None) -> dict:
    return run_book(book, hold_k=hold_k, signal_k=signal_k, target_ratio=1.08,
                    exit_rule=exit_rule, slippage_bps=0.0, min_score=min_score)[0]


def test_without_a_deeper_shortlist_an_unfillable_pick_is_not_replaced():
    """signal_k == hold_k is the no-substitution convention: submit k, keep what fills."""
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, False, False],
                 hit=(1, 0, 1), to_close=(0.09, -0.04, 0.09))
    res = _run(book, hold_k=2, signal_k=2)
    assert res["avg_positions"] == 1.0                      # not 2: the third is not pulled in
    assert res["gross_per_trade"] == pytest.approx(-0.04)   # only the score-2.0 row
    assert res["hit_rate"] == 0.0


def test_a_deeper_shortlist_substitutes_down_the_ranking_in_order():
    """The concentrated-book case: hold 1, shortlist 3, top two names unfillable.

    Substitution must follow the ranking, not pick the best outcome among survivors.
    """
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, True, False],
                 hit=(1, 1, 0), to_close=(0.09, 0.20, -0.04))
    res = _run(book, hold_k=1, signal_k=3)
    assert res["avg_positions"] == 1.0
    assert res["gross_per_trade"] == pytest.approx(-0.04)   # the rank-3 name, not the +20%
    assert res["avg_fill_depth"] == 3.0
    assert res["short_day_rate"] == 0.0


def test_a_shortlist_that_runs_out_leaves_the_day_short_and_says_so():
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, True, False],
                 hit=(1, 1, 0), to_close=(0.09, 0.09, -0.04))
    res = _run(book, hold_k=2, signal_k=3)
    assert res["avg_positions"] == 1.0
    assert res["short_day_rate"] == 1.0


def test_substitution_never_reaches_past_the_shortlist():
    """A name ranked below signal_k is not a candidate, however good it turned out."""
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, False, False],
                 hit=(1, 0, 1), to_close=(0.09, -0.04, 0.30))
    res = _run(book, hold_k=1, signal_k=2)
    assert res["gross_per_trade"] == pytest.approx(-0.04)   # not the +30% at rank 3


def test_base_rate_is_measured_over_fillable_rows_only():
    """Otherwise lift compares a fill-aware numerator to an as-is denominator."""
    book = _book(scores=[3.0, 2.0, 1.0], unfillable=[True, False, False],
                 hit=(1, 1, 0), to_close=(0.09, 0.09, -0.04))
    res = _run(book, hold_k=2, signal_k=2)
    assert res["base_rate"] == pytest.approx(0.5)   # 1 of the 2 fillable, not 2 of 3


def test_the_conviction_gate_keeps_a_prefix_of_the_ranking():
    """Not a filter over the whole shortlist: the list is score-descending, so the gate
    truncates it and cannot reorder which names get bought."""
    book = _book(scores=[0.7, 0.5, 0.3], unfillable=[False, False, False],
                 hit=(1, 0, 1), to_close=(0.09, -0.04, 0.20))
    res = _run(book, hold_k=3, signal_k=3, min_score=0.6)
    assert res["avg_positions"] == 1.0                      # only the 0.7 clears it
    assert res["gross_per_trade"] == pytest.approx(0.09)    # not the +20% scoring 0.3


def test_a_day_where_nothing_clears_the_gate_is_flat_not_absent():
    """The distinction the parameter exists for. Dropping the day would shorten the track
    and score the strategy only on days it chose to trade -- so a threshold would always
    look like it cut drawdown, whether or not it sat out the bad days."""
    book = _book(scores=[0.4, 0.3], unfillable=[False, False],
                 hit=(0, 0), to_close=(-0.05, -0.06))
    res, daily = run_book(book, hold_k=2, signal_k=2, target_ratio=1.08,
                          exit_rule="close", slippage_bps=0.0, min_score=0.9)
    assert res["n_days"] == 1 and res["n_traded_days"] == 0   # the day is still on the books
    assert res["flat_day_rate"] == 1.0
    assert float(daily["portfolio_return"].iloc[0]) == 0.0
    assert float(daily["equity"].iloc[0]) == 1.0             # flat, and it costs nothing


def test_an_all_unfillable_day_is_flat_rather_than_skipped():
    """Every order sitting unfilled at the limit is a real day out of the market, and the
    equity curve has to show it. Without a gate this is rare; it is not impossible."""
    book = _book(scores=[3.0, 2.0], unfillable=[True, True],
                 hit=(1, 1), to_close=(0.09, 0.09))
    res = _run(book, hold_k=2, signal_k=2)
    assert res["n_days"] == 1 and res["n_traded_days"] == 0
    assert res["flat_day_rate"] == 1.0


def test_per_trade_stats_exclude_flat_days_but_the_curve_does_not():
    """A flat day is a zero in the curve and a non-event in the per-trade average. Folding
    it into the latter would dilute the loss and make the gate look like alpha."""
    book = _book(scores=[0.9, 0.2], unfillable=[False, False],
                 hit=(0, 0), to_close=(-0.10, -0.10))
    book.loc[1, "trade_date"] = "20250102"     # day 2 has only the 0.2 name: gated out
    res, daily = run_book(book, hold_k=1, signal_k=1, target_ratio=1.08,
                          exit_rule="close", slippage_bps=0.0, min_score=0.5)
    assert res["n_days"] == 2 and res["n_traded_days"] == 1
    # -0.1009, i.e. the one trade. Folding the flat day in would halve it to -0.0505.
    assert res["net_per_trade"] == pytest.approx(
        float(net_return(np.array([-0.10]), 0.0)[0]), abs=1e-6)
    assert list(daily["portfolio_return"]) == pytest.approx(
        [float(net_return(np.array([-0.10]), 0.0)[0]) / 2, 0.0])


def test_the_daily_series_is_the_curve_the_scalars_were_computed_from():
    """The summary alone cannot be re-cut into windows, and a drawdown is only meaningful
    against the length it was observed over. So the series has to survive, and it has to be
    the same series -- not a second, subtly different reconstruction."""
    book = _book(scores=[3.0, 2.0], unfillable=[False, False],
                 hit=(1, 0), to_close=(0.10, -0.05))
    book.loc[1, "trade_date"] = "20250102"        # one position on each of two days
    res, daily = run_book(book, hold_k=1, signal_k=1, target_ratio=1.08,
                          exit_rule="close", slippage_bps=0.0)

    assert list(daily["date"]) == ["20250101", "20250102"]
    assert res["n_days"] == len(daily) == 2
    # Capital is split across OVERLAP concurrent tranches, so the daily line is halved.
    assert daily["portfolio_return"].iloc[0] == pytest.approx(
        float(net_return(np.array([0.10]), 0.0)[0]) / 2)
    assert float(daily["equity"].iloc[-1]) == pytest.approx(res["final_equity"])
    assert res["sharpe"] == pytest.approx(
        daily["portfolio_return"].mean() / daily["portfolio_return"].std(ddof=1)
        * np.sqrt(252.0), rel=1e-4)


def _two_days() -> pd.DataFrame:
    # Same within-day ordering, shifted by a +10% market move on the second day.
    return pd.DataFrame({
        "trade_date": ["d1"] * 3 + ["d2"] * 3,
        "r": [-0.01, 0.00, 0.01, 0.09, 0.10, 0.11],
    })


def test_cross_sectional_z_removes_the_shared_daily_move():
    """A day that was broadly up must not outrank a flat day; only the within-day rank is
    information the book can act on."""
    z = cross_sectional_z(_two_days(), "r")
    assert z[:3] == pytest.approx(z[3:])
    assert z.mean() == pytest.approx(0.0, abs=1e-12)


def test_cross_sectional_z_is_zero_not_nan_when_a_day_has_no_dispersion():
    """Every name moving identically carries no cross-sectional signal. Dividing by a zero
    std would emit NaN/inf and poison the whole training target."""
    frame = pd.DataFrame({"trade_date": ["d1"] * 3, "r": [0.05, 0.05, 0.05]})
    z = cross_sectional_z(frame, "r")
    assert np.isfinite(z).all()
    assert z == pytest.approx([0.0, 0.0, 0.0])
