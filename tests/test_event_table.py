"""Slot packing, the no-time-series guard, and IC arithmetic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from helix import pipeline_events
from helix.config import Config
from helix.data.event_lineage import (
    EventAuditColumns,
    EventLineageError,
    audit_column_names,
    load_event_calendar,
    load_event_lineage,
    validate_event_fields,
)
from helix.data.event_table import (
    EventPanel,
    assert_no_label_columns,
    build_event_panel,
    is_label_column,
    load_event_panel,
    numeric_feature_columns,
    open_event_source,
    stream_feature_grids,
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


VERSION = "raw-times-same-day-adj-v1:" + "a" * 64
CALENDAR = ["20240102", "20240103", "20240104", "20240105"]


def _governed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "stock_code": ["000001.SZ", "000002.SZ"],
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
            "label_d2_return_hfq": [0.1, -0.1],
            "feature_source_date": ["20240102", "2024-01-03"],
            "feature_as_of_time": [
                "2024-01-02T15:00:00+08:00",
                "2024-01-03T14:59:59+08:00",
            ],
            "feature_price_basis": ["hfq", "hfq"],
            "feature_adj_factor_version": [VERSION, VERSION],
            "outcome_source_date": ["20240104", "2024-01-05"],
            "outcome_as_of_time": [
                "2024-01-04T15:00:00+08:00",
                "2024-01-05T15:00:00+08:00",
            ],
            "outcome_price_basis": ["hfq", "hfq"],
            "outcome_adj_factor_version": [VERSION, VERSION],
        }
    )


def _manifest_dict() -> dict:
    feature_audit = {
        "source_date": "feature_source_date",
        "as_of_time": "feature_as_of_time",
        "price_basis": "feature_price_basis",
        "adj_factor_version": "feature_adj_factor_version",
        "horizon": 0,
    }
    return {
        "schema_version": 1,
        "fields": {
            "feat_a": feature_audit,
            "feat_b": feature_audit.copy(),
            "label_d2_return_hfq": {
                "source_date": "outcome_source_date",
                "as_of_time": "outcome_as_of_time",
                "price_basis": "outcome_price_basis",
                "adj_factor_version": "outcome_adj_factor_version",
                "horizon": 2,
            },
        },
    }


def _write_manifest(tmp_path, payload: dict | str | None = None) -> Path:
    path = tmp_path / "event-lineage.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(_manifest_dict() if payload is None else payload), encoding="utf-8")
    return path


def _write_calendar(tmp_path) -> Path:
    path = tmp_path / "calendar.parquet"
    pd.DataFrame({"cal_date": CALENDAR, "is_open": 1}).to_parquet(path, index=False)
    return path


def test_event_lineage_manifest_is_required():
    with pytest.raises(EventLineageError, match="event lineage manifest is required"):
        load_event_lineage(None)


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        {"schema_version": 2, "fields": {}},
        {"schema_version": 1},
        {"schema_version": 1, "fields": [], "extra": True},
        {"schema_version": 1, "fields": {"feat_a": {"horizon": 0}}},
        {
            "schema_version": 1,
            "fields": {
                "feat_a": {
                    "source_date": "source",
                    "as_of_time": "asof",
                    "price_basis": "basis",
                    "adj_factor_version": "version",
                    "horizon": -1,
                }
            },
        },
    ],
)
def test_invalid_event_lineage_manifest_fails_with_governed_error(tmp_path, payload):
    path = _write_manifest(tmp_path, payload)
    with pytest.raises(EventLineageError, match="event lineage manifest"):
        load_event_lineage(path)


def test_manifest_loads_frozen_audit_entries_and_exposes_shared_columns(tmp_path):
    manifest = load_event_lineage(_write_manifest(tmp_path))

    assert manifest["feat_a"] == EventAuditColumns(
        "feature_source_date",
        "feature_as_of_time",
        "feature_price_basis",
        "feature_adj_factor_version",
        0,
    )
    assert manifest["feat_a"] == manifest["feat_b"]
    assert audit_column_names(manifest) == {
        "feature_source_date",
        "feature_as_of_time",
        "feature_price_basis",
        "feature_adj_factor_version",
        "outcome_source_date",
        "outcome_as_of_time",
        "outcome_price_basis",
        "outcome_adj_factor_version",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="feature_source_date"), "feat_a.*source_date.*missing"),
        (lambda frame: frame.drop(columns="feature_as_of_time"), "feat_a.*as_of_time.*missing"),
        (lambda frame: frame.drop(columns="feature_price_basis"), "feat_a.*price_basis.*missing"),
        (
            lambda frame: frame.drop(columns="feature_adj_factor_version"),
            "feat_a.*adj_factor_version.*missing",
        ),
        (lambda frame: frame.assign(feature_price_basis="raw"), "feat_a.*price_basis.*20240102"),
        (
            lambda frame: frame.assign(feature_price_basis=["hfq", "raw"]),
            "feat_a.*price_basis.*20240103",
        ),
        (
            lambda frame: frame.assign(feature_price_basis=["hfq", None]),
            "feat_a.*price_basis.*20240103",
        ),
        (
            lambda frame: frame.assign(feature_adj_factor_version="bad"),
            "feat_a.*adj_factor_version.*20240102",
        ),
        (
            lambda frame: frame.assign(feature_adj_factor_version=[VERSION, VERSION[:-1] + "b"]),
            "feat_a.*adjustment version.*20240103",
        ),
        (
            lambda frame: frame.assign(feature_source_date=["20240102", "2024/01/03"]),
            "feat_a.*source_date.*20240103",
        ),
        (
            lambda frame: frame.assign(
                feature_as_of_time=["2024-01-02T15:00:00+08:00", "not-a-time"]
            ),
            "feat_a.*as_of_time.*20240103",
        ),
        (
            lambda frame: frame.assign(
                feature_as_of_time=[
                    "2024-01-02T15:00:00+08:00",
                    "2024-01-03T15:00:00+00:00",
                ]
            ),
            r"feat_a.*\+08:00.*20240103",
        ),
        (
            lambda frame: frame.assign(
                feature_as_of_time=[
                    "2024-01-02T15:00:01+08:00",
                    "2024-01-03T15:00:00+08:00",
                ]
            ),
            "feat_a.*after.*close.*20240102",
        ),
        (
            lambda frame: frame.assign(
                feature_source_date=["20240101", "20240103"],
                feature_as_of_time=[
                    "2024-01-01T15:00:00+08:00",
                    "2024-01-03T14:59:59+08:00",
                ],
            ),
            "feat_a.*horizon=0.*20240102",
        ),
        (
            lambda frame: frame.assign(
                outcome_source_date=["20240103", "20240105"],
                outcome_as_of_time=[
                    "2024-01-03T15:00:00+08:00",
                    "2024-01-05T15:00:00+08:00",
                ],
            ),
            "label_d2_return_hfq.*horizon=2.*20240102",
        ),
    ],
)
def test_event_field_validation_fails_closed(tmp_path, mutation, message):
    manifest = load_event_lineage(_write_manifest(tmp_path))
    with pytest.raises(EventLineageError, match=message):
        validate_event_fields(
            mutation(_governed_frame()),
            manifest,
            ["feat_a", "label_d2_return_hfq"],
            calendar=CALENDAR,
        )


def test_event_field_validation_requires_manifest_entry(tmp_path):
    manifest = load_event_lineage(_write_manifest(tmp_path))
    with pytest.raises(EventLineageError, match="unknown.*manifest entry"):
        validate_event_fields(_governed_frame(), manifest, ["unknown"], calendar=CALENDAR)


def test_event_field_validation_rejects_outcomes_beyond_training_cutoff(tmp_path):
    manifest = load_event_lineage(_write_manifest(tmp_path))
    with pytest.raises(EventLineageError, match="label_d2_return_hfq.*train_end.*20240102"):
        validate_event_fields(
            _governed_frame(),
            manifest,
            ["label_d2_return_hfq"],
            calendar=CALENDAR,
            train_end="20240103",
        )


def test_positive_horizon_requires_independent_trading_calendar(tmp_path):
    manifest = load_event_lineage(_write_manifest(tmp_path))
    with pytest.raises(EventLineageError, match="authoritative event trading calendar is required"):
        validate_event_fields(_governed_frame(), manifest, ["label_d2_return_hfq"])


def test_authoritative_calendar_rejects_omitted_intervening_session():
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"],
            "label_d1": [1.0],
            "source": ["20240104"],
            "asof": ["2024-01-04T15:00:00+08:00"],
            "basis": ["hfq"],
            "version": [VERSION],
        }
    )
    manifest = {"label_d1": EventAuditColumns("source", "asof", "basis", "version", 1)}

    with pytest.raises(EventLineageError, match="label_d1.*horizon=1.*expected 2024-01-03"):
        validate_event_fields(
            frame, manifest, ["label_d1"], calendar=["20240102", "20240103", "20240104"]
        )


def test_authoritative_calendar_accepts_exact_d1_and_d2():
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"],
            "label_d1": [1.0],
            "label_d2": [2.0],
            "d1_source": ["20240103"],
            "d1_asof": ["2024-01-03T15:00:00+08:00"],
            "d2_source": ["20240104"],
            "d2_asof": ["2024-01-04T15:00:00+08:00"],
            "basis": ["hfq"],
            "version": [VERSION],
        }
    )
    manifest = {
        "label_d1": EventAuditColumns("d1_source", "d1_asof", "basis", "version", 1),
        "label_d2": EventAuditColumns("d2_source", "d2_asof", "basis", "version", 2),
    }

    validate_event_fields(
        frame,
        manifest,
        ["label_d1", "label_d2"],
        calendar=["20240102", "20240103", "20240104"],
    )


def test_event_calendar_loader_reads_only_open_sessions(tmp_path):
    path = tmp_path / "calendar.parquet"
    pd.DataFrame(
        {
            "cal_date": ["20240102", "20240103", "20240104"],
            "is_open": [1, 0, 1],
        }
    ).to_parquet(path, index=False)

    assert load_event_calendar(path) == ("20240102", "20240104")


@pytest.mark.parametrize("bad_is_open", [None, "oops", 0.5, -1, 2])
def test_event_calendar_rejects_invalid_is_open_instead_of_dropping_session(
    tmp_path, bad_is_open
):
    path = tmp_path / "calendar.parquet"
    pd.DataFrame(
        {
            "cal_date": ["20240102", "20240103", "20240104"],
            "is_open": ["1", None if bad_is_open is None else str(bad_is_open), "1"],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(EventLineageError, match="is_open.*20240103"):
        load_event_calendar(path)


def test_row_validation_error_names_position_trade_date_and_stock(tmp_path):
    manifest = load_event_lineage(_write_manifest(tmp_path))
    frame = _governed_frame()
    frame.loc[1, "feature_price_basis"] = "raw"

    with pytest.raises(
        EventLineageError,
        match=r"feat_a.*price_basis.*row 1.*trade_date=.?20240103.*stock_code=.?000002\.SZ",
    ):
        validate_event_fields(frame, manifest, ["feat_a"], calendar=CALENDAR)


def test_governed_load_packs_fields_and_labels_without_audit_columns(tmp_path):
    frame = _governed_frame()
    path = tmp_path / "events.parquet"
    frame.to_parquet(path, index=False)
    lineage_path = _write_manifest(tmp_path)

    panel = load_event_panel(
        path,
        label_columns=["label_d2_return_hfq"],
        feature_columns=["feat_a", "feat_b"],
        lineage_path=lineage_path,
        calendar_path=_write_calendar(tmp_path),
    )

    assert panel.field_names() == ["feat_a", "feat_b"]
    assert set(panel.labels) == {"label_d2_return_hfq"}
    assert not (set(panel.fields) & audit_column_names(load_event_lineage(lineage_path)))


def test_formal_load_rejects_legacy_table_without_manifest(tmp_path, frame):
    path = tmp_path / "legacy.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(EventLineageError, match="event lineage manifest is required"):
        load_event_panel(path, label_columns=["label_hit"])


def test_formal_load_rejects_positive_horizon_without_calendar(tmp_path):
    path = tmp_path / "events.parquet"
    _governed_frame().to_parquet(path, index=False)

    with pytest.raises(EventLineageError, match="authoritative event trading calendar is required"):
        load_event_panel(
            path,
            label_columns=["label_d2_return_hfq"],
            feature_columns=["feat_a"],
            lineage_path=_write_manifest(tmp_path),
        )


def test_auto_feature_discovery_excludes_shared_audit_columns(tmp_path):
    frame = _governed_frame()
    path = tmp_path / "events.parquet"
    frame.to_parquet(path, index=False)
    lineage_path = _write_manifest(tmp_path)

    panel = load_event_panel(
        path,
        label_columns=["label_d2_return_hfq"],
        lineage_path=lineage_path,
        calendar_path=_write_calendar(tmp_path),
    )

    assert panel.field_names() == ["feat_a", "feat_b"]


def test_auto_feature_discovery_excludes_numeric_manifest_audit_sentinel(tmp_path):
    frame = _governed_frame().assign(numeric_audit_sentinel=[101, 102])
    path = tmp_path / "events.parquet"
    frame.to_parquet(path, index=False)
    payload = _manifest_dict()
    payload["fields"]["unused_governed_field"] = {
        "source_date": "numeric_audit_sentinel",
        "as_of_time": "feature_as_of_time",
        "price_basis": "feature_price_basis",
        "adj_factor_version": "feature_adj_factor_version",
        "horizon": 0,
    }

    panel = load_event_panel(
        path,
        label_columns=["label_d2_return_hfq"],
        lineage_path=_write_manifest(tmp_path, payload),
        calendar_path=_write_calendar(tmp_path),
    )

    assert "numeric_audit_sentinel" not in panel.fields


def test_streaming_audit_validation_projects_one_bounded_group_at_a_time(
    tmp_path, monkeypatch
):
    group_count = 459
    data: dict[str, list] = {
        "trade_date": ["20240102", "20240103"],
        "stock_code": ["000001.SZ", "000002.SZ"],
        "label_h0": [0.0, 1.0],
    }
    fields: dict[str, dict] = {}
    for group in range(group_count):
        name = f"feat_{group:03d}"
        data[name] = [float(group), float(group + 1)]
        data[f"source_{group}"] = ["20240102", "20240103"]
        data[f"asof_{group}"] = [
            "2024-01-02T15:00:00+08:00",
            "2024-01-03T15:00:00+08:00",
        ]
        data[f"basis_{group}"] = ["hfq", "hfq"]
        data[f"version_{group}"] = [VERSION, VERSION]
        fields[name] = {
            "source_date": f"source_{group}",
            "as_of_time": f"asof_{group}",
            "price_basis": f"basis_{group}",
            "adj_factor_version": f"version_{group}",
            "horizon": 0,
        }
    fields["label_h0"] = fields["feat_000"].copy()
    path = tmp_path / "wide-events.parquet"
    pd.DataFrame(data).to_parquet(path, index=False)
    manifest_path = _write_manifest(
        tmp_path, {"schema_version": 1, "fields": fields}
    )

    import pyarrow.parquet as pq

    projected: list[tuple[str, ...]] = []
    original = pq.ParquetFile.iter_batches

    def recording_iter_batches(parquet_file, *args, **kwargs):
        projected.append(tuple(kwargs["columns"]))
        yield from original(parquet_file, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", recording_iter_batches)

    open_event_source(
        path,
        ["label_h0"],
        lineage_path=manifest_path,
        feature_columns=[f"feat_{group:03d}" for group in range(group_count)],
    )

    assert len(projected) == group_count
    assert max(map(len, projected)) <= 6


def test_explicit_feature_columns_reject_physical_audit_columns(tmp_path):
    frame = _governed_frame()
    path = tmp_path / "events.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(EventLineageError, match="feature_columns.*audit.*feature_source_date"):
        load_event_panel(
            path,
            label_columns=["label_d2_return_hfq"],
            feature_columns=["feat_a", "feature_source_date"],
            lineage_path=_write_manifest(tmp_path),
            calendar_path=tmp_path / "calendar.parquet",
        )


def test_streaming_source_rejects_explicit_audit_feature_before_read(tmp_path):
    path = tmp_path / "events.parquet"
    _governed_frame().to_parquet(path, index=False)

    with pytest.raises(EventLineageError, match="feature_columns.*audit.*feature_source_date"):
        open_event_source(
            path,
            ["label_d2_return_hfq"],
            lineage_path=_write_manifest(tmp_path),
            calendar_path=_write_calendar(tmp_path),
            feature_columns=["feat_a", "feature_source_date"],
        )


def test_streaming_grids_cannot_emit_a_physical_audit_column(tmp_path):
    path = tmp_path / "events.parquet"
    _governed_frame().to_parquet(path, index=False)
    index, _, keys = open_event_source(
        path,
        ["label_d2_return_hfq"],
        lineage_path=_write_manifest(tmp_path),
        calendar_path=_write_calendar(tmp_path),
        feature_columns=["feat_a"],
    )

    with pytest.raises(EventLineageError, match="stream.*audit.*feature_source_date"):
        list(stream_feature_grids(path, keys, index, ["feature_source_date"]))


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


def test_event_mining_uses_only_complete_training_outcomes(monkeypatch):
    shape = (12, 5)
    dates = np.array([f"202401{day:02d}" for day in range(2, 14)])
    occupied = np.ones(shape, dtype=bool)
    occupied[:, -1] = False
    close_return = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10_000
    panel = EventPanel(
        dates=dates,
        codes=np.full(shape, "000001.SZ"),
        occupied=occupied,
        fields={"feat_a": np.ones(shape)},
        labels={
            pipeline_events.PRIMARY_TARGET: close_return + 0.02,
            pipeline_events.BINARY_TARGET: (close_return > 0.002).astype(float),
            pipeline_events.RETURN_TARGET: close_return,
        },
    )
    captured: dict[str, object] = {}

    def fake_select_features(**kwargs):
        captured["screen"] = kwargs
        return ["feat_a"], []

    def fake_run_search(**kwargs):
        captured["search"] = kwargs
        return SimpleNamespace(
            library=FactorLibrary([], field_names=["feat_a"], windows=[], kind="event")
        )

    monkeypatch.setattr(pipeline_events, "select_features", fake_select_features)
    monkeypatch.setattr(pipeline_events, "run_search", fake_run_search)

    run = pipeline_events.mine_events(panel, Config(), search_fraction=0.75)

    assert run.search_rows == slice(0, 7)
    screen = captured["screen"]
    assert screen["fields"]["feat_a"].shape == (7, 5)
    np.testing.assert_array_equal(screen["mask"], occupied[:7])
    search = captured["search"]
    np.testing.assert_array_equal(search["candidate_mask"], occupied[:7])
    np.testing.assert_array_equal(search["gross_returns"], close_return[:7])
    np.testing.assert_array_equal(search["dates"], dates[:7])
    assert search["backtest_cfg"].top_k == 4
