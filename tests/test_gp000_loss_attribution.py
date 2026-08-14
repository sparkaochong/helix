from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix.gp.library import FactorLibrary, FactorSpec
from scripts.gp000_loss_attribution import (
    audit_adjustment_chain,
    build_price_lookup,
    outcome_complete_dates,
    validate_formal_factor,
)


def test_outcome_complete_dates_never_cross_training_end() -> None:
    calendar = np.array(
        [
            "2024-08-21",
            "2024-08-22",
            "2024-08-23",
            "2024-08-26",
            "2024-08-27",
            "2024-08-28",
            "2024-08-29",
            "2024-08-30",
            "2024-09-02",
            "2024-09-03",
            "2024-09-04",
        ]
    )

    d2 = outcome_complete_dates(calendar, calendar, 2)
    d10 = outcome_complete_dates(calendar, calendar, 10)

    assert d2.tolist() == calendar[:-2].tolist()
    assert d10.tolist() == ["2024-08-21"]


def test_validate_formal_factor_rejects_other_gp000_library() -> None:
    library = FactorLibrary(
        factors=[FactorSpec("gp_000", "neg(x)", 1.0)],
        field_names=["x"],
        windows=[],
        kind="event",
    )

    with pytest.raises(ValueError, match="expression"):
        validate_formal_factor(library)


def _market_for_adjustment_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-05-10", "2024-05-13", "2024-05-14"],
            "ts_code": ["000001.SZ"] * 3,
            "open": [9.8, 10.0, 8.9],
            "high": [10.0, 10.2, 9.2],
            "close": [9.9, 10.0, 9.0],
            "adj_factor": [1.0, 1.0, 1.12],
        }
    )


def test_adjusted_return_removes_ex_right_gap() -> None:
    events = pd.DataFrame(
        {
            "trade_date": ["2024-05-10"],
            "stock_code": ["000001.SZ"],
            "label_px_d1_open": [10.0],
            "label_px_d2_high": [9.2],
            "label_px_d2_close": [9.0],
            "label_d2_return": [-0.1],
            "label_d2_hit_8pct": [0.0],
        }
    )
    prices = build_price_lookup(
        _market_for_adjustment_test(),
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )

    audit, aligned = audit_adjustment_chain(events, prices)

    assert aligned.loc[0, "raw_return"] == pytest.approx(-0.1)
    assert aligned.loc[0, "hfq_return"] == pytest.approx(0.008)
    assert audit["return_mismatch_count"] == 1
    assert audit["event_prices_match_raw"] is True


def test_ex_right_detection_uses_adj_factor_change_on_same_stock() -> None:
    prices = build_price_lookup(
        _market_for_adjustment_test(),
        ["2024-05-10", "2024-05-13", "2024-05-14"],
        ["000001.SZ"],
    )

    assert prices.ex_right[:, 0].tolist() == [False, False, True]
