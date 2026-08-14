from __future__ import annotations

import numpy as np
import pytest

from helix.gp.library import FactorLibrary, FactorSpec
from scripts.gp000_loss_attribution import (
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
