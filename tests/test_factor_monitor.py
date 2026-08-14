"""Training-only auxiliary metrics around the production economic objective."""

from __future__ import annotations

import numpy as np

from helix.config import BacktestConfig
from helix.eval.factor_monitor import evaluate_training_monitors


def test_monitor_reports_production_top4_and_supplemental_top10_roles():
    n_dates, n_names = 150, 12
    score_row = np.arange(n_names, dtype=float)
    score = np.tile(score_row, (n_dates, 1))
    gross_row = np.linspace(-0.02, 0.03, n_names)
    gross = np.tile(gross_row, (n_dates, 1))
    hit = (gross > 0.01).astype(float)
    peak = gross + 0.04
    candidate = np.ones_like(score, dtype=bool)
    dates = np.array([f"{20200101 + index:08d}" for index in range(n_dates)])
    config = BacktestConfig(
        top_k=4,
        commission_bps=0,
        transfer_bps=0,
        stamp_sell_bps=0,
        stamp_sell_bps_before_cut=0,
        slippage_bps=0,
    )

    report = evaluate_training_monitors(
        score=score,
        hit_label=hit,
        peak_return=peak,
        gross_return=gross,
        candidate_mask=candidate,
        dates=dates,
        config=config,
        entry_offset=1,
        touch_offset=2,
        embargo_days=5,
        min_samples=5,
    )

    assert set(report) == {"fit", "selection", "training_full"}
    for block in report.values():
        production = block["production_objective"]
        supplemental = block["supplemental_top10"]
        assert production["role"] == "production_objective"
        assert production["top_k"] == 4
        assert supplemental["role"] == "supplemental_only"
        assert supplemental["top_k"] == 10
        assert production["net"]["mean"] > supplemental["net"]["mean"]
        assert block["hit_ic"]["ic_mean"] > 0
        assert block["hit_gini"]["mean"] > 0
        assert block["peak_return_ic"]["ic_mean"] > 0
        assert block["close_return_ic"]["ic_mean"] > 0
        assert block["top_k_hit"]["lift"] >= 1


def test_monitor_has_no_out_of_sample_section():
    shape = (150, 12)
    rng = np.random.default_rng(9)
    score = rng.normal(size=shape)
    gross = rng.normal(size=shape)

    report = evaluate_training_monitors(
        score=score,
        hit_label=(gross > 0).astype(float),
        peak_return=gross,
        gross_return=gross,
        candidate_mask=np.ones(shape, dtype=bool),
        dates=np.array([f"{20200101 + index:08d}" for index in range(shape[0])]),
        config=BacktestConfig(top_k=4),
        entry_offset=1,
        touch_offset=2,
        embargo_days=5,
        min_samples=5,
    )

    assert "out_of_sample" not in report
    assert "post_search" not in report
