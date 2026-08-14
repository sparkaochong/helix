"""End-to-end integration on a synthetic market with a planted, discoverable edge.

This is the test that catches wiring bugs: panel -> base fields -> labels -> GP ->
factor library -> normalised sequences -> walk-forward training -> backtest, using the
same functions the CLI calls. No Tushare access required.
"""

from __future__ import annotations

import numpy as np
import pytest

from helix import pipeline
from helix.config import (
    BacktestConfig,
    Config,
    DataConfig,
    DLConfig,
    GPConfig,
    LabelConfig,
    SplitConfig,
)
from helix.data.panel import Panel
from helix.dl.dataset import normalize_factors
from helix.dl.train import train_walk_forward
from helix.eval.backtest import run_backtest
from helix.eval.metrics import daily_gini, summarize_daily
from helix.features.base_fields import compute_base_fields, field_names
from helix.gp.engine import run_search
from helix.gp.library import compute_factors
from helix.labels.touch_label import build_touch_label
from helix.splits import search_window, walk_forward

N_DATES, N_CODES = 280, 60


@pytest.fixture(scope="module")
def market() -> Panel:
    """Random-walk prices where a large D0 move predicts a large D+2 high."""
    rng = np.random.default_rng(42)
    shape = (N_DATES, N_CODES)

    innovations = rng.normal(0.0, 0.025, size=shape)
    daily_ret = innovations.copy()
    daily_ret[2:] += 0.6 * innovations[:-2]
    close = 10.0 * np.exp(np.cumsum(daily_ret, axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.01, size=shape))

    # Planted edge: yesterday's return lifts the intraday range two days later.
    momentum = np.vstack([np.zeros((2, N_CODES)), daily_ret[:-2]])
    upside = np.abs(rng.normal(0.0, 0.03, size=shape)) + 0.9 * np.maximum(momentum, 0.0)
    high = np.maximum(open_, close) * (1.0 + upside)
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.02, size=shape)))

    volume = rng.lognormal(11.0, 0.6, size=shape)
    fields = {
        "open": open_, "high": high, "low": low, "close": close, "pre_close": prev_close,
        "open_hfq": open_, "high_hfq": high, "low_hfq": low, "close_hfq": close,
        "pre_close_hfq": prev_close,
        "vol": volume, "amount": volume * close / 10.0,
        "up_limit": np.round(prev_close * 1.1, 2),
        "down_limit": np.round(prev_close * 0.9, 2),
        "limit_price_observed": np.ones(shape),
        "is_trading": np.ones(shape),
        "turnover_rate_f": rng.uniform(0.5, 8.0, size=shape),
        "volume_ratio": rng.uniform(0.5, 3.0, size=shape),
        "circ_mv": rng.lognormal(13.0, 0.8, size=shape),
        "pb": rng.uniform(0.8, 8.0, size=shape),
        "pe_ttm": rng.uniform(8.0, 60.0, size=shape),
    }
    return Panel(
        dates=np.array([f"{20200101 + i:08d}" for i in range(N_DATES)]),
        codes=np.array([f"{600000 + j:06d}.SH" for j in range(N_CODES)]),
        fields={k: np.asarray(v, dtype=np.float64) for k, v in fields.items()},
    )


@pytest.fixture(scope="module")
def split() -> SplitConfig:
    return SplitConfig(train_days=150, valid_days=20, test_days=20, step_days=20, embargo_days=5)


@pytest.fixture(scope="module")
def prepared(market):
    fields = compute_base_fields(market)
    universe = np.ones(market.shape, dtype=bool)
    labels = build_touch_label(market, universe, LabelConfig(target_ratio=1.08))
    return fields, field_names(fields), labels


def test_label_base_rate_is_plausible(prepared):
    _, _, labels = prepared
    assert 0.0 < labels.base_rate < 0.6
    assert labels.valid.sum() > 10_000
    # The final touch_offset rows can never be resolved.
    assert not labels.valid[-2:].any()


@pytest.fixture(scope="module")
def library(prepared, split, market):
    fields, names, labels = prepared
    rows = search_window(N_DATES, split)
    cfg = GPConfig(
        population=80, generations=5, hall_of_fame=25, n_keep=6,
        windows=[3, 5, 10], max_nodes=15, max_depth=4,
        min_daily_samples=20, search_max_stocks=N_CODES, seed=5,
    )
    gross = np.where(
        labels.valid[rows],
        labels.exit_price[rows] / labels.entry_price[rows] - 1.0,
        np.nan,
    )
    result = run_search(
        fields={name: values[rows] for name, values in fields.items()},
        field_names=names,
        gross_returns=gross,
        candidate_mask=np.ones_like(gross, dtype=bool),
        dates=market.dates[rows],
        cfg=cfg,
        backtest_cfg=BacktestConfig(
            top_k=4,
            commission_bps=0,
            transfer_bps=0,
            stamp_sell_bps=0,
            stamp_sell_bps_before_cut=0,
            slippage_bps=0,
        ),
        entry_offset=1,
        touch_offset=2,
        embargo_days=split.embargo_days,
    )
    return result.library


def test_gp_discovers_factors_that_hold_up_out_of_sample(library, prepared, split):
    assert library.factors, "GP found nothing that survived the selection block"

    fields, _, labels = prepared
    names, values = compute_factors(library, fields)
    after = slice(search_window(N_DATES, split).stop, N_DATES)

    ginis = [
        summarize_daily(
            daily_gini(values[:, :, k].astype(np.float64)[after], labels.y[after],
                       labels.valid[after], min_samples=20)
        )["mean"]
        for k in range(len(names))
    ]
    assert max(ginis) > 0.05, f"no factor generalised past the search window: {ginis}"


def test_walk_forward_training_and_backtest_complete(library, prepared, market, split):
    fields, _, labels = prepared
    _, values = compute_factors(library, fields)
    normalized = normalize_factors(values, np.ones(market.shape, dtype=bool), n_sigma=4.0)
    traded = (market["is_trading"] > 0).astype(np.float32)

    folds = walk_forward(N_DATES, split)
    assert folds
    predictions, results = train_walk_forward(
        folds=folds,
        values=normalized,
        traded=traded,
        y=labels.y,
        label_mask=labels.valid,
        prediction_mask=np.ones(market.shape, dtype=bool),
        cfg=DLConfig(seq_len=5, hidden_size=16, num_layers=1, epochs=2,
                     batch_size=512, early_stopping_patience=2, seed=1),
    )

    assert len(results) == len(folds)
    # Predictions exist only inside test windows, and are probabilities.
    tested = np.isfinite(predictions)
    assert tested.any()
    assert not tested[folds[0].train].any()
    assert predictions[tested].min() >= 0.0 and predictions[tested].max() <= 1.0

    summary = run_backtest(
        predictions, labels, np.ones(market.shape, dtype=bool), market.dates,
        LabelConfig(target_ratio=1.08),
        BacktestConfig(top_k=5),
    ).summary
    assert summary["n_days"] > 0
    assert 0.0 <= summary["hit_rate"] <= 1.0
    assert np.isfinite(summary["mean_trade_return_net"])


def test_live_scoring_ranks_the_latest_date_which_has_no_label(
    tmp_path, library, prepared, market, split
):
    """The newest bar is unlabelled by construction, and must still be scorable."""
    fields, names, labels = prepared
    cfg = Config(
        data=DataConfig(root=tmp_path),
        split=split,
        dl=DLConfig(seq_len=5, hidden_size=16, num_layers=1, epochs=2,
                    batch_size=512, early_stopping_patience=2, seed=1),
    )
    prep = pipeline.Prepared(
        panel=market,
        universe=np.ones(market.shape, dtype=bool),
        fields=fields,
        names=names,
        labels=labels,
    )
    predictions, _ = pipeline.train(cfg, prep, library)

    latest = market.dates[-1]
    assert not labels.valid[-1].any(), "the final bar cannot have a resolved label"
    assert np.isfinite(predictions[-1]).any(), (
        "OOS scoring must not require the latest row's future label to be observable"
    )

    frame = pipeline.score(cfg, prep, library)
    assert len(frame) > 0
    assert (frame["date"] == latest).all()
    assert frame["probability"].between(0.0, 1.0).all()
    assert frame["rank"].tolist() == sorted(frame["rank"].tolist())
    assert frame["probability"].is_monotonic_decreasing
    assert (pipeline.artifacts_dir(cfg) / f"scores_{latest}.csv").exists()


def test_scoring_rejects_a_date_outside_the_panel(tmp_path, library, prepared, market, split):
    fields, names, labels = prepared
    cfg = Config(
        data=DataConfig(root=tmp_path),
        split=split,
        dl=DLConfig(seq_len=5, hidden_size=16, num_layers=1, epochs=1,
                    batch_size=512, early_stopping_patience=1, seed=1),
    )
    prep = pipeline.Prepared(
        panel=market,
        universe=np.ones(market.shape, dtype=bool),
        fields=fields,
        names=names,
        labels=labels,
    )
    pipeline.train(cfg, prep, library)

    with pytest.raises(ValueError, match="not a trade date"):
        pipeline.score(cfg, prep, library, date="19000101")
