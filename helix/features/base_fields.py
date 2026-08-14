"""The raw material handed to the GP search.

These are deliberately *primitive*: returns, ranges, volume and valuation observables
that a human would recognise, each roughly stationary and cross-sectionally
comparable. The genetic program's job is to discover the combinations -- pre-baking
clever composites here just biases the search toward what we already believed.

Every field is computed from information available at the D0 close.
"""

from __future__ import annotations

import numpy as np

from ..data.panel import Panel
from ..logging_setup import get_logger
from . import operators as ops

log = get_logger(__name__)

EPS = 1e-9


def compute_base_fields(panel: Panel) -> dict[str, np.ndarray]:
    """Return ``{field_name: (T, N) float64}``, skipping fields whose inputs are absent."""
    panel.require_adjusted_prices(
        ("open_hfq", "high_hfq", "low_hfq", "close_hfq"), "compute_base_fields"
    )
    close_h = panel.f64("close_hfq")
    high_h = panel.f64("high_hfq")
    low_h = panel.f64("low_hfq")
    open_h = panel.f64("open_hfq")
    raw_close = panel.f64("close")
    amount = panel.f64("amount")
    up_limit = panel.f64("up_limit")

    ret1 = ops.div(close_h, ops.delay(close_h, 1)) - 1.0
    delayed_close_h = ops.delay(close_h, 1)
    hl = high_h - low_h

    fields: dict[str, np.ndarray] = {
        # --- returns over several horizons
        "ret1": ret1,
        "ret5": ops.div(close_h, ops.delay(close_h, 5)) - 1.0,
        "ret20": ops.div(close_h, ops.delay(close_h, 20)) - 1.0,
        # --- intraday shape of the D0 bar
        "gap": ops.div(open_h, delayed_close_h) - 1.0,
        "intraday": ops.div(close_h, open_h) - 1.0,
        "hl_range": ops.div(hl, delayed_close_h),
        "close_pos": ops.div(close_h - low_h, hl),
        "upper_shadow": ops.div(high_h - np.maximum(open_h, close_h), delayed_close_h),
        "lower_shadow": ops.div(np.minimum(open_h, close_h) - low_h, delayed_close_h),
        # --- trend / mean reversion
        "ma_dev5": ops.div(close_h, ops.ts_mean(close_h, 5)) - 1.0,
        "ma_dev20": ops.div(close_h, ops.ts_mean(close_h, 20)) - 1.0,
        "ma_dev60": ops.div(close_h, ops.ts_mean(close_h, 60)) - 1.0,
        "rsv20": ops.div(
            close_h - ops.ts_min(low_h, 20), ops.ts_max(high_h, 20) - ops.ts_min(low_h, 20)
        ),
        "vola20": ops.ts_std(ret1, 20),
        "max_ret20": ops.ts_max(ret1, 20),
        # --- liquidity / participation
        "log_amount": np.log(np.maximum(amount, EPS)),
        "amount_z20": ops.ts_zscore(np.log(np.maximum(amount, EPS)), 20),
        "amihud20": ops.ts_mean(ops.div(np.abs(ret1), np.maximum(amount, EPS)), 20) * 1e6,
        # --- distance to the daily price limit, which caps how far D+2 can travel
        "to_up_limit": ops.div(up_limit - raw_close, raw_close),
        "limitup_cnt20": ops.ts_sum(
            (raw_close >= up_limit - 0.001).astype(np.float64), 20
        ),
        # --- overnight/open behaviour, directly relevant to a D+1-open entry
        "open_gap_mean5": ops.ts_mean(ops.div(open_h, delayed_close_h) - 1.0, 5),
        "oc_corr20": ops.ts_corr(open_h, close_h, 20),
    }

    if "turnover_rate_f" in panel:
        turnover = panel.f64("turnover_rate_f")
        fields["turnover"] = turnover
        fields["turnover_z20"] = ops.ts_zscore(turnover, 20)
    if "volume_ratio" in panel:
        fields["volume_ratio"] = panel.f64("volume_ratio")
    if "circ_mv" in panel:
        fields["log_circ_mv"] = np.log(np.maximum(panel.f64("circ_mv"), EPS))
    if "pb" in panel:
        fields["bp"] = ops.div(np.ones_like(close_h), panel.f64("pb"))
    if "pe_ttm" in panel:
        fields["ep"] = ops.div(np.ones_like(close_h), panel.f64("pe_ttm"))

    for name, arr in fields.items():
        fields[name] = np.where(np.isfinite(arr), arr, np.nan)

    log.info("computed %d base fields: %s", len(fields), ", ".join(sorted(fields)))
    return fields


def field_names(fields: dict[str, np.ndarray]) -> list[str]:
    """Stable ordering -- GP terminals are positional, so this must never be a set."""
    return sorted(fields)


def stack_fields(fields: dict[str, np.ndarray], names: list[str]) -> list[np.ndarray]:
    return [fields[n] for n in names]
