"""Trade-level backtest of the top-``k`` daily selection.

Exit rule mirrors the label exactly: enter at the D+1 open, exit at the target if the
D+2 high reaches it, otherwise exit at the D+2 close. That keeps the backtest honest
about the losing tail -- a "hit rate" alone hides the fact that the misses are the
trades that gapped down.

The equity curve divides capital across the overlapping tranches (a new book opens
every day while the previous one is still held), so it is not the sum of daily
per-trade returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BacktestConfig, LabelConfig
from ..labels.touch_label import LabelSet
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    summary: dict[str, float]


def _trade_returns(labels: LabelSet, target_ratio: float) -> np.ndarray:
    """Gross per-trade return: the target if touched, otherwise the D+2 close."""
    touched = labels.y == 1.0
    hit_return = np.full(labels.y.shape, target_ratio - 1.0)
    miss_return = labels.exit_price / labels.entry_price - 1.0
    return np.where(touched, hit_return, miss_return)


def run_backtest(
    predictions: np.ndarray,
    labels: LabelSet,
    dates: np.ndarray,
    label_cfg: LabelConfig,
    cfg: BacktestConfig,
) -> BacktestResult:
    gross = _trade_returns(labels, label_cfg.target_ratio)
    cost = 2.0 * cfg.cost_bps / 10_000.0  # round trip
    usable = labels.valid & np.isfinite(predictions) & np.isfinite(labels.y)

    n_dates = predictions.shape[0]
    rows: list[dict] = []
    scores = np.where(usable, predictions, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")

    for t in range(n_dates):
        n_available = int(usable[t].sum())
        if n_available < cfg.top_k:
            continue
        picked = order[t, : cfg.top_k]
        trade_ret = gross[t, picked]
        if not np.isfinite(trade_ret).all():
            continue
        rows.append(
            {
                "date": dates[t],
                "n_available": n_available,
                "hit_rate": float(np.mean(labels.y[t, picked])),
                "base_rate": float(np.mean(labels.y[t, usable[t]])),
                "gross_return": float(np.mean(trade_ret)),
                "net_return": float(np.mean(trade_ret) - cost),
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        log.warning("backtest produced no tradable dates; check top_k against universe size")
        return BacktestResult(daily=daily, summary={})

    overlap = max(label_cfg.touch_offset - label_cfg.entry_offset + 1, 1)
    daily["portfolio_return"] = daily["net_return"] / overlap
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()

    net = daily["net_return"].to_numpy()
    portfolio = daily["portfolio_return"].to_numpy()
    ann = 252.0
    vol = float(portfolio.std(ddof=1)) if len(portfolio) > 1 else float("nan")
    summary = {
        "n_days": float(len(daily)),
        "top_k": float(cfg.top_k),
        "hit_rate": float(daily["hit_rate"].mean()),
        "base_rate": float(daily["base_rate"].mean()),
        "lift": float(daily["hit_rate"].mean() / max(daily["base_rate"].mean(), 1e-12)),
        "mean_trade_return_net": float(net.mean()),
        "trade_win_rate": float((net > 0).mean()),
        "ann_return": float(portfolio.mean() * ann),
        "ann_vol": float(vol * np.sqrt(ann)),
        "sharpe": float(portfolio.mean() / vol * np.sqrt(ann)) if vol > 0 else float("nan"),
        "max_drawdown": float(_max_drawdown(daily["equity"].to_numpy())),
        "final_equity": float(daily["equity"].iloc[-1]),
    }
    log.info(
        "backtest | days %d | hit %.2f%% vs base %.2f%% (lift %.2fx) | net/trade %.3f%% | "
        "ann %.1f%% | sharpe %.2f | maxDD %.1f%%",
        len(daily), 100 * summary["hit_rate"], 100 * summary["base_rate"], summary["lift"],
        100 * summary["mean_trade_return_net"], 100 * summary["ann_return"],
        summary["sharpe"], 100 * summary["max_drawdown"],
    )
    return BacktestResult(daily=daily, summary=summary)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))
