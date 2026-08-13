"""Trade-level backtest of the top-``k`` daily selection.

The shortlist is fixed from the D0 point-in-time universe before any D+1/D+2
tradability checks. Future execution data may reject a shortlisted trade, but can never
promote a lower-ranked name into the book.

Everyone enters at the D+1 open. Two accounting choices decide the answer, and the
first version of this file got both of them wrong in the optimistic direction:

* **Exit rule.** ``target`` mirrors the label: the D+2 high reaching ``entry ×
  target_ratio`` is booked as a fill at that price. Touching a price is not being
  filled at it, and taking profit there caps the only side that pays. Measured on the
  argus event table, the same predictions return −46.7% CAGR under ``target`` and
  +51.6% under ``close``, because selection picks high-volatility momentum names: the
  winners run past +8% (+10.1% to the close) while the losers are left to run (−4.6%).
  ``close`` -- every position exits at the D+2 close, hit or miss -- is the default.
* **Costs.** Charged per side against notional, not subtracted from the gross return,
  and split along the statutory A-share schedule. Stamp duty is sell-side only and
  halved on 2023-08-28, so the sell rate is resolved per trade date rather than assumed
  constant across a panel that starts in 2018.

Neither is free to get wrong: on this strategy the break-even point sits at roughly
19bp of slippage per side, so a 5bp error in the stamp rate is a quarter of the margin.

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

#: Stamp duty on A-share sales halved from 10bp to 5bp with effect from this date.
STAMP_CUT_DATE = "20230828"


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    summary: dict[str, float | str]


def _trade_returns(labels: LabelSet, target_ratio: float, exit_rule: str) -> np.ndarray:
    """Gross per-trade return under the chosen exit assumption."""
    to_close = labels.exit_price / labels.entry_price - 1.0
    if exit_rule == "close":
        return to_close
    return np.where(labels.y == 1.0, target_ratio - 1.0, to_close)


def _cost_rates(cfg: BacktestConfig, dates: np.ndarray) -> tuple[float, np.ndarray]:
    """``(buy, sell)`` cost as a fraction of notional; ``sell`` varies by trade date.

    Keyed on the decision date rather than on the D+2 settlement date. The two disagree
    only for the two books that straddle the stamp-duty cut, which is far below the
    resolution of anything reported here.
    """
    both_sides = cfg.commission_bps + cfg.transfer_bps + cfg.slippage_bps
    stamp = np.where(
        np.asarray(dates).astype(str) < STAMP_CUT_DATE,
        cfg.stamp_sell_bps_before_cut,
        cfg.stamp_sell_bps,
    )
    return both_sides / 10_000.0, (both_sides + stamp) / 10_000.0


def _net_returns(gross: np.ndarray, buy: float, sell: float) -> np.ndarray:
    """Cost-adjusted trade return. Costs scale with notional, so this is not ``gross - c``."""
    return (1.0 + gross) * (1.0 - sell) / (1.0 + buy) - 1.0


def run_backtest(
    predictions: np.ndarray,
    labels: LabelSet,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    label_cfg: LabelConfig,
    cfg: BacktestConfig,
) -> BacktestResult:
    gross = _trade_returns(labels, label_cfg.target_ratio, cfg.exit_rule)
    buy_rate, sell_rates = _cost_rates(cfg, dates)
    candidates = candidate_mask & np.isfinite(predictions)

    n_dates = predictions.shape[0]
    rows: list[dict] = []
    executed_net_returns: list[np.ndarray] = []
    scores = np.where(candidates, predictions, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")
    overlap = max(label_cfg.touch_offset - label_cfg.entry_offset + 1, 1)

    for t in range(n_dates):
        n_available = int(candidates[t].sum())
        if n_available < cfg.top_k:
            continue
        selected = order[t, : cfg.top_k]
        execution_valid = (
            labels.valid[t, selected]
            & labels.touch_tradable[t, selected]
            & np.isfinite(gross[t, selected])
        )
        executed = selected[execution_valid]
        trade_ret = gross[t, executed]
        net_ret = _net_returns(trade_ret, buy_rate, float(sell_rates[t]))
        if net_ret.size:
            executed_net_returns.append(net_ret)
        observable = candidates[t] & labels.valid[t] & np.isfinite(labels.y[t])
        base_rate = float(np.mean(labels.y[t, observable])) if observable.any() else float("nan")
        rows.append(
            {
                "date": dates[t],
                "n_available": n_available,
                "n_selected": cfg.top_k,
                "n_executed": int(executed.size),
                "hit_rate": (
                    float(np.mean(labels.y[t, executed])) if executed.size else float("nan")
                ),
                "base_rate": base_rate,
                "gross_return": float(np.mean(trade_ret)) if executed.size else float("nan"),
                "net_return": float(np.mean(net_ret)) if executed.size else float("nan"),
                # Every D0 selection reserves one equal-weight slot. A failed execution
                # stays in cash; its capital is not reassigned to another name or fill.
                "portfolio_return": float(np.sum(net_ret) / cfg.top_k / overlap),
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        log.warning("backtest produced no selectable dates; check top_k against universe size")
        return BacktestResult(daily=daily, summary={})

    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()

    net = (
        np.concatenate(executed_net_returns)
        if executed_net_returns
        else np.empty(0, dtype=np.float64)
    )
    portfolio = daily["portfolio_return"].to_numpy()
    hit_rate = float(daily["hit_rate"].mean())
    base_rate = float(daily["base_rate"].mean())
    ann = 252.0
    vol = float(portfolio.std(ddof=1)) if len(portfolio) > 1 else float("nan")
    summary: dict[str, float | str] = {
        "n_days": float(len(daily)),
        "top_k": float(cfg.top_k),
        # Recorded so two summary files are distinguishable: the exit rule alone moves
        # this strategy's CAGR by ~100 percentage points.
        "exit_rule": cfg.exit_rule,
        "slippage_bps": float(cfg.slippage_bps),
        "avg_executed": float(daily["n_executed"].mean()),
        "fill_rate": float(daily["n_executed"].sum() / daily["n_selected"].sum()),
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": float(hit_rate / base_rate) if base_rate > 0 else float("nan"),
        "mean_trade_return_net": float(net.mean()) if net.size else float("nan"),
        "trade_win_rate": float((net > 0).mean()) if net.size else float("nan"),
        "ann_return": float(portfolio.mean() * ann),
        "ann_vol": float(vol * np.sqrt(ann)),
        "sharpe": float(portfolio.mean() / vol * np.sqrt(ann)) if vol > 0 else float("nan"),
        "max_drawdown": float(_max_drawdown(daily["equity"].to_numpy())),
        "final_equity": float(daily["equity"].iloc[-1]),
    }
    log.info(
        "backtest | exit=%s slippage=%.1fbp | days %d | hit %.2f%% vs base %.2f%% (lift %.2fx) | "
        "net/trade %.3f%% | ann %.1f%% | sharpe %.2f | maxDD %.1f%%",
        cfg.exit_rule, cfg.slippage_bps,
        len(daily), 100 * summary["hit_rate"], 100 * summary["base_rate"], summary["lift"],
        100 * summary["mean_trade_return_net"], 100 * summary["ann_return"],
        summary["sharpe"], 100 * summary["max_drawdown"],
    )
    return BacktestResult(daily=daily, summary=summary)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))
