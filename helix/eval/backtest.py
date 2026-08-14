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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig, LabelConfig
from ..data.panel import Panel
from ..data.price_lineage import PriceLineageError, require_hfq_adjustment_stamp
from ..labels.touch_label import LabelSet
from ..logging_setup import get_logger
from .shared_entry_check import entry_is_fillable

log = get_logger(__name__)

#: Stamp duty on A-share sales halved from 10bp to 5bp with effect from this date.
STAMP_CUT_DATE = "20230828"


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    summary: dict[str, float | str]
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class ExitResolution:
    """Observed exit for one D+1 entry, resolved without looking past each attempt."""

    exit_index: int | None
    exit_price: float
    exit_session: str
    d2_limit_down: bool
    d2_suspended: bool
    encountered_suspension: bool
    missing_limit_data: bool
    delay_days: int | None
    holding_days: int
    unresolved_at_end: bool


_REALISTIC_EXIT_FIELDS = (
    "open",
    "high",
    "close",
    "open_hfq",
    "close_hfq",
    "down_limit",
    "limit_price_observed",
    "is_trading",
)


def _unresolved_exit(
    panel: Panel,
    d0_index: int,
    label_cfg: LabelConfig,
    *,
    d2_limit_down: bool,
    d2_suspended: bool,
    encountered_suspension: bool,
    missing_limit_data: bool,
) -> ExitResolution:
    entry_index = d0_index + label_cfg.entry_offset
    return ExitResolution(
        exit_index=None,
        exit_price=float("nan"),
        exit_session="unresolved",
        d2_limit_down=d2_limit_down,
        d2_suspended=d2_suspended,
        encountered_suspension=encountered_suspension,
        missing_limit_data=missing_limit_data,
        delay_days=None,
        holding_days=max(len(panel.dates) - entry_index, 0),
        unresolved_at_end=True,
    )


def _resolved_exit(
    exit_index: int,
    exit_price: float,
    exit_session: str,
    d0_index: int,
    label_cfg: LabelConfig,
    *,
    d2_limit_down: bool,
    d2_suspended: bool,
    encountered_suspension: bool,
    missing_limit_data: bool,
) -> ExitResolution:
    return ExitResolution(
        exit_index=exit_index,
        exit_price=float(exit_price),
        exit_session=exit_session,
        d2_limit_down=d2_limit_down,
        d2_suspended=d2_suspended,
        encountered_suspension=encountered_suspension,
        missing_limit_data=missing_limit_data,
        delay_days=exit_index - (d0_index + label_cfg.touch_offset),
        holding_days=exit_index - (d0_index + label_cfg.entry_offset) + 1,
        unresolved_at_end=False,
    )


def resolve_realistic_exit(
    panel: Panel,
    d0_index: int,
    stock_index: int,
    label_cfg: LabelConfig,
) -> ExitResolution:
    """Resolve one exit serially from D+2 using only each attempted day's bars."""
    missing = [field for field in _REALISTIC_EXIT_FIELDS if field not in panel]
    if missing:
        raise ValueError(f"realistic exit requires panel fields {missing}")

    touch_index = d0_index + label_cfg.touch_offset
    if touch_index >= len(panel.dates):
        return _unresolved_exit(
            panel,
            d0_index,
            label_cfg,
            d2_limit_down=False,
            d2_suspended=False,
            encountered_suspension=False,
            missing_limit_data=False,
        )

    encountered_suspension = False
    missing_limit_data = False
    d2_limit_down = False
    d2_suspended = False
    eps = label_cfg.limit_price_eps

    for exit_index in range(touch_index, len(panel.dates)):
        if not bool(panel["is_trading"][exit_index, stock_index] > 0):
            encountered_suspension = True
            if exit_index == touch_index:
                d2_suspended = True
            continue

        down_limit = float(panel["down_limit"][exit_index, stock_index])
        limit_observed = bool(panel["limit_price_observed"][exit_index, stock_index] > 0)
        if not limit_observed or not np.isfinite(down_limit):
            missing_limit_data = True
            continue

        raw_open = float(panel["open"][exit_index, stock_index])
        raw_high = float(panel["high"][exit_index, stock_index])
        raw_close = float(panel["close"][exit_index, stock_index])

        if exit_index == touch_index:
            has_bar = np.isfinite(raw_high) and np.isfinite(raw_close)
            if not has_bar:
                continue
            d2_limit_down = bool(
                np.isclose(raw_high, down_limit, rtol=0.0, atol=eps)
                and np.isclose(raw_close, down_limit, rtol=0.0, atol=eps)
            )
            if d2_limit_down:
                continue
            close_hfq = float(panel["close_hfq"][exit_index, stock_index])
            if not np.isfinite(close_hfq):
                continue
            return _resolved_exit(
                exit_index,
                close_hfq,
                "d2_close",
                d0_index,
                label_cfg,
                d2_limit_down=False,
                d2_suspended=d2_suspended,
                encountered_suspension=encountered_suspension,
                missing_limit_data=missing_limit_data,
            )

        open_hfq = float(panel["open_hfq"][exit_index, stock_index])
        if np.isfinite(raw_open) and np.isfinite(open_hfq) and raw_open > down_limit + eps:
            return _resolved_exit(
                exit_index,
                open_hfq,
                "open",
                d0_index,
                label_cfg,
                d2_limit_down=d2_limit_down,
                d2_suspended=d2_suspended,
                encountered_suspension=encountered_suspension,
                missing_limit_data=missing_limit_data,
            )

        close_hfq = float(panel["close_hfq"][exit_index, stock_index])
        if np.isfinite(raw_close) and np.isfinite(close_hfq) and raw_close > down_limit + eps:
            return _resolved_exit(
                exit_index,
                close_hfq,
                "close",
                d0_index,
                label_cfg,
                d2_limit_down=d2_limit_down,
                d2_suspended=d2_suspended,
                encountered_suspension=encountered_suspension,
                missing_limit_data=missing_limit_data,
            )

    return _unresolved_exit(
        panel,
        d0_index,
        label_cfg,
        d2_limit_down=d2_limit_down,
        d2_suspended=d2_suspended,
        encountered_suspension=encountered_suspension,
        missing_limit_data=missing_limit_data,
    )


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


def summarize_portfolio_returns(returns: np.ndarray) -> dict[str, float]:
    """Requested strategy metrics, all defined on the aligned daily return series."""
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        return {
            "cagr": float("nan"),
            "day_win_rate": float("nan"),
            "profit_loss_ratio": float("nan"),
            "max_consecutive_losses": 0.0,
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "final_equity": 1.0,
        }

    equity = np.cumprod(1.0 + values)
    final_equity = float(equity[-1])
    cagr = final_equity ** (252.0 / values.size) - 1.0 if final_equity > 0 else -1.0
    positive = values[values > 0]
    negative = values[values < 0]
    profit_loss_ratio = (
        float(positive.mean() / abs(negative.mean()))
        if positive.size and negative.size
        else float("nan")
    )
    longest = current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    vol = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        "cagr": float(cagr),
        "day_win_rate": float((values > 0).mean()),
        "profit_loss_ratio": profit_loss_ratio,
        "max_consecutive_losses": float(longest),
        "ann_return": float(values.mean() * 252.0),
        "ann_vol": float(vol * np.sqrt(252.0)),
        "sharpe": float(values.mean() / vol * np.sqrt(252.0)) if vol > 0 else float("nan"),
        "max_drawdown": float(_max_drawdown(equity)),
        "final_equity": final_equity,
    }


def run_backtest(
    predictions: np.ndarray,
    labels: LabelSet,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    label_cfg: LabelConfig,
    cfg: BacktestConfig,
    panel: Panel | None = None,
) -> BacktestResult:
    require_hfq_adjustment_stamp(labels.adjustment, "run_backtest")
    if cfg.enable_realistic_exit:
        if cfg.exit_rule != "close":
            raise ValueError("enable_realistic_exit requires exit_rule='close'")
        if panel is None:
            raise ValueError("enable_realistic_exit requires an aligned market panel")
        return _run_realistic_backtest(
            predictions, labels, candidate_mask, dates, label_cfg, cfg, panel
        )

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
        # Compatibility contract: the opt-in feature must not alter the historical
        # switch-off summary, including its legacy omission of the initial 1.0 peak.
        "max_drawdown": float(_legacy_max_drawdown(daily["equity"].to_numpy())),
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


def _run_realistic_backtest(
    predictions: np.ndarray,
    labels: LabelSet,
    candidate_mask: np.ndarray,
    dates: np.ndarray,
    label_cfg: LabelConfig,
    cfg: BacktestConfig,
    panel: Panel,
) -> BacktestResult:
    if predictions.shape != labels.y.shape or predictions.shape != candidate_mask.shape:
        raise ValueError("predictions, labels, and candidate_mask must have the same shape")
    if predictions.shape != panel.shape:
        raise ValueError("realistic exit panel must align with predictions")
    if not np.array_equal(np.asarray(dates).astype(str), panel.dates.astype(str)):
        raise ValueError("realistic exit panel dates must align with backtest dates")
    panel_adjustment = panel.require_adjusted_prices(
        ("open_hfq", "close_hfq"), "run_backtest.realistic_exit"
    )
    if panel_adjustment != labels.adjustment:
        raise PriceLineageError(
            "run_backtest.realistic_exit: panel adj_factor_version "
            f"{panel_adjustment.adj_factor_version!r} does not match label adj_factor_version "
            f"{labels.adjustment.adj_factor_version!r}"
        )

    candidates = candidate_mask & np.isfinite(predictions)
    scores = np.where(candidates, predictions, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")
    overlap = max(label_cfg.touch_offset - label_cfg.entry_offset + 1, 1)
    buy_rate, _ = _cost_rates(cfg, dates)
    rows: list[dict] = []
    trade_rows: list[dict] = []

    for t in range(predictions.shape[0]):
        n_available = int(candidates[t].sum())
        if n_available < cfg.top_k:
            continue
        selected = order[t, : cfg.top_k]
        entry_valid = np.asarray(
            [
                entry_is_fillable(panel, t, int(stock_index), label_cfg)
                for stock_index in selected
            ],
            dtype=bool,
        )
        executed = selected[entry_valid]
        day_trade_rows: list[dict] = []

        for rank, (stock_index, is_fillable) in enumerate(
            zip(selected, entry_valid, strict=True), start=1
        ):
            if not is_fillable:
                continue
            entry_index = t + label_cfg.entry_offset
            entry_price = float(panel["open_hfq"][entry_index, stock_index])

            resolved = resolve_realistic_exit(panel, t, int(stock_index), label_cfg)
            baseline_exit_price = float(labels.exit_price[t, stock_index])
            baseline_gross = (
                baseline_exit_price / entry_price - 1.0
                if np.isfinite(baseline_exit_price)
                else float("nan")
            )
            realistic_gross = (
                resolved.exit_price / entry_price - 1.0
                if not resolved.unresolved_at_end
                else float("nan")
            )

            touch_index = t + label_cfg.touch_offset
            baseline_sell = (
                float(_cost_rates(cfg, np.asarray([panel.dates[touch_index]]))[1][0])
                if touch_index < len(panel.dates)
                else float("nan")
            )
            actual_sell = (
                float(_cost_rates(cfg, np.asarray([panel.dates[resolved.exit_index]]))[1][0])
                if resolved.exit_index is not None
                else float("nan")
            )
            baseline_net = (
                float(_net_returns(np.asarray([baseline_gross]), buy_rate, baseline_sell)[0])
                if np.isfinite(baseline_gross) and np.isfinite(baseline_sell)
                else float("nan")
            )
            realistic_net = (
                float(_net_returns(np.asarray([realistic_gross]), buy_rate, actual_sell)[0])
                if np.isfinite(realistic_gross) and np.isfinite(actual_sell)
                else float("nan")
            )
            comparable = np.isfinite(baseline_net) and np.isfinite(realistic_net)
            trade = {
                "d0_index": t,
                "d0_date": str(dates[t]),
                "stock_index": int(stock_index),
                "ts_code": str(panel.codes[stock_index]),
                "selected_rank": rank,
                "entry_price": entry_price,
                "baseline_exit_price": baseline_exit_price,
                "exit_index": resolved.exit_index,
                "exit_date": (
                    str(panel.dates[resolved.exit_index])
                    if resolved.exit_index is not None
                    else ""
                ),
                "exit_price": resolved.exit_price,
                "exit_session": resolved.exit_session,
                "d2_limit_down": resolved.d2_limit_down,
                "d2_suspended": resolved.d2_suspended,
                "encountered_suspension": resolved.encountered_suspension,
                "missing_limit_data": resolved.missing_limit_data,
                "delay_days": resolved.delay_days,
                "holding_days": resolved.holding_days,
                "unresolved_at_end": resolved.unresolved_at_end,
                "baseline_gross_return": baseline_gross,
                "realistic_gross_return": realistic_gross,
                "baseline_net_return": baseline_net,
                "realistic_net_return": realistic_net,
                "matched_baseline_net_return": baseline_net if comparable else 0.0,
                "matched_realistic_net_return": realistic_net if comparable else 0.0,
            }
            trade_rows.append(trade)
            day_trade_rows.append(trade)

        observable_y = labels.y[t, executed]
        observable_y = observable_y[np.isfinite(observable_y)]
        observable = candidates[t] & labels.valid[t] & np.isfinite(labels.y[t])
        base_rate = float(np.mean(labels.y[t, observable])) if observable.any() else float("nan")
        actual_returns = np.asarray(
            [r["realistic_net_return"] for r in day_trade_rows], dtype=np.float64
        )
        actual_gross = np.asarray(
            [r["realistic_gross_return"] for r in day_trade_rows], dtype=np.float64
        )
        finite_actual = np.isfinite(actual_returns)
        matched_realistic = np.asarray(
            [r["matched_realistic_net_return"] for r in day_trade_rows], dtype=np.float64
        )
        matched_baseline = np.asarray(
            [r["matched_baseline_net_return"] for r in day_trade_rows], dtype=np.float64
        )
        rows.append(
            {
                "date": dates[t],
                "n_available": n_available,
                "n_selected": cfg.top_k,
                "n_executed": int(executed.size),
                "hit_rate": (
                    float(observable_y.mean()) if observable_y.size else float("nan")
                ),
                "base_rate": base_rate,
                "gross_return": (
                    float(actual_gross[np.isfinite(actual_gross)].mean())
                    if np.isfinite(actual_gross).any()
                    else float("nan")
                ),
                "net_return": (
                    float(actual_returns[finite_actual].mean())
                    if finite_actual.any()
                    else float("nan")
                ),
                "baseline_portfolio_return": float(
                    matched_baseline.sum() / cfg.top_k / overlap
                ),
                "portfolio_return": float(
                    matched_realistic.sum() / cfg.top_k / overlap
                ),
            }
        )

    daily = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)
    if daily.empty:
        log.warning("backtest produced no selectable dates; check top_k against universe size")
        return BacktestResult(daily=daily, summary={}, trades=trades)

    daily["baseline_equity"] = (1.0 + daily["baseline_portfolio_return"]).cumprod()
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    performance = summarize_portfolio_returns(daily["portfolio_return"].to_numpy())
    baseline_performance = summarize_portfolio_returns(
        daily["baseline_portfolio_return"].to_numpy()
    )
    resolved_trades = trades[~trades["unresolved_at_end"]] if not trades.empty else trades
    limit_down = trades[trades["d2_limit_down"]] if not trades.empty else trades
    resolved_limit_down = (
        limit_down[~limit_down["unresolved_at_end"]]
        if not limit_down.empty
        else limit_down
    )
    hit_rate = float(daily["hit_rate"].mean())
    base_rate = float(daily["base_rate"].mean())
    summary: dict[str, float | str] = {
        "n_days": float(len(daily)),
        "top_k": float(cfg.top_k),
        "exit_rule": cfg.exit_rule,
        "enable_realistic_exit": "true",
        "slippage_bps": float(cfg.slippage_bps),
        "avg_executed": float(daily["n_executed"].mean()),
        "fill_rate": float(daily["n_executed"].sum() / daily["n_selected"].sum()),
        "hit_rate": hit_rate,
        "base_rate": base_rate,
        "lift": float(hit_rate / base_rate) if base_rate > 0 else float("nan"),
        "mean_trade_return_net": (
            float(resolved_trades["realistic_net_return"].mean())
            if not resolved_trades.empty
            else float("nan")
        ),
        "trade_win_rate": (
            float((resolved_trades["realistic_net_return"] > 0).mean())
            if not resolved_trades.empty
            else float("nan")
        ),
        **performance,
        "limit_down_exit_share": (
            float(len(limit_down) / len(trades)) if not trades.empty else float("nan")
        ),
        "avg_delay_days": (
            float(resolved_limit_down["delay_days"].mean())
            if not resolved_limit_down.empty
            else float("nan")
        ),
        "avg_limit_down_loss": (
            float(resolved_limit_down["realistic_gross_return"].mean())
            if not resolved_limit_down.empty
            else float("nan")
        ),
        "avg_limit_down_return_impact": (
            float(
                (
                    resolved_limit_down["realistic_gross_return"]
                    - resolved_limit_down["baseline_gross_return"]
                ).mean()
            )
            if not resolved_limit_down.empty
            else float("nan")
        ),
        "d2_suspension_count": float(trades["d2_suspended"].sum()) if not trades.empty else 0.0,
        "unresolved_at_end": (
            float(trades["unresolved_at_end"].sum()) if not trades.empty else 0.0
        ),
    }
    summary.update({f"baseline_{key}": value for key, value in baseline_performance.items()})
    return BacktestResult(daily=daily, summary=summary, trades=trades)


def _max_drawdown(equity: np.ndarray) -> float:
    values = np.concatenate(([1.0], np.asarray(equity, dtype=np.float64)))
    peak = np.maximum.accumulate(values)
    return float(np.min(values / peak - 1.0))


def _legacy_max_drawdown(equity: np.ndarray) -> float:
    values = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(values)
    return float(np.min(values / peak - 1.0))
