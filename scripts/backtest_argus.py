#!/usr/bin/env python3
"""Does the top-20 selection still make money after costs, or only after hit rate?

A hit rate of 35% against a 17% base is a real edge in ranking. It is not yet a return.
Each trade pays commission, stamp duty, transfer fee and slippage, and -- the part a hit
rate cannot show -- the 65% that miss exit at the D+2 close, which is where the momentum
names that failed have already gapped down. Pool-wide the average gross trade is
**-0.07%**, so the entire result lives in whether selection improves the losing tail and
not merely the winning count.

Scoring is deliberately imported from `fill_impact.py` rather than reimplemented: same
features, same split, same seeds, same model. Only the accounting is new, so any
difference from the published hit rate is accounting and nothing else.

Two assumptions get their own sensitivity axis, because both are optimistic and neither
is verifiable from this table:

* **exit=target** mirrors the label -- the D+2 high touching `entry * 1.08` is treated as
  a fill at `entry * 1.08`. Touching a price is not being filled at it.
  **exit=close** is the floor: every position, hit or miss, exits at the D+2 close.
* **slippage** is charged per side on top of the statutory costs. Entering at the
  official open is itself an idealisation.

Positions are fill-aware: a pick whose D+1 opens at its up-limit never fills and is
dropped rather than counted as a win (`fillability.py`, and the audit in
`check_fillability.py`).

Standalone: numpy / pandas / pyarrow / xgboost, so it runs on the training host.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from fill_impact import build_model, daily_ic, feature_columns
from fillability import REQUIRED_COLUMNS, unfillable_mask

#: Statutory A-share round-trip costs, in basis points of notional.
#: Stamp duty is sell-side only and was halved to 5bp on 2023-08-28; every out-of-sample
#: date here is after that, so a single rate is correct for this window and would not be
#: for a backtest reaching further back.
COMMISSION_BPS = 2.5     # 佣金，双边
TRANSFER_BPS = 0.1       # 过户费，双边
STAMP_SELL_BPS = 5.0     # 印花税，仅卖出
STAMP_CUT_DATE = "20230828"


def _digits(date: str) -> str:
    return "".join(ch for ch in date if ch.isdigit())

#: A new book opens every day while the previous one is still held, so capital is split
#: across this many overlapping tranches. touch_offset - entry_offset + 1 = 2.
OVERLAP = 2
TRADING_DAYS = 252.0


def cost_rates(slippage_bps: float) -> tuple[float, float]:
    """(buy, sell) cost as a fraction of notional."""
    buy = (COMMISSION_BPS + TRANSFER_BPS + slippage_bps) / 10_000.0
    sell = (COMMISSION_BPS + TRANSFER_BPS + STAMP_SELL_BPS + slippage_bps) / 10_000.0
    return buy, sell


def net_return(gross: np.ndarray, slippage_bps: float) -> np.ndarray:
    """Cost-adjusted trade return. Costs scale with notional, so this is not `gross - c`."""
    buy, sell = cost_rates(slippage_bps)
    return (1.0 + gross) * (1.0 - sell) / (1.0 + buy) - 1.0


def gross_returns(frame: pd.DataFrame, label: str, target_ratio: float,
                  exit_rule: str) -> np.ndarray:
    """Per-trade gross return under the chosen exit assumption."""
    to_close = (frame["label_px_d2_close"].to_numpy(dtype=float)
                / frame["label_px_d1_open"].to_numpy(dtype=float)) - 1.0
    if exit_rule == "close":
        return to_close
    hit = frame[label].to_numpy(dtype=float) == 1.0
    return np.where(hit, target_ratio - 1.0, to_close)


def max_drawdown(equity: np.ndarray) -> float:
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def run_book(scored: pd.DataFrame, label: str, k: int, target_ratio: float,
             exit_rule: str, slippage_bps: float) -> dict:
    """Daily fill-aware top-k book, then the tranche-split equity curve."""
    rows = []
    for date, g in scored.groupby("trade_date", sort=True):
        if len(g) < k:
            continue
        # Fill-aware, not queue-deeper: submit k orders and keep what fills. Ranking among
        # fillable rows instead would assume you knew at D0 close which names gap to the
        # limit, which is exactly what you cannot know.
        submitted = g.nlargest(k, "score")
        picked = submitted[~submitted["unfillable"]]
        if picked.empty:
            continue
        gross = gross_returns(picked, label, target_ratio, exit_rule)
        if not np.isfinite(gross).all():
            continue
        net = net_return(gross, slippage_bps)
        hit = picked[label].to_numpy(dtype=float) == 1.0
        to_close = (picked["label_px_d2_close"].to_numpy(dtype=float)
                    / picked["label_px_d1_open"].to_numpy(dtype=float)) - 1.0
        rows.append({
            "date": date,
            "positions": len(picked),
            "hit_rate": float(hit.mean()),
            "base_rate": float(g[~g["unfillable"]][label].mean()),
            "gross_return": float(gross.mean()),
            "net_return": float(net.mean()),
            # The decomposition that explains the result: selection lifts the winners a
            # little and the losers a lot, so where the exit sits decides the sign.
            "win_to_close": float(to_close[hit].mean()) if hit.any() else np.nan,
            "loss_to_close": float(to_close[~hit].mean()) if (~hit).any() else np.nan,
        })

    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"n_days": 0}

    daily["portfolio_return"] = daily["net_return"] / OVERLAP
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    portfolio = daily["portfolio_return"].to_numpy()
    vol = float(portfolio.std(ddof=1)) if len(portfolio) > 1 else float("nan")
    hit, base = float(daily["hit_rate"].mean()), float(daily["base_rate"].mean())
    years = len(daily) / TRADING_DAYS
    final = float(daily["equity"].iloc[-1])
    return {
        "n_days": int(len(daily)),
        "avg_positions": round(float(daily["positions"].mean()), 3),
        "hit_rate": round(hit, 6),
        "base_rate": round(base, 6),
        "lift": round(hit / max(base, 1e-12), 4),
        "win_to_close": round(float(daily["win_to_close"].mean()), 6),
        "loss_to_close": round(float(daily["loss_to_close"].mean()), 6),
        "gross_per_trade": round(float(daily["gross_return"].mean()), 6),
        "net_per_trade": round(float(daily["net_return"].mean()), 6),
        # Arithmetic annualisation flatters a losing curve; report the compounded one too.
        "cagr": round(final ** (1.0 / years) - 1.0, 6) if final > 0 else -1.0,
        "day_win_rate": round(float((daily["net_return"] > 0).mean()), 6),
        "ann_return": round(float(portfolio.mean() * TRADING_DAYS), 6),
        "ann_vol": round(float(vol * np.sqrt(TRADING_DAYS)), 6),
        "sharpe": round(float(portfolio.mean() / vol * np.sqrt(TRADING_DAYS)), 4)
        if vol > 0 else float("nan"),
        "max_drawdown": round(max_drawdown(daily["equity"].to_numpy()), 6),
        "final_equity": round(float(daily["equity"].iloc[-1]), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", default="label_d2_hit_8pct")
    ap.add_argument("--ic-target", default="label_d2_peak_return")
    ap.add_argument("--target-ratio", type=float, default=1.08)
    ap.add_argument("--split-date", default="2024-09-04")
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seeds", default="7,13,42")
    ap.add_argument("--slippage-grid", default="0,5,10,20",
                    help="Per-side slippage in bps, on top of statutory costs.")
    ap.add_argument("--report", default="backtest_argus_report.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    slippages = [float(s) for s in args.slippage_grid.split(",") if s.strip()]

    features = feature_columns(args.input)
    price_cols = ["label_px_d1_open", "label_px_d2_close"]
    needed = sorted({"trade_date", args.label, args.ic_target,
                     *price_cols, *REQUIRED_COLUMNS, *features})
    df = pd.read_parquet(args.input, columns=needed)
    df["trade_date"] = df["trade_date"].astype(str)
    df["unfillable"] = unfillable_mask(df)
    df = df.dropna(subset=[args.label, args.ic_target, *price_cols])
    print(f"features {len(features)}  rows {len(df):,}")

    dates = np.array(sorted(df["trade_date"].unique()))
    cut = int(np.searchsorted(dates, args.split_date, "right"))
    train_end = dates[max(cut - 1, 0)]
    test_start = dates[min(cut + args.embargo_days, len(dates) - 1)]
    train = df[df["trade_date"] <= train_end]
    test = df[df["trade_date"] >= test_start]
    print(f"train ~{train_end} ({len(train):,}) | test {test_start}~ ({len(test):,})")
    if _digits(test_start) < STAMP_CUT_DATE:
        raise SystemExit(f"测试窗口起点 {test_start} 早于 2023-08-28 印花税减半，"
                         f"{STAMP_SELL_BPS}bp 单一税率会低估这段时间的成本")

    report = {"features": len(features), "train_end": train_end, "test_start": test_start,
              "seeds": seeds, "top_k": args.top_k, "target_ratio": args.target_ratio,
              "cost_bps": {"commission": COMMISSION_BPS, "transfer": TRANSFER_BPS,
                           "stamp_sell": STAMP_SELL_BPS},
              "runs": [], "summary": {}}

    books: dict[tuple[str, float], list[dict]] = {}
    for seed in seeds:
        model = build_model(seed)
        model.fit(train[features].to_numpy(dtype=np.float32), train[args.label].to_numpy())
        scored = test[["trade_date", args.label, args.ic_target, "unfillable",
                       *price_cols]].copy()
        scored["score"] = model.predict_proba(
            test[features].to_numpy(dtype=np.float32))[:, 1]
        ic = daily_ic(scored, "score", args.ic_target)

        for exit_rule in ("target", "close"):
            for slip in slippages:
                book = run_book(scored, args.label, args.top_k, args.target_ratio,
                                exit_rule, slip)
                book |= {"seed": seed, "exit": exit_rule, "slippage_bps": slip,
                         "ic_mean": round(ic, 6)}
                report["runs"].append(book)
                books.setdefault((exit_rule, slip), []).append(book)
        base = books[("target", slippages[0])][-1]
        print(f"  seed {seed}  IC {ic:+.5f}  命中 {100 * base['hit_rate']:.2f}%  "
              f"毛 {100 * base['gross_per_trade']:+.3f}%/笔")

    sample = books[("target", slippages[0])]
    print(f"\ntop{args.top_k} 收益分解（{len(seeds)} 个种子均值）：命中组持到 D+2 收盘 "
          f"{100 * float(np.mean([r['win_to_close'] for r in sample])):+.2f}%  |  "
          f"未命中组 {100 * float(np.mean([r['loss_to_close'] for r in sample])):+.2f}%  |  "
          f"实际建仓 {float(np.mean([r['avg_positions'] for r in sample])):.2f}/{args.top_k}")

    print(f"\n=== {len(seeds)} 个种子 均值 ± 标准差 ===")
    print(f"{'退出假设':<16}{'滑点':>6}{'净收益/笔':>17}{'CAGR':>17}"
          f"{'Sharpe':>15}{'最大回撤':>10}")
    for (exit_rule, slip), runs in books.items():
        agg = {}
        for key in ("net_per_trade", "ann_return", "cagr", "sharpe", "max_drawdown",
                    "day_win_rate"):
            values = [r[key] for r in runs]
            agg[key] = (float(np.mean(values)),
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        report["summary"][f"{exit_rule}|{slip}"] = {
            k: {"mean": round(m, 6), "std": round(s, 6)} for k, (m, s) in agg.items()}
        rule = "触及即成交" if exit_rule == "target" else "全部按 D+2 收盘"
        print(f"{rule:<16}{slip:4.0f}bp"
              f"{100 * agg['net_per_trade'][0]:>11.3f}% ±{100 * agg['net_per_trade'][1]:.3f}"
              f"{100 * agg['cagr'][0]:>11.1f}% ±{100 * agg['cagr'][1]:.1f}"
              f"{agg['sharpe'][0]:>10.2f} ±{agg['sharpe'][1]:.2f}"
              f"{100 * agg['max_drawdown'][0]:>9.1f}%")

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
