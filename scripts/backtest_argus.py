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

`--rank-by` then asks the follow-up question. The published ranker is trained on
`P(touch 8%)`, which matches the label but not the exit that actually makes money:

* **classify**   -- P(touch 8%), the published ranker.
* **regress**    -- squared error on `close[D+2]/open[D+1] - 1` directly.
* **regress-cs** -- the same return, z-scored within each trading day. The book picks k
  names *inside* a day, so the market's shared daily move is variance the ranker should
  not be spending capacity on.

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
from fill_impact import build_model, build_regressor, daily_ic, feature_columns
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


def target_hit(frame: pd.DataFrame, target_ratio: float) -> np.ndarray:
    """Did the D+2 high reach `open[D+1] * target_ratio`?

    Recomputed from prices rather than read off `label_d2_hit_8pct`, so the take-profit
    level is a free parameter instead of being welded to the label. Reading the label
    would make `--target-ratio 1.10` pay out 10% on the set of trades that touched 8% --
    a strictly better strategy than any that exists. `main` checks this reproduces the
    published label at 1.08 before trusting it anywhere else.
    """
    return ((frame["label_px_d2_high"].to_numpy(dtype=float)
             / frame["label_px_d1_open"].to_numpy(dtype=float)) >= target_ratio)


def gross_returns(frame: pd.DataFrame, target_ratio: float, exit_rule: str) -> np.ndarray:
    """Per-trade gross return under the chosen exit assumption."""
    to_close = (frame["label_px_d2_close"].to_numpy(dtype=float)
                / frame["label_px_d1_open"].to_numpy(dtype=float)) - 1.0
    if exit_rule == "close":
        return to_close
    return np.where(target_hit(frame, target_ratio), target_ratio - 1.0, to_close)


def max_drawdown(equity: np.ndarray) -> float:
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def cross_sectional_z(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Per-day z-score of `column`, so the target carries no market-direction component.

    Ranking happens *within* a trading day: knowing that tomorrow is broadly up helps pick
    nothing. A raw-return regression spends capacity on that shared component anyway,
    because it is the largest single term in the variance. Demeaning by day removes it and
    leaves the only part the book can act on. Uses that day's own cross-section, so it is
    applied to the training target only -- never to build a feature.
    """
    g = frame.groupby("trade_date", sort=False)[column]
    std = g.transform("std").to_numpy(dtype=float)
    centred = frame[column].to_numpy(dtype=float) - g.transform("mean").to_numpy(dtype=float)
    return np.where(std > 0, centred / np.where(std > 0, std, 1.0), 0.0)


def fit_and_score(rank_by: str, seed: int, train: pd.DataFrame, test: pd.DataFrame,
                  features: list[str], label: str, return_col: str) -> np.ndarray:
    """Train one ranker and return its out-of-sample score, higher = buy first."""
    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)
    if rank_by == "classify":
        model = build_model(seed)
        model.fit(x_train, train[label].to_numpy())
        return model.predict_proba(x_test)[:, 1]

    target = (cross_sectional_z(train, return_col) if rank_by == "regress-cs"
              else train[return_col].to_numpy(dtype=float))
    model = build_regressor(seed)
    model.fit(x_train, target)
    return model.predict(x_test)


def run_book(scored: pd.DataFrame, hold_k: int, signal_k: int,
             target_ratio: float, exit_rule: str, slippage_bps: float,
             min_score: float | None = None) -> tuple[dict, pd.DataFrame]:
    """Submit the ranked shortlist, fill down it until `hold_k` positions, then compound.

    `min_score` is the conviction gate: a candidate below it is not bought, and a day where
    nothing clears the bar is held flat rather than dropped. That distinction is the whole
    point of the parameter. Skipping the day would quietly shorten the track and measure
    the strategy only on days it chose to trade; a flat day is a real day with a zero
    return, and it is what lets a threshold reduce drawdown instead of just reducing `n`.

    Returns the summary *and* the per-day series behind it. The scalars alone cannot answer
    the question a short live track record raises -- a headline drawdown is the minimum
    over the observation window, so a 28-day number and a 431-day number are not the same
    quantity. Keeping the series lets `window_stats.py` cut this one into windows of the
    same length and compare like with like.

    `signal_k == hold_k` is the plain fill-aware convention: submit k orders, keep
    whatever fills, do not substitute. `signal_k > hold_k` is what a concentrated book
    needs -- a shortlist deep enough that a limit-up open on the top name does not leave
    the day flat.

    Walking down a *predefined* shortlist is not the foreknowledge that made the
    `queue-deeper` view of `fill_impact.py` an upper bound. The D+1 opening call auction
    prints at 09:25, so which candidates gapped to their limit is known before continuous
    trading starts and the substitution is executable. What it does cost is precision on
    the entry price: the backup is bought after the open, not at it. That is what the
    slippage grid is for.
    """
    rows = []
    for date, g in scored.groupby("trade_date", sort=True):
        if len(g) < signal_k:
            continue
        fillable = g[~g["unfillable"]]
        base_rate = (float(target_hit(fillable, target_ratio).mean())
                     if len(fillable) else np.nan)
        flat = {"date": date, "positions": 0, "fill_depth": 0, "short": 0.0,
                "hit_rate": np.nan, "base_rate": base_rate,
                "gross_return": np.nan, "net_return": 0.0,
                "win_to_close": np.nan, "loss_to_close": np.nan}

        shortlist = g.nlargest(signal_k, "score").reset_index(drop=True)
        usable = shortlist.index[~shortlist["unfillable"].to_numpy()][:hold_k]
        picked = shortlist.loc[usable]
        if min_score is not None:
            # The shortlist is score-descending, so the gate keeps a prefix of it and the
            # substitution order above is untouched.
            keep = picked["score"].to_numpy(dtype=float) >= min_score
            usable, picked = usable[keep], picked[keep]
        if picked.empty:
            rows.append(flat)
            continue
        # How deep the shortlist had to go, and whether it ran out before filling hold_k.
        depth = int(usable[-1]) + 1
        short = len(picked) < hold_k
        gross = gross_returns(picked, target_ratio, exit_rule)
        if not np.isfinite(gross).all():
            continue
        net = net_return(gross, slippage_bps)
        # Hit rate follows `target_ratio` too, so lift describes the level being traded
        # rather than the 8% the table happens to label.
        hit = target_hit(picked, target_ratio)
        to_close = (picked["label_px_d2_close"].to_numpy(dtype=float)
                    / picked["label_px_d1_open"].to_numpy(dtype=float)) - 1.0
        rows.append({
            "date": date,
            "positions": len(picked),
            "fill_depth": depth,
            "short": float(short),
            "hit_rate": float(hit.mean()),
            "base_rate": base_rate,
            "gross_return": float(gross.mean()),
            "net_return": float(net.mean()),
            # The decomposition that explains the result: selection lifts the winners a
            # little and the losers a lot, so where the exit sits decides the sign.
            "win_to_close": float(to_close[hit].mean()) if hit.any() else np.nan,
            "loss_to_close": float(to_close[~hit].mean()) if (~hit).any() else np.nan,
        })

    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"n_days": 0}, daily

    daily["portfolio_return"] = daily["net_return"] / OVERLAP
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    portfolio = daily["portfolio_return"].to_numpy()
    vol = float(portfolio.std(ddof=1)) if len(portfolio) > 1 else float("nan")
    # Per-trade statistics average over days that traded; the curve averages over all of
    # them. Mixing the two would let a threshold flatter its own per-trade number by
    # sitting out the days it expected to lose, while the equity curve says otherwise.
    traded = daily[daily["positions"] > 0]
    hit, base = float(traded["hit_rate"].mean()), float(daily["base_rate"].mean())
    years = len(daily) / TRADING_DAYS
    final = float(daily["equity"].iloc[-1])
    summary = {
        "n_days": int(len(daily)),
        "n_traded_days": int(len(traded)),
        "flat_day_rate": round(float((daily["positions"] == 0).mean()), 6),
        "avg_positions": round(float(traded["positions"].mean()), 3),
        # Both are noise at hold_k=20 and decisive at hold_k=1: if the top name is often
        # unfillable, the shortlist is doing the work and its depth is a real assumption.
        "avg_fill_depth": round(float(traded["fill_depth"].mean()), 3),
        "short_day_rate": round(float(traded["short"].mean()), 6),
        "hit_rate": round(hit, 6),
        "base_rate": round(base, 6),
        "lift": round(hit / max(base, 1e-12), 4),
        "win_to_close": round(float(traded["win_to_close"].mean()), 6),
        "loss_to_close": round(float(traded["loss_to_close"].mean()), 6),
        "gross_per_trade": round(float(traded["gross_return"].mean()), 6),
        "net_per_trade": round(float(traded["net_return"].mean()), 6),
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
    return summary, daily[["date", "positions", "portfolio_return", "equity"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", default="label_d2_hit_8pct")
    ap.add_argument("--ic-target", default="label_d2_peak_return")
    ap.add_argument("--return-col", default="label_d2_return",
                    help="close[D+2]/open[D+1] - 1, i.e. what the close exit earns.")
    ap.add_argument("--rank-by", default="classify",
                    choices=("classify", "regress", "regress-cs"),
                    help="What to rank on: P(touch 8%%), the raw D+2 return, or its "
                         "per-day cross-sectional z-score.")
    ap.add_argument("--target-ratio", type=float, default=1.08)
    ap.add_argument("--split-date", default="2024-09-04")
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=20,
                    help="Positions actually held per day.")
    ap.add_argument("--signal-k", type=int, default=None,
                    help="Length of the ranked shortlist submitted. Defaults to --top-k "
                         "(no substitution). Set larger to allow filling down the list "
                         "when a higher-ranked name opens at its limit.")
    ap.add_argument("--min-score-grid", default="",
                    help="Comma-separated conviction gates, swept without refitting: a "
                         "candidate scoring below the gate is not bought, and a day where "
                         "none clear it is held flat. The no-gate case is always included. "
                         "Only comparable across runs for --rank-by classify, where the "
                         "score is a probability.")
    ap.add_argument("--seeds", default="7,13,42")
    ap.add_argument("--slippage-grid", default="0,5,10,20",
                    help="Per-side slippage in bps, on top of statutory costs.")
    ap.add_argument("--report", default="backtest_argus_report.json")
    ap.add_argument("--daily-out", default="backtest_argus_daily.csv",
                    help="Per-day portfolio return for every (seed, exit, slippage). "
                         "Feeds window_stats.py.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    slippages = [float(s) for s in args.slippage_grid.split(",") if s.strip()]
    # None first, so the ungated book is always the reference every gate is read against.
    gates: list[float | None] = [None] + [float(s) for s in args.min_score_grid.split(",")
                                          if s.strip()]
    signal_k = args.signal_k if args.signal_k is not None else args.top_k
    if signal_k < args.top_k:
        raise SystemExit(f"--signal-k {signal_k} 小于持仓数 {args.top_k}，候选不够建仓")

    features = feature_columns(args.input)
    price_cols = ["label_px_d1_open", "label_px_d2_high", "label_px_d2_close"]
    needed = sorted({"trade_date", args.label, args.ic_target, args.return_col,
                     *price_cols, *REQUIRED_COLUMNS, *features})
    df = pd.read_parquet(args.input, columns=needed)
    df["trade_date"] = df["trade_date"].astype(str)
    df["unfillable"] = unfillable_mask(df)
    df = df.dropna(subset=[args.label, args.ic_target, args.return_col, *price_cols])
    print(f"features {len(features)}  rows {len(df):,}  rank-by {args.rank_by}")

    # The return column has to be the quantity the close exit actually earns, or ranking
    # on it optimises something else. Cheap to check, so check rather than trust the name.
    implied = (df["label_px_d2_close"].to_numpy(dtype=float)
               / df["label_px_d1_open"].to_numpy(dtype=float)) - 1.0
    if not np.allclose(implied, df[args.return_col].to_numpy(dtype=float), atol=1e-6):
        raise SystemExit(f"{args.return_col} 不等于 close[D+2]/open[D+1]-1，不能当回归目标")

    # The take-profit level is now recomputed from prices instead of read off the label, so
    # tie the recomputation back to the published label at the one ratio where both exist.
    # A float tie on `high == open * 1.08` may disagree; a systematic difference may not.
    disagree = float((target_hit(df, 1.08) != (df[args.label].to_numpy(dtype=float) == 1.0)).mean())
    if disagree > 1e-4:
        raise SystemExit(f"按价格重算 1.08 触及与 {args.label} 有 {disagree:.4%} 不一致，"
                         f"止盈判定不可信")
    print(f"1.08 触及重算 vs {args.label}：不一致 {disagree:.6%}")

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
              "seeds": seeds, "top_k": args.top_k, "signal_k": signal_k,
              "target_ratio": args.target_ratio, "min_score_grid": gates,
              "rank_by": args.rank_by, "return_col": args.return_col,
              "cost_bps": {"commission": COMMISSION_BPS, "transfer": TRANSFER_BPS,
                           "stamp_sell": STAMP_SELL_BPS},
              "runs": [], "summary": {}}

    books: dict[tuple[str, float, float | None], list[dict]] = {}
    series: list[pd.DataFrame] = []
    for seed in seeds:
        scored = test[["trade_date", args.label, args.ic_target, args.return_col,
                       "unfillable", *price_cols]].copy()
        scored["score"] = fit_and_score(args.rank_by, seed, train, test, features,
                                        args.label, args.return_col)
        ic = daily_ic(scored, "score", args.ic_target)
        # Against the return too: a ranker trained on returns should not be judged only on
        # its correlation with the peak, which is the classifier's home turf.
        ic_ret = daily_ic(scored, "score", args.return_col)

        for exit_rule in ("target", "close"):
            for slip in slippages:
                for gate in gates:
                    book, daily = run_book(scored, args.top_k, signal_k, args.target_ratio,
                                           exit_rule, slip, gate)
                    book |= {"seed": seed, "exit": exit_rule, "slippage_bps": slip,
                             "min_score": gate,
                             "ic_mean": round(ic, 6), "ic_vs_return": round(ic_ret, 6)}
                    report["runs"].append(book)
                    books.setdefault((exit_rule, slip, gate), []).append(book)
                    series.append(daily.assign(seed=seed, exit=exit_rule,
                                               slippage_bps=slip, min_score=gate))
        base = books[("close", slippages[0], gates[0])][-1]
        print(f"  seed {seed}  IC(peak) {ic:+.5f}  IC(ret) {ic_ret:+.5f}  "
              f"命中 {100 * base['hit_rate']:.2f}%  "
              f"收盘口径毛 {100 * base['gross_per_trade']:+.3f}%/笔")

    sample = books[("close", slippages[0], gates[0])]
    print(f"\n持仓 {args.top_k} / 候选 {signal_k} 收益分解（{len(seeds)} 个种子均值）：命中组"
          f"持到 D+2 收盘 {100 * float(np.mean([r['win_to_close'] for r in sample])):+.2f}%"
          f"  |  未命中组 {100 * float(np.mean([r['loss_to_close'] for r in sample])):+.2f}%")
    print(f"实际建仓 {float(np.mean([r['avg_positions'] for r in sample])):.2f}/{args.top_k}"
          f"  |  平均用到候选第 {float(np.mean([r['avg_fill_depth'] for r in sample])):.2f} 名"
          f"  |  候选不够建满的交易日 "
          f"{100 * float(np.mean([r['short_day_rate'] for r in sample])):.2f}%")
    print(f"交易日 {float(np.mean([r['n_days'] for r in sample])):.0f}"
          f"  |  止盈档 {args.target_ratio}  |  门槛网格 {gates}")

    print(f"\n=== {len(seeds)} 个种子 均值 ± 标准差 ===")
    print(f"{'退出假设':<16}{'滑点':>6}{'门槛':>7}{'空仓日':>8}{'净收益/笔':>17}{'CAGR':>17}"
          f"{'Sharpe':>15}{'最大回撤':>10}")
    for (exit_rule, slip, gate), runs in books.items():
        agg = {}
        for key in ("net_per_trade", "ann_return", "cagr", "sharpe", "max_drawdown",
                    "day_win_rate", "flat_day_rate"):
            values = [r[key] for r in runs]
            agg[key] = (float(np.mean(values)),
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        report["summary"][f"{exit_rule}|{slip}|{gate}"] = {
            k: {"mean": round(m, 6), "std": round(s, 6)} for k, (m, s) in agg.items()}
        rule = "触及即成交" if exit_rule == "target" else "全部按 D+2 收盘"
        print(f"{rule:<16}{slip:4.0f}bp{'—' if gate is None else f'{gate:.2f}':>7}"
              f"{100 * agg['flat_day_rate'][0]:>7.1f}%"
              f"{100 * agg['net_per_trade'][0]:>11.3f}% ±{100 * agg['net_per_trade'][1]:.3f}"
              f"{100 * agg['cagr'][0]:>11.1f}% ±{100 * agg['cagr'][1]:.1f}"
              f"{agg['sharpe'][0]:>10.2f} ±{agg['sharpe'][1]:.2f}"
              f"{100 * agg['max_drawdown'][0]:>9.1f}%")

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}")

    # Long format keyed by (seed, exit, slippage, gate): one file feeds every window
    # without re-running the model, which is the expensive half.
    pd.concat(series, ignore_index=True).to_csv(args.daily_out, index=False)
    print(f"wrote {args.daily_out}")


if __name__ == "__main__":
    main()
