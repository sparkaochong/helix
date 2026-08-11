#!/usr/bin/env python3
"""How does a 431-day track look when you only get to see 28 days of it?

A live track record is quoted over the window it happens to have run. Cumulative return,
positive-day rate and especially **maximum drawdown are all functions of that length** --
drawdown is a minimum over the observation window, so it can only deepen as the window
grows. Comparing a 28-day drawdown against a 431-day one is not a comparison between two
strategies; it is a comparison between two window lengths.

This script removes that confound the only honest way available: cut the long series into
windows of the short one's length and ask where the short track's numbers sit inside the
resulting distribution. If a large fraction of 28-day slices of your own curve clear the
reference's headline, the reference is inside your noise and there is nothing to explain.
If none do, the gap is real and worth chasing.

Two things this deliberately does not do:

* **No p-values.** Rolling windows overlap by construction -- 28-day windows one day apart
  share 27 days -- so the windows are nowhere near independent and any test built on
  counting them would be badly overconfident. The output is a description of a
  distribution, not an inference from a sample.
* **No claim that a high percentile is luck.** It says how often the number occurs in a
  strategy already known to be this good, which bounds how much evidence a single
  short window carries either way.

Reads the long-format CSV that `backtest_argus.py --daily-out` writes. Standalone: numpy
and pandas only.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def window_metrics(returns: np.ndarray, window: int) -> pd.DataFrame:
    """Cumulative return, positive-day rate and max drawdown for every rolling window."""
    if len(returns) < window:
        return pd.DataFrame(columns=["start", "cum_return", "pos_day_rate", "max_drawdown"])
    panes = sliding_window_view(returns, window)
    # Prepend the pre-window value of 1.0: a window that opens by falling has drawn down
    # from its own starting capital, and a running max that starts at day 1 would miss it.
    growth = np.concatenate(
        [np.ones((len(panes), 1)), np.cumprod(1.0 + panes, axis=1)], axis=1)
    peak = np.maximum.accumulate(growth, axis=1)
    return pd.DataFrame({
        "start": np.arange(len(panes)),
        "cum_return": growth[:, -1] - 1.0,
        "pos_day_rate": (panes > 0).mean(axis=1),
        "max_drawdown": (growth / peak - 1.0).min(axis=1),
    })


def select_curve(df: pd.DataFrame, exit_rule: str, slippage: float,
                 min_score: str) -> pd.DataFrame:
    """Narrow the long CSV down to exactly one equity curve per seed.

    The file holds every (exit, slippage, gate) variant the backtest swept, so a partial
    filter silently concatenates several curves into one series -- three gates would look
    like a track three times as long, with a fabricated jump wherever one variant ends and
    the next begins. Whatever survives here is checked to have one row per (seed, date).
    """
    # The ungated book writes an empty min_score, which reads back as NaN and compares
    # unequal to everything -- including itself -- so it needs isna() rather than ==.
    gate = (df["min_score"].isna() if not min_score.strip()
            else df["min_score"] == float(min_score))
    sel = df[(df["exit"] == exit_rule) & (df["slippage_bps"] == slippage) & gate]
    if sel.empty:
        raise SystemExit(f"没有 exit={exit_rule} slippage={slippage} "
                         f"门槛={min_score or '无'} 的行")
    if sel.duplicated(subset=["seed", "date"]).any():
        raise SystemExit("同一个种子的同一天出现多行，筛选没有唯一确定一条曲线")
    return sel


def describe(values: np.ndarray, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    return {f"p{int(100 * q)}": float(np.quantile(values, q)) for q in quantiles} | {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Daily CSV from backtest_argus.py.")
    ap.add_argument("--window", type=int, default=28,
                    help="Window length in trading days. 28 is what a 71.4%%/60.7%% pair "
                         "of positive-day rates back-solves to (20/28 and 17/28).")
    ap.add_argument("--exit", default="close", choices=("close", "target"))
    ap.add_argument("--slippage", type=float, default=10.0)
    ap.add_argument("--min-score", default="",
                    help="Which conviction gate to slice, blank for the ungated book. The "
                         "daily CSV holds every gate in one file, so without this the "
                         "variants would be concatenated into one impossibly long series.")
    # Defaults are the reference track's headline: +25.00% cumulative, 20/28 positive days,
    # -7.61% max drawdown.
    ap.add_argument("--ref-return", type=float, default=0.25)
    ap.add_argument("--ref-pos-rate", type=float, default=20.0 / 28.0)
    ap.add_argument("--ref-drawdown", type=float, default=-0.0761)
    args = ap.parse_args()

    # Trade dates are YYYYMMDD digits; left to itself pandas reads them as floats and the
    # window's end date prints as 20250103.0.
    df = pd.read_csv(args.input, dtype={"date": str})
    sel = select_curve(df, args.exit, args.slippage, args.min_score)

    per_seed = {}
    for seed, g in sel.groupby("seed", sort=True):
        g = g.sort_values("date")
        per_seed[int(seed)] = window_metrics(g["portfolio_return"].to_numpy(dtype=float),
                                             args.window).assign(seed=int(seed),
                                                                 date=g["date"].to_numpy()[
                                                                     args.window - 1:])
    windows = pd.concat(per_seed.values(), ignore_index=True)
    if windows.empty:
        raise SystemExit(f"序列不足 {args.window} 个交易日，切不出窗口")

    n_days = len(sel) // len(per_seed)
    print(f"=== exit={args.exit} 滑点={args.slippage:.0f}bp "
          f"门槛={args.min_score or '无'} 窗口={args.window} 交易日 ===")
    print(f"原始序列 {n_days} 天 × {len(per_seed)} 个种子 → {len(windows):,} 个重叠窗口")
    print("（窗口高度重叠，下面是分布描述，不是独立样本，不做显著性检验）\n")

    rows = [("累计收益", "cum_return", 100.0, "%"),
            ("正收益日", "pos_day_rate", 100.0, "%"),
            ("最大回撤", "max_drawdown", 100.0, "%")]
    print(f"{'指标':<10}{'p5':>9}{'p25':>9}{'中位':>9}{'p75':>9}{'p95':>9}{'最好':>9}")
    for name, col, scale, unit in rows:
        d = describe(windows[col].to_numpy())
        # "Best" is the maximum for all three: for drawdown that is the shallowest window.
        print(f"{name:<10}" + "".join(f"{scale * d[k]:>8.1f}{unit}"
                                      for k in ("p5", "p25", "p50", "p75", "p95"))
              + f"{scale * d['max']:>8.1f}{unit}")

    ref = {"cum_return": (args.ref_return, "累计收益不低于对照"),
           "pos_day_rate": (args.ref_pos_rate, "正收益日不低于对照"),
           "max_drawdown": (args.ref_drawdown, "回撤不深于对照")}
    print(f"\n对照基准：累计 {100 * args.ref_return:+.2f}%  "
          f"正收益日 {100 * args.ref_pos_rate:.1f}%  回撤 {100 * args.ref_drawdown:.2f}%")
    meets = pd.Series(True, index=windows.index)
    for col, (threshold, label) in ref.items():
        ok = windows[col] >= threshold
        meets &= ok
        print(f"  {label:<18} {100 * float(ok.mean()):6.2f}% 的窗口达到")
    print(f"  {'三条同时满足':<18} {100 * float(meets.mean()):6.2f}% 的窗口达到")

    best = windows.loc[windows["cum_return"].idxmax()]
    print(f"\n最好的 {args.window} 天窗口（种子 {int(best['seed'])}，截至 {best['date']}）："
          f"累计 {100 * best['cum_return']:+.2f}%  "
          f"正收益日 {100 * best['pos_day_rate']:.1f}%  "
          f"回撤 {100 * best['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()
