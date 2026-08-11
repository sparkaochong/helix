#!/usr/bin/env python3
"""Audit whether the argus_quant label admits entries that cannot actually be filled.

The label says `high[D+2] >= open[D+1] * 1.08`. That is only tradeable if you can buy at
`open[D+1]`, which you cannot when D+1 opens at its up-limit. Those rows are undefined,
not wins, and counting them as wins inflates every hit rate downstream.

The reconstruction lives in `fillability.py`; this script measures it and, critically,
validates it two independent ways before believing any of the numbers:

1. the limit base backed out of the gap must agree with the one backed out of
   `label_px_d1_close / (1 + label_d1_pct_chg/100)`. Both recover `pre_close[D+1]` rather
   than the raw D0 close, which is what the exchange quotes limits against;
2. a row that opens at its limit cannot trade higher that day, so `open[D+1] ==
   high[D+1]` must hold for essentially every flagged row. If it does not, the assumed
   board rate is wrong and every count below is wrong with it.

Standalone: numpy / pandas / pyarrow only, so it runs on the training host.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from fillability import PRICE_EPS, st_suspect_count, unfillable_mask

COLUMNS = [
    "stock_code", "trade_date",
    "label_px_d1_open", "label_px_d1_high", "label_px_d1_low", "label_px_d1_close",
    "label_open_gap", "label_d1_pct_chg",
    "label_d2_hit_8pct",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", default="label_d2_hit_8pct")
    ap.add_argument("--split-date", default="2024-09-10",
                    help="Report the out-of-sample slice separately from the whole table.")
    ap.add_argument("--report", default="fillability_report.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.input, columns=COLUMNS)
    df["trade_date"] = df["trade_date"].astype(str)
    n_all = len(df)
    print(f"rows {n_all:,}  dates {df['trade_date'].nunique():,}  "
          f"{df['trade_date'].min()} ~ {df['trade_date'].max()}")

    label_null = int(df[args.label].isna().sum())
    print(f"\nNaN: {args.label} {label_null:,} | "
          f"px_d1_open {int(df['label_px_d1_open'].isna().sum()):,} | "
          f"open_gap {int(df['label_open_gap'].isna().sum()):,}")

    open_d1 = df["label_px_d1_open"].to_numpy(dtype=float)
    high_d1 = df["label_px_d1_high"].to_numpy(dtype=float)
    low_d1 = df["label_px_d1_low"].to_numpy(dtype=float)
    close_d1 = df["label_px_d1_close"].to_numpy(dtype=float)
    gap = df["label_open_gap"].to_numpy(dtype=float)
    pct = df["label_d1_pct_chg"].to_numpy(dtype=float)

    # ------------------------------------------------------- 校验一：两条推导 --
    from_gap = open_d1 / (1.0 + gap)
    from_pct = close_d1 / (1.0 + pct / 100.0)
    both = np.isfinite(from_gap) & np.isfinite(from_pct) & (from_pct > 0)
    rel = np.abs(from_gap[both] - from_pct[both]) / from_pct[both]
    print(f"\npre_close[D+1] 两条推导路径一致性: median {np.median(rel):.2e}  "
          f"p99 {np.quantile(rel, 0.99):.2e}  >1e-3 的比例 {(rel > 1e-3).mean():.4%}")

    # --------------------------------------------- 校验二：开盘涨停则 open==high --
    unfillable = unfillable_mask(df)
    flagged = unfillable & np.isfinite(high_d1)
    open_is_high = np.isclose(open_d1[flagged], high_d1[flagged], atol=PRICE_EPS)
    print(f"自校验 open[D+1] == high[D+1]（被标记样本应几乎全部满足）: "
          f"{open_is_high.mean():.4%} of {int(flagged.sum()):,}")

    flat = (np.isclose(open_d1, high_d1, atol=PRICE_EPS)
            & np.isclose(open_d1, low_d1, atol=PRICE_EPS)
            & np.isclose(open_d1, close_d1, atol=PRICE_EPS))
    st_suspects = st_suspect_count(df)
    print(f"一字板（open==high==low==close）: {int(flat.sum()):,} ({flat.mean():.4%})")
    print(f"gap≈+5%（可能是 ST，本表按 10% 处理会漏标）: {st_suspects:,}  "
          f"-> 买不进的上界 {int(unfillable.sum()) + st_suspects:,}")

    # 上界：把 ST 疑似样本也算成买不进，看结论会不会翻转。
    upper = unfillable | (np.abs(gap - 0.05) < 0.002)

    label = df[args.label].to_numpy(dtype=float)
    report: dict = {
        "rows": int(n_all),
        "label_nan": label_null,
        "limit_base_disagree_rate": round(float((rel > 1e-3).mean()), 6),
        "open_equals_high_among_flagged": round(float(open_is_high.mean()), 6),
        "flat_limit_rows": int(flat.sum()),
        "st_suspects": st_suspects,
        "slices": {},
    }

    for tag, rows in (("全样本", np.ones(n_all, dtype=bool)),
                      ("样本外", (df["trade_date"] >= args.split_date).to_numpy())):
        sub_label = label[rows]
        observed = np.isfinite(sub_label)
        sub_bad = unfillable[rows]
        bad_obs = sub_bad & observed

        hit_bad = float(np.mean(sub_label[bad_obs])) if bad_obs.any() else float("nan")
        hit_ok = float(np.mean(sub_label[observed & ~sub_bad]))
        hit_all = float(np.mean(sub_label[observed]))
        hit_ok_upper = float(np.mean(sub_label[observed & ~upper[rows]]))
        nan_share = float((~observed)[sub_bad].mean()) if sub_bad.any() else float("nan")

        print(f"\n=== {tag} ({int(rows.sum()):,} 行) ===")
        print(f"  D+1 开盘即涨停（买不进）      {int(sub_bad.sum()):,}  占 {sub_bad.mean():.4%}")
        print(f"  这批的标签为 NaN 的比例        {nan_share:.4%}"
              "   <- 数据集若已剔除，这里应接近 100%")
        print(f"  这批的命中率                  {hit_bad:.4%}")
        print(f"  可成交样本的命中率            {hit_ok:.4%}")
        print(f"  当前口径的整体命中率          {hit_all:.4%}")
        print(f"  剔除后整体命中率              {hit_ok:.4%}   "
              f"(变化 {100 * (hit_ok - hit_all):+.3f}pp)")
        print(f"  连 ST 疑似一并剔除            {hit_ok_upper:.4%}   "
              f"(变化 {100 * (hit_ok_upper - hit_all):+.3f}pp)")

        report["slices"][tag] = {
            "rows": int(rows.sum()),
            "unfillable": int(sub_bad.sum()),
            "unfillable_share": round(float(sub_bad.mean()), 6),
            "unfillable_label_nan_share": round(nan_share, 6) if sub_bad.any() else None,
            "hit_rate_unfillable": round(hit_bad, 6),
            "hit_rate_fillable": round(hit_ok, 6),
            "hit_rate_asis": round(hit_all, 6),
            "hit_rate_fillable_upper_bound": round(hit_ok_upper, 6),
            "base_rate_shift_pp": round(100 * (hit_ok - hit_all), 4),
        }

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
