#!/usr/bin/env python3
"""Model ablation: does adding the mined factor columns actually help?

Residual IC says a factor carries information the base columns do not span *linearly*.
That is necessary but not sufficient -- a gradient-boosted model already captures
nonlinear structure the linear projection leaves behind, so a factor can show residual
IC and still be redundant to the model that will consume it.

This is the deciding experiment: train the same model on the existing features, then on
the existing features plus the factors, and compare out-of-sample. Everything else is
held fixed -- same rows, same split, same hyper-parameters, same seed.

Runs on the training host; needs numpy / pandas / pyarrow / scikit-learn (xgboost or
lightgbm are used when available, otherwise HistGradientBoosting from sklearn).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def daily_ic(frame: pd.DataFrame, score: str, target: str, min_n: int = 30) -> np.ndarray:
    """Per-date Spearman IC of a score column against the target."""
    out = []
    for _, g in frame.groupby("trade_date", sort=True):
        if len(g) < min_n:
            continue
        out.append(stats.spearmanr(g[score], g[target]).statistic)
    return np.asarray(out, dtype=float)


def summarize(ic: np.ndarray) -> dict:
    finite = ic[np.isfinite(ic)]
    if finite.size == 0:
        return {}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1))
    return {
        "ic_mean": round(mean, 6),
        "icir": round(mean / std, 4) if std > 0 else float("nan"),
        "positive_rate": round(float((finite > 0).mean()), 4),
        "n_days": int(finite.size),
    }


def top_k_hit_rate(frame: pd.DataFrame, score: str, label: str, k: int) -> dict:
    hits, base = [], []
    for _, g in frame.groupby("trade_date", sort=True):
        if len(g) < k:
            continue
        picked = g.nlargest(k, score)
        hits.append(picked[label].mean())
        base.append(g[label].mean())
    if not hits:
        return {}
    hits, base = np.asarray(hits), np.asarray(base)
    return {
        "hit_rate": round(float(hits.mean()), 5),
        "base_rate": round(float(base.mean()), 5),
        "lift": round(float(hits.mean() / max(base.mean(), 1e-12)), 4),
        "n_days": int(len(hits)),
    }


def build_model(seed: int):
    """Prefer the gradient-boosting library the host actually has installed."""
    try:
        from xgboost import XGBClassifier

        return "xgboost", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed, n_jobs=-1, tree_method="hist",
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        return "lightgbm", LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier

    return "sklearn-hgb", HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.05, random_state=seed
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Parquet already carrying the factor columns")
    ap.add_argument("--factors", required=True, help="JSON list of factor column names")
    ap.add_argument("--features", required=True, help="JSON list of base feature column names")
    ap.add_argument("--label", default="label_d2_hit_8pct")
    ap.add_argument("--ic-target", default="label_d2_peak_return")
    ap.add_argument("--split-date", required=True, help="Train on dates <= this, test after")
    ap.add_argument("--embargo-days", type=int, default=3,
                    help="Trade dates dropped either side of the split; a D0 row resolves on D+2.")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", default="ablation_report.json")
    args = ap.parse_args()

    with open(args.factors) as fh:
        factors = json.load(fh)
    with open(args.features) as fh:
        features = json.load(fh)
    factors = [c for c in factors if c not in features]

    needed = ["trade_date", args.label, args.ic_target, *features, *factors]
    df = pd.read_parquet(args.input, columns=sorted(set(needed)))
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.dropna(subset=[args.label, args.ic_target])

    dates = np.array(sorted(df["trade_date"].unique()))
    cut = int(np.searchsorted(dates, args.split_date, "right"))
    train_end = dates[max(cut - 1, 0)]
    test_start = dates[min(cut + args.embargo_days, len(dates) - 1)]
    train = df[df["trade_date"] <= train_end]
    test = df[df["trade_date"] >= test_start]
    print(f"train {train['trade_date'].min()}~{train_end} ({len(train):,} rows) | "
          f"embargo {args.embargo_days}d | test {test_start}~{test['trade_date'].max()} "
          f"({len(test):,} rows)")
    print(f"train positive rate {train[args.label].mean():.4f} | "
          f"test positive rate {test[args.label].mean():.4f}")

    name, _ = build_model(args.seed)
    print(f"model: {name}")

    report = {"model": name, "train_end": train_end, "test_start": test_start,
              "n_features": len(features), "n_factors": len(factors), "arms": {}}

    for arm, columns in (("base", features), ("base+factors", features + factors)):
        _, model = build_model(args.seed)
        model.fit(train[columns].to_numpy(dtype=np.float32), train[args.label].to_numpy())
        scored = test[["trade_date", args.label, args.ic_target]].copy()
        scored["score"] = model.predict_proba(test[columns].to_numpy(dtype=np.float32))[:, 1]

        entry = {
            "n_columns": len(columns),
            "ic": summarize(daily_ic(scored, "score", args.ic_target)),
            "hit": top_k_hit_rate(scored, "score", args.label, args.top_k),
        }
        report["arms"][arm] = entry
        print(f"\n{arm}: {len(columns)} columns")
        print(f"  IC   {entry['ic'].get('ic_mean'):+.5f}  ICIR {entry['ic'].get('icir'):+.3f}  "
              f"pos {100 * entry['ic'].get('positive_rate', 0):.1f}%")
        print(f"  top{args.top_k} 命中率 {100 * entry['hit'].get('hit_rate', 0):.2f}%  "
              f"base {100 * entry['hit'].get('base_rate', 0):.2f}%  "
              f"lift {entry['hit'].get('lift'):.3f}")

    a, b = report["arms"]["base"], report["arms"]["base+factors"]
    report["delta"] = {
        "ic_mean": round(b["ic"]["ic_mean"] - a["ic"]["ic_mean"], 6),
        "icir": round(b["ic"]["icir"] - a["ic"]["icir"], 4),
        "hit_rate": round(b["hit"]["hit_rate"] - a["hit"]["hit_rate"], 5),
        "lift": round(b["hit"]["lift"] - a["hit"]["lift"], 4),
    }
    print("\n=== 增量 (base+factors - base) ===")
    for k, v in report["delta"].items():
        print(f"  {k:10s} {v:+.5f}")

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
