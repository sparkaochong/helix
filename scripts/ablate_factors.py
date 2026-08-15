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


#: Above this many distinct values one-hot stops being an encoding and starts being noise.
MAX_ONEHOT_LEVELS = 32


def one_hot_columns(frame: pd.DataFrame, column: str,
                    max_levels: int = MAX_ONEHOT_LEVELS) -> pd.DataFrame:
    """Indicator columns for a categorical, encoded over train and test together.

    Encoding the halves separately is the quiet way to break this: a level present on only
    one side yields a column the other side lacks, and the arm is then trained and scored
    on different feature spaces.

    Deliberately not target encoding. The whole reason a categorical is worth testing here
    is that its levels have different base rates, so any encoding derived from the label
    hands the model the answer it is supposed to be predicting.
    """
    levels = frame[column].astype(str)
    n = levels.nunique()
    if n < 2:
        raise SystemExit(f"{column} 只有 {n} 个取值，编码出来是常数列")
    if n > max_levels:
        raise SystemExit(f"{column} 有 {n} 个取值，超过 {max_levels}，one-hot 会炸开")
    return pd.get_dummies(levels, prefix=column, prefix_sep="==").astype(np.float32)


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
    ap.add_argument("--factors", default="", help="JSON list of factor column names")
    ap.add_argument("--onehot", default="",
                    help="A categorical column to one-hot encode into the added arm. A "
                         "string column is invisible to a numeric feature list, so a "
                         "categorical the model should have been using can sit unread in "
                         "the table indefinitely without anything reporting it missing.")
    ap.add_argument("--features", required=True, help="JSON list of base feature column names")
    ap.add_argument("--label", default="label_d2_hit_8pct_hfq")
    ap.add_argument("--ic-target", default="label_d2_peak_return_hfq")
    ap.add_argument("--split-date", required=True, help="Train on dates <= this, test after")
    ap.add_argument("--embargo-days", type=int, default=3,
                    help="Trade dates dropped either side of the split; a D0 row resolves on D+2.")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seeds", default="7",
                    help="Comma-separated seeds. Several are worth the time: a single "
                         "run cannot tell a real gain from seed-to-seed variance, and the "
                         "deltas at stake here are smaller than that variance.")
    ap.add_argument("--report", default="ablation_report.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not args.factors and not args.onehot:
        raise SystemExit("--factors 和 --onehot 至少要给一个，否则两条臂完全相同")
    factors = []
    if args.factors:
        with open(args.factors) as fh:
            factors = json.load(fh)
    with open(args.features) as fh:
        features = json.load(fh)
    factors = [c for c in factors if c not in features]

    needed = ["trade_date", args.label, args.ic_target, *features, *factors]
    if args.onehot:
        needed.append(args.onehot)
    df = pd.read_parquet(args.input, columns=sorted(set(needed)))
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.dropna(subset=[args.label, args.ic_target])

    if args.onehot:
        dummies = one_hot_columns(df, args.onehot)
        clash = [c for c in dummies.columns if c in features]
        if clash:
            raise SystemExit(f"one-hot 列名和已有特征撞车：{clash}")
        # Base rates per level are the reason to run this at all, so print them: if they
        # are flat, the arm cannot help and the run is not worth the GPU time.
        rates = df.groupby(args.onehot)[args.label].agg(["mean", "size"]).sort_values("mean")
        print(f"{args.onehot} 各取值的基准命中率：")
        for level, row in rates.iterrows():
            print(f"  {100 * row['mean']:6.2f}%  ({int(row['size']):>7,} 行)  {level}")
        df = pd.concat([df.drop(columns=[args.onehot]), dummies], axis=1)
        factors = factors + list(dummies.columns)

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

    name, _ = build_model(seeds[0])
    print(f"model: {name} | seeds: {seeds}")

    report = {"model": name, "train_end": train_end, "test_start": test_start,
              "n_features": len(features), "n_factors": len(factors),
              "seeds": seeds, "runs": [], "arms": {}}

    collected: dict[str, dict[str, list[float]]] = {
        "base": {"ic_mean": [], "icir": [], "hit_rate": [], "lift": []},
        "base+factors": {"ic_mean": [], "icir": [], "hit_rate": [], "lift": []},
    }

    for seed in seeds:
        for arm, columns in (("base", features), ("base+factors", features + factors)):
            _, model = build_model(seed)
            model.fit(train[columns].to_numpy(dtype=np.float32), train[args.label].to_numpy())
            scored = test[["trade_date", args.label, args.ic_target]].copy()
            scored["score"] = model.predict_proba(test[columns].to_numpy(dtype=np.float32))[:, 1]

            ic = summarize(daily_ic(scored, "score", args.ic_target))
            hit = top_k_hit_rate(scored, "score", args.label, args.top_k)
            report["runs"].append({"seed": seed, "arm": arm, "ic": ic, "hit": hit})
            for key, value in (("ic_mean", ic["ic_mean"]), ("icir", ic["icir"]),
                               ("hit_rate", hit["hit_rate"]), ("lift", hit["lift"])):
                collected[arm][key].append(value)
            print(f"  seed {seed} {arm:13s} IC {ic['ic_mean']:+.5f}  "
                  f"ICIR {ic['icir']:+.3f}  top{args.top_k} {100 * hit['hit_rate']:.2f}%  "
                  f"lift {hit['lift']:.3f}")

    print(f"\n=== 均值 ± 标准差（{len(seeds)} 个种子）===")
    for arm in ("base", "base+factors"):
        stats_ = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                  for k, v in collected[arm].items()}
        report["arms"][arm] = {k: {"mean": round(m, 6), "std": round(s, 6)}
                               for k, (m, s) in stats_.items()}
        print(f"  {arm:13s} IC {stats_['ic_mean'][0]:+.5f}±{stats_['ic_mean'][1]:.5f}  "
              f"top{args.top_k} {100 * stats_['hit_rate'][0]:.2f}%±{100 * stats_['hit_rate'][1]:.2f}  "
              f"lift {stats_['lift'][0]:.3f}±{stats_['lift'][1]:.3f}")

    print("\n=== 增量 (base+factors - base) ===")
    report["delta"] = {}
    for key in ("ic_mean", "icir", "hit_rate", "lift"):
        a = np.array(collected["base"][key])
        b = np.array(collected["base+factors"][key])
        diff = b - a
        noise = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
        report["delta"][key] = {"mean": round(float(diff.mean()), 6),
                                "base_seed_std": round(noise, 6)}
        verdict = "" if not np.isfinite(noise) else (
            "  <- 在噪声内" if abs(diff.mean()) <= noise else "  <- 超出种子噪声")
        print(f"  {key:10s} {diff.mean():+.5f}   (基线种子间标准差 {noise:.5f}){verdict}")

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
