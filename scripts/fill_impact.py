#!/usr/bin/env python3
"""How much of the reported top-k hit rate is orders you could never have filled?

`check_fillability.py` establishes that 0.475% of argus_quant rows open D+1 at their
up-limit and that the dataset labels them anyway, at a 37% hit rate against a 15% pool
base. Diluted across the whole table that shifts the base rate by only -0.1pp, which
looks harmless.

It is not necessarily harmless where it matters. A model ranks on exactly the momentum
that produces a limit-up open, so those rows concentrate in the top of the book. The
question this script answers is what share of the **top-k picks** they are, and what the
hit rate becomes once they are treated as unfilled orders rather than as wins.

Three views of the same predictions:

* ``as-is``        -- top k by score, every pick counted. Reproduces the published number.
* ``fill-aware``   -- top k by score, unfillable picks dropped as orders that never
                      filled. This is what the strategy actually earns.
* ``queue-deeper`` -- top k among fillable rows only, i.e. what you would get by
                      submitting past the failures. An upper bound: you cannot know at
                      D0 close which names will gap to the limit.

Crossed with two training regimes, because an undefined label is bad training data too:

* ``train-as-is``  -- unfillable rows left in, as they are today.
* ``train-clean``  -- unfillable rows dropped from training entirely.

Standalone: numpy / pandas / pyarrow / xgboost, so it runs on the training host.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from fillability import REQUIRED_COLUMNS, unfillable_mask
from scipy import stats

#: Same prefix rule as helix/data/event_table.py. Enumerating outcome columns by hand is
#: how label_d2_hit_5pct got into a feature set here once and produced a fake IC of 0.63.
LABEL_PREFIXES = ("label", "target", "y_", "fwd_", "future_")
META_COLUMNS = ("trade_date", "stock_code", "strategy_name", "sector_name")


def feature_columns(path: str) -> list[str]:
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    names = [
        n for n in schema.names
        if n not in META_COLUMNS
        and not any(n.lower().startswith(p) for p in LABEL_PREFIXES)
        and any(k in str(schema.field(n).type) for k in ("int", "float", "double"))
    ]
    leaked = [n for n in names if any(n.lower().startswith(p) for p in LABEL_PREFIXES)]
    assert not leaked, leaked
    return names


#: Shared by the classifier and the regressor so that swapping the objective is the only
#: difference between them. Duplicating these would make "same model, different target"
#: an assertion instead of something you can read off the file.
MODEL_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=-1, tree_method="hist",
)


def build_model(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(**MODEL_PARAMS, eval_metric="logloss", random_state=seed)


def build_regressor(seed: int):
    """Same trees, squared-error objective -- for ranking on a return instead of a hit."""
    from xgboost import XGBRegressor

    return XGBRegressor(**MODEL_PARAMS, objective="reg:squarederror",
                        eval_metric="rmse", random_state=seed)


def daily_ic(frame: pd.DataFrame, score: str, target: str, min_n: int = 30) -> float:
    out = [stats.spearmanr(g[score], g[target]).statistic
           for _, g in frame.groupby("trade_date", sort=True) if len(g) >= min_n]
    finite = np.asarray([v for v in out if np.isfinite(v)])
    return float(finite.mean()) if finite.size else float("nan")


def evaluate(scored: pd.DataFrame, label: str, k: int) -> dict[str, dict]:
    """Per-date top-k under the three fill conventions, then averaged across dates."""
    rows: dict[str, list[list[float]]] = {v: [] for v in ("as-is", "fill-aware", "queue-deeper")}
    unfilled_share, base_all, base_fillable = [], [], []

    for _, g in scored.groupby("trade_date", sort=True):
        if len(g) < k:
            continue
        base_all.append(g[label].mean())
        ok = g[~g["unfillable"]]
        base_fillable.append(ok[label].mean() if len(ok) else np.nan)

        picked = g.nlargest(k, "score")
        unfilled_share.append(picked["unfillable"].mean())
        rows["as-is"].append([picked[label].mean(), float(k)])

        filled = picked[~picked["unfillable"]]
        rows["fill-aware"].append(
            [filled[label].mean(), float(len(filled))] if len(filled) else [np.nan, 0.0])

        deeper = ok.nlargest(k, "score") if len(ok) >= k else ok
        rows["queue-deeper"].append(
            [deeper[label].mean(), float(len(deeper))] if len(deeper) else [np.nan, 0.0])

    base_all_mean = float(np.nanmean(base_all))
    base_ok_mean = float(np.nanmean(base_fillable))
    out: dict[str, dict] = {
        "unfillable_share_of_topk": round(float(np.nanmean(unfilled_share)), 6),
        "base_rate_pool": round(base_all_mean, 6),
        "base_rate_fillable": round(base_ok_mean, 6),
        "n_days": int(len(base_all)),
    }
    for view, records in rows.items():
        arr = np.asarray(records, dtype=float)
        hit = float(np.nanmean(arr[:, 0]))
        base = base_all_mean if view == "as-is" else base_ok_mean
        out[view] = {
            "hit_rate": round(hit, 6),
            "lift": round(hit / max(base, 1e-12), 4),
            "avg_positions": round(float(arr[:, 1].mean()), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", default="label_d2_hit_8pct")
    ap.add_argument("--ic-target", default="label_d2_peak_return")
    ap.add_argument("--split-date", default="2024-09-04")
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seeds", default="7,13,42")
    ap.add_argument("--report", default="fill_impact_report.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    features = feature_columns(args.input)
    print(f"features {len(features)}")

    # The flag is computed here rather than joined in: 514 rows share
    # (stock_code, trade_date) even including strategy_name, so no join is one-to-one.
    needed = sorted({"trade_date", args.label, args.ic_target, *REQUIRED_COLUMNS, *features})
    df = pd.read_parquet(args.input, columns=needed)
    df["trade_date"] = df["trade_date"].astype(str)
    df["unfillable"] = unfillable_mask(df)
    df = df.dropna(subset=[args.label, args.ic_target])

    dates = np.array(sorted(df["trade_date"].unique()))
    cut = int(np.searchsorted(dates, args.split_date, "right"))
    train_end = dates[max(cut - 1, 0)]
    test_start = dates[min(cut + args.embargo_days, len(dates) - 1)]
    train_all = df[df["trade_date"] <= train_end]
    test = df[df["trade_date"] >= test_start]
    print(f"train ~{train_end} ({len(train_all):,}) | test {test_start}~ ({len(test):,}) | "
          f"unfillable in test {test['unfillable'].mean():.4%}")

    regimes = {
        "train-as-is": train_all,
        "train-clean": train_all[~train_all["unfillable"]],
    }
    report = {"features": len(features), "train_end": train_end, "test_start": test_start,
              "seeds": seeds, "top_k": args.top_k, "runs": [], "summary": {}}

    collected: dict[tuple[str, str], list[float]] = {}
    for regime, train in regimes.items():
        print(f"\n### {regime} ({len(train):,} train rows)")
        for seed in seeds:
            model = build_model(seed)
            model.fit(train[features].to_numpy(dtype=np.float32),
                      train[args.label].to_numpy())
            scored = test[["trade_date", args.label, args.ic_target, "unfillable"]].copy()
            scored["score"] = model.predict_proba(
                test[features].to_numpy(dtype=np.float32))[:, 1]

            res = evaluate(scored, args.label, args.top_k)
            res["ic_mean"] = round(daily_ic(scored, "score", args.ic_target), 6)
            report["runs"].append({"regime": regime, "seed": seed, **res})
            for view in ("as-is", "fill-aware", "queue-deeper"):
                collected.setdefault((regime, view), []).append(res[view]["hit_rate"])
            collected.setdefault((regime, "unfilled_share"), []).append(
                res["unfillable_share_of_topk"])
            print(f"  seed {seed}  top{args.top_k} 中买不进 "
                  f"{100 * res['unfillable_share_of_topk']:.2f}%  |  "
                  f"as-is {100 * res['as-is']['hit_rate']:.2f}%  "
                  f"fill-aware {100 * res['fill-aware']['hit_rate']:.2f}%  "
                  f"queue-deeper {100 * res['queue-deeper']['hit_rate']:.2f}%")

    print(f"\n=== 均值 ± 标准差（{len(seeds)} 个种子）===")
    for (regime, view), values in collected.items():
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        report["summary"][f"{regime}|{view}"] = {"mean": round(mean, 6), "std": round(std, 6)}
        print(f"  {regime:12s} {view:15s} {100 * mean:6.2f}% ± {100 * std:.2f}")

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
