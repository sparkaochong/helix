#!/usr/bin/env python3
"""Does the argus_quant label handle a suspended D+1 or D+2, or silently score it 0?

`label_d2_hit_8pct` has zero NaN across all 617,416 rows. A label that correctly handles
suspension should not: if D+1 is halted you cannot enter, and if D+2 is halted you cannot
observe the outcome. Either the table excludes those rows upstream, or it treats an
unobservable outcome as a miss -- which would understate the hit rate.

Three checks, in the order that matters:

1. **Overlap.** Map every row's D0 onto the exchange calendar, take the next two trading
   days, and count how many are full-day halts according to Tushare `suspend_d`.

2. **Positive control.** A count of zero is also what a broken join returns, so the same
   lookup runs at offsets the pipeline has no reason to filter (D-10 ... D+20). If those
   are zero too, the join is broken and check 1 means nothing.

3. **Independent reconstruction.** Fetch raw bars for a couple of windows and compare the
   table's own price columns against them on the *calendar* D+1/D+2. Prices that match
   prove the window is not shifted -- a pipeline that skipped halted days would land on a
   later bar. Recomputing the label from those bars then checks the formula end to end,
   under both raw and back-adjusted prices, which is also how the missing dividend
   adjustment shows up.

Needs a Tushare token (`HELIX_TUSHARE_TOKEN`); downloads are cached under `--cache` so
reruns cost nothing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from helix.config import Config
from helix.data.tushare_source import TushareSource

#: Offsets probed by the positive control. D+1/D+2 are the hypothesis; the rest exist to
#: prove the lookup can find a halt at all.
CONTROL_OFFSETS = (-10, -5, -2, -1, 0, 1, 2, 3, 5, 10, 20)

PRICE_COLUMNS = ("label_px_d1_open", "label_px_d2_high", "label_px_d2_close")
RETRIES = 4
RETRY_SLEEP = 15.0
#: Two prices agree if they are within this relative distance. Quotes carry two decimals,
#: so anything real is either exact or off by percent.
PRICE_TOL = 5e-4


def _call(src: TushareSource, api: str, **kwargs) -> pd.DataFrame:
    for attempt in range(RETRIES):
        try:
            return src._call(api, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the API raises bare Exception on throttle
            print(f"  {api} {kwargs} retry {attempt}: {type(exc).__name__} {str(exc)[:80]}",
                  flush=True)
            time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"{api} failed after {RETRIES} attempts: {kwargs}")


def _cached(cache: Path, name: str, build) -> pd.DataFrame:
    path = cache / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = build()
    cache.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def load_calendar(src: TushareSource, cache: Path, start: str, end: str) -> np.ndarray:
    frame = _cached(cache, f"calendar_{start}_{end}", lambda: _call(
        src, "trade_cal", exchange="SSE", start_date=start, end_date=end, is_open="1"))
    return np.array(sorted(frame["cal_date"].astype(str)))


def load_halts(src: TushareSource, cache: Path, start: str, end: str) -> pd.DataFrame:
    """Full-day suspensions as (ts_code, trade_date) rows.

    `suspend_d` emits one row per suspended trading day, so no interval reconstruction is
    needed. Rows carrying a `suspend_timing` are intraday halts -- the stock traded that
    day, so an entry or an observation was still possible, and they are excluded here.
    """
    def build() -> pd.DataFrame:
        months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
        frames = []
        for i, month in enumerate(months):
            lo = month.strftime("%Y%m%d")
            hi = (months[i + 1] - pd.Timedelta(days=1)).strftime("%Y%m%d") \
                if i + 1 < len(months) else end
            frames.append(_call(src, "suspend_d", start_date=lo, end_date=hi))
            time.sleep(0.3)
        return pd.concat(frames, ignore_index=True)

    raw = _cached(cache, f"suspend_{start}_{end}", build)
    raw["trade_date"] = raw["trade_date"].astype(str)
    return raw[(raw["suspend_type"] == "S") & (raw["suspend_timing"].isna())]


def load_bars(src: TushareSource, cache: Path, dates: list[str]) -> pd.DataFrame:
    """Raw daily bars plus adjustment factors, one API pair per trading day."""
    def build() -> pd.DataFrame:
        bars, adjs = [], []
        for i, date in enumerate(dates):
            bars.append(_call(src, "daily", trade_date=date))
            adjs.append(_call(src, "adj_factor", trade_date=date))
            if i % 20 == 0:
                print(f"  bars {i}/{len(dates)} {date}", flush=True)
        left = pd.concat(bars, ignore_index=True)
        right = pd.concat(adjs, ignore_index=True)
        for frame in (left, right):
            frame["trade_date"] = frame["trade_date"].astype(str)
        return left.merge(right[["ts_code", "trade_date", "adj_factor"]],
                          on=["ts_code", "trade_date"], how="left")

    return _cached(cache, "bars_" + "_".join((dates[0], dates[-1], str(len(dates)))), build)


def load_table(path: str, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["stock_code", "trade_date", label, *PRICE_COLUMNS])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y%m%d")
    return frame


def shifted(calendar: np.ndarray, dates: np.ndarray, offset: int) -> np.ndarray:
    idx = np.searchsorted(calendar, dates)
    if not (calendar[idx] == dates).all():
        raise ValueError("some D0 dates are not SSE trading days; wrong calendar range?")
    return calendar[np.clip(idx + offset, 0, len(calendar) - 1)]


def parse_windows(spec: str) -> list[tuple[str, str]]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition(":")
        out.append((lo, hi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", default="label_d2_hit_8pct")
    ap.add_argument("--cache", default="data/raw/suspension_cache")
    ap.add_argument("--verify-windows", default="20240902:20241031,20220301:20220429",
                    help="Date ranges to pull full bars for, comma separated lo:hi.")
    ap.add_argument("--report", default="suspension_report.json")
    args = ap.parse_args()

    df = load_table(args.input, args.label)
    d0 = df["trade_date"].to_numpy()
    code = df["stock_code"].to_numpy()
    print(f"rows {len(df):,}  {d0.min()} ~ {d0.max()}  stocks {df['stock_code'].nunique():,}")

    src = TushareSource(Config.load(None))
    cache = Path(args.cache)
    # The calendar has to reach past the table on both sides: the control probes D-10..D+20.
    lo = (pd.Timestamp(d0.min()) - pd.Timedelta(days=90)).strftime("%Y%m%d")
    hi = (pd.Timestamp(d0.max()) + pd.Timedelta(days=90)).strftime("%Y%m%d")
    calendar = load_calendar(src, cache, lo, hi)
    halted = load_halts(src, cache, lo, hi)
    halt = set(zip(halted["ts_code"], halted["trade_date"], strict=True))
    overlap = len(set(code) & set(halted["ts_code"]))
    print(f"calendar {len(calendar):,} days | full-day halts {len(halt):,} over "
          f"{halted['ts_code'].nunique():,} stocks | 与池内股票交集 {overlap:,}")

    # ------------------------------------------------ 检查一、二：重叠与正对照 --
    print("\n各偏移量上命中停牌的行数（D+1/D+2 若被上游剔除应为 0，其他偏移不该为 0）：")
    control: dict[str, dict] = {}
    for offset in CONTROL_OFFSETS:
        dates = shifted(calendar, d0, offset)
        hits = int(sum((c, d) in halt for c, d in zip(code, dates, strict=True)))
        control[f"D{offset:+d}"] = {"rows": hits, "share": round(hits / len(df), 6)}
        flag = "  <- 假设：应为 0" if offset in (1, 2) else ""
        print(f"  D{offset:+3d}  {hits:7,}  {hits / len(df):8.4%}{flag}")

    forward = control["D+1"]["rows"] + control["D+2"]["rows"]
    elsewhere = sum(v["rows"] for k, v in control.items() if k not in ("D+1", "D+2"))
    if elsewhere == 0:
        raise SystemExit("正对照全为 0：匹配逻辑本身有问题，D+1/D+2 的零无意义")
    verdict = "excluded_upstream" if forward == 0 else "present_in_table"
    print(f"\n正对照在其他偏移上共命中 {elsewhere:,} 行 -> 匹配逻辑有效。"
          f"\n判定：D+1/D+2 停牌样本 {'不在表内（上游已剔除）' if forward == 0 else f'仍在表内（{forward:,} 行）'}")

    if forward:
        d1 = shifted(calendar, d0, 1)
        d2 = shifted(calendar, d0, 2)
        bad = np.array([(c, a) in halt or (c, b) in halt
                        for c, a, b in zip(code, d1, d2, strict=True)])
        sub = df[bad]
        print(f"  这批的标签 NaN 比例 {sub[args.label].isna().mean():.4%}"
              "   <- 正确处理应接近 100%")
        print(f"  这批的命中率 {sub[args.label].mean():.4%} vs 其余 {df[~bad][args.label].mean():.4%}")

    # ------------------------------------------------- 检查三：独立重建价格 --
    windows = parse_windows(args.verify_windows)
    dates = [d for d in calendar if any(a <= d <= b for a, b in windows)]
    bars = load_bars(src, cache, dates)
    verify = verify_prices(df, calendar, bars, args.label)

    report = {
        "rows": len(df),
        "halts_total": len(halt),
        "stocks_overlap": overlap,
        "control": control,
        "verdict": verdict,
        "verify": verify,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.report}")


def verify_prices(df: pd.DataFrame, calendar: np.ndarray, bars: pd.DataFrame,
                  label: str) -> dict:
    """Compare the table's D+1/D+2 prices against independently fetched raw bars."""
    frame = df.copy()
    frame["d1"] = shifted(calendar, frame["trade_date"].to_numpy(), 1)
    frame["d2"] = shifted(calendar, frame["trade_date"].to_numpy(), 2)
    have = set(bars["trade_date"])
    frame = frame[frame["d1"].isin(have) & frame["d2"].isin(have)]

    left = bars.rename(columns={"ts_code": "stock_code", "trade_date": "d1",
                                "open": "o1", "adj_factor": "a1"})
    right = bars.rename(columns={"ts_code": "stock_code", "trade_date": "d2",
                                 "high": "h2", "close": "c2", "adj_factor": "a2"})
    frame = frame.merge(left[["stock_code", "d1", "o1", "a1"]], on=["stock_code", "d1"], how="left")
    frame = frame.merge(right[["stock_code", "d2", "h2", "c2", "a2"]],
                        on=["stock_code", "d2"], how="left")

    missing_d1 = int(frame["o1"].isna().sum())
    missing_d2 = int(frame["h2"].isna().sum())
    print(f"\n=== 独立重建（{len(frame):,} 行落在抓取窗口内）===")
    print(f"  日历 D+1 无行情 {missing_d1:,}   日历 D+2 无行情 {missing_d2:,}"
          "   <- 管线若跳过停牌日取后面的 bar，这里会非零")
    frame = frame[frame["o1"].notna() & frame["h2"].notna()]

    out: dict = {"rows": len(frame), "missing_bar_d1": missing_d1, "missing_bar_d2": missing_d2}
    for column, actual in zip(PRICE_COLUMNS, ("o1", "h2", "c2"), strict=True):
        rel = (frame[column] - frame[actual]).abs() / frame[actual].abs().clip(lower=1e-9)
        out[column] = round(float((rel <= PRICE_TOL).mean()), 6)
        print(f"  {column:18s} == 原始价  {out[column]:8.4%}  中位相对差 {np.median(rel):.2e}")

    ex_div = np.asarray(~np.isclose(frame["a1"], frame["a2"]))
    truth = frame[label].to_numpy()
    out["ex_div_rows"] = int(ex_div.sum())
    print(f"\n  D+1→D+2 之间除权除息 {int(ex_div.sum()):,} 行 ({ex_div.mean():.4%})")
    for tag, ratio in (("raw", frame["h2"] / frame["o1"]),
                       ("hfq", (frame["h2"] * frame["a2"]) / (frame["o1"] * frame["a1"]))):
        recomputed = (ratio.to_numpy() >= 1.08).astype(float)
        out[f"label_agreement_{tag}"] = round(float((recomputed == truth).mean()), 6)
        on_ex = float((recomputed[ex_div] == truth[ex_div]).mean()) if ex_div.any() else float("nan")
        out[f"label_agreement_{tag}_ex_div"] = round(on_ex, 6)
        print(f"  重算命中（{tag}）与表内标签一致 {out[f'label_agreement_{tag}']:8.4%}"
              f"   除权行上 {on_ex:7.3%}")
    return out


if __name__ == "__main__":
    main()
