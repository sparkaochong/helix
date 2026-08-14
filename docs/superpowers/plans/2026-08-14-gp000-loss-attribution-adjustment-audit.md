# gp_000 Loss Attribution and Adjustment Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, training-only audit that identifies whether adjustment bugs, configuration choices, or weak alpha explain formal `gp_000` losses.

**Architecture:** Add one standalone orchestration/report script whose statistical units are pure functions over pandas/numpy objects. Reuse the canonical factor replay, IC, cost, portfolio-summary, and style-neutralization implementations; load the large local datasets only in the CLI boundary. Produce one Markdown report, two SVG charts, one JSON evidence file, and one daily parquet file from a shared evidence payload.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy, pyarrow, DEAP, existing Helix evaluation modules, pytest, Ruff.

---

## File map

- Create `scripts/gp000_loss_attribution.py`: all boundary, adjustment, attribution, style, rendering, and CLI orchestration.
- Create `tests/test_gp000_loss_attribution.py`: synthetic-data contracts for every new pure function and report section.
- Create `docs/risk/gp000_loss_attribution.md`: generated final report.
- Create `docs/risk/assets/gp000_loss_attribution_equity.svg`: generated D+2 equity curves.
- Create `docs/risk/assets/gp000_loss_attribution_decay.svg`: generated D+1..D+10 decay view.
- Generate ignored `data/artifacts/gp000_loss_attribution.json`: machine-readable evidence.
- Generate ignored `data/artifacts/gp000_loss_attribution_daily.parquet`: daily return evidence.

### Task 1: Boundary and formal-factor contracts

**Files:**
- Create: `tests/test_gp000_loss_attribution.py`
- Create: `scripts/gp000_loss_attribution.py`

- [ ] **Step 1: Write failing boundary and library tests**

```python
def test_outcome_complete_dates_never_cross_training_end():
    calendar = np.array([
        "2024-08-21", "2024-08-22", "2024-08-23", "2024-08-26",
        "2024-08-27", "2024-08-28", "2024-08-29", "2024-08-30",
        "2024-09-02", "2024-09-03", "2024-09-04",
    ])
    d0 = outcome_complete_dates(calendar, calendar, 2, "2024-09-04")
    assert d0.tolist() == calendar[:-2].tolist()
    d10 = outcome_complete_dates(calendar, calendar, 10, "2024-09-04")
    assert d10.tolist() == ["2024-08-21"]


def test_validate_formal_factor_rejects_other_gp000_library():
    library = FactorLibrary(
        factors=[FactorSpec("gp_000", "neg(x)", 1.0)],
        field_names=["x"],
        windows=[],
        kind="event",
    )
    with pytest.raises(ValueError, match="expression"):
        validate_formal_factor(library)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -v`

Expected: collection/import failure because `scripts.gp000_loss_attribution` does not exist.

- [ ] **Step 3: Implement minimal contracts**

```python
TRAIN_START = "2022-01-04"
TRAIN_END = "2024-09-04"
TRAIN_DATES = 649
FORMAL_FACTOR = "gp_000"
FORMAL_EXPRESSION = (
    "add(add(stock_intra_amp_d1d3_mean, "
    "div(stock_vwap_dev_d1, vol_burst_count_20d)), stock_intra_amp_d0)"
)


def hyphenated(value):
    text = str(value)
    return text if "-" in text else f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def outcome_complete_dates(d0_dates, calendar, horizon, train_end=TRAIN_END):
    dates = np.asarray(d0_dates).astype(str)
    sessions = np.asarray(calendar).astype(str)
    positions = np.searchsorted(sessions, dates)
    safe = np.clip(positions, 0, max(len(sessions) - 1, 0))
    exits = positions + horizon
    valid = (
        (horizon >= 1)
        & (positions < len(sessions))
        & (sessions[safe] == dates)
        & (exits < len(sessions))
    )
    valid &= np.where(exits < len(sessions), sessions[np.clip(exits, 0, len(sessions) - 1)] <= train_end, False)
    return dates[valid]


def validate_formal_factor(library):
    if library.kind != "event" or len(library.factors) != 1:
        raise ValueError("formal library must contain one event factor")
    factor = library.factors[0]
    if factor.name != FORMAL_FACTOR or factor.sign != 1.0:
        raise ValueError("formal factor identity or direction changed")
    if factor.expression != FORMAL_EXPRESSION:
        raise ValueError("formal factor expression changed")
    return factor
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -v`

Expected: boundary and formal-library tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gp000_loss_attribution.py tests/test_gp000_loss_attribution.py
git commit -m "feat: enforce gp000 audit boundaries"
```

### Task 2: Point-in-time price reconstruction and adjustment-chain audit

**Files:**
- Modify: `scripts/gp000_loss_attribution.py`
- Modify: `tests/test_gp000_loss_attribution.py`

- [ ] **Step 1: Write failing adjustment tests**

```python
def test_adjusted_return_removes_ex_right_gap():
    events = pd.DataFrame({
        "trade_date": ["2024-05-10"],
        "stock_code": ["000001.SZ"],
        "label_px_d1_open": [10.0],
        "label_px_d2_high": [9.2],
        "label_px_d2_close": [9.0],
        "label_d2_return": [-0.1],
        "label_d2_hit_8pct": [0.0],
    })
    prices = make_price_lookup_for_test(
        dates=["2024-05-10", "2024-05-13", "2024-05-14"],
        open_=[9.8, 10.0, 8.9], high=[10.0, 10.2, 9.2], close=[9.9, 10.0, 9.0],
        adj=[1.0, 1.0, 1.12],
    )
    audit, aligned = audit_adjustment_chain(events, prices)
    assert aligned.loc[0, "raw_return"] == pytest.approx(-0.1)
    assert aligned.loc[0, "hfq_return"] == pytest.approx(0.008)
    assert audit["return_mismatch_count"] == 1
    assert audit["event_prices_match_raw"] is True


def test_ex_right_detection_uses_adj_factor_change_on_same_stock():
    prices = make_price_lookup_for_test(
        dates=["2024-05-10", "2024-05-13", "2024-05-14"],
        open_=[10, 10, 9], high=[10, 10, 9], close=[10, 10, 9], adj=[1, 1, 1.1],
    )
    assert prices.ex_right[:, 0].tolist() == [False, False, True]
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'adjusted or ex_right' -v`

Expected: missing `PriceLookup`, `build_price_lookup`, or `audit_adjustment_chain`.

- [ ] **Step 3: Implement price lookup and audit**

```python
@dataclass(frozen=True)
class PriceLookup:
    dates: np.ndarray
    codes: np.ndarray
    raw_open: np.ndarray
    raw_high: np.ndarray
    raw_close: np.ndarray
    adj_factor: np.ndarray
    hfq_open: np.ndarray
    hfq_high: np.ndarray
    hfq_close: np.ndarray
    ex_right: np.ndarray
    date_positions: dict[str, int]
    code_positions: dict[str, int]


def build_price_lookup(market, calendar, codes):
    frame = market.copy()
    frame["trade_date"] = frame["trade_date"].astype(str).map(hyphenated)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market cache contains duplicate date/stock rows")
    dates = np.asarray(calendar).astype(str)
    names = np.asarray(sorted(codes)).astype(str)

    def pivot(column):
        return frame.pivot(
            index="trade_date", columns="ts_code", values=column
        ).reindex(index=dates, columns=names).to_numpy(dtype=np.float64)

    raw_open, raw_high, raw_close = (pivot(name) for name in ("open", "high", "close"))
    adj = pivot("adj_factor")
    ex_right = np.zeros(adj.shape, dtype=bool)
    ex_right[1:] = (
        np.isfinite(adj[1:])
        & np.isfinite(adj[:-1])
        & ~np.isclose(adj[1:], adj[:-1], rtol=0.0, atol=1e-12)
    )
    return PriceLookup(
        dates, names, raw_open, raw_high, raw_close, adj,
        raw_open * adj, raw_high * adj, raw_close * adj, ex_right,
        {date: index for index, date in enumerate(dates)},
        {code: index for index, code in enumerate(names)},
    )


def align_event_prices(events, prices, horizon):
    work = events.copy()
    d0 = work["trade_date"].astype(str).map(prices.date_positions).to_numpy()
    code = work["stock_code"].astype(str).map(prices.code_positions).to_numpy()
    if pd.isna(d0).any() or pd.isna(code).any():
        raise ValueError("event keys are absent from the market cache")
    d0, code = d0.astype(int), code.astype(int)
    entry, exit_ = d0 + 1, d0 + horizon
    in_bounds = exit_ < len(prices.dates)
    work["entry_date"] = np.where(in_bounds, prices.dates[np.minimum(entry, len(prices.dates) - 1)], "")
    work["exit_date"] = np.where(in_bounds, prices.dates[np.minimum(exit_, len(prices.dates) - 1)], "")
    work["raw_entry"] = prices.raw_open[np.minimum(entry, len(prices.dates) - 1), code]
    work["raw_exit"] = prices.raw_close[np.minimum(exit_, len(prices.dates) - 1), code]
    work["raw_exit_high"] = prices.raw_high[np.minimum(exit_, len(prices.dates) - 1), code]
    work["entry_adj_factor"] = prices.adj_factor[np.minimum(entry, len(prices.dates) - 1), code]
    work["exit_adj_factor"] = prices.adj_factor[np.minimum(exit_, len(prices.dates) - 1), code]
    work["hfq_entry"] = prices.hfq_open[np.minimum(entry, len(prices.dates) - 1), code]
    work["hfq_exit"] = prices.hfq_close[np.minimum(exit_, len(prices.dates) - 1), code]
    work["raw_return"] = work["raw_exit"] / work["raw_entry"] - 1.0
    work["hfq_return"] = work["hfq_exit"] / work["hfq_entry"] - 1.0
    work["entry_ex_right"] = prices.ex_right[np.minimum(entry, len(prices.dates) - 1), code]
    work["exit_ex_right"] = prices.ex_right[np.minimum(exit_, len(prices.dates) - 1), code]
    return work.loc[in_bounds & (work["exit_date"] <= TRAIN_END)].reset_index(drop=True)


def audit_adjustment_chain(events, prices):
    aligned = align_event_prices(events, prices, horizon=2)
    np.testing.assert_allclose(
        aligned["label_px_d1_open"], aligned["raw_entry"], rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        aligned["label_px_d2_close"], aligned["raw_exit"], rtol=0.0, atol=1e-6
    )
    aligned["return_delta"] = aligned["hfq_return"] - aligned["raw_return"]
    aligned["raw_hit"] = aligned["label_d2_hit_8pct"].astype(bool)
    aligned["hfq_hit"] = (
        aligned["raw_exit_high"]
        * aligned["exit_adj_factor"]
        >= aligned["raw_entry"] * aligned["entry_adj_factor"] * 1.08
    )
    summary = {
        "event_prices_match_raw": True,
        "return_mismatch_count": int(
            ~np.isclose(aligned["raw_return"], aligned["hfq_return"], atol=1e-12)
        ),
        "hit_flip_count": int((aligned["raw_hit"] != aligned["hfq_hit"]).sum()),
        "mean_return_delta": float(aligned["return_delta"].mean()),
        "max_abs_return_delta": float(aligned["return_delta"].abs().max()),
    }
    return summary, aligned
```

Implementation must use finite-value tolerances (`rtol=0`, `atol=1e-6`) and must not
normalize historical prices with the training-end factor.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'adjusted or ex_right' -v`

Expected: all adjustment tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gp000_loss_attribution.py tests/test_gp000_loss_attribution.py
git commit -m "feat: audit point in time adjustment chain"
```

### Task 3: Quintiles, fixed Top4, cost split, and monthly table

**Files:**
- Modify: `scripts/gp000_loss_attribution.py`
- Modify: `tests/test_gp000_loss_attribution.py`

- [ ] **Step 1: Write failing portfolio tests**

```python
def test_quintiles_are_daily_and_ordered_low_to_high():
    frame = synthetic_factor_returns(days=2, names=10)
    result = evaluate_quintiles(frame, BacktestConfig())
    assert result["quintile"].tolist() == [1, 2, 3, 4, 5]
    assert result["n"].tolist() == [4, 4, 4, 4, 4]
    assert result.loc[4, "gross_return"] > result.loc[0, "gross_return"]


def test_top4_missing_exit_stays_cash_without_replacement():
    frame = pd.DataFrame({
        "trade_date": ["2024-01-02"] * 5,
        "stock_code": list("ABCDE"),
        "factor_score": [5, 4, 3, 2, 1],
        "gross_return": [0.1, np.nan, 0.03, 0.02, 9.0],
    })
    _, daily = evaluate_top_k_book(frame, BacktestConfig(top_k=4), gross=True, overlap=2)
    assert daily.loc[0, "n_executed"] == 3
    assert daily.loc[0, "portfolio_return"] == pytest.approx((0.1 + 0.03 + 0.02) / 4 / 2)


def test_monthly_returns_compound_daily_returns():
    daily = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-02-01"],
        "gross_portfolio_return": [0.1, -0.1, 0.2],
        "net_portfolio_return": [0.08, -0.12, 0.18],
    })
    monthly = evaluate_monthly_returns(daily)
    assert monthly.loc[0, "gross_return"] == pytest.approx(0.1 * 0.9 - 1)
    assert monthly.loc[1, "net_return"] == pytest.approx(0.18)
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'quintile or top4 or monthly' -v`

Expected: missing attribution functions.

- [ ] **Step 3: Implement attribution functions**

```python
def evaluate_quintiles(frame, config):
    work = frame.copy()
    work["quintile"] = work.groupby("trade_date")["factor_score"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False) + 1
    )
    work["net_return"] = apply_cost_by_d0(work["gross_return"], work["trade_date"], config)
    return work.groupby("quintile").agg(
        n=("factor_score", "size"), n_dates=("trade_date", "nunique"),
        gross_return=("gross_return", "mean"), net_return=("net_return", "mean"),
        hit_rate=("hit_hfq", "mean"),
    ).reset_index()


def apply_cost_by_d0(returns, d0_dates, config):
    rates = _cost_rates(np.asarray(d0_dates).astype(str), config)
    values = np.asarray(returns, dtype=float)
    return (1.0 + values) * (1.0 - rates) - 1.0


def evaluate_top_k_book(frame, config, *, gross, overlap):
    rows, trades = [], []
    for date, block in frame.groupby("trade_date", sort=True):
        eligible = np.flatnonzero(np.isfinite(block["factor_score"]))
        if eligible.size < config.top_k:
            continue
        order = eligible[np.argsort(-block["factor_score"].to_numpy()[eligible], kind="stable")]
        picked = block.iloc[order[: config.top_k]]
        values = picked["gross_return"].to_numpy(dtype=float)
        if not gross:
            values = apply_cost_by_d0(values, np.repeat(date, len(values)), config)
        finite = values[np.isfinite(values)]
        trades.extend(finite.tolist())
        rows.append({
            "date": date,
            "n_selected": config.top_k,
            "n_executed": finite.size,
            "portfolio_return": float(finite.sum() / config.top_k / overlap),
        })
    daily = pd.DataFrame(rows)
    performance = summarize_portfolio_returns(daily["portfolio_return"].to_numpy())
    performance["mean_trade_return"] = float(np.mean(trades)) if trades else np.nan
    performance["n_days"] = float(len(daily))
    performance["execution_rate"] = float(
        daily["n_executed"].sum() / daily["n_selected"].sum()
    )
    return performance, daily


def evaluate_monthly_returns(daily):
    work = daily.copy()
    work["month"] = work["date"].astype(str).str[:7]
    rows = []
    for month, block in work.groupby("month", sort=True):
        rows.append({
            "month": month,
            "n_days": len(block),
            "gross_return": float(np.prod(1 + block["gross_portfolio_return"]) - 1),
            "net_return": float(np.prod(1 + block["net_portfolio_return"]) - 1),
            "day_win_rate": float((block["net_portfolio_return"] > 0).mean()),
        })
    result = pd.DataFrame(rows)
    result["gross_equity"] = (1 + result["gross_return"]).cumprod()
    result["net_equity"] = (1 + result["net_return"]).cumprod()
    return result
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'quintile or top4 or monthly' -v`

Expected: all portfolio attribution tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gp000_loss_attribution.py tests/test_gp000_loss_attribution.py
git commit -m "feat: add gp000 pnl attribution metrics"
```

### Task 4: D+1..D+10 decay and style-neutral Top4

**Files:**
- Modify: `scripts/gp000_loss_attribution.py`
- Modify: `tests/test_gp000_loss_attribution.py`

- [ ] **Step 1: Write failing decay and style tests**

```python
def test_horizon_decay_uses_horizon_as_overlap_and_truncates_exit():
    evidence = evaluate_horizon_decay(events, prices, BacktestConfig(top_k=2), range(1, 4))
    assert evidence["summary"]["horizon"].tolist() == [1, 2, 3]
    assert evidence["summary"].loc[2, "d0_end"] < TRAIN_END
    assert evidence["daily"].query("horizon == 3")["exit_date"].max() <= TRAIN_END


def test_style_neutral_book_uses_common_mask_and_is_orthogonal():
    result = evaluate_style_neutral_book(
        events=style_test_events(),
        styles=style_test_styles(),
        members=style_test_members(),
        config=BacktestConfig(top_k=2),
    )
    assert result["raw"]["n_days"] == result["style_neutral"]["n_days"]
    assert result["orthogonality"]["max_abs_normalized_exposure"] < 1e-10
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'horizon or style' -v`

Expected: missing decay and style functions.

- [ ] **Step 3: Implement decay and style evaluation**

```python
def evaluate_horizon_decay(events, prices, config, horizons=range(1, 11)):
    summaries, daily_frames = [], []
    for horizon in horizons:
        aligned = align_event_prices(events, prices, horizon)
        aligned["gross_return"] = aligned["hfq_return"]
        dates, score, target, mask = event_grids(
            aligned, "factor_score", "gross_return"
        )
        ic = summarize_ic(daily_ic(score, target, mask, min_samples=30))
        gross_summary, gross_daily = evaluate_top_k_book(
            aligned, config, gross=True, overlap=horizon
        )
        net_summary, net_daily = evaluate_top_k_book(
            aligned, config, gross=False, overlap=horizon
        )
        merged = gross_daily.merge(net_daily, on="date", suffixes=("_gross", "_net"))
        merged["horizon"] = horizon
        merged["exit_date"] = aligned.groupby("trade_date")["exit_date"].first().reindex(merged["date"]).to_numpy()
        daily_frames.append(merged)
        summaries.append({
            "horizon": horizon,
            "n_days": len(merged),
            "d0_end": str(aligned["trade_date"].max()),
            "ic_mean": ic["ic_mean"],
            "icir": ic["icir"],
            "gross_per_trade": gross_summary["mean_trade_return"],
            "net_per_trade": net_summary["mean_trade_return"],
            "cagr": net_summary["cagr"],
            "sharpe": net_summary["sharpe"],
            "final_equity": net_summary["final_equity"],
        })
    return {"summary": pd.DataFrame(summaries), "daily": pd.concat(daily_frames)}


def event_grids(frame, score_column, target_column):
    dates = np.asarray(sorted(frame["trade_date"].unique())).astype(str)
    codes = np.asarray(sorted(frame["stock_code"].unique())).astype(str)
    index = pd.MultiIndex.from_product([dates, codes])
    keyed = frame.set_index(["trade_date", "stock_code"]).reindex(index)
    score = keyed[score_column].to_numpy(dtype=float).reshape(len(dates), len(codes))
    target = keyed[target_column].to_numpy(dtype=float).reshape(len(dates), len(codes))
    return dates, score, target, np.isfinite(score) & np.isfinite(target)


def evaluate_style_neutral_book(events, styles, members, config):
    aligned = events.merge(styles, on=["trade_date", "stock_code"], how="left")
    industry_frame, industry_names = _align_industries(
        aligned[["trade_date", "stock_code"]], members
    )
    aligned["industry_code"] = industry_frame["industry_code"].to_numpy()
    panel = build_event_panel(
        aligned,
        ["factor_score", *STYLE_COLUMNS, "industry_code"],
        ["hfq_return", "hit_hfq"],
    )
    raw = panel.f64("factor_score")
    continuous = np.stack([panel.f64(name) for name in STYLE_COLUMNS], axis=2)
    industry = panel.f64("industry_code")
    common = panel.occupied & np.isfinite(raw) & np.isfinite(continuous).all(2) & np.isfinite(industry)
    levels = np.arange(len(industry_names), dtype=float)
    neutral = style_residualize(raw, continuous, industry, common, industry_levels=levels)
    arms = {}
    for name, score in {"raw": raw, "style_neutral": neutral}.items():
        long = panel.to_long({
            "factor_score": score,
            "gross_return": panel.f64("hfq_return"),
            "hit_hfq": panel.f64("hit_hfq"),
            "common": common.astype(float),
        })
        long = long[long["common"] == 1]
        arms[name], _ = evaluate_top_k_book(long, config, gross=False, overlap=2)
    exposure = style_orthogonality(neutral, continuous, industry, common, levels)
    return {**arms, "orthogonality": exposure}


def style_orthogonality(neutral, continuous, industry, mask, levels):
    design = build_style_design(continuous, industry, mask, levels)
    normalized = normalize_daily_exposure(neutral, design.matrix, mask)
    return {
        "max_abs_normalized_exposure": float(np.nanmax(np.abs(normalized))),
        "n_valid_dates": int(np.isfinite(normalized).all(axis=1).sum()),
    }
```

The implementation must reuse `style_residualize`, `daily_ic`, and
`summarize_portfolio_returns`; it must not fit styles across dates. The concrete
orthogonality helper may directly reuse the design/exposure primitives already in
`scripts/g3_style_ablation.py`; the test contract above is authoritative if their
names differ.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'horizon or style' -v`

Expected: all decay and style tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gp000_loss_attribution.py tests/test_gp000_loss_attribution.py
git commit -m "feat: add gp000 decay and style attribution"
```

### Task 5: Root-cause ranking, report, artifacts, and charts

**Files:**
- Modify: `scripts/gp000_loss_attribution.py`
- Modify: `tests/test_gp000_loss_attribution.py`

- [ ] **Step 1: Write failing report tests**

```python
def test_root_causes_rank_engineering_before_config_before_alpha():
    causes = rank_root_causes(minimal_evidence())
    assert [cause["category"] for cause in causes] == [
        "工程 bug", "参数配置", "因子 alpha",
    ]


def test_report_contains_every_required_section():
    report = render_report(minimal_evidence())
    for heading in (
        "复权全链路审计", "除权日专项校验", "五分位单调性",
        "成本拆分", "收益衰减", "时间分布", "风格中性收益",
        "根因优先级", "修复路径与预期效果", "复现方式",
    ):
        assert heading in report
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -k 'root_causes or report' -v`

Expected: missing ranking/rendering functions or required headings.

- [ ] **Step 3: Implement ranking, rendering, SVG, and atomic writes**

```python
CATEGORY_ORDER = {"工程 bug": 0, "参数配置": 1, "因子 alpha": 2}


def rank_root_causes(evidence):
    causes = build_root_cause_rows(evidence)
    return sorted(causes, key=lambda row: (CATEGORY_ORDER[row["category"]], row["priority"]))


def render_report(evidence):
    sections = [
        "# gp_000 亏损根因排查与复权全链路审计",
        "## 执行摘要",
        render_summary(evidence),
        "## 第一部分：复权全链路审计",
        render_adjustment_tables(evidence),
        "### 除权日专项校验",
        markdown_table(evidence["ex_right_samples"]),
        "## 第二部分：gp_000 亏损归因",
        "### 五分位单调性",
        markdown_table(evidence["quintiles"]),
        "### 成本拆分",
        markdown_table(evidence["cost_split"]),
        "### 收益衰减",
        markdown_table(evidence["decay"]["summary"]),
        "![D+1 至 D+10 衰减](assets/gp000_loss_attribution_decay.svg)",
        "### 时间分布",
        markdown_table(evidence["monthly"]),
        "![累计收益](assets/gp000_loss_attribution_equity.svg)",
        "### 风格中性收益",
        markdown_table(evidence["style_table"]),
        "## 根因优先级",
        markdown_table(pd.DataFrame(rank_root_causes(evidence))),
        "## 修复路径与预期效果",
        render_repairs(evidence),
        "## 复现方式",
        f"```bash\n{evidence['metadata']['command']}\n```",
    ]
    return "\n\n".join(sections) + "\n"


def render_equity_svg(daily):
    series = {
        "gross": np.r_[1.0, np.cumprod(1 + daily["gross_portfolio_return"])],
        "net": np.r_[1.0, np.cumprod(1 + daily["net_portfolio_return"])],
        "style_neutral": np.r_[1.0, np.cumprod(1 + daily["style_neutral_return"])],
    }
    return line_chart_svg(series, title="gp_000 Top4 累计收益", y_label="净值")


def line_chart_svg(series, title, y_label):
    """Return deterministic, escaped SVG with a shared x-axis and labelled paths.

    Scale all finite values into a 960x480 viewBox, include the 1.0 reference line,
    and raise ValueError when no finite point is available.  Use only stdlib XML
    escaping so the chart is reproducible without a plotting backend.
    """


def render_decay_svg(decay):
    """Render ten labelled net-equity paths from decay['daily'], one per horizon."""


def markdown_table(frame):
    """Render a DataFrame after formatting dates, counts, percentages, and NaN."""


def render_summary(evidence):
    """State the adjustment verdict, the measured loss driver, and boundary counts."""


def render_adjustment_tables(evidence):
    """Render the four-node matrix, mismatch statistics, and future-risk verdict."""


def build_root_cause_rows(evidence):
    """Build evidence-backed engineering, configuration, and alpha cause rows."""


def render_repairs(evidence):
    """Render one executable repair and measured/guardrail effect per root cause."""


def json_ready(value):
    """Recursively convert frames, numpy scalars, dates, and non-finite values."""


def atomic_text(path, content):
    """Write UTF-8 content through a sibling temporary file and os.replace."""


def atomic_parquet(path, frame):
    """Write a DataFrame through a sibling temporary parquet and os.replace."""


def write_outputs(evidence, paths):
    payload = json.dumps(
        json_ready(evidence), indent=2, ensure_ascii=False, allow_nan=False
    )
    atomic_text(paths.report, render_report(evidence))
    atomic_text(paths.json, payload)
    atomic_parquet(paths.daily, evidence["daily"])
    atomic_text(paths.equity_svg, render_equity_svg(evidence["daily"]))
    atomic_text(paths.decay_svg, render_decay_svg(evidence["decay"]))
```

The rendering helpers are deliberately backend-free. Their focused tests must also
parse both SVG strings with `xml.etree.ElementTree.fromstring`, assert that JSON has
no `NaN` tokens, and verify that atomic writes create parent directories before
replacement.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -v`

Expected: all专项 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/gp000_loss_attribution.py tests/test_gp000_loss_attribution.py
git commit -m "feat: render gp000 loss attribution report"
```

### Task 6: Run the full-data audit and inspect the conclusions

**Files:**
- Create: `docs/risk/gp000_loss_attribution.md`
- Create: `docs/risk/assets/gp000_loss_attribution_equity.svg`
- Create: `docs/risk/assets/gp000_loss_attribution_decay.svg`
- Generate: `data/artifacts/gp000_loss_attribution.json`
- Generate: `data/artifacts/gp000_loss_attribution_daily.parquet`

- [ ] **Step 1: Make ignored data visible in the isolated worktree**

Run from the worktree only if `data` is absent:

```bash
ln -s ../../data data
ln -s ../../.venv .venv
```

Expected: the worktree resolves the main checkout's ignored local datasets and environment.

- [ ] **Step 2: Run the audit**

Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/gp000_loss_attribution.py
```

Expected: exit 0; JSON summary reports the fixed formal factor, 649 nominal dates,
647 D+2-complete dates, no forward exit after 2024-09-04, and all five attribution tables.

- [ ] **Step 3: Inspect generated evidence**

Run:

```bash
jq '{metadata, adjustment_audit, cost_split, style_neutral, root_causes}' \
  data/artifacts/gp000_loss_attribution.json
rg -n '^#|^##|越界|未来|复权|核心原因|修复' docs/risk/gp000_loss_attribution.md
```

Expected: root causes are ordered engineering, configuration, alpha; any adjustment bug
is explicitly separated from whether it explains the loss.

- [ ] **Step 4: Verify report assets are well-formed**

Run:

```bash
xmllint --noout docs/risk/assets/gp000_loss_attribution_equity.svg
xmllint --noout docs/risk/assets/gp000_loss_attribution_decay.svg
```

Expected: both commands exit 0.

- [ ] **Step 5: Commit generated tracked deliverables**

```bash
git add docs/risk/gp000_loss_attribution.md docs/risk/assets/gp000_loss_attribution_*.svg
git commit -m "docs: publish gp000 loss attribution audit"
```

### Task 7: Fresh full verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run the dedicated suite**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gp000_loss_attribution.py -v`

Expected: all dedicated tests pass.

- [ ] **Step 2: Run all tests**

Run: `PYTHONPATH=. .venv/bin/pytest`

Expected: zero failures; the pre-change baseline was 416 passed, 1 skipped.

- [ ] **Step 3: Run Ruff**

Run: `PYTHONPATH=. .venv/bin/ruff check .`

Expected: `All checks passed!`

- [ ] **Step 4: Check repository integrity**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -8
```

Expected: no whitespace errors; only intended tracked files differ from the base branch;
ignored machine-readable artifacts are present but not staged.

- [ ] **Step 5: Review requirements line by line**

Confirm the final report contains:

1. Four-node adjustment matrix and point-in-time/future-function conclusion.
2. Adjustment consistency and ex-right sample table.
3. Five quintiles with D+2 gross/net/sample count.
4. Gross/net CAGR, Sharpe, and per-trade return.
5. D+1..D+10 IC and Top4 net curves.
6. Monthly returns and cumulative equity.
7. Style-neutral Top4 comparison.
8. Engineering > configuration > alpha root-cause ordering.
9. Concrete repair path and measured expected effect for every cause.
10. Reproduction command and hashes.
