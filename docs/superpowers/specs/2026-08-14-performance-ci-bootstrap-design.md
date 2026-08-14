# Training Performance Confidence-Interval Bootstrap Design

## Goal

Quantify whether the formal `gp_000` Top4 strategy has statistically robust positive
training-window performance. The experiment reports deterministic CAGR, annualised
Sharpe, daily win rate, and mean net return per resolved trade, plus ten-seed circular
moving-block-bootstrap mean, sample standard deviation, and percentile 95% confidence
intervals. The mandatory downgrade is lifted only when the Sharpe interval lower bound
is strictly greater than zero.

## Scope and non-goals

The experiment is a read-only consumer of the formal factor, production configuration,
event data, and observed market caches. It does not change GP fitness, factor direction,
portfolio selection, backtest accounting, realistic-exit resolution, or any production
default. The only governance mutation is the measured status in
`docs/factor-governance.md` §7.5.

This experiment measures in-sample sampling uncertainty. It does not establish
out-of-sample profitability, capacity, or impact-cost robustness and cannot close D6.

## Canonical inputs

- Event source: `data/raw/argus_quant_working.parquet`.
- Formal library: `data/artifacts/argus/event_factors.json`, which must contain exactly
  one event factor named `gp_000`; `compute_factors` applies its stored production sign.
- Configuration: `configs/default.yaml`.
- Calendar: `data/raw/suspension_cache/calendar_20211006_20261022.parquet`.
- Observed daily/adjustment/limit cache: `data/raw/d2_exit_cache/<YYYYMMDD>.parquet`.
- Nominal training calendar: `2022-01-04` through `2024-09-04` inclusive.
- Complete D+2 decisions: apply `complete_outcome_window(..., horizon=2)`, leaving 647
  D0 dates through `2024-09-02`.

The command may populate missing date caches through the repository's existing Tushare
client and `d2_limit_down_bias.load_or_fetch_market`. `--cache-only` refuses to fetch
and is the report reproduction mode once the local observed cache exists.

## Production-consistent replay

1. Pack the complete-D+2 event rows with `build_event_panel` and evaluate the formal
   factor through `compute_factors`.
2. Align factor scores and the event occupancy mask to the observed market panel's
   fixed `(date, stock)` coordinates. A real event is a candidate; no outcome field is
   used to construct the shortlist.
3. Build labels from the observed market panel with `build_touch_label` so D+1 entry
   fillability uses the production limit-up and trading checks.
4. Call the canonical `run_backtest` once with the loaded production costs, `top_k=4`,
   `exit_rule="close"`, and an experiment-local `enable_realistic_exit=True` copy.
5. Keep the market panel hard-truncated at `2024-09-04`. A D+2 limit-down or suspension
   that cannot resolve by that boundary remains unresolved; no later observation is
   read. Failed entries and unresolved exits stay cash and never promote rank five.

The deterministic metric values come directly from the returned `BacktestResult`:
`summary["cagr"]`, `summary["sharpe"]`, `summary["day_win_rate"]`, and
`summary["mean_trade_return_net"]`.

## Bootstrap design

The reusable numerical layer lives in `helix/eval/bootstrap.py` and has no dependency
on factors, backtests, pandas, or files.

`circular_block_bootstrap_indices(n_dates, block_length, seeds)` returns a
`(n_seeds, n_dates)` integer matrix. For each seed it samples
`ceil(n_dates / block_length)` starts uniformly from `[0, n_dates)`, expands each start
into a circular run of consecutive dates, flattens, and truncates to `n_dates`.

The fixed seeds are `[7, 13, 42, 101, 211, 307, 419, 523, 631, 743]`; each seed creates
one complete replicate. Block length is 20 dates. This matches the repository's
existing governance interpretation: ten independent random seeds produce ten
date-resampling estimates, not ten groups of an additional unspecified replicate
count.

Realistic exits are resolved on the original calendar before bootstrap. Each sampled
D0 carries its entire daily portfolio return and every resolved trade selected on that
date. This preserves full daily cross-sections without trying to run the serial exit
state machine on a duplicated, non-monotonic pseudo-calendar.

The statistical aggregation is vectorised over the seed-by-date index matrix:

- CAGR: `prod(1 + daily_return) ** (252 / n_dates) - 1`;
- annualised Sharpe: `mean / sample_std * sqrt(252)`;
- daily win rate: fraction of sampled daily returns strictly greater than zero;
- mean net trade return: sampled per-day resolved-trade sums divided by sampled
  per-day resolved-trade counts.

For each metric, the output contains the deterministic value, bootstrap mean, sample
standard deviation (`ddof=1`), and linear 2.5%/97.5% percentiles. Non-finite inputs,
empty trade counts, fewer than two seeds, duplicate seeds, invalid blocks, or malformed
index shapes fail loudly.

## Decision and reporting

`sharpe_ci_low > 0` produces `LIFT_DOWNGRADE`; `sharpe_ci_low <= 0` or a non-finite
bound produces `KEEP_DOWNGRADE`.

`docs/risk/performance_ci_bootstrap.md` records:

- data and configuration provenance;
- deterministic and bootstrap metric table;
- all ten per-seed values;
- the exact Sharpe lower bound and decision;
- training/D+2/realistic-exit boundaries;
- observed execution diagnostics;
- the cache-only reproduction command;
- limitations covering in-sample inference and unmodelled impact costs.

`docs/factor-governance.md` §7.5 is updated with the measured lower bound and current
status, without weakening the `> 0` rule.

## Test strategy

- fixed seeds reproduce byte-identical circular index matrices;
- every block is consecutive modulo `n_dates`, including wraparound;
- invalid seed, block, date, and metric inputs are rejected;
- vectorised CAGR, Sharpe, win rate, and trade mean match canonical scalar references;
- percentile intervals and sample standard deviations match NumPy references;
- D+2 truncation admits `2024-09-02` and rejects later D0 rows;
- decision is lifted only for a finite strictly positive Sharpe lower bound;
- the experiment config forces realistic exits while preserving every production cost;
- report rendering includes seeds, CI, decision, boundaries, and reproduction command.

## Rejected alternatives

- Importing private helpers from `scripts/g3_style_ablation.py` preserves duplicated
  ownership and creates script-to-script coupling.
- Reimplementing bootstrap functions in the new script violates the reuse requirement.
- Adding bootstrap behavior to `run_backtest` mixes inference with production
  accounting and risks changing core behavior.
- Re-running realistic exit resolution on resampled pseudo-dates is invalid because the
  resampled calendar is duplicated and non-monotonic.
- Stock-level resampling breaks complete daily cross-sections and changes the Top4
  selection population.
