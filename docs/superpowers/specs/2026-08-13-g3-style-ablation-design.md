# G3 Style Ablation Design

## Objective

Determine whether the formal factor `gp_000` in
`data/artifacts/argus/event_factors.json` retains independent predictive and economic
value after removing five prescribed style exposures. The experiment produces a strict
training-window go/no-go decision, an out-of-sample appendix that cannot affect that
decision, and a reproducible audit trail in `docs/risk/g3_style_ablation.md`.

The style set is fixed:

1. log total market capitalization;
2. Shenwan 2021 level-one industry indicators;
3. trailing 20-trading-day compounded momentum;
4. trailing 20-trading-day return volatility;
5. trailing 20-trading-day mean free-float turnover rate.

The formal D0 calendar is exactly 2022-01-04 through 2024-09-04 (649 dates). Primary
decision metrics additionally require the D+2 label date to be no later than
2024-09-04, leaving 647 eligible D0 dates through 2024-09-02. D0 rows on 2024-09-03
and 2024-09-04 are moved to the reference-only appendix before metric or bootstrap
arrays are built. The same exit-date rule applies independently at every decay horizon.
The appendix may evaluate later outcomes but is not read by the decision function.

## Chosen Statistical Design

The factor replay and cross-sectional regression are deterministic. To make random
seeds measure genuine sampling uncertainty rather than inject arbitrary score noise,
the experiment reports:

- one deterministic full-training-window estimate for each arm and metric; and
- mean and sample standard deviation across ten fixed-seed circular moving-block
  bootstrap replicates of trading dates, using 20-date blocks and a replicate length
  equal to the original number of eligible dates.

Ten seeds exceed the requested minimum of three and satisfy the existing governance
rule for performance conclusions. Circular moving blocks preserve complete daily cross
sections and local serial dependence. A bootstrap replicate concatenates whole blocks;
portfolio returns follow that pseudo-time order when CAGR, Sharpe, and drawdown are
calculated. The deterministic full-window estimate, not a bootstrap average, drives the
go/no-go rule. Seed values and block length are command-line parameters stored in the
report so the table is reproducible.

Rejected alternatives are per-date stock subsampling, which changes the population on
which neutralisation is defined, and random score jitter, which measures tie breaking
rather than factor robustness.

## Components and Boundaries

### `helix/eval/style_neutralize.py`

This new module contains only reusable, fully vectorised numerical operations:

- cross-sectional standardisation of continuous style columns;
- construction of an intercept plus continuous exposures plus a fixed-width industry
  one-hot tensor;
- batched per-date Hermitian pseudoinverse projection; and
- residual extraction with rank-deficiency and negligible-residual guards.

Inputs use the shapes `(T, N)` for the factor and mask, `(T, N, C)` for continuous
styles, and `(T, N)` for integer industry codes. The output is `(T, N)`, with NaN
outside rows where the factor and every required style are observed. Industry levels
are determined once from the approved SW2021 L1 taxonomy, not inferred independently
on each date. One reference category is omitted to avoid the intercept/one-hot dummy
trap; a batched pseudoinverse of each date's Gram matrix handles absent and collinear
industry directions without dropping later populated dummy columns. No loop over dates or
stocks is permitted in the neutralisation path.

This module does not change `helix/gp/neutralize.py`: GP residual fitness remains a
separate rank-space feature-basis operation, while this experiment uses explicit
economic style exposures.

### `scripts/g3_style_ablation.py`

The experiment script owns orchestration and publication:

1. validate the formal window, library identity, factor name, seed count, metric
   thresholds, and output paths;
2. load only the formal factor's required source columns and replay its recorded sign
   through the existing factor-library evaluator;
3. load or build a local market/style cache;
4. map each training D0 to its market-calendar D+2 and move boundary outcomes beyond
   `train_end` to the appendix;
5. align style data to the event-table `(trade_date, stock_code)` population;
6. calculate the raw and style-neutral factor arms;
7. calculate deterministic metrics and seeded bootstrap summaries from the 647-date
   label-complete training subset;
8. calculate the decay table;
9. calculate a separately tagged OOS appendix when OOS rows and style data are
   available; and
10. atomically write the Markdown report and compact machine-readable result artifact.

The script exposes explicit CLI options for the input event table, formal factor
library, style cache, report path, result path, training bounds, seeds, bootstrap block
length, Top-K, decay horizons, and refresh behaviour. Defaults reproduce the published
report. A cache-only mode fails with an actionable error if required coverage is absent;
refresh mode retrieves missing dates without replacing already cached dates.

### Style-data cache

The cache is a derived, ignored artifact rather than a committed data file. Source data
comes from Tushare Pro:

- `daily_basic`: `total_mv` and `turnover_rate_f`;
- `daily`: daily percentage return and raw open/close needed for return targets;
- `adj_factor`: point-in-time adjustment factors for D+1-open-to-exit-close returns;
- `index_classify(level="L1", src="SW2021")`; and
- `index_member`: membership intervals using `in_date <= D0` and
  (`out_date` missing or `D0 <= out_date`).

All trailing style windows include D0 and the prior 19 market trading dates. Momentum
is the product of `(1 + pct_chg / 100)` minus one; volatility is sample standard
deviation of the same returns; mean turnover uses `turnover_rate_f`. Each rolling value
requires all 20 observations. Log market cap requires finite positive `total_mv` on D0.
The cache begins at least 19 market sessions before the first evaluated D0 date.

The report records source names, cache coverage, row counts, missingness, SW taxonomy,
library file hash, event-date digest, style-cache digest, CLI parameters, and software
version/commit when available. No token or other secret is persisted.

## Daily Neutralisation Contract

For date `t`, let `f_t` be the signed raw factor and `X_t` contain an intercept, the four
cross-sectionally standardised continuous styles, and SW L1 indicators. Using only
stocks whose factor and all styles are observed on date `t`, compute

`f_neutral,t = f_t - X_t (X_t' X_t)^+ X_t' f_t`.

The implementation uses batched Hermitian pseudoinverses of the daily Gram matrices but
is tested against `numpy.linalg.lstsq` date by date as a reference. Required invariants are:

- the residual is orthogonal to every retained design column within numerical
  tolerance on every usable date;
- changing another date cannot change the current date's residual;
- no input array is mutated;
- masked or incomplete observations remain NaN and do not alter other rows;
- a fully explained factor produces NaN for that date rather than floating-point rank
  noise; and
- rank-deficient or absent industries do not produce non-finite fitted values.

## Metrics and Backtest

The primary predictive target is `label_d2_hit_8pct`, matching the existing placebo
calibration. Before these labels enter any primary metric, the script resolves each D0
against the market calendar and retains it only if D0+2 is no later than `train_end`.
Metrics are calculated per eligible date and then averaged:

- Spearman IC mean and ICIR via the canonical IC functions;
- signed mean daily gini via the canonical gini function;
- Top10 daily hit rate, observable-pool base rate, and lift; and
- coverage and number of usable dates as audit fields.

The economic comparison ranks the same event-table candidate population by each arm,
selects Top10 without replacement or deeper-name substitution, and uses the existing
D+1-open to D+2-close label prices. The existing `BacktestConfig` statutory costs,
historical stamp-duty cut, and default 10 bps per-side slippage are applied through the
canonical multiplicative cost helper. With a two-day book, daily portfolio return is
the sum of executed net trade returns divided by Top10 and by overlap two; failed or
unobservable selections remain cash. The report includes cumulative net return
(`final_equity - 1`), net return per executed trade, CAGR, annualised Sharpe, maximum
drawdown, hit rate, base rate, lift, execution count, and coverage.

The decay horizons are 1, 2, 3, 5, 10, and 20 trading days. For horizon `h`, gross
return is adjusted close on D+h divided by adjusted D+1 open minus one. D0 rows are
eligible only when both dates are within the evaluated window. Each arm reports daily
Spearman IC mean, ICIR, and Top10 cost-adjusted net return per trade at every horizon.
Portfolio CAGR and drawdown are not promoted as decay statistics because overlapping
capital changes with `h`; the main D+2 backtest remains the economic headline.

## Decision Rule

The training-window decision is mechanical:

1. read the current training-matched placebo ICIR p95 threshold, expected to be
   `0.311614377566`, from the saved calibration distribution rather than hard-code it;
2. require `abs(style_neutral_icir) > placebo_icir_p95` (strict inequality); and
3. require the deterministic style-neutral Top10 mean net trade return to have the same
   non-zero sign as the deterministic raw-factor value.

Both conditions true yields **GO**. Any false, missing, or non-finite condition yields
**NO-GO**. Bootstrap dispersion, gini, hit rate, CAGR, Sharpe, drawdown, decay shape,
and the OOS appendix are diagnostic and cannot override this rule.

## Report and Governance Update

`docs/risk/g3_style_ablation.md` contains:

- an executive GO/NO-GO statement;
- exact decision-rule inputs;
- data lineage and leakage controls;
- a complete raw-versus-neutral deterministic table;
- the ten-seed mean ± sample-standard-deviation table;
- a style-orthogonality audit;
- the decay curve;
- robustness and limitations;
- a clearly separated OOS appendix; and
- reproduction commands and artifact hashes.

After the report is generated, `docs/factor-governance.md` §8 D7 changes from “未测 / 未
处理” to the measured effect size and status with a link to the report. D13 remains
unchanged.

## Tests and Verification

Tests are written before production code. The numerical suite covers exact continuous
and industry exposure removal, per-date isolation, missingness, rank deficiency,
negligible residuals, input immutability, and agreement with a small looped reference
solver. Script-level tests cover the strict D+2-complete training bounds, boundary
routing, forward-horizon truncation,
minimum seed count, deterministic bootstrap indices, metric aggregation, decision-rule
strictness and sign handling, formal-library identity, cache coverage, and report
sections.

Completion requires fresh successful runs of the targeted tests, the full `pytest`
suite, and Ruff over the repository. Generated report numbers are audited by rerunning
the script in cache-only mode and comparing the machine-readable artifact to the
Markdown tables before any completion claim.
