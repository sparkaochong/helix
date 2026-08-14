# Objective–P&L Alignment Design

## 1. Objective and Scope

Replace the GP search objective that rewards `label_d2_hit_8pct` IC/gini with the
cost-adjusted D+2-close net return of the production Top-K long portfolio. Production
`K` is read from `BacktestConfig.top_k`; the current default is **4**. Top10 remains a
supplemental validation view and cannot enter fitness, sign selection, survival, or
factor ordering.

The change covers GP fitness, GP survival, factor evaluation, a reproducible
training-only diagnosis/validation report, unit tests, and the D1 governance ledger.
It deliberately does not change:

- `label_d2_hit_8pct` or any other label definition;
- the backtest engine's ranking, no-replacement selection, costs, or accounting;
- the live candidate universe;
- the GP primitive set; or
- the formal/supplemental status of any existing factor library.

All diagnosis, calibration, factor-library comparison, and acceptance tests use only
the formal training calendar `2022-01-04..2024-09-04`. Because D0 outcomes settle on
D+2, objective-bearing D0 rows end on `2024-09-02`; D0 rows on September 3 and 4 are
excluded before any objective, auxiliary metric, bootstrap, or correlation array is
built. No sample-out rows or prices may enter this work.

## 2. Confirmed Root Cause

The defect is a target/payoff mismatch, not a sign bug in the four existing direction
layers.

The current path is internally consistent:

1. `gp_000` is stored with `sign=+1`;
2. `label_d2_hit_8pct=1` means D+2 intraday high touched 108% of D+1 open;
3. IC/gini treats a larger factor value as a higher touch probability; and
4. the long portfolio buys the largest signed scores.

That coherent path optimises the wrong economic event. Training-only replay on the
647 D+2-complete D0 dates gives:

| Diagnostic | Result |
|---|---:|
| `gp_000` IC against hit label | +0.129251 |
| `gp_000` IC against D+2 close return | -0.062899 |
| Top10 D+2 gross return per trade | -0.1912% |
| Top10 D+2 net return per trade | -0.5230% |
| All-candidate D+2 gross return | -0.0846% |
| Top10 gross excess over candidates | -0.1066% |
| Binary hit-label IC against D+2 close return | +0.332453 |
| Hit/miss mean D+2 close return | +8.8690% / -1.4996% |

The label itself therefore points in the right average return direction. The loss
appears inside the model ranking: factor-score quintiles raise hit rate from 7.17% to
25.29% and peak return from 2.13% to 4.14%, while mean close return deteriorates from
+0.030% to -0.096%. Among touched names, 48.59% close back below the 8% target level.
The factor expression is rewarding amplitude/touch propensity without pricing the
subsequent giveback and miss loss.

The other investigated mechanisms are secondary:

- D+2 Top10 gross return is already negative, so approximately 33 bp of round-trip
  statutory cost and slippage worsens but does not create the reversal.
- Top10 underperforms the same-date candidate pool, so an overall falling market is
  not a complete explanation.
- reconstructed D+1 limit-up opens are 0.37% of formal Top10 selections; removing them
  cannot reverse the result;
- formal Top10 is not concentrated in the bottom liquidity decile (0.02% by turnover,
  6.23% by market value); and
- the highest score quintile has lower, not higher, day-to-day membership turnover than
  the middle quintiles. Moreover, a fixed D+2 exit pays both sides for every executed
  cohort regardless of score persistence.

The documented “approximately -0.064” is specifically the cross-sectional score-to-
D+2-return IC. It is not the time-series correlation between daily hit-label IC and
daily portfolio return; the latter is positive because high-IC days lose less or make
more while the unconditional portfolio level remains negative. The final report will
publish both quantities under distinct names to prevent another semantic collision.

## 3. Considered Objective Designs

### A. Mean production Top4 D+2-close net portfolio return — selected

For each D0, rank the point-in-time candidate mask by the candidate factor, fix the
first `BacktestConfig.top_k` names, apply the canonical multiplicative buy/sell costs to
their D+2-close gross returns, leave failed/unobservable selections in cash without
replacement, and average the resulting daily portfolio return over the fitness window.

This is the only option exactly aligned with the production action and payoff. It sees
the hard Top-K tail, absolute return level, both-side costs, and failed execution. It is
also naturally vectorisable and requires one cross-sectional sort, the same asymptotic
work as current daily gini.

### B. Top4 net information ratio

Net IR improves stability and, on the existing 30 factors, has a strong positive
relationship with held-out net return. It is rejected as the core objective because it
can rank a low-return/low-volatility factor above a higher-return factor, violating the
required monotonic relationship with real net P&L. IR remains an auxiliary monitor and
may later become an admission gate only after separate pre-registration.

### C. Cost-adjusted D+2 return IC

Close-return IC is dense and correlates in the correct direction with held-out Top4
P&L. It is rejected as the core objective because fixed per-date costs are a
cross-sectional constant and therefore disappear under ranks. IC also weights the
entire candidate population rather than the production Top4 tail, so it cannot guarantee
that a higher score means a better executable book. Close-return IC remains an auxiliary
diagnostic.

The selected design uses A alone for evolution. It does not blend A/B/C or tune weights
on these 30 factors; doing so would turn the validation library into a hyperparameter
selection set.

## 4. Economic Objective Contract

### 4.1 Inputs

The reusable objective receives aligned two-dimensional arrays:

- `score[T, N]`: a candidate GP expression;
- `gross_return[T, N]`: D+1 open to D+2 close gross return, NaN when no executable or
  observable outcome exists;
- `candidate_mask[T, N]`: information available at D0, never a D+1/D+2 validity mask;
- `dates[T]`: D0 trading dates used by the historical stamp-duty schedule; and
- `BacktestConfig`: the single source for Top-K, commission, transfer fee, stamp duty,
  and slippage.

Shape mismatch, non-monotonic dates, a non-positive Top-K, fewer than K candidates on
every date, no finite realised outcomes, or any date beyond the declared training
boundary is a hard error at context construction. Candidate expressions that return a
wrong shape or insufficient finite coverage receive `INVALID` as today.

### 4.2 Fixed selection and cost accounting

For each D0 `t`:

1. define candidates as `candidate_mask[t] & isfinite(score[t])`;
2. perform one stable descending sort and fix its first K names;
3. never replace a selected name whose outcome is NaN;
4. calculate every finite selected net trade return with the existing backtest helpers:
   `(1 + gross) * (1 - sell_cost) / (1 + buy_cost) - 1`;
5. treat an unexecuted slot as cash with return zero; and
6. calculate daily portfolio return as `sum(selected_net) / K / overlap`, where
   `overlap = touch_offset - entry_offset + 1 = 2`.

The constant overlap does not affect factor ordering but is retained so the objective's
unit is the production daily portfolio return. Reports additionally expose net return
per executed trade and execution coverage for interpretation.

Costs are precomputed once when `EvalContext` is built, not once per expression. The
objective imports and reuses the canonical cost helpers from `helix.eval.backtest`; it
does not copy the formulas. The engine and label modules themselves are unchanged.

### 4.3 Direction and complexity

Net P&L is directional and must not use an absolute value. A negative mean remains
negative. The candidate expression's printed direction is the long direction evaluated
by GP; the existing `neg` primitive can express the inverse when economically useful.
Newly mined factors are stored with `sign=+1` because direction is already represented
by the expression. Existing libraries keep replaying their recorded legacy signs.

This removes the old “sign is a free parameter” rule for new searches. Automatically
trying both tails and keeping the better in the same fit window is deliberately avoided:
it adds a hidden binary selection, weakens held-out sign stability, and doubles sorting
work.

The primary DEAP fitness component is mean Top4 net portfolio return in basis points.
Expression size is only a lexicographic tie-breaker; `max_nodes=40` and `max_depth=8`
remain hard bounds. A subtractive node penalty is not allowed to move a lower-P&L factor
above a higher-P&L factor, because that would break the required monotonic ordering.

### 4.4 Fit/selection discipline

After removing the two D0 boundary dates, the 647-date objective window is divided by
the existing rule:

```text
[ fit: 517 dates ][ embargo: 5 dates ][ selection: 125 dates ]
2022-01-04..2024-02-23               2024-03-04..2024-09-02
```

Evolution ranks expressions only by fit-period Top4 mean net P&L. Hall-of-fame factors
survive only when selection-period Top4 mean net P&L is finite and strictly positive.
Survivors are ordered by selection-period mean net P&L, then by smaller node count, and
then deduplicated with the existing rank-correlation rule. A valid outcome can be that
no factor survives; the pipeline must report that honestly rather than fall back to IC.

## 5. Vectorised Implementation and Performance Budget

`helix/eval/objective.py` will own two pure numerical operations:

1. precompute the cost-adjusted outcome grid from gross returns, D0 dates, and
   `BacktestConfig`; and
2. calculate the daily fixed-Top-K portfolio return and coverage from a score grid.

The hot path contains no Python loop over dates, stocks, or selections. It uses a stable
`np.argsort(..., axis=1)`, `np.take_along_axis`, boolean finite masks, and reductions.
This exactly matches the production backtest's stable tie rule and no-replacement
semantics. Fit and selection slices reuse the same ordered result from one evaluation.

A pre-design benchmark at the formal search shape `(647, 1990)`, 23% occupancy, gives a
median 28.0 ms for vectorised stable Top4 selection versus 37.8 ms for the current daily
gini implementation (five repeats of ten calls). The new evolution hot metric is thus
approximately 74% of the old metric's standalone runtime. Costs are outside this timed
loop. Auxiliary metrics are not calculated for every population member.

IC, gini, and hit monitors are calculated only for the hall of fame and retained factor
reports, at most 60 expressions rather than roughly 10,000 population evaluations. This
keeps total GP scoring cost at or below the present order while preserving all legacy
diagnostics.

## 6. GP and Pipeline Changes

### `helix/eval/objective.py` — new

Define the vectorised cost target, fixed-Top-K daily returns, and objective summary
(mean, standard deviation, IR, positive-day rate, execution rate, date coverage, and
number of dates). A small looped reference exists only in tests.

### `helix/gp/fitness.py`

`EvalContext` receives precomputed net outcomes, D0 candidate mask, production Top-K,
and overlap. `score_values` residualises the expression as before, then calculates fit
and selection Top4 net objective summaries. `FactorScore` exposes `fit_net_return`,
`fit_net_ir`, `sel_net_return`, coverage, and node count. It never reads hit labels to
produce `fitness`.

### `helix/gp/engine.py`

`make_context` and `run_search` require D+2 gross returns, D0 dates,
`BacktestConfig`, and the label offsets needed for overlap. Population and hall-of-fame
fitness use net P&L as the first component and smaller node count only for exact ties.
`_select_factors` replaces the positive-selection-gini gate with a strictly positive
selection-net-return gate. New factor specs use `sign=+1`.

### `helix/pipeline.py`, `helix/pipeline_events.py`, and `scripts/mine_argus.py`

All three callers pass D0-only candidate masks and D+2-close gross return grids. The
panel path derives gross returns from the existing label prices and leaves invalid
outcomes NaN. The event path consumes `label_d2_return`; no label is redefined.

Before feature screening, basis construction, context construction, or evaluation, the
nominal training/search slice is shortened by `label.touch_offset` dates so every D+2
outcome is inside the training boundary. Event mining therefore changes from 649 D0
dates to the same 647 D+2-complete dates already enforced by the G3 audit.

Factor evaluation is changed from “search vs everything after search” to three
training-only blocks: fit, selection, and full training. Rows after the formal training
end are neither read nor reported by the objective-alignment workflow.

## 7. Auxiliary Monitoring Contract

The following metrics remain visible but cannot affect DEAP fitness, tournament
selection, hall-of-fame order, sign, survival, or factor deduplication:

- Spearman IC and ICIR against `label_d2_hit_8pct`;
- daily gini against `label_d2_hit_8pct`;
- Top-K hit rate, base rate, and lift;
- IC and ICIR against `label_d2_peak_return`;
- IC and ICIR against `label_d2_return`;
- Top4 gross return, net return per executed trade, net portfolio return, and net IR;
- supplemental Top10 versions of the economic and hit metrics; and
- coverage, execution count, and selected-slot cash rate.

The report labels Top4 as `production_objective` and Top10 as
`supplemental_top10`. No shared generic `top_k` column may make the two roles ambiguous.
Changing hit labels while holding scores and returns fixed must leave fitness bitwise
unchanged; a unit test pins this separation.

## 8. Training-Only Diagnosis and Validation Report

`scripts/objective_pnl_alignment.py` will reproduce and write
`docs/risk/objective_pnl_alignment.md` plus a machine-readable ignored artifact. It
loads only the formal factor's required source columns, the three existing factor
libraries, training-period event rows, and market-cache rows needed to resolve outcomes
that end no later than `2024-09-04`.

The report contains:

1. exact calendars, hashes, factor expression/sign, costs, K, and execution rules;
2. direction audit across sign config, label direction, IC/gini direction, and long
   selection;
3. both score-to-return cross-sectional IC and daily-hit-IC-to-P&L time-series
   correlation under unambiguous names;
4. D+1 through D+10 tables, independently dropping D0 rows whose exit exceeds the
   training end;
5. monthly statistics and within-month correlations;
6. ex-ante market regimes based on D0-observable trailing 20-session equal-weight
   market return: bull above +5%, bear below -5%, neutral otherwise;
7. Top4 and supplemental Top10 gross-versus-net comparisons;
8. candidate-pool return and Top-K excess return, separating absolute-market and
   ranking effects;
9. score quintiles, hit/miss payoff, peak-to-close giveback, turnover, D+1 fillability,
   and liquidity concentration;
10. root causes ranked by measured impact; and
11. old-versus-new objective validation on the formal library plus the 17-factor
    `argus_multi` and 12-factor `argus_n40` supplemental libraries, without changing
    their governance status.

No conclusion function accepts an OOS frame. The script rejects a source row later than
the training end, an objective D0 later than September 2, or any forward-return exit
date later than September 4.

## 9. Pre-registered Acceptance Criteria

The 30 existing signed factor outputs are evaluated on the exact 517/5/125
fit/embargo/selection partition. The acceptance statistic compares each fit-period
objective with its selection-period production Top4 net P&L:

- Pearson correlation must be strictly positive with two-sided `p < 0.05`; and
- Spearman rank correlation must be strictly positive with two-sided `p < 0.05`.

The preliminary training-only result is Pearson `r=0.857423`,
`p=1.44e-9`, and Spearman `rho=0.722803`, `p=6.45e-6`. The old hit-gini fit objective
has Spearman `rho=-0.348610` against held-out Top4 P&L. These values must be reproduced
by the committed script from source data; the design values are not accepted as final
evidence.

Supplemental Top10 must also be reported but has no pass/fail authority. Its preliminary
fit-objective versus selection-P&L correlations are Pearson `r=0.860961` and Spearman
`rho=0.714794`, both positive and significant.

No requirement says an existing factor must become profitable. In the preliminary
Top4 replay, `gp_000` remains negative and only one of 30 existing factors has a
slightly positive full-window net return. The repair passes when the search ordering is
economically aligned and rejects non-positive selection performance; it must not turn a
methodological correction into a fabricated alpha claim.

## 10. Tests

Tests are written and observed failing before production changes.

### Objective arithmetic

- vectorised Top4 results equal a looped reference on random and hand-built arrays;
- K comes from `BacktestConfig.top_k`, with tests at K=1, 4, and 10;
- commission, transfer fee, historical stamp-duty cut, and per-side slippage equal the
  canonical backtest helpers exactly;
- selection is stable on ties;
- an invalid selected outcome remains cash and is never replaced;
- fewer than K candidates yields an unusable date;
- arrays are not mutated; and
- malformed shapes, dates, masks, and non-finite context metadata fail loudly.

### Fitness and direction

- a factor that ranks higher D+2 net outcomes above lower ones has greater fitness;
- negating that score reverses the economic direction rather than being rescued by
  `abs` or an implicit sign flip;
- changing hit labels cannot change fitness;
- a lower-P&L factor cannot outrank a higher-P&L factor because of node count;
- node count resolves exact P&L ties only;
- the selection gate requires finite, strictly positive Top4 net P&L; and
- new factor specs store `sign=+1`, while legacy factor replay still honours saved
  signs.

### Leakage and reporting

- the last objective D0 maps to D+2 on the training end;
- September 3/4 boundary D0 rows never reach objective or auxiliary arrays;
- each D+1..D+10 horizon drops exits after the training end independently;
- market regimes use only D0 and earlier returns;
- Top10 output cannot affect the Top4 objective or admission result;
- the 30-factor validation uses fit metrics against selection P&L, not same-window
  self-correlation; and
- the generated report contains the root-cause ranking, validation p-values, exact
  training metadata, and no OOS section.

## 11. Documentation and Governance

`docs/risk/objective_pnl_alignment.md` is the evidence report. After its numbers are
reproduced, `docs/factor-governance.md` is updated as follows:

- D1 changes from **未决** to **已关闭** with the root cause, Top4 acceptance
  correlations, and a link to the report;
- the GP objective section replaces absolute hit-gini fitness with production Top4
  D+2-close net P&L and lexicographic complexity handling;
- the sign rule states that new P&L-mined factors encode direction in the expression;
- the auxiliary metric section preserves IC/gini/hit as monitoring only; and
- the global invariants add D+2-complete objective boundaries and exact alignment of
  fitness K/costs with `BacktestConfig`.

D5, D6, D13, D14, and D15 remain separate debts. In particular, this change uses the
currently configured statutory costs and fixed slippage; it does not claim to close the
unmodelled market-impact/capacity gap.

## 12. Completion Gate

The implementation is complete only when all of the following fresh checks succeed:

1. targeted red/green unit tests for objective arithmetic and direction;
2. regenerated training-only report reproducing all metadata and acceptance statistics;
3. explicit audit that no input/output row exceeds the formal training boundary;
4. full `pytest` with zero failures;
5. repository-wide `ruff check` with zero violations; and
6. a final diff review confirming that `helix/eval/backtest.py` execution behaviour and
   `helix/labels/touch_label.py` are unchanged.

