# Placebo IC/Gini Calibration

## Calibration Inputs and Training Window

- Input: `data/raw/argus_quant_working.parquet`
- Target: `label_d2_hit_8pct`
- Seed: `20260813`
- Permutations: `1000`
- Minimum daily samples: `50`
- Training window: `2022-01-04` through `2024-09-04`
- Training dates: `649`
- Formal factors: `1`
- Supplemental factors: `29`

## Null Distribution Summary

| metric | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| ic_mean | 0.0205605152662 | 0.00149325345527 | 0.0155146784462 | 0.0205389149195 | 0.0249286004641 |
| icir | 0.276079058311 | 0.0214670393613 | 0.209565841888 | 0.276393181664 | 0.337391423632 |
| gini | 0.00328707941712 | 0.00245494157274 | 4.76208741945e-06 | 0.0026924750963 | 0.0128763615596 |

## Core Thresholds

| metric | p95 | p99 | p99.9 |
| --- | --- | --- | --- |
| ic_mean | 0.0230458043517 | 0.0238587331471 | 0.0246969818477 |
| icir | 0.311614377566 | 0.326832368246 | 0.334098790299 |
| gini | 0.00806281828144 | 0.0102819758641 | 0.0118939341546 |

## Formal Library Screening

### Formal factor details

| library_path | factor_name | ic_mean_signed | ic_mean | icir_signed | icir | gini_signed | gini | ic_level | icir_level | gini_level | overall_level | candidate_eligible | suggest_evict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data/artifacts/argus/event_factors.json | gp_000 | 0.129047992946 | 0.129047992946 | 1.41355629575 | 1.41355629575 | 0.294826247568 | 0.294826247568 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | True | False |

### Formal level counts

| overall_level | count |
| --- | --- |
| 超 p99.9 | 1 |

## Supplemental: argus_n40

Factors: 12

| library_path | factor_name | ic_mean_signed | ic_mean | icir_signed | icir | gini_signed | gini | ic_level | icir_level | gini_level | overall_level | candidate_eligible | suggest_evict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data/artifacts/argus_n40/event_factors.json | gp_000 | 0.0844682825972 | 0.0844682825972 | 1.12159684339 | 1.12159684339 | 0.217793616616 | 0.217793616616 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_001 | 0.0984283507281 | 0.0984283507281 | 1.12597503783 | 1.12597503783 | 0.261449354453 | 0.261449354453 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_002 | 0.0457037775452 | 0.0457037775452 | 0.648961641491 | 0.648961641491 | 0.161404396231 | 0.161404396231 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_003 | 0.105505991118 | 0.105505991118 | 1.51726351833 | 1.51726351833 | 0.234143735382 | 0.234143735382 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_004 | -0.0498090556893 | 0.0498090556893 | -0.615705547864 | 0.615705547864 | -0.0970722564674 | 0.0970722564674 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_005 | 0.0969822885653 | 0.0969822885653 | 1.22182450573 | 1.22182450573 | 0.239587383492 | 0.239587383492 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_006 | -0.0124525471679 | 0.0124525471679 | -0.154380628274 | 0.154380628274 | -0.06912850795 | 0.06912850795 | 低于随机水平 | 低于随机水平 | 超 p99.9 | 低于随机水平 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_007 | -0.0125173749792 | 0.0125173749792 | -0.155990047273 | 0.155990047273 | -0.0733360089624 | 0.0733360089624 | 低于随机水平 | 低于随机水平 | 超 p99.9 | 低于随机水平 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_008 | 0.0899537361939 | 0.0899537361939 | 1.16453732392 | 1.16453732392 | 0.228122329667 | 0.228122329667 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_009 | 0.102876938452 | 0.102876938452 | 1.30654186042 | 1.30654186042 | 0.251967135778 | 0.251967135778 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_010 | 0.0551373765162 | 0.0551373765162 | 0.744421515596 | 0.744421515596 | 0.121358530161 | 0.121358530161 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_n40/event_factors.json | gp_011 | 0.0776406282853 | 0.0776406282853 | 0.889028489207 | 0.889028489207 | 0.131659447671 | 0.131659447671 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |

### argus_n40 level counts

| overall_level | count |
| --- | --- |
| 超 p99.9 | 10 |
| 低于随机水平 | 2 |

## Supplemental: argus_multi

Factors: 17

| library_path | factor_name | ic_mean_signed | ic_mean | icir_signed | icir | gini_signed | gini | ic_level | icir_level | gini_level | overall_level | candidate_eligible | suggest_evict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data/artifacts/argus_multi/event_factors.json | gp_000 | 0.0904903453815 | 0.0904903453815 | 1.10502919539 | 1.10502919539 | 0.210001327576 | 0.210001327576 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_001 | 0.0857552997423 | 0.0857552997423 | 1.00290364139 | 1.00290364139 | 0.182056255293 | 0.182056255293 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_002 | 0.0863777863342 | 0.0863777863342 | 0.998078564658 | 0.998078564658 | 0.134898018157 | 0.134898018157 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_003 | 0.0833551133845 | 0.0833551133845 | 0.911700661825 | 0.911700661825 | 0.143643121039 | 0.143643121039 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_004 | 0.083569565306 | 0.083569565306 | 0.990907322487 | 0.990907322487 | 0.178772575711 | 0.178772575711 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_005 | 0.0676500907549 | 0.0676500907549 | 0.939889653338 | 0.939889653338 | 0.135384048692 | 0.135384048692 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_006 | 0.0668079593873 | 0.0668079593873 | 0.798321567872 | 0.798321567872 | 0.152380940164 | 0.152380940164 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_007 | 0.0892933782352 | 0.0892933782352 | 0.967105642345 | 0.967105642345 | 0.155818546185 | 0.155818546185 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_008 | 0.0854710239168 | 0.0854710239168 | 1.0224703225 | 1.0224703225 | 0.180100277564 | 0.180100277564 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_009 | 0.0706191623646 | 0.0706191623646 | 0.797859195569 | 0.797859195569 | 0.133188968494 | 0.133188968494 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_010 | 0.139513663932 | 0.139513663932 | 1.80600743685 | 1.80600743685 | 0.183292711556 | 0.183292711556 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_011 | 0.045795482424 | 0.045795482424 | 0.590067361253 | 0.590067361253 | 0.105946682263 | 0.105946682263 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_012 | 0.0898559849103 | 0.0898559849103 | 1.04477015195 | 1.04477015195 | 0.193256121399 | 0.193256121399 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_013 | 0.107908700017 | 0.107908700017 | 1.30702348636 | 1.30702348636 | 0.243816602907 | 0.243816602907 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_014 | 0.123939057427 | 0.123939057427 | 1.34425662193 | 1.34425662193 | 0.234646827439 | 0.234646827439 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_015 | 0.0762280163654 | 0.0762280163654 | 1.01265403013 | 1.01265403013 | 0.202027069102 | 0.202027069102 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |
| data/artifacts/argus_multi/event_factors.json | gp_016 | 0.0704922925985 | 0.0704922925985 | 1.02903961376 | 1.02903961376 | 0.20607083501 | 0.20607083501 | 超 p99.9 | 超 p99.9 | 超 p99.9 | 超 p99.9 | False |  |

### argus_multi level counts

| overall_level | count |
| --- | --- |
| 超 p99.9 | 17 |

## Governance Caveats

Only the formal library and training dates generated these thresholds. Out-of-sample rows
never entered the panel, factor computation, permutation RNG, null statistics, or thresholds.

Supplemental scopes never participated in threshold generation and are never eligible for formal admission.
Their levels are counterfactual comparisons against the formal thresholds only.

The canonical `daily_ic` currently applies ordinal ranks to the binary label. Consequently,
these IC placebo thresholds depend on the current event-slot layout, missingness structure,
and formal factor family. They are only the G1 baseline for the current event-table formal library.
A change to the dataset, event-slot layout, missingness structure, or formal factor family
requires recalibration.

Passing G1 does not override G3. A factor rejected by G3 remains rejected even when its G1
statistics exceed the placebo thresholds.
