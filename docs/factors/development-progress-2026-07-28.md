# AQuant development progress - 2026-07-28

## Completed

- Built the factor platform core: immutable `FactorSpec`, registry, lineage,
  lifecycle, safe formula DSL, L1 operators, materialization, evaluation,
  controlled generation, clustering, feature sets and L3 aggregation.
- Added 73 versioned L2 baseline factors across 10 families.
- Added Standard/PIT loading, deterministic atomic publication, corporate-action
  accounting, unified tradability controls and realistic A-share execution.
- Added 1/5/10/20/40-day institutional evaluation with explicit T+1 timing,
  Newey-West statistics, regimes, costs, exposures and a final 20% OOS holdout.
- Added three-view convergence, configurable production gates, leakage
  attestation, PIT neutralization contracts, performance scenarios and purged
  date-grouped walk-forward Alpha baselines.

## Validation evidence

- The content-addressed five-year, all-A-share evaluation completed for all 73
  factors and all five horizons with `status=PASS` and no stderr output.
- Audited runtime: 1,374.88 seconds; peak RSS: approximately 4.50 GiB.
- Three-view convergence completed in 352.99 seconds (approximately 5.83 GiB
  peak RSS) and selected 15 cluster representatives.
- The unchanged `production_gate_v1` admitted `net_profit_growth_yoy` and
  `return_skewness_60d`. Aggregate status is `INSUFFICIENT_EVIDENCE`; no
  production set was forced.
- `baseline_raw_v1`, `baseline_compact_v1` and `baseline_neutral_v1` were
  published atomically as content-addressed `DRAFT` specifications.
- The full gate is 599 passed with 88.31% coverage; Ruff, formatting and mypy
  pass across 204 source files.
- The real-data execution/accounting smoke completed 34 sessions, 2,416 orders
  and 2,414 fills. Its return is wiring evidence, not a research claim.

## Remaining acceptance work

1. Add a versioned historical industry-membership source. The current release
   cannot support honest PIT industry neutralization, so the neutral feature set
   must remain `DRAFT`.
2. Investigate the recorded OOS and stability rejections and collect new
   evidence; do not weaken the existing gate to reach the target count.
3. Train full-data L3 comparisons only after a feature set becomes `VALIDATED`.

Large Standard data, factor caches, reports and local research artifacts are
deliberately excluded from Git.

## 2026-07-29 continuation

- `PIT industry data`: PARTIAL. Added SW2014/SW2021 Raw ingestion, Standard/PIT
  schema, `[effective_from,effective_to)` standardization, repository, atomic
  publisher, quarantine, manifests and quality validation.
- `Real data`: published `sw2021_industry_pit_20260717_v2`, bound to
  `cn_equity_20260717_001`, content hash
  `7bf8b896d1694036abbbbc3c973540dabbe727860bd1c9aae9d0650b45097f98`.
  The 2024-01-01 through 2026-07-17 quality window passes at 99.92%沪深
  stock-day coverage with zero conflicts, invalid intervals or orphan codes.
- `Five-year quality`: BLOCKED. Overall 2021-12-13 onward coverage is 96.22%,
  but annual coverage for 2021/2022/2023 is 91.72%/90.61%/92.14%, primarily
  missing early STAR Market memberships. The 95% annual gate was not changed.
- `SW2014 backfill`: BLOCKED by Tushare error 2002 (`token expired`). The
  resumable structured failure report is under
  `artifacts/industry-backfill/sw2014-industry-backfill-20260729-v1/`.
- `PIT neutralization`: implemented raw/size/industry/industry+size views,
  small-industry and missing-exposure statuses, daily diagnostics and quality
  attestation enforcement in `evaluate_factors.py`.
- `Real smoke`: one-day PIT query/neutralization passed for 5,090沪深 names
  (5,085 valid; weighted size exposure `-3.7e-15`). A 2024-Q1, 3-factor,
  5D/10D/20D neutral evaluation passed in 13.90 seconds with 455.5 MiB peak RSS.
- `73-factor five-year neutral reevaluation`: BLOCKED by the failed five-year
  industry quality gate; no convergence, admission or neutral feature-set
  lifecycle promotion was performed.
- `Production Alpha`: unchanged at 2. Formal L3 Alpha remains NOT STARTED.
- `Git commits`: `f7933a0`, `c4c7e2a`; the current neutralization commit is
  recorded after its quality gate completes.
