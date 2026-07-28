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
