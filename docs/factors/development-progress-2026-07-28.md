# AQuant development progress — 2026-07-28

## Completed

- Built the factor platform core: immutable `FactorSpec`, registry, lineage,
  lifecycle, safe formula DSL, L1 operators, materialization, evaluation,
  controlled generation, clustering, feature sets and L3 aggregation.
- Added 73 versioned L2 baseline factors across 10 families.
- Added Standard/PIT loading with strict financial availability handling and
  deterministic, incremental, atomic factor publication.
- Added corporate-action accounting for dividends, stock distributions,
  capitalization, rights issues, splits, reverse splits and delisting settlement.
- Added a unified A-share tradability gate and execution constraints covering
  listing state, suspension, ST, price limits, one-price boards, liquidity,
  board lots, participation limits, partial fills, fees and slippage.
- Added institutional evaluation for 1/5/10/20/40-day horizons, explicit T+1
  execution timing, Newey-West statistics, regime slices, cost-adjusted returns,
  style exposures and a fixed final 20% out-of-sample holdout.
- Added three-view factor convergence using value Spearman correlation,
  long-short return correlation and RankIC-series correlation.
- Added configurable production gates, PIT neutralization, disk-backed
  convergence caches, versioned baseline feature-set builders and purged
  walk-forward alpha baselines.
- Added reproducible performance scenarios and updated operating documentation.

## Validation evidence

- The first full five-year, all-A-share evaluation completed for all 73 factors
  and all five horizons with no stderr output.
- First-pass runtime: 1,347.8 seconds; peak RSS: approximately 4.49 GiB.
- Measured compute scenarios include one-year/50-factor, five-year/50-factor,
  ten-year/73-factor, 500-workload cache and single-day incremental runs.
- Corporate-action, tradability and ledger tests: 23 targeted tests passed.
- The current full project gate is 594 passed with 88.40% total coverage.
- Ruff passes, all 295 files match Ruff formatting, and mypy reports no issues
  across 202 source files.

## Interrupted audited run

The content-addressed five-year audited run was interrupted when the Codex
desktop process exited. It stopped cleanly with:

- 20 of 73 audited factor reports completed;
- 21 of 73 factors written to the convergence cache;
- an intact `RUNNING` manifest and empty stderr;
- no evidence of data corruption or a Python exception.

The evaluation CLI validates report, configuration, code and cache hashes before
resuming. The next run can safely continue without trusting stale artifacts.

## Remaining acceptance work

1. Resume and complete the audited five-year run.
2. Produce the three correlation matrices and evidence-based 15–25 factor
   compact selection.
3. Apply the configured hard production gate; do not force 8–15 production
   factors if evidence is insufficient.
4. Publish `baseline_raw_v1`, `baseline_compact_v1` and
   `baseline_neutral_v1`.
5. Rerun the full quality gate and real-data backtest smoke test after the
   remaining audited research artifacts are complete.

Large Standard data, factor caches, generated reports and other local research
artifacts are deliberately excluded from Git.
