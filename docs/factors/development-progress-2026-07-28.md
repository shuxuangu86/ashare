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

### Second-wave continuation

- Added 28 economically distinct, PIT-safe L2 candidates across residual
  momentum/reversal, quality-growth interactions, liquidity, tail risk,
  micro-cap interactions and daily price-volume structure. Exact numerical
  duplicate tests and future-mutation invariance tests pass.
- Five-year materialization published 182,899,752 rows in 263.59 seconds with
  3,167,252 KiB peak RSS. Materialization hash:
  `985aab0300310be3276c14d1e1bdc23c3a717b7a038f33d8f1a5c0cea873b07c`.
- The 28-factor, five-horizon evaluation completed 28/28 PASS in 696.38 seconds
  with 3,526,252 KiB peak RSS. Convergence cache hash:
  `da790cb93e5bb8f16383c4aefb2a07c6974d345db00f65c9fdc8d4c8c730c7f6`.
- Fixed the research split so convergence uses only the first 80% isolated
  selection interval and leaves the final 20% holdout untouched.
- Added structured feature roles, per-factor admission reason codes,
  lifecycle counts, Markdown admission summaries and Parquet decision tables.
- Added versioned Feature Set roles, evaluation windows, lineage, lifecycle
  status and deterministic hashes that exclude only the observational
  `created_at` timestamp.
- A unified 101-factor沪深 five-year reevaluation is `RUNNING`; its reports and
  convergence cache checkpoint each completed factor.

### Unified evaluation and admission result

- Unified five-year evaluation completed 101/101 factors across
  1D/5D/10D/20D/40D in 29:34.26 with 4,693,632 KiB peak RSS. Evaluation hash:
  `9f55322d54a8e74a28d26f0287e205d2e167587cb6a98398814466a2211590a9`.
- Holdout-safe three-view convergence completed in 5:41.36 with 2,547,884 KiB
  peak RSS and selected 17 compact candidates. Selection hash:
  `f5a637a4cca288334bc520577bbcee4bcaa8a20c2e7b804c599ae8ecab8e40dc`.
- Strict admission returned `INSUFFICIENT_EVIDENCE`: one Production Alpha
  (`amount_concentration_20d`), one Risk Factor, 11 `FAILED_GATE`, and four
  `INSUFFICIENT_EVIDENCE`. Gate thresholds were unchanged.
- Rebuilt feature sets deterministically: raw 101-factor DRAFT hash
  `c901f6c7...57ab7`, compact 17-factor VALIDATED hash
  `7780356a...60254`, and neutral 17-factor DRAFT hash
  `ba1de144...d64b4`. A second build produced identical hashes.
- Formal L3 Alpha training is `BLOCKED` because Production Alpha count is one,
  below the attested minimum of eight. The CLI now rejects formal training
  without a passing admission artifact.

### TinyShare source recovery

- Installed and locked `tinyshare==0.1036.0`; the authorization code is stored
  only in ignored local `.env`. A real daily query returned six expected rows.
- Added a TinyShare transport adapter without changing immutable Raw response,
  checkpoint, manifest or normalization contracts.
- SW2014 backfill completed 359 classifications and 58,086 membership rows with
  zero failures in 48.12 seconds (124,908 KiB peak RSS).
- Published combined `sw_industry_pit_20260717_v3`, hash
  `3dcd98d0461e926d9c6322be261ef22302d2b5dc23123de07f4bba6369e86a6b`.
- Five-year industry quality remains `BLOCKED`: overall coverage is 95.93%,
  while 2021/2022/2023 are 92.18%/90.61%/92.14%. This is a source-history
  completeness failure, not an access failure; the annual 95% gate is intact.
- Final quality gate: 635 tests passed, total coverage 88.32%, Ruff and format
  checks passed, and mypy passed for 211 source files.
- Real event/accounting smoke passed 34 sessions, 2,416 orders and 2,414 fills
  in 17.67 seconds with 1,155,308 KiB peak RSS. It remains wiring evidence only.
