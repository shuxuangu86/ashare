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

### Single-factor L4 chain validation

- Added a verified materialized-factor reader and a generic PIT single-factor
  selector/execution strategy. The reader checks the immutable manifest,
  partition hashes, factor version, data release, valid rows and unique keys.
- Ran admitted `amount_concentration_20d@1.0.0` as a negative-direction,
  50-stock equal-weight portfolio with weekly T-close signals and T+1-open
  execution. Base costs include fees, 5 bps slippage, board lots and 0.5%
  volume participation; the existing limit, status and corporate-action ledger
  remain active.
- Real-data run `2026-06-01` through `2026-07-17`: 34 sessions, six
  rebalances, 491 orders/fills, 97.08% quantity fill rate, 7.2623 gross
  turnover and RMB 38,491.09 fees. T+1 attestation passed.
- Total return was -11.10% and maximum drawdown was 11.14%. This is a chain
  validation result, not evidence that a one-factor portfolio is deployable.
- Two identical runs produced content hash
  `c2ed845d489969865964648458f227f395c2144b2ecdf7221f7f1a835f14a6fc`
  and final-state hash
  `0fcee19f33f2d948200bc19088b1223981a469a63e932c33d9ac327845f29c23`.
- Final gate: 648 tests passed, 88.33% coverage, Ruff/format passed and mypy
  passed for 213 source files.

### Full-market factor-quintile L4 comparison

- Added direction-aware fractional selection and compared the admitted
  `amount_concentration_20d` best/worst 20% tails over all eligible Shanghai
  and Shenzhen A shares. Weekly T-close/T+1-open execution used RMB 1 billion
  per portfolio and the unchanged base-cost matcher.
- Real-data period `2026-01-01` through the latest common verified date
  `2026-07-17`: 129 sessions, 26 rebalances, 660,924 factor rows and 4,929
  eligible names at the first rebalance; each tail selected 985 names.
- Best-direction 20% returned -9.04% after costs; worst-direction 20% returned
  -20.55%; CSI All Share (`000985.CSI`) returned -4.18%. The factor spread was
  +11.51 percentage points, while the best tail lagged the benchmark by 4.87
  points. This remains `PASS_RESEARCH_ONLY`.
- Monthly JSON/Markdown output includes net return, filled-notional turnover,
  fees, modeled slippage and total friction. Artifact hash:
  `9f84f263a18549406bb86484ef4c7e53c4e9221888321eadad05ad902157f9d5`.
- The broad-universe run exposed fractional stock-dividend accounting without
  a cash-in-lieu reference. The loader now uses the ex-date unadjusted open,
  falling back only to the last prior visible close when that bar is absent;
  all 288 real 2026 stock-dividend records now have a reference price.
- Runtime was 1:59.47 with 2,760,436 KiB peak RSS.
- Final gate: 653 tests passed with 88.42% coverage; Ruff and format passed,
  and mypy passed for 214 source files.

### CSI 300 replication backtest calibration

- Added an immutable, hash-verified reader for local `index_weight` and
  `index_daily` Raw pages plus a deterministic CSI 300 replication strategy.
- Reconstructed regular June/December changes using the official next-session
  rule and explicitly configured the 2025-03-04 and 2025-09-05 temporary
  replacements. Later snapshot weights are backcast to effective-open weights
  using unadjusted prices and share-action multipliers.
- Fixed two backtest defects exposed by the calibration: rebalance sizing used
  stale signal-close equity, and missing/suspended constituent bars prevented
  index-like valuation and left orders unfilled. Portfolio sizing now accepts
  next-open valuation prices; calibration supplies deterministic last-visible
  closes while preserving the execution price contract.
- Zero-cost, zero-slippage replication from 2024-01-01 through 2025-12-31
  produced 2,148 orders and 2,148 fills. Annual returns were 14.6823% versus
  14.6833% in 2024 (-0.10 bp) and 17.7727% versus 17.6631% in 2025
  (+10.95 bp). Annualized daily tracking error was 4.83 bp; maximum monthly
  difference was 5.34 bp.
- Quality status: `PASS_DEBUG_CALIBRATION`. Artifact:
  `artifacts/backtest_debug/csi300/csi300-replication-8fed6755f7cc.json`;
  content hash:
  `8fed6755f7ccb71f2281332ff3b7e0674162c3778acaaa0b7048314e693333bf`.
- Runtime was 17.92 seconds with 1,041,428 KiB peak RSS.
- Final gate: 663 tests passed with 88.47% coverage; Ruff and format passed,
  and mypy passed for 216 source files.

### Wind microcap daily equal-weight replication

- Added a deterministic reference implementation for Wind
  `8841431.WI`: prior-close PIT selection, the smallest 400 Shanghai/Shenzhen
  A shares by total market capitalization, daily arithmetic equal weighting,
  zero fees/slippage, and no price-limit execution blocking.
- Historical eligibility excludes PIT ST/delisting-risk names and unopened IPO
  limit boards. Suspended names retain their last visible capitalization and
  contribute zero return until their next visible close.
- Real-data run from 2024-01-01 through 2025-12-31 processed 485 sessions and
  194,000 constituent-day rows. Replication returned 8.6437% versus Wind
  9.9972% in 2024, and 77.3652% versus Wind 81.6455% in 2025.
- Wind's exact public daily series overlaps August–December 2025; maximum
  monthly difference in that exact overlap is only 0.0696 percentage point.
  Older public chart data are weekly sampled and are diagnostic only.
- Quality status remains `FAILED_DEBUG_CALIBRATION`: maximum annual difference
  is 4.2802 percentage points, exceeding the configured 1-point gate. This
  indicates unresolved historical constituent-eligibility/source differences;
  no filter was tuned to force a pass.
- Artifact:
  `artifacts/backtest_debug/wind_microcap/wind-microcap-replication-47af475f7025.json`;
  content hash:
  `47af475f70259e902ca5b687baf92d11bae92271a84f872b085f89fdc534d918`.
- Runtime was 20.70 seconds with 1,821,568 KiB peak RSS.
- Final gate: 666 tests passed with 88.52% coverage; Ruff and format passed,
  and mypy passed for 217 source files.
