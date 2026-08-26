# AQuant L2 Factor Library v2

## Purpose

L2 v2 separates broad research feature eligibility from the strict single-factor
production gate:

```text
DRAFT -> COMPUTED -> RESEARCH_VALIDATED -> FEATURE_ELIGIBLE
                                      \-> STANDALONE_PRODUCTION_ALPHA
```

`STANDALONE_PRODUCTION_ALPHA` retains the existing strict production evidence gate.
It is not the only input to L3. A factor may remain useful as an alpha candidate, risk
factor, control feature, or state feature even when it cannot trade independently.

## Library scope

The pre-existing library contains 73 baseline factors and 28 second-wave candidates.
The v2 controlled technical generator adds 1,470 materializable candidates: 98 base
indicator/parameter variants times 15 representations. It covers SMA, EMA, WMA, MACD,
RSI, BOLL, KDJ, TR, ATR, ADX, DMI, CCI, ROC, MTM, CMO, RVI, TRIX, BIAS, OBV, VR, PSY,
Force Index, VHF, AMA, Coppock, Donchian, and a documented Hurst proxy.

The representations are level, normalized level, delta, slope, acceleration, trailing
percentile, z-score, threshold distance, cross state, state duration, recovery,
price/volume divergence, compression, and expansion. Parameter sets are finite and
source/research-hypothesis based. Generation aborts on duplicate factor identifiers or
expression hashes.

The published-formula layer adds 101 Alpha101 and 191 GTJA Alpha191 registrations.
Runtime golden-panel audit passes 291/292: Alpha101 passes 101/101 and GTJA passes
190/191. GTJA Alpha143 is cataloged but non-executable because `SELF` is undefined.
Faithfulness is split into `EXACT`, `NORMALIZED_EQUIVALENT`, `A_SHARE_ADAPTED`, and
`CORRECTED_AMBIGUITY`; raw and adopted expressions are stored side by side.

The academic extension layer adds 707 registrations. Han–Yang–Zhou contributes seven
auditable components and six two-component interactions. The China 7,000-rules source
contributes a declared, hash-stable 687-trial daily search space across filter, moving
average, support/resistance, channel breakout, and OBV-average families. Technical
sentiment contributes seven aggregates over ten pre-declared signals, without selecting
rules on the test period. Fama–French reuses the existing PIT size and book-to-market
features; operating profitability and investment remain dependency-gated.

The catalog maps candidates into the requested representation packs and the family packs
`technical_trend_v1`, `technical_momentum_v1`, `technical_reversal_v1`,
`technical_oscillator_v1`, `technical_channel_v1`, `technical_volatility_v1`,
`technical_volume_price_v1`, `technical_divergence_v1`,
`technical_state_duration_v1`, and `technical_regime_interaction_v1`.

## Safety and evaluation funnel

Research validation rejects leakage, non-reproducibility, formula errors, fabricated
dependencies, degenerate output, and severe unexplained coverage loss. Feature eligibility
then rejects exact duplicates, clear OOS directional collapse, and unacceptable
instability. It does not require high standalone ICIR or cost-adjusted strategy returns.

Family selection combines a Pareto frontier with required archetypes: strongest,
most stable, lowest turnover, most independent, and most regime-complementary.
Benjamini-Hochberg FDR, family-level summaries, and a deterministic stationary-bootstrap
SPA interface are available in the evaluation package. L3 uses time-ordered walk-forward
aggregation with purge and embargo; all L3 results remain `RESEARCH_ONLY` until L4
strategy validation.

## Reproducibility and batch operation

Run:

```bash
uv run python scripts/build_l2_factor_library_v2.py \
  --factor-pack technical_v2 --batch-size 32 --max-workers 2 --resume
```

The command also supports `--family`, `--start-date`, `--end-date`, `--dry-run`, and
`--force`. Its checkpoint binds factor hashes, source-registry hash, code version, and
configuration. Unchanged completed builds are skipped with `--resume`.

The size-neutral five-year evaluation is complete for 2,569 executable factors. It
produced 2,517 `RESEARCH_VALIDATED`, 140 `FEATURE_ELIGIBLE`, and 52 `REJECTED` factors.
The strict standalone-production gate remains separate and was not relaxed or inferred
from feature eligibility.

Behavioral deduplication uses only the first 80% daily RankIC series inside each family;
762 non-representative candidates are mapped to a near-duplicate representative at
absolute correlation 0.995. The 140-member pool is selected by required family
archetypes and Pareto diversity, with at most nine members per family. BH-FDR results
are recorded but weak single-factor significance does not itself hard-reject an L2.

The L3 integration smoke covers 19 multi-member families and five methods over three
expanding time-ordered folds. All preprocessing and model fitting use the training
window only. Results are `RESEARCH_ONLY`: the L2 shortlist used OOS evidence, so the
same history is not an untouched final L3 test. A nested L3 selection run or later data
release is required before L4 claims.

## Explicit partial/deferred scope

- `IMPLEMENTED`: Alpha101 101/101 and GTJA Alpha191 190/191 are executable.
- `IMPLEMENTED`: Han–Yang–Zhou 13, China controlled rules 687, technical sentiment 7.
- `PARTIAL`: GTJA Alpha143 remains `FORMULA_AMBIGUOUS`; no guessed `SELF` semantics.
- `PARTIAL`: Fama–French size and book-to-market are implemented; profitability and
  investment await standardized statement fields.
- `DEFERRED`: minute UTD/UTR and high-frequency indicator aggregation require minute data.
- `DEFERRED`: gross-profitability and asset-growth extensions require broader financial PIT fields.
- `COMPLETE`: size-neutral output/predictive-behavior deduplication and pool publication.
- `COMPLETE`: size-neutral L3 family integration smoke; result remains `RESEARCH_ONLY`.
- `RUNNING`: industry/style proxy robustness evaluation, 943/2,569 at last audit. This
  proxy is not true industry neutralization and does not block the size-neutral pool.
