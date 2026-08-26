# AQuant factor engineering platform

## Boundaries

The factor system consumes only immutable Standard/PIT releases. It never calls an
online provider. Financial inputs are selected with `ann_date < trade_date`; the common
contract is `available_at <= evaluation_time`. L0 data cleaning remains owned by the
history materializer.

The layers are:

- L0: Standard/PIT data and `data_release_id`.
- L1: deterministic mathematical, trailing time-series, cross-sectional and regression
  operators.
- L2: independently registered atomic and formula factors.
- L3: version-pinned feature sets, redundancy compression and alpha aggregation.

Registry membership, materialized factors and model feature sets are separate concepts.

## Reproducibility

A factor value is identified by trade date, security, factor id and factor version.
Materialization also records the data release, code version, configuration hash and
computation time. Output is written to a temporary directory, validated, and then
published atomically. Repeating an identical request returns the immutable manifest.

Formula factors use a Python-AST parser restricted to whitelisted fields and operators.
No `eval` or dynamic import is used. Depth, node, input-field and historical-window
limits are validated before execution.

## Baseline library

The first library contains 73 factors across size, value, quality, growth, momentum,
reversal, volatility, liquidity, price-volume and microcap-risk families. Only fields
present in the current Tushare Standard/PIT release are used. Unsupported event and
fundamental factors are not fabricated.

## Materialization

```bash
uv run --extra data --extra research python scripts/materialize_factors.py \
  --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
  --data-release-id cn_equity_20260717_001 \
  --factor-set baseline_v1 \
  --start-date 20260601 \
  --end-date 20260717
```

Use `--dry-run` to validate selection and print required fields without reading data.
Published Parquet is partitioned by factor family, year and month. The command prints
the manifest path and content hash.

## Evaluation

```bash
uv run --extra data --extra research python scripts/evaluate_factors.py \
  --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
  --data-release-id cn_equity_20260717_001 \
  --factor-set momentum_20d \
  --universe all_a_share \
  --start-date 20260601 \
  --end-date 20260717 \
  --horizons 1,5,10,20,40 \
  --batch-size 8
```

Reports are emitted as machine-readable JSON and human-readable Markdown. Evaluation
uses ordered forward labels. A factor observed after date T's close is first executable
on T+1; `return_start` is T+1 and `return_end` is the requested later trading session.
The CLI writes `evaluation_manifest.json` after each factor. Use `--resume` after an
interruption; random time-series splits are prohibited.

## Candidate and feature-set management

```bash
uv run python scripts/generate_factor_candidates.py \
  --parent-factor-id momentum_20d \
  --template-set price_volume_v1 \
  --budget 7

uv run python scripts/build_factor_feature_set.py \
  --feature-set-id microcap_weekly_v1 \
  --factor-set baseline_v1 \
  --data-release-id cn_equity_20260717_001 \
  --target-horizon 10

uv run python scripts/build_baseline_feature_sets.py \
  --selection artifacts/convergence/five_year_v1.json \
  --data-release-id cn_equity_20260717_001 \
  --effective-from 20260717 \
  --code-version <git-sha>
```

Candidate generation changes one dimension per batch, uses sparse windows and enforces
a family budget. Feature sets pin every factor version and the data release. The
baseline bundle command refuses compact selections outside 15–25 factors and publishes
`baseline_raw_v1`, `baseline_compact_v1` and `baseline_neutral_v1` atomically per
individual immutable specification. Its selection artifact must come from convergence;
the command does not invent or rank members.

## L3 training

The training dataset is an NPZ file containing ordered arrays `X`, `y`, and optional
historical `ic` or `icir`. The CLI uses purged walk-forward folds. The purge equals the
target horizon and prevents overlapping labels from entering training.

```bash
uv run --extra research python scripts/train_alpha_model.py \
  --feature-set-id microcap_weekly_v1 \
  --dataset artifacts/microcap_weekly_v1.npz \
  --model ridge \
  --target-horizon 10
```

Supported baselines are equal weight, IC weight, ICIR weight, Ridge, Elastic Net and
LightGBM.
L3 produces predictions and explanatory contributions; portfolio weights and orders
remain L4 responsibilities.

## Trading realism

The event backtest applies corporate actions at session open before matching. The
ledger supports cash-dividend entitlement/payment, stock dividends, capitalization,
rights issues, splits, reverse splits and delisting settlement. Current history data
loads implemented cash/stock dividends; rights, split and delisting event feeds remain
unavailable and are therefore not fabricated.

Matching uses unadjusted prices and a unified tradability decision covering listing and
delisting state, suspension, ST, limit-up buys, limit-down sells, one-price boards,
missing quotes, minimum listing age, minimum amount and exchange scope. Commission,
minimum commission, stamp duty, transfer fee, slippage, board lots, participation caps,
partial fills and delayed fills remain enforced by the execution layer.

## Research admission and convergence

Production thresholds live in `config/factors/production_gate_v1.yaml`. The gate
records a configuration hash, evidence hash, component scores and hard-veto reasons.
Future leakage, inadequate history, excessive missingness, weak out-of-sample evidence,
cost failure, redundancy without residual information, concentrated style exposure
and unstable extreme regimes can block production.

Convergence combines average cross-sectional factor-value Spearman correlation,
long-short return correlation and RankIC-series correlation. Hierarchical clusters
retain representatives while reporting marginal and conditional RankIC; redundant
definitions are archived, not deleted.

## Performance

Run a bounded, reproducible scenario with:

```bash
uv run --extra data --extra research python scripts/benchmark_factors.py \
  --scenario all_a_1y_50 \
  --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
  --data-release-id cn_equity_20260717_001
```

The JSON result records wall/CPU time, peak RSS, process I/O, panel bytes, workload and
unique expression counts, cache hit rate and throughput. Compute-only benchmarks report
zero Factor Store bytes; storage throughput must be measured with materialization.

## Safety checks

Run `make quality` before publishing research artifacts. A research release must also
pass the real Standard/PIT load, factor materialization, evaluation and event-backtest
smoke tests. A smoke-test return is only an engineering invariant; it is not production
research evidence.
