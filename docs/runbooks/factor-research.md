# Factor research runbook

1. Confirm the immutable Standard/PIT release manifest and quality report.
2. Run factor materialization with `--dry-run`, then run the requested date range.
3. Preserve the printed materialization manifest and content hash.
4. Evaluate horizons 1, 5, 10, 20 and 40 using T+1-start ordered labels.
5. Reject quality-failed factors before predictive evaluation.
6. Record every candidate attempt, including failures.
7. Apply FDR, walk-forward, purge and embargo controls.
8. Cluster correlated factors; archive redundant members without deleting history.
9. Build a version-pinned feature set.
10. Compare every L3 model with equal-weight, IC-weight and Ridge baselines.

For a long evaluation, use `--batch-size 8`. If interrupted, rerun the identical
command with `--resume`; only reports with both JSON and Markdown artifacts are skipped.
Inspect `evaluation_manifest.json` and require `status=PASS` and matching factor counts.

Before promotion, load `config/factors/production_gate_v1.yaml`, preserve its hash and
the evidence hash, and require a passing hard gate. Threshold changes require a new
versioned configuration, never an edit to an already published research artifact.

```bash
uv run python scripts/converge_factors.py \
  --cache-root artifacts/convergence/five_year_v2 \
  --report-dir reports/institutional-5y-audited-v2 \
  --output artifacts/convergence/five_year_v2.json --horizon 5

uv run python scripts/attest_factor_leakage.py \
  --data-release-id cn_equity_20260717_001 --code-version <evaluated-revision> \
  --output artifacts/admission/leakage.json

uv run python scripts/admit_factors.py \
  --convergence artifacts/convergence/five_year_v2.json \
  --report-dir reports/institutional-5y-audited-v2 \
  --evaluation-manifest reports/institutional-5y-audited-v2/evaluation_manifest.json \
  --gate-config config/factors/production_gate_v1.yaml \
  --leakage-attestation artifacts/admission/leakage.json \
  --output artifacts/admission/five_year_v2.json
```

Admission is deliberately non-forcing: fewer than eight passing factors produces
`INSUFFICIENT_EVIDENCE`, never an automatically weakened gate.

## PIT industry neutral evaluation

Publish and validate industry data before requesting an industry view. The
quality report must cover the complete evaluation range and attest the exact
industry release hash; a blocked report is rejected by the evaluation CLI.

```bash
uv run python scripts/materialize_industry_pit.py \
  --start-date 20211213 --end-date 20260717 \
  --classification-system SW2021 --industry-level L1,L2,L3 \
  --release-id sw2021_industry_pit_20260717_v2 \
  --data-release-id cn_equity_20260717_001 \
  --raw-state artifacts/tushare-backfill/state.sqlite3 \
  --history-release-dir data/standard/history-release=cn_equity_history_20260717_001

uv run python scripts/validate_industry_pit.py \
  --industry-release data/standard/industry-release=sw2021_industry_pit_20260717_v2 \
  --history-release data/standard/history-release=cn_equity_history_20260717_001 \
  --start-date 20240101 --end-date 20260717

uv run python scripts/evaluate_factors.py \
  --release-dir data/standard/history-release=cn_equity_history_20260717_001 \
  --data-release-id cn_equity_20260717_001 --factor-set baseline_v1 \
  --start-date 20240101 --end-date 20260717 --horizons 1,5,10,20,40 \
  --neutralization industry_size \
  --industry-release data/standard/industry-release=sw2021_industry_pit_20260717_v2 \
  --industry-quality-report \
  artifacts/data_quality/industry_pit/sw2021_industry_pit_20260717_v2/20240101_20260717/quality.json
```

`--neutralization size` uses only same-day float market capitalization and does
not require an industry release. Supported views are `raw`, `size`, `industry`
and `industry_size`.

## Release gate

```bash
make quality
uv run --extra data --extra research python scripts/materialize_factors.py --help
uv run --extra data --extra research python scripts/evaluate_factors.py --help
uv run --extra data --extra research python scripts/benchmark_factors.py --help
make microcap-history-smoke
```

A failed materialization must leave no published directory. Do not manually rename
temporary directories into place. Do not edit a historical manifest.

## Troubleshooting

- `unsupported Standard/PIT fields`: the factor is not supported by this data release;
  do not substitute or fabricate a field.
- `range must be visible as-of`: move the evaluation time forward or shorten the range.
- low coverage or all-null: inspect PIT availability and universe alignment before IC.
- duplicate expression: reuse the existing factor version or register an explicit,
  economically justified variant.
- model dataset error: regenerate finite, ordered `X`/`y`; never shuffle dates.
- interrupted evaluation: verify arguments are unchanged, then use `--resume`.
- corporate-action mismatch: block the run; never force balance by editing cash or
  positions outside the ledger.

## Artifact locations

- Factor values: `data/factors/materialization=<sha256>/`
- Materialization metadata: `manifest.json` inside the immutable directory
- Factor reports: `reports/factors/`
- Candidate batches: `artifacts/candidates.json`
- Feature sets: `artifacts/feature_sets/`
- L3 predictions: `artifacts/alpha_predictions.npy`
- Evaluation checkpoint: `reports/<run>/evaluation_manifest.json`
- Performance records: `reports/benchmarks/*.json`
- Production gate configuration: `config/factors/production_gate_v1.yaml`

## TinyShare source configuration

Install the locked data extra and configure the ignored local environment:

```bash
uv sync --extra data
TUSHARE_ENDPOINT=tinyshare://pro
```

`TUSHARE_TOKEN` carries the TinyShare authorization code. Never place it in a
tracked configuration file or command log. The adapter preserves AQuant's
existing immutable Raw envelope, checkpoint and manifest contracts.

## Single-factor L4 chain smoke

Only factors explicitly marked `PRODUCTION` in an attested admission artifact
may enter this command. It is still research-only until a production feature
set and portfolio policy are approved.

```bash
uv run python scripts/run_single_factor_l4_backtest.py \
  --history-release data/standard/history-release=cn_equity_history_20260717_001 \
  --factor-materialization data/factors/materialization=<materialization-id> \
  --admission artifacts/admission/<run>/admission_summary.json \
  --factor-id amount_concentration_20d \
  --start-date 20260601 \
  --end-date 20260717 \
  --target-count 50 \
  --frequency weekly \
  --cost-scenario base-cost
```

The command verifies materialized partition hashes, uses the factor's declared
direction, ranks at T close, trades at T+1 open and writes deterministic
JSON/Markdown evidence under `artifacts/l4_single_factor/`.
