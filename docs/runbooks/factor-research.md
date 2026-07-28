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
