from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from aquant.factors.atomic import (
    baseline_factor_library,
    second_wave_candidate_library,
    technical_factor_library_v2,
)
from aquant.factors.atomic.models import AtomicFactor
from aquant.factors.sources import SourceRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish the AQuant L2 Factor Library v2 catalog")
    parser.add_argument("--factor-pack", default="all")
    parser.add_argument("--family")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/factor_library_v2"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0 or not 1 <= args.max_workers <= 2:
        raise ValueError("batch-size must be positive and max-workers must be in [1, 2]")
    started = time.perf_counter()
    baseline = baseline_factor_library()
    second_wave = second_wave_candidate_library()
    technical = technical_factor_library_v2()
    packs = {
        "baseline_v1": baseline,
        "second_wave_v1": second_wave,
        "technical_v2": technical,
        "all": (*baseline, *second_wave, *technical),
    }
    if args.factor_pack not in packs:
        raise ValueError(f"unknown factor pack: {args.factor_pack}")
    factors = packs[args.factor_pack]
    if args.family:
        factors = tuple(factor for factor in factors if factor.spec.family == args.family)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args, factors), indent=2, sort_keys=True))
        return 0
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry.from_yaml(Path("configs/factors/source_registry.yaml"))
    input_hash = _input_hash(factors, registry.content_hash, args)
    checkpoint = output / "build_checkpoint.json"
    if args.resume and not args.force and checkpoint.exists():
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous.get("input_hash") == input_hash and previous.get("status") == "PASS":
            print(json.dumps(previous, indent=2, sort_keys=True))
            return 0

    rows = [_catalog_row(factor) for factor in factors]
    pq.write_table(
        pa.Table.from_pylist(rows), output / "factor_catalog.parquet", compression="zstd"
    )
    _write_json(output / "source_coverage.json", _source_coverage(registry, factors))
    _write_json(output / "implementation_status.json", _implementation_status(factors))
    _write_json(output / "duplicate_report.json", _duplicate_report(factors))
    _write_json(
        output / "evaluation_summary.json",
        {
            "schema_version": "aquant.factor-evaluation-summary.v2",
            "status": "NOT_STARTED",
            "reason_code": "AWAITING_MATERIALIZATION_AND_OOS_EVALUATION",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "factor_count": len(factors),
            "data_release_id": None,
            "code_version": _git_revision(),
            "config_hash": input_hash,
        },
    )
    for name, tier in (
        ("research_validated_pool.json", "RESEARCH_VALIDATED"),
        ("feature_eligible_pool.json", "FEATURE_ELIGIBLE"),
        ("standalone_production_pool.json", "STANDALONE_PRODUCTION_ALPHA"),
        ("rejected_pool.json", "REJECTED"),
    ):
        _write_json(
            output / name,
            {
                "schema_version": "aquant.factor-pool.v2",
                "tier": tier,
                "status": "NOT_EVALUATED",
                "factor_ids": [],
                "reason_codes": ["AWAITING_OOS_EVALUATION"],
                "content_hash": hashlib.sha256(b"[]").hexdigest(),
            },
        )
    _write_json(
        output / "l3_family_smoke.json",
        {
            "schema_version": "aquant.l3-family-smoke.v1",
            "status": "RESEARCH_ONLY_NOT_STARTED",
            "methods": [
                "equal_weight",
                "rolling_ic_weight",
                "rolling_icir_weight",
                "ridge",
                "elastic_net",
            ],
            "reason_code": "AWAITING_FEATURE_ELIGIBLE_POOL",
        },
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": "aquant.factor-library-build.v2",
        "status": "PASS",
        "input_hash": input_hash,
        "factor_count": len(factors),
        "new_technical_factor_count": len(technical),
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "elapsed_seconds": elapsed,
        "code_version": _git_revision(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(checkpoint, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _dry_run_payload(args: argparse.Namespace, factors: tuple[AtomicFactor, ...]) -> dict[str, Any]:
    return {
        "status": "DRY_RUN",
        "factor_pack": args.factor_pack,
        "family": args.family,
        "factor_count": len(factors),
        "batch_count": (len(factors) + args.batch_size - 1) // args.batch_size,
        "max_workers": args.max_workers,
    }


def _catalog_row(factor: AtomicFactor) -> dict[str, Any]:
    spec = factor.spec
    return {
        "factor_id": spec.factor_id,
        "version": spec.version,
        "display_name": spec.name,
        "family": spec.family,
        "subfamily": spec.subfamily,
        "role": str(spec.role),
        "source_id": spec.source_id,
        "source_formula_id": spec.source_formula_id,
        "source_faithfulness": str(spec.source_faithfulness),
        "formula": spec.expression or spec.implementation,
        "parameters_json": json.dumps(spec.parameters, sort_keys=True, separators=(",", ":")),
        "required_columns": list(spec.input_fields),
        "required_datasets": list(spec.required_datasets),
        "lookback": spec.required_history,
        "minimum_periods": spec.minimum_periods,
        "availability_lag": spec.availability_lag,
        "normalization": spec.normalization,
        "complexity_score": spec.complexity_score,
        "variant_of": spec.variant_of,
        "implementation_status": str(spec.implementation_status),
        "coverage": None,
        "oos_rank_ic": None,
        "oos_rank_icir": None,
        "correlation_cluster": None,
        "lifecycle_tier": str(spec.status),
        "reason_codes": ["NOT_EVALUATED"],
        "content_hash": spec.expression_hash,
    }


def _source_coverage(
    registry: SourceRegistry,
    factors: tuple[AtomicFactor, ...],
) -> dict[str, Any]:
    by_source: dict[str, list[AtomicFactor]] = defaultdict(list)
    for factor in factors:
        by_source[factor.spec.source_id].append(factor)
    sources = []
    for source in registry.sources:
        source_factors = by_source[source.source_id]
        sources.append(
            {
                "source_id": source.source_id,
                "implementation_status": str(source.implementation_status),
                "registered_variants": len(source_factors),
                "base_formulas": len(
                    {
                        factor.spec.source_formula_id
                        for factor in source_factors
                        if factor.spec.source_formula_id
                    }
                ),
                "factor_ids": [factor.spec.factor_id for factor in source_factors],
            }
        )
    return {
        "schema_version": "aquant.factor-source-coverage.v2",
        "sources": sources,
        "unregistered_internal": sum(
            factor.spec.source_id == "SRC_AQUANT_INTERNAL" for factor in factors
        ),
    }


def _implementation_status(factors: tuple[AtomicFactor, ...]) -> dict[str, Any]:
    counts = Counter(str(factor.spec.implementation_status) for factor in factors)
    return {
        "schema_version": "aquant.factor-implementation-status.v2",
        "counts": dict(sorted(counts.items())),
        "formula_ambiguities": [
            {
                "source_id": "SRC_ALPHA101",
                "status": "FORMULA_AMBIGUOUS",
                "notes": (
                    "Formula-by-formula transcription and VWAP/industry semantics "
                    "remain pending."
                ),
            },
            {
                "source_id": "SRC_GTJA_ALPHA191",
                "status": "SOURCE_UNAVAILABLE",
                "notes": "No licensed local report; public transcriptions are not marked EXACT.",
            },
        ],
        "deferred_intraday": [
            {
                "source_id": "SRC_DONGWU_TECH_SERIES",
                "status": "DEFERRED_INTRADAY",
            },
            {
                "source_id": "SRC_CSC_TECH_HF_20240623",
                "status": "DEFERRED_INTRADAY",
            },
        ],
        "missing_data_dependencies": [
            "gross_profit financial PIT fields",
            "asset growth financial PIT fields",
            "minute bars and minute turnover",
        ],
    }


def _duplicate_report(factors: tuple[AtomicFactor, ...]) -> dict[str, Any]:
    expressions: dict[str, list[str]] = defaultdict(list)
    for factor in factors:
        expressions[factor.spec.expression_hash].append(factor.spec.factor_id)
    duplicates = [ids for ids in expressions.values() if len(ids) > 1]
    return {
        "schema_version": "aquant.factor-duplicates.v2",
        "semantic_duplicate_groups": [],
        "expression_duplicate_groups": duplicates,
        "output_duplicate_groups": [],
        "predictive_duplicate_groups": [],
        "status": "EXPRESSION_COMPLETE_OUTPUT_PENDING",
    }


def _input_hash(
    factors: tuple[AtomicFactor, ...],
    source_hash: str,
    args: argparse.Namespace,
) -> str:
    payload = {
        "factor_hashes": sorted(factor.spec.expression_hash for factor in factors),
        "source_hash": source_hash,
        "factor_pack": args.factor_pack,
        "family": args.family,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
