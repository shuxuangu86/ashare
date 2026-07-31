import argparse
import gc
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import numpy as np

from aquant.data.industry import IndustryPITRepository
from aquant.domain.data_release import DataReleaseId
from aquant.factors.aggregation import ModelKind, walk_forward_predict
from aquant.factors.atomic import (
    AtomicFactor,
    academic_extension_library,
    alpha101_original_library,
    baseline_factor_library,
    gtja191_original_library,
    second_wave_candidate_library,
    technical_factor_library_v2,
)
from aquant.factors.data_loader import StandardPITFactorLoader
from aquant.factors.evaluation import (
    basic_style_exposures,
    evaluate_institutional,
    historical_market_regimes,
    pit_forward_return_labels,
)
from aquant.factors.evaluation.ic import information_coefficient
from aquant.factors.evaluation.quality import evaluate_quality
from aquant.factors.feature_sets import (
    FactorMember,
    FeatureRole,
    FeatureSetRegistry,
    FeatureSetSpec,
    FeatureSetStatus,
    build_baseline_feature_sets,
)
from aquant.factors.generation import generate_window_variants
from aquant.factors.materialization import FactorMaterializationEngine, MaterializationRequest
from aquant.factors.preprocessing import (
    load_pit_exposures,
    load_size_exposures,
    neutralize_pit_factor,
)
from aquant.factors.reporting import FactorReport, write_factor_report, write_feature_set_report
from aquant.factors.selection import ConvergenceCache

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc


def selected_factors(value: str) -> tuple[AtomicFactor, ...]:
    baseline = baseline_factor_library()
    second_wave = second_wave_candidate_library()
    technical = technical_factor_library_v2()
    academic = academic_extension_library()
    alpha101 = alpha101_original_library()
    alpha191 = gtja191_original_library()
    legacy = (*baseline, *second_wave)
    published = (*alpha101, *alpha191)
    library = (*legacy, *technical, *published, *academic)
    if value == "baseline_v1":
        return baseline
    if value == "second_wave_v1":
        return second_wave
    if value == "all":
        return legacy
    if value == "technical_v2":
        return technical
    if value == "alpha101_original_v1":
        return alpha101
    if value == "alpha191_original_v1":
        return alpha191
    if value == "published_formulas_v1":
        return published
    if value == "academic_extensions_v1":
        return academic
    if value == "han_yang_zhou_2013_v1":
        return tuple(
            factor for factor in academic if factor.spec.source_id == "SRC_HAN_YANG_ZHOU_2013"
        )
    if value == "china_7000_rules_controlled_v1":
        return tuple(
            factor for factor in academic if factor.spec.source_id == "SRC_CHINA_7000_RULES"
        )
    if value == "technical_sentiment_2023_v1":
        return tuple(
            factor for factor in academic if factor.spec.source_id == "SRC_TECH_SENTIMENT_2023"
        )
    if value == "fama_french_2015_partial_v1":
        return tuple(
            factor for factor in baseline if factor.spec.source_id == "SRC_FAMA_FRENCH_2015"
        )
    if value == "l2_v2":
        return library
    if value.startswith("technical_") and value.endswith("_v1"):
        selected = tuple(
            factor
            for factor in technical
            if value in factor.spec.parameters.get("factor_packs", ())
        )
        if selected:
            return selected
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    by_id = {factor.spec.factor_id: factor for factor in library}
    missing = set(requested) - by_id.keys()
    if missing:
        raise ValueError(f"unknown factor ids: {sorted(missing)}")
    return tuple(by_id[item] for item in requested)


def _release(value: str) -> DataReleaseId:
    return DataReleaseId(value)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluation_report_matches(
    json_path: Path,
    markdown_path: Path,
    *,
    data_release_id: str,
    config_hash: str,
    code_version: str,
) -> bool:
    if not json_path.is_file() or not markdown_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        expected_content_hash = payload.pop("content_hash")
        actual_content_hash = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        provenance = payload["extra_metrics"]["provenance"]
        return bool(
            expected_content_hash == actual_content_hash
            and payload["data_release_id"] == data_release_id
            and provenance["config_hash"] == config_hash
            and provenance["code_version"] == code_version
        )
    except (json.JSONDecodeError, KeyError, OSError, TypeError):
        return False


def _verified_industry_quality(
    path: Path,
    *,
    industry_repository: IndustryPITRepository,
    data_release_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("content_hash", None)
    payload.pop("created_at", None)
    if expected != _hash(payload):
        raise ValueError("industry quality report content hash mismatch")
    if payload.get("status") != "PASS":
        raise ValueError("neutral evaluation requires a passing industry quality report")
    if payload.get("data_release_id") != data_release_id:
        raise ValueError("industry quality data release does not match evaluation")
    if payload.get("industry_release_hash") != industry_repository.content_hash:
        raise ValueError("industry quality report does not attest the selected industry release")
    if (
        date.fromisoformat(payload["start_date"]) > start_date
        or date.fromisoformat(payload["end_date"]) < end_date
    ):
        raise ValueError("industry quality report does not cover the evaluation range")
    return {**payload, "content_hash": expected}


def materialize_factors_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize audited AQuant factors")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/factors"))
    parser.add_argument("--data-release-id", type=_release, required=True)
    parser.add_argument("--factor-set", default="baseline_v1")
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--universe", default="all_a_share")
    parser.add_argument("--code-version", default="working-tree")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        factors = selected_factors(args.factor_set)
        config = {
            "factor_set": args.factor_set,
            "universe": args.universe,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
        }
        if args.dry_run:
            _print(
                {
                    "status": "DRY_RUN",
                    "factor_count": len(factors),
                    "required_fields": sorted(
                        {field for factor in factors for field in factor.spec.input_fields}
                    ),
                    "config_hash": _hash(config),
                }
            )
            return 0
        as_of = datetime.combine(args.end_date, datetime.max.time(), _SHANGHAI)
        request = MaterializationRequest(
            args.data_release_id,
            factors,
            args.start_date,
            args.end_date,
            as_of,
            args.universe,
            StandardPITFactorLoader(args.release_dir),
            None,
            config,
            args.code_version,
            _hash(config),
        )
        manifest = FactorMaterializationEngine(args.output_root).materialize(request)
        _print(
            {
                "status": manifest.status,
                "manifest": str(
                    args.output_root
                    / f"materialization={manifest.materialization_id}"
                    / "manifest.json"
                ),
                "rows": manifest.total_rows,
                "content_hash": manifest.content_hash,
            }
        )
        return 0
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


def evaluate_factors_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate factors on ordered forward labels")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/factors"))
    parser.add_argument("--data-release-id", type=_release, required=True)
    parser.add_argument("--factor-set", default="baseline_v1")
    parser.add_argument("--universe", default="all_a_share")
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--horizons", default="1,5,10,20,40")
    parser.add_argument("--cost-bps", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--oos-fraction", type=float, default=0.2)
    parser.add_argument("--code-version", default="working-tree")
    parser.add_argument("--convergence-cache", type=Path)
    parser.add_argument(
        "--reload-per-batch",
        action="store_true",
        help="trade I/O for lower retained memory; default loads all factor inputs once",
    )
    parser.add_argument(
        "--neutralization",
        choices=("raw", "size", "industry", "industry_size"),
        default="raw",
    )
    parser.add_argument("--industry-release", type=Path)
    parser.add_argument("--industry-quality-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        factors = selected_factors(args.factor_set)
        horizons = tuple(int(item) for item in args.horizons.split(","))
        if (
            not horizons
            or any(item <= 0 for item in horizons)
            or args.cost_bps < 0
            or args.batch_size <= 0
            or not 0 < args.oos_fraction <= 0.5
        ):
            raise ValueError("evaluation horizons, costs, batch and OOS fraction are invalid")
        industry_repository: IndustryPITRepository | None = None
        industry_quality: dict[str, Any] | None = None
        if args.neutralization in {"industry", "industry_size"}:
            if args.industry_release is None or args.industry_quality_report is None:
                raise ValueError(
                    "neutral evaluation requires --industry-release and --industry-quality-report"
                )
            industry_repository = IndustryPITRepository(args.industry_release)
            industry_quality = _verified_industry_quality(
                args.industry_quality_report,
                industry_repository=industry_repository,
                data_release_id=str(args.data_release_id),
                start_date=args.start_date,
                end_date=args.end_date,
            )
        if args.dry_run:
            _print({"status": "DRY_RUN", "factor_count": len(factors), "horizons": horizons})
            return 0
        reports: list[tuple[str, str]] = []
        completed: list[str] = []
        manifest_path = args.report_dir / "evaluation_manifest.json"
        primary = 5 if 5 in horizons else horizons[0]
        evaluation_config = {
            "factor_set": args.factor_set,
            "universe": args.universe,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "horizons": horizons,
            "cost_bps": args.cost_bps,
            "oos_fraction": args.oos_fraction,
            "neutralization": args.neutralization,
            "industry_release_hash": (
                industry_repository.content_hash if industry_repository is not None else None
            ),
            "industry_quality_hash": (
                industry_quality["content_hash"] if industry_quality is not None else None
            ),
            "reload_per_batch": args.reload_per_batch,
        }
        config_hash = _hash(evaluation_config)
        convergence_cache: ConvergenceCache | None = None
        if (
            args.convergence_cache is not None
            and (args.convergence_cache / "metadata.json").is_file()
        ):
            convergence_cache = ConvergenceCache(args.convergence_cache)
            if (
                convergence_cache.factor_ids != tuple(factor.spec.factor_id for factor in factors)
                or convergence_cache.data_release_id != str(args.data_release_id)
                or convergence_cache.config_hash != config_hash
            ):
                raise ValueError("existing convergence cache does not match evaluation")
        resumable: set[str] = set()
        if args.resume:
            for factor in factors:
                json_report = (
                    args.report_dir / f"{factor.spec.factor_id}-{factor.spec.version}.json"
                )
                markdown_report = (
                    args.report_dir / f"{factor.spec.factor_id}-{factor.spec.version}.md"
                )
                report_matches = _evaluation_report_matches(
                    json_report,
                    markdown_report,
                    data_release_id=str(args.data_release_id),
                    config_hash=config_hash,
                    code_version=args.code_version,
                )
                cache_matches = args.convergence_cache is None or (
                    convergence_cache is not None
                    and factor.spec.factor_id in convergence_cache.factor_hashes
                )
                if report_matches and cache_matches:
                    resumable.add(factor.spec.factor_id)
                    completed.append(factor.spec.factor_id)
                    reports.append((str(json_report), str(markdown_report)))
        shared_panel = None
        if not args.reload_per_batch:
            shared_pending = tuple(
                factor for factor in factors if factor.spec.factor_id not in resumable
            )
            if shared_pending:
                shared_fields = tuple(
                    sorted(
                        {"close", "float_market_cap", "turnover_rate"}
                        | {field for factor in shared_pending for field in factor.spec.input_fields}
                    )
                )
                shared_panel = StandardPITFactorLoader(args.release_dir).load(
                    fields=shared_fields,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    as_of_time=datetime.combine(
                        args.end_date,
                        datetime.max.time(),
                        _SHANGHAI,
                    ),
                    universe_id=args.universe,
                    data_release_id=args.data_release_id,
                )
        for offset in range(0, len(factors), args.batch_size):
            batch = factors[offset : offset + args.batch_size]
            pending = tuple(factor for factor in batch if factor.spec.factor_id not in resumable)
            if not pending:
                continue
            fields = tuple(
                sorted(
                    {"close", "float_market_cap", "turnover_rate"}
                    | {field for factor in pending for field in factor.spec.input_fields}
                )
            )
            panel = shared_panel
            if panel is None:
                panel = StandardPITFactorLoader(args.release_dir).load(
                    fields=fields,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    as_of_time=datetime.combine(
                        args.end_date,
                        datetime.max.time(),
                        _SHANGHAI,
                    ),
                    universe_id=args.universe,
                    data_release_id=args.data_release_id,
                )
            pit_exposures = None
            if industry_repository is not None:
                pit_exposures = load_pit_exposures(panel, industry_repository)
            elif args.neutralization == "size":
                pit_exposures = load_size_exposures(panel)
            labelled = {
                horizon: pit_forward_return_labels(
                    panel.fields["close"],
                    panel.trade_dates,
                    horizon,
                )
                for horizon in horizons
            }
            exposures = basic_style_exposures(
                panel.fields["close"],
                panel.fields["float_market_cap"],
                panel.fields["turnover_rate"],
            )
            regimes = historical_market_regimes(panel.fields["close"])
            oos_start = max(1, int(len(panel.trade_dates) * (1 - args.oos_fraction)))
            if oos_start >= len(panel.trade_dates):
                raise ValueError("evaluation range is too short for an OOS holdout")
            if args.convergence_cache is not None and convergence_cache is None:
                convergence_cache = ConvergenceCache.create(
                    args.convergence_cache,
                    factor_ids=tuple(factor.spec.factor_id for factor in factors),
                    trade_dates=panel.trade_dates,
                    ts_codes=panel.ts_codes,
                    close=panel.fields["close"],
                    data_release_id=str(args.data_release_id),
                    config_hash=config_hash,
                )
            for factor in pending:
                values = factor.compute_array(panel)
                if pit_exposures is not None:
                    values = neutralize_pit_factor(
                        values,
                        pit_exposures,
                        method=cast(
                            Literal["industry", "size", "industry_size"],
                            args.neutralization,
                        ),
                    ).neutralized_value
                if convergence_cache is not None:
                    convergence_cache.write(factor.spec.factor_id, values)
                warmup = max(factor.spec.required_history - 1, 0)
                quality = evaluate_quality(values[warmup:])
                pearson = information_coefficient(values, labelled[primary][0])
                ranked = information_coefficient(values, labelled[primary][0], rank=True)
                evaluations = {
                    horizon: evaluate_institutional(
                        values,
                        labelled[horizon][0],
                        panel.trade_dates,
                        horizon=horizon,
                        cost_bps=args.cost_bps,
                        regimes=regimes,
                        exposures=exposures if horizon == primary else None,
                    )
                    for horizon in horizons
                }
                oos_evaluations = {
                    horizon: evaluate_institutional(
                        values[oos_start:],
                        labelled[horizon][0][oos_start:],
                        panel.trade_dates[oos_start:],
                        horizon=horizon,
                        cost_bps=args.cost_bps,
                        regimes=regimes[oos_start:],
                        exposures=(
                            {name: exposure[oos_start:] for name, exposure in exposures.items()}
                            if horizon == primary
                            else None
                        ),
                    )
                    for horizon in horizons
                }
                primary_timing = labelled[primary][1]
                paths = write_factor_report(
                    FactorReport(
                        factor.spec,
                        str(args.data_release_id),
                        quality=quality,
                        pearson_ic=pearson,
                        rank_ic=ranked,
                        extra_metrics={
                            "horizons": {
                                str(horizon): evaluation.payload()
                                for horizon, evaluation in evaluations.items()
                            },
                            "oos_horizons": {
                                str(horizon): evaluation.payload()
                                for horizon, evaluation in oos_evaluations.items()
                            },
                            "oos_timing": {
                                "method": "fixed_final_time_holdout",
                                "fraction": args.oos_fraction,
                                "start_date": panel.trade_dates[oos_start].isoformat(),
                                "end_date": panel.trade_dates[-1].isoformat(),
                            },
                            "timing": {
                                "definition": (
                                    "factor_date close -> next-session execution/return_start -> "
                                    "horizon-session return_end"
                                ),
                                "first": primary_timing[0].payload() if primary_timing else None,
                                "last": primary_timing[-1].payload() if primary_timing else None,
                            },
                            "provenance": {
                                "code_version": args.code_version,
                                "config_hash": config_hash,
                                "neutralization": args.neutralization,
                                "industry_release_hash": (
                                    industry_repository.content_hash
                                    if industry_repository is not None
                                    else None
                                ),
                                "industry_quality_hash": (
                                    industry_quality["content_hash"]
                                    if industry_quality is not None
                                    else None
                                ),
                            },
                        },
                    ),
                    args.report_dir,
                )
                reports.append((str(paths[0]), str(paths[1])))
                completed.append(factor.spec.factor_id)
                _write_json_atomic(
                    manifest_path,
                    {
                        "status": "RUNNING",
                        "data_release_id": str(args.data_release_id),
                        "factor_set": args.factor_set,
                        "start_date": args.start_date.isoformat(),
                        "end_date": args.end_date.isoformat(),
                        "horizons": horizons,
                        "code_version": args.code_version,
                        "config_hash": config_hash,
                        "completed_factor_ids": completed,
                        "completed_count": len(completed),
                        "factor_count": len(factors),
                    },
                )
                del values, evaluations, oos_evaluations
            del panel, labelled, exposures, regimes, pit_exposures
            gc.collect()
        del shared_panel
        convergence_content_hash = (
            convergence_cache.finalize() if convergence_cache is not None else None
        )
        _write_json_atomic(
            manifest_path,
            {
                "status": "PASS",
                "data_release_id": str(args.data_release_id),
                "factor_set": args.factor_set,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "horizons": horizons,
                "code_version": args.code_version,
                "config_hash": config_hash,
                "convergence_cache": (
                    str(args.convergence_cache) if args.convergence_cache is not None else None
                ),
                "convergence_content_hash": convergence_content_hash,
                "completed_factor_ids": completed,
                "completed_count": len(completed),
                "factor_count": len(factors),
            },
        )
        _print(
            {
                "status": "PASS",
                "factor_count": len(factors),
                "reports": reports,
                "manifest": str(manifest_path),
            }
        )
        return 0
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


def generate_candidates_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate controlled single-dimension candidates")
    parser.add_argument("--parent-factor-id", default="momentum_20d")
    parser.add_argument("--template-set", default="price_volume_v1")
    parser.add_argument("--budget", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidates.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        parent = selected_factors(args.parent_factor_id)[0].spec
        windows = (5, 10, 20, 40, 60, 120, 250)[: args.budget]
        batch = generate_window_variants(
            parent,
            template="Rank(Delta(Close, {window}))",
            windows=windows,
            batch_id=args.template_set,
            family_budget="price_volume",
            created_at=datetime.now(_SHANGHAI),
        )
        payload = {
            "batch_id": batch.batch_id,
            "attempted": batch.attempted,
            "failures": batch.failures,
            "candidates": [json.loads(item.model_dump_json()) for item in batch.candidates],
        }
        if args.dry_run:
            _print({"status": "DRY_RUN", "attempted": batch.attempted})
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print({"status": "PASS", "output": str(args.output), "candidates": len(batch.candidates)})
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def build_feature_set_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a version-pinned factor feature set")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--factor-set", default="baseline_v1")
    parser.add_argument("--data-release-id", type=_release, required=True)
    parser.add_argument("--target-horizon", type=int, default=20)
    parser.add_argument("--universe", default="all_a_share")
    parser.add_argument("--standardization", default="cross_sectional_zscore")
    parser.add_argument("--winsorization", default="mad_5")
    parser.add_argument("--missing-strategy", default="preserve")
    parser.add_argument("--effective-from", type=parse_date)
    parser.add_argument("--training-window-days", type=int, default=1260)
    parser.add_argument("--code-version", default="working-tree")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/feature_sets"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        factors = selected_factors(args.factor_set)
        feature_config = {
            "factor_set": args.factor_set,
            "standardization": args.standardization,
            "winsorization": args.winsorization,
            "missing_strategy": args.missing_strategy,
            "effective_from": args.effective_from.isoformat() if args.effective_from else None,
            "training_window_days": args.training_window_days,
        }
        spec = FeatureSetSpec(
            feature_set_id=args.feature_set_id,
            version="1.0.0",
            description="Version-pinned AQuant factor feature set",
            target_horizon=args.target_horizon,
            universe=args.universe,
            factor_members=tuple(
                FactorMember(factor_id=item.spec.factor_id, factor_version=item.spec.version)
                for item in factors
            ),
            selection_method="explicit_cli_selection",
            created_from_experiment="cli",
            data_release_id=args.data_release_id,
            standardization_method=args.standardization,
            winsorization_method=args.winsorization,
            missing_value_strategy=args.missing_strategy,
            effective_from=args.effective_from,
            training_window_days=args.training_window_days,
            code_version=args.code_version,
            config_hash=_hash(feature_config),
        )
        if args.dry_run:
            _print({"status": "DRY_RUN", "members": len(spec.factor_members)})
            return 0
        json_path = FeatureSetRegistry(args.output_dir).publish(spec)
        markdown_path = args.output_dir / f"{spec.feature_set_id}-{spec.version}.md"
        write_feature_set_report(spec, markdown_path)
        _print({"status": "PASS", "spec": str(json_path), "report": str(markdown_path)})
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def build_baseline_feature_sets_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish raw, compact and PIT-neutral baseline feature sets"
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--industry-quality-report", type=Path)
    parser.add_argument("--data-release-id", type=_release, required=True)
    parser.add_argument("--effective-from", type=parse_date, required=True)
    parser.add_argument("--target-horizon", type=int, default=20)
    parser.add_argument("--training-window-days", type=int, default=1260)
    parser.add_argument("--code-version", default="working-tree")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/feature_sets"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        selection = _verified_selection(args.selection)
        if selection.get("status") != "PASS":
            raise ValueError("feature sets require a passing convergence selection")
        release_in_selection = selection.get("data_release_id")
        if release_in_selection and release_in_selection != str(args.data_release_id):
            raise ValueError("selection data release does not match requested release")
        compact_ids = tuple(selection["compact_factor_ids"])
        experiment_id = str(selection["experiment_id"])
        admission = _verified_selection(args.admission) if args.admission is not None else None
        if admission is not None and admission.get("data_release_id") != str(args.data_release_id):
            raise ValueError("admission data release does not match requested release")
        industry_quality = (
            _verified_stable_payload(
                args.industry_quality_report,
                volatile_fields=("created_at",),
            )
            if args.industry_quality_report is not None
            else None
        )
        if industry_quality is not None and industry_quality.get("data_release_id") != str(
            args.data_release_id
        ):
            raise ValueError("industry quality data release does not match requested release")
        roles: dict[str, FeatureRole] = {}
        if admission is not None:
            for item in admission["factors"]:
                role = str(item["feature_role"])
                roles[str(item["factor_id"])] = {
                    "ALPHA_CANDIDATE": FeatureRole.ALPHA_CANDIDATE,
                    "RISK_FACTOR": FeatureRole.RISK_CONTROL,
                    "CONTROL_FEATURE": FeatureRole.CONTROL_FEATURE,
                    "EXPECTED_DIRECTION_UNKNOWN": FeatureRole.CONTROL_FEATURE,
                }[role]
        selection_window = selection.get("selection_window")
        evaluation_window = (
            (
                date.fromisoformat(selection_window["start_date"]),
                date.fromisoformat(selection_window["end_date"]),
            )
            if selection_window
            else None
        )
        lineage = tuple(
            (name, value)
            for name, value in (
                ("selection_hash", str(selection["content_hash"])),
                (
                    "admission_hash",
                    str(admission["content_hash"]) if admission is not None else "",
                ),
                (
                    "industry_quality_hash",
                    str(industry_quality["content_hash"]) if industry_quality is not None else "",
                ),
            )
            if value
        )
        neutral_validated = bool(
            industry_quality is not None
            and industry_quality.get("status") == "PASS"
            and admission is not None
            and selection.get("holdout_window", {}).get("used_for_selection") is False
        )
        bundle = build_baseline_feature_sets(
            tuple(
                factor.spec
                for factor in (*baseline_factor_library(), *second_wave_candidate_library())
            ),
            compact_ids,
            data_release_id=args.data_release_id,
            effective_from=args.effective_from,
            created_from_experiment=experiment_id,
            target_horizon=args.target_horizon,
            training_window_days=args.training_window_days,
            code_version=args.code_version,
            compact_status=FeatureSetStatus.VALIDATED,
            neutral_status=(
                FeatureSetStatus.VALIDATED if neutral_validated else FeatureSetStatus.DRAFT
            ),
            evaluation_window=evaluation_window,
            feature_roles=roles,
            lineage=lineage,
        )
        specs = (bundle.raw, bundle.compact, bundle.neutral)
        if args.dry_run:
            _print(
                {
                    "status": "DRY_RUN",
                    "feature_sets": {
                        spec.feature_set_id: len(spec.factor_members) for spec in specs
                    },
                }
            )
            return 0
        registry = FeatureSetRegistry(args.output_dir)
        outputs: list[dict[str, str]] = []
        for spec in specs:
            json_path = registry.publish(spec)
            markdown_path = args.output_dir / f"{spec.feature_set_id}-{spec.version}.md"
            write_feature_set_report(spec, markdown_path)
            outputs.append(
                {
                    "feature_set_id": spec.feature_set_id,
                    "content_hash": spec.content_hash,
                    "spec": str(json_path),
                    "report": str(markdown_path),
                }
            )
        _print({"status": "PASS", "feature_sets": outputs})
        return 0
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


def _verified_selection(path: Path) -> dict[str, Any]:
    return _verified_stable_payload(path)


def _verified_stable_payload(
    path: Path,
    *,
    volatile_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("selection must be a JSON object")
    expected = selection.pop("content_hash", None)
    stable = dict(selection)
    for field in volatile_fields:
        stable.pop(field, None)
    actual = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("selection content hash mismatch")
    selection["content_hash"] = expected
    return selection


def train_alpha_model_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an ordered L3 alpha baseline")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", choices=[item.value for item in ModelKind], default="ridge")
    parser.add_argument("--target-horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/alpha_predictions.npy"))
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.target_horizon <= 0:
        parser.error("target horizon must be positive")
    if args.dry_run:
        _print(
            {
                "status": "DRY_RUN",
                "feature_set_id": args.feature_set_id,
                "model": args.model,
                "target_horizon": args.target_horizon,
            }
        )
        return 0
    if args.dataset is None:
        parser.error("--dataset NPZ is required unless --dry-run; expected X, y and optional ic")
    try:
        admission = _verified_selection(args.admission) if args.admission is not None else None
        if args.research_only:
            if args.model not in {
                ModelKind.EQUAL_WEIGHT.value,
                ModelKind.IC_WEIGHT.value,
                ModelKind.ICIR_WEIGHT.value,
            }:
                raise ValueError(
                    "research-only Alpha is limited to equal/IC/ICIR weighted baselines"
                )
        elif admission is None or int(admission["production_core_count"]) < 8:
            raise ValueError(
                "formal L3 training requires an attested admission with at least "
                "8 Production Alpha factors"
            )
        dataset = np.load(args.dataset)
        features, target = dataset["X"], dataset["y"]
        groups = dataset.get("dates", None)
        period_count = len(np.unique(groups)) if groups is not None else len(target)
        train_size = args.train_size or int(period_count * 0.6)
        validation_size = args.validation_size or max(1, int(period_count * 0.2))
        step = args.step or validation_size
        outcome = walk_forward_predict(
            ModelKind(args.model),
            features,
            target,
            train_size=train_size,
            validation_size=validation_size,
            step=step,
            purge=args.target_horizon,
            groups=groups,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, outcome.predictions)
        _print(
            {
                "status": "PASS",
                "output": str(args.output),
                "folds": len(outcome.splits),
                "oos_coverage": outcome.coverage,
                "oos_rank_correlation": outcome.rank_correlation,
                "lifecycle_status": ("RESEARCH_ONLY" if args.research_only else "FORMAL_BASELINE"),
                "admission_hash": (admission["content_hash"] if admission is not None else None),
            }
        )
        return 0
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
