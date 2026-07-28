import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from aquant.factors.evaluation.attestation import verify_leakage_attestation
from aquant.factors.evaluation.gates import (
    ProductionEvidence,
    ProductionGateConfig,
    evaluate_production_gate,
)


def admit_factor_candidates(
    *,
    convergence_path: Path,
    report_dir: Path,
    evaluation_manifest_path: Path,
    gate_config_path: Path,
    leakage_attestation_path: Path,
    output: Path,
    horizon: int = 5,
) -> dict[str, object]:
    convergence = _verified_payload(convergence_path)
    if convergence.get("status") != "PASS":
        raise ValueError("factor convergence must pass before production admission")
    release_id = str(convergence["data_release_id"])
    manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("data_release_id") != release_id
        or manifest.get("completed_count") != manifest.get("factor_count")
    ):
        raise ValueError("audited evaluation manifest is incomplete or mismatched")
    attestation = verify_leakage_attestation(
        leakage_attestation_path,
        data_release_id=release_id,
        code_version=str(manifest["code_version"]),
    )
    config = ProductionGateConfig.from_yaml(gate_config_path)
    history_years = _history_years(manifest)
    factor_ids = tuple(convergence["factor_ids"])
    combined = np.asarray(convergence["combined_correlation"], dtype=float)
    if combined.shape != (len(factor_ids), len(factor_ids)):
        raise ValueError("convergence correlation matrix is invalid")
    convergence_factors = {item["factor_id"]: item for item in convergence["factors"]}
    report_hashes = convergence["report_content_hashes"]
    admitted: list[dict[str, Any]] = []
    for factor_id in convergence["compact_factor_ids"]:
        report = _single_report(report_dir, factor_id)
        if report.get("content_hash") != report_hashes.get(factor_id):
            raise ValueError(f"audited report hash mismatch for {factor_id}")
        direction = int(report["factor"]["expected_direction"])
        if direction not in {-1, 1}:
            admitted.append(_non_directional_rejection(report, config))
            continue
        evidence = _production_evidence(
            report,
            convergence_factors[factor_id],
            combined,
            factor_ids.index(factor_id),
            history_years=history_years,
            horizon=horizon,
        )
        decision = evaluate_production_gate(evidence, config)
        admitted.append(
            {
                "factor_id": factor_id,
                "factor_version": report["factor"]["version"],
                "evidence": asdict(evidence),
                "decision": asdict(decision),
            }
        )
    passed = sorted(
        (item for item in admitted if item["decision"]["passed"]),
        key=lambda item: (-float(item["decision"]["overall_score"]), item["factor_id"]),
    )
    core_ids = tuple(item["factor_id"] for item in passed[:15])
    status = "PASS" if 8 <= len(core_ids) <= 15 else "INSUFFICIENT_EVIDENCE"
    payload: dict[str, object] = {
        "status": status,
        "data_release_id": release_id,
        "code_version": manifest["code_version"],
        "horizon": horizon,
        "evaluation_config_hash": manifest["config_hash"],
        "convergence_content_hash": convergence["content_hash"],
        "gate_config_hash": config.config_hash,
        "leakage_attestation_hash": attestation["content_hash"],
        "compact_factor_ids": convergence["compact_factor_ids"],
        "production_core_factor_ids": core_ids,
        "production_core_count": len(core_ids),
        "factors": admitted,
    }
    payload["content_hash"] = _content_hash(payload)
    _write_json_atomic(output, payload)
    return payload


def _non_directional_rejection(
    report: dict[str, Any], config: ProductionGateConfig
) -> dict[str, Any]:
    identity = {
        "factor_id": report["factor"]["factor_id"],
        "factor_version": report["factor"]["version"],
        "expected_direction": report["factor"]["expected_direction"],
    }
    return {
        "factor_id": identity["factor_id"],
        "factor_version": identity["factor_version"],
        "evidence": {"expected_direction": identity["expected_direction"]},
        "decision": {
            "passed": False,
            "overall_score": 0.0,
            "components": (),
            "rejections": ("MISSING_EXPECTED_DIRECTION",),
            "config_hash": config.config_hash,
            "evidence_hash": _content_hash(identity),
        },
    }


def _production_evidence(
    report: dict[str, Any],
    convergence_factor: dict[str, Any],
    combined: npt.NDArray[np.float64],
    factor_index: int,
    *,
    history_years: float,
    horizon: int,
) -> ProductionEvidence:
    direction = int(report["factor"]["expected_direction"])
    if direction not in {-1, 1}:
        raise ValueError("production factor expected direction must be -1 or 1")
    metrics = report["extra_metrics"]["oos_horizons"][str(horizon)]
    rank_by_date = np.asarray(report["rank_ic"]["by_date"], dtype=float)
    annual = tuple(metrics["annual_rank_ic"])
    regimes = {
        name: float(value)
        for name, value in metrics["regime_rank_ic"]
        if name in {"BULL", "BEAR", "SIDEWAYS"}
    }
    style = tuple(abs(float(value)) for _, value in metrics["style_exposures"])
    peer_values = np.delete(np.abs(combined[factor_index]), factor_index)
    return ProductionEvidence(
        history_years=history_years,
        cross_sections=int(np.count_nonzero(np.isfinite(rank_by_date))),
        missing_rate=float(report["quality"]["missing_rate"]),
        oos_rank_ic=direction * float(metrics["rank_ic_mean"]),
        rank_icir=direction * float(metrics["rank_icir"]),
        directional_ic_win_rate=(
            float(metrics["ic_positive_ratio"])
            if direction > 0
            else 1 - float(metrics["ic_positive_ratio"])
        ),
        positive_year_ratio=(
            float(np.mean([direction * float(value) > 0 for _, value in annual]))
            if annual
            else float("nan")
        ),
        monotonicity=direction * float(metrics["monotonicity"]),
        turnover=float(metrics["turnover"]),
        directional_net_return=direction * float(metrics["net_long_short_return"]),
        maximum_peer_correlation=(float(np.max(peer_values)) if len(peer_values) else 0.0),
        conditional_rank_ic=direction * float(convergence_factor["conditional_rank_ic"]),
        complexity=float(report["factor"]["complexity_score"]),
        maximum_style_exposure=max(style, default=float("nan")),
        extreme_regime_rank_ic=(
            min(direction * value for value in regimes.values()) if regimes else float("nan")
        ),
        leakage_detected=False,
    )


def _history_years(manifest: dict[str, Any]) -> float:
    start = date.fromisoformat(manifest["start_date"])
    end = date.fromisoformat(manifest["end_date"])
    if end < start:
        raise ValueError("evaluation manifest date range is invalid")
    return round(((end - start).days + 1) / 365.25, 2)


def _single_report(report_dir: Path, factor_id: str) -> dict[str, Any]:
    matches = tuple(report_dir.glob(f"{factor_id}-*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one audited report for {factor_id}, found {len(matches)}")
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audited factor report must be a JSON object")
    return cast(dict[str, Any], payload)


def _verified_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("convergence result must be a JSON object")
    payload = cast(dict[str, Any], raw)
    expected = payload.pop("content_hash")
    if expected != _content_hash(payload):
        raise ValueError("convergence content hash mismatch")
    payload["content_hash"] = expected
    return payload


def _content_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".factor-admission-",
        suffix=".json",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
