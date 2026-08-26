import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aquant.factors.atomic import baseline_factor_library
from aquant.factors.evaluation import (
    admit_factor_candidates,
    verify_leakage_attestation,
    write_leakage_attestation,
)
from aquant.factors.evaluation.admission import (
    _evidence_failure,
    _failed_lifecycle,
    _feature_role,
    _history_years,
    _non_directional_rejection,
    _role_rejection,
    _single_report,
    _verified_payload,
)
from aquant.factors.evaluation.gates import ProductionGateConfig
from aquant.factors.evaluation.ic import ICStatistics
from aquant.factors.evaluation.quality import evaluate_quality
from aquant.factors.reporting import FactorReport, write_factor_report


def test_admission_uses_directional_oos_evidence_and_does_not_force_core(
    tmp_path: Path,
) -> None:
    library = baseline_factor_library()
    factors = (
        next(factor for factor in library if factor.spec.factor_id == "net_profit_growth_yoy"),
        next(factor for factor in library if factor.spec.factor_id == "momentum_20d"),
    )
    reports = tmp_path / "reports"
    report_hashes: dict[str, str] = {}
    convergence_factors: list[dict[str, object]] = []
    for index, factor in enumerate(factors):
        spec = (
            factor.spec if index == 0 else factor.spec.model_copy(update={"expected_direction": 0})
        )
        direction = int(factor.spec.expected_direction)
        values = np.arange(100.0).reshape(10, 10)
        rank = ICStatistics(tuple([direction * 0.03] * 800), 0.03, 0.1, 0.3, 0.6, 3.0)
        report = FactorReport(
            spec,
            "cn_equity_20260717_001",
            quality=evaluate_quality(values),
            rank_ic=rank,
            extra_metrics={
                "oos_horizons": {
                    "5": {
                        "rank_ic_mean": direction * 0.03,
                        "rank_icir": direction * 0.4,
                        "ic_positive_ratio": 0.6 if direction > 0 else 0.4,
                        "annual_rank_ic": [[2025, direction * 0.02], [2026, direction * 0.03]],
                        "monotonicity": direction * 0.8,
                        "turnover": 0.3,
                        "long_short_return": direction * 0.0012,
                        "net_long_short_return": direction * 0.001,
                        "regime_rank_ic": [
                            ["BULL", direction * 0.02],
                            ["BEAR", direction * 0.01],
                            ["SIDEWAYS", direction * 0.015],
                        ],
                        "style_exposures": [["size", 0.5]],
                    }
                }
            },
        )
        json_path, _ = write_factor_report(report, reports)
        report_hashes[factor.spec.factor_id] = json.loads(json_path.read_text())["content_hash"]
        convergence_factors.append(
            {
                "factor_id": factor.spec.factor_id,
                "conditional_rank_ic": direction * 0.01,
            }
        )
    factor_ids = [factor.spec.factor_id for factor in factors]
    convergence: dict[str, object] = {
        "status": "PASS",
        "data_release_id": "cn_equity_20260717_001",
        "factor_ids": factor_ids,
        "combined_correlation": [[1.0, 0.5], [0.5, 1.0]],
        "factors": convergence_factors,
        "compact_factor_ids": factor_ids,
        "report_content_hashes": report_hashes,
    }
    convergence["content_hash"] = _hash(convergence)
    convergence_path = tmp_path / "convergence.json"
    convergence_path.write_text(json.dumps(convergence))
    manifest = {
        "status": "PASS",
        "data_release_id": "cn_equity_20260717_001",
        "completed_count": 73,
        "factor_count": 73,
        "start_date": "2021-07-18",
        "end_date": "2026-07-17",
        "code_version": "test",
        "config_hash": "a" * 64,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    attestation_path = tmp_path / "attestation.json"
    write_leakage_attestation(
        output=attestation_path,
        data_release_id="cn_equity_20260717_001",
        code_version="test",
        test_files=("test_future.py",),
        test_summary="1 passed",
    )
    payload = admit_factor_candidates(
        convergence_path=convergence_path,
        report_dir=reports,
        evaluation_manifest_path=manifest_path,
        gate_config_path=Path("config/factors/production_gate_v1.yaml"),
        leakage_attestation_path=attestation_path,
        output=tmp_path / "admission.json",
    )
    assert payload["status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["production_core_count"] == 1
    assert payload["factors"][0]["decision"]["passed"]
    assert payload["factors"][1]["decision"]["rejections"] == ("MISSING_EXPECTED_DIRECTION",)
    assert (tmp_path / "admission_summary.md").is_file()
    assert (tmp_path / "factor_level_decisions.parquet").is_file()


def test_negative_direction_does_not_reverse_transaction_cost() -> None:
    from aquant.factors.evaluation.admission import _production_evidence

    report = {
        "factor": {"expected_direction": -1, "complexity_score": 1.0},
        "rank_ic": {"by_date": [0.1, -0.1]},
        "extra_metrics": {
            "oos_horizons": {
                "5": {
                    "rank_ic_mean": -0.03,
                    "rank_icir": -0.2,
                    "ic_positive_ratio": 0.4,
                    "annual_rank_ic": [[2025, -0.03]],
                    "monotonicity": -0.8,
                    "turnover": 0.3,
                    "long_short_return": -0.002,
                    "net_long_short_return": -0.0023,
                    "regime_rank_ic": [],
                    "style_exposures": [],
                }
            }
        },
        "quality": {"missing_rate": 0.1},
    }
    evidence = _production_evidence(
        report,
        {"conditional_rank_ic": -0.01},
        np.asarray([[1.0]]),
        0,
        history_years=5.0,
        horizon=5,
    )
    assert evidence.directional_net_return == pytest.approx(0.0017)


def test_leakage_attestation_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    write_leakage_attestation(
        output=path,
        data_release_id="cn_equity_20260717_001",
        code_version="test",
        test_files=("test_future.py",),
        test_summary="1 passed",
    )
    payload = json.loads(path.read_text())
    payload["checks"]["future_mutation_invariance"] = "FAIL"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_leakage_attestation(path, data_release_id="cn_equity_20260717_001")


def test_leakage_attestation_rejects_another_code_version(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    write_leakage_attestation(
        output=path,
        data_release_id="cn_equity_20260717_001",
        code_version="evaluated-revision",
        test_files=("test_future.py",),
        test_summary="1 passed",
    )
    with pytest.raises(ValueError, match="code version"):
        verify_leakage_attestation(
            path,
            data_release_id="cn_equity_20260717_001",
            code_version="different-revision",
        )


def test_admission_roles_and_lifecycle_are_auditable() -> None:
    config = ProductionGateConfig.from_yaml(Path("config/factors/production_gate_v1.yaml"))
    base = {
        "factor": {
            "factor_id": "factor",
            "version": "1.0.0",
            "expected_direction": 0,
            "family": "size",
            "tags": [],
        }
    }
    assert _feature_role(base) == "RISK_FACTOR"
    assert (
        _feature_role(
            {**base, "factor": {**base["factor"], "family": "quality", "tags": ["control"]}}
        )
        == "CONTROL_FEATURE"
    )
    assert (
        _feature_role({**base, "factor": {**base["factor"], "family": "quality", "tags": []}})
        == "ALPHA_CANDIDATE"
    )

    unknown = _non_directional_rejection(base, config)
    assert unknown["feature_role"] == "EXPECTED_DIRECTION_UNKNOWN"
    assert unknown["decision"]["rejections"] == ("MISSING_EXPECTED_DIRECTION",)
    risk = _role_rejection(base, config, "RISK_FACTOR")
    assert risk["lifecycle_status"] == "RISK_ONLY"
    control = _role_rejection(base, config, "CONTROL_FEATURE")
    assert control["lifecycle_status"] == "VALIDATED"
    failed = _evidence_failure(base, config, KeyError("missing metric"))
    assert failed["evidence"]["error_type"] == "KeyError"
    assert _failed_lifecycle(("INSUFFICIENT_HISTORY",)) == "COMPUTED"
    assert _failed_lifecycle(("OOS_RANK_IC_BELOW_THRESHOLD",)) == "VALIDATED"


def test_admission_artifact_validation_rejects_ambiguous_or_tampered_input(
    tmp_path: Path,
) -> None:
    assert _history_years({"start_date": "2021-01-01", "end_date": "2022-01-01"}) == 1.0
    with pytest.raises(ValueError, match="date range"):
        _history_years({"start_date": "2022-01-01", "end_date": "2021-01-01"})

    with pytest.raises(ValueError, match="found 0"):
        _single_report(tmp_path, "missing")
    (tmp_path / "factor-a.json").write_text("{}")
    (tmp_path / "factor-b.json").write_text("{}")
    with pytest.raises(ValueError, match="found 2"):
        _single_report(tmp_path, "factor")

    valid: dict[str, object] = {"status": "PASS", "data_release_id": "release"}
    valid["content_hash"] = _hash(valid)
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(valid))
    assert _verified_payload(path)["status"] == "PASS"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="JSON object"):
        _verified_payload(path)


def _hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
