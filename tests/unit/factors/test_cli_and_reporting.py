import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aquant.factors.atomic import baseline_factor_library
from aquant.factors.cli_tools import (
    _evaluation_report_matches,
    build_baseline_feature_sets_main,
    build_feature_set_main,
    evaluate_factors_main,
    generate_candidates_main,
    materialize_factors_main,
    train_alpha_model_main,
)
from aquant.factors.evaluation.ic import information_coefficient
from aquant.factors.evaluation.quality import evaluate_quality
from aquant.factors.reporting import FactorReport, write_factor_report

RELEASE = "cn_equity_20260728_001"


def test_all_factor_cli_dry_runs_emit_machine_readable_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = (
        (
            materialize_factors_main,
            [
                "--release-dir",
                str(tmp_path),
                "--data-release-id",
                RELEASE,
                "--factor-set",
                "momentum_20d",
                "--start-date",
                "20260101",
                "--end-date",
                "20260131",
                "--dry-run",
            ],
        ),
        (
            evaluate_factors_main,
            [
                "--release-dir",
                str(tmp_path),
                "--data-release-id",
                RELEASE,
                "--factor-set",
                "momentum_20d",
                "--start-date",
                "20260101",
                "--end-date",
                "20260131",
                "--dry-run",
            ],
        ),
        (
            generate_candidates_main,
            ["--parent-factor-id", "momentum_20d", "--budget", "3", "--dry-run"],
        ),
        (
            build_feature_set_main,
            [
                "--feature-set-id",
                "baseline_simple_v1",
                "--factor-set",
                "momentum_20d",
                "--data-release-id",
                RELEASE,
                "--dry-run",
            ],
        ),
        (
            train_alpha_model_main,
            ["--feature-set-id", "baseline_simple_v1", "--model", "ridge", "--dry-run"],
        ),
    )
    for main, arguments in cases:
        assert main(arguments) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "DRY_RUN"


def test_industry_neutral_evaluation_requires_attested_pit_release(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        evaluate_factors_main(
            [
                "--release-dir",
                str(tmp_path),
                "--data-release-id",
                RELEASE,
                "--factor-set",
                "momentum_20d",
                "--start-date",
                "20260101",
                "--end-date",
                "20260131",
                "--neutralization",
                "industry_size",
                "--dry-run",
            ]
        )


def test_evaluation_cli_accepts_nested_pool_factor_subset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = tmp_path / "nested-pool.json"
    pool.write_text(json.dumps({"union_factor_ids": ["momentum_20d", "reversal_5d"]}))

    assert (
        evaluate_factors_main(
            [
                "--release-dir",
                str(tmp_path),
                "--data-release-id",
                RELEASE,
                "--factor-set",
                "l2_v2_executable",
                "--factor-ids-file",
                str(pool),
                "--start-date",
                "20260101",
                "--end-date",
                "20260131",
                "--dry-run",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output == {"factor_count": 2, "horizons": [1, 5, 10, 20, 40], "status": "DRY_RUN"}


def test_train_cli_uses_purged_walk_forward_and_writes_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    generator = np.random.default_rng(3)
    features = generator.normal(size=(100, 4))
    target = features @ np.arange(1.0, 5.0)
    dataset = tmp_path / "dataset.npz"
    output = tmp_path / "prediction.npy"
    np.savez(dataset, X=features, y=target)
    assert (
        train_alpha_model_main(
            [
                "--feature-set-id",
                "baseline",
                "--dataset",
                str(dataset),
                "--model",
                "equal_weight",
                "--output",
                str(output),
                "--research-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["folds"] == 1
    assert payload["oos_coverage"] == pytest.approx(0.2)
    predictions = np.load(output)
    assert predictions.shape == (100,)
    assert np.count_nonzero(np.isfinite(predictions)) == 20


def test_formal_alpha_training_requires_eight_admitted_factors(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, X=np.ones((20, 2)), y=np.arange(20.0))
    with pytest.raises(SystemExit):
        train_alpha_model_main(
            [
                "--feature-set-id",
                "baseline_compact_v1",
                "--dataset",
                str(dataset),
                "--model",
                "ridge",
            ]
        )


def test_baseline_bundle_cli_publishes_three_immutable_specs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    selection = tmp_path / "selection.json"
    compact = [factor.spec.factor_id for factor in baseline_factor_library()[:20]]
    selection_payload = {
        "status": "PASS",
        "experiment_id": "five_year_convergence_v1",
        "data_release_id": RELEASE,
        "compact_factor_ids": compact,
    }
    selection_payload["content_hash"] = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    selection.write_text(json.dumps(selection_payload))
    output = tmp_path / "sets"
    assert (
        build_baseline_feature_sets_main(
            [
                "--selection",
                str(selection),
                "--data-release-id",
                RELEASE,
                "--effective-from",
                "20260717",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert {item["feature_set_id"] for item in payload["feature_sets"]} == {
        "baseline_raw_v1",
        "baseline_compact_v1",
        "baseline_neutral_v1",
    }
    assert len(tuple(output.glob("*.json"))) == 3
    assert "Content hash:" in (output / "baseline_neutral_v1-1.0.0.md").read_text()


def test_baseline_bundle_cli_rejects_tampered_selection(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": "tampered",
                "data_release_id": RELEASE,
                "compact_factor_ids": [],
                "content_hash": "0" * 64,
            }
        )
    )
    with pytest.raises(SystemExit):
        build_baseline_feature_sets_main(
            [
                "--selection",
                str(selection),
                "--data-release-id",
                RELEASE,
                "--effective-from",
                "20260717",
            ]
        )


def test_factor_report_writes_json_and_markdown(tmp_path: Path) -> None:
    factor = baseline_factor_library()[0]
    values = np.arange(20.0).reshape(4, 5)
    report = FactorReport(
        factor.spec,
        RELEASE,
        quality=evaluate_quality(values),
        rank_ic=information_coefficient(values, values, rank=True),
    )
    json_path, markdown_path = write_factor_report(report, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["factor"]["factor_id"] == factor.spec.factor_id
    assert len(payload["content_hash"]) == 64
    assert "PIT rule" in markdown_path.read_text()


def test_evaluation_resume_rejects_stale_or_tampered_reports(tmp_path: Path) -> None:
    factor = baseline_factor_library()[0]
    report = FactorReport(
        factor.spec,
        RELEASE,
        extra_metrics={
            "provenance": {
                "config_hash": "a" * 64,
                "code_version": "abc123",
            }
        },
    )
    json_path, markdown_path = write_factor_report(report, tmp_path)
    assert _evaluation_report_matches(
        json_path,
        markdown_path,
        data_release_id=RELEASE,
        config_hash="a" * 64,
        code_version="abc123",
    )
    payload = json.loads(json_path.read_text())
    payload["factor"]["description"] = "tampered"
    json_path.write_text(json.dumps(payload))
    assert not _evaluation_report_matches(
        json_path,
        markdown_path,
        data_release_id=RELEASE,
        config_hash="a" * 64,
        code_version="abc123",
    )


def test_cli_invalid_factor_has_nonzero_exit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        materialize_factors_main(
            [
                "--release-dir",
                str(tmp_path),
                "--data-release-id",
                RELEASE,
                "--factor-set",
                "unknown",
                "--start-date",
                "20260101",
                "--end-date",
                "20260131",
                "--dry-run",
            ]
        )
    assert error.value.code == 2
