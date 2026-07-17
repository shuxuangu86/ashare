from datetime import date, timedelta

import numpy as np
import pytest

from aquant.domain.data_release import DataReleaseId
from aquant.ml import (
    FeatureStandardizer,
    MLDataset,
    ModelArtifact,
    ModelKind,
    ModelRegistry,
    WalkForwardSplitter,
    create_model,
    train_fold,
)


def _dataset(samples: int = 40) -> MLDataset:
    x = np.arange(samples, dtype=float)
    features = np.column_stack((x, x**2 / samples))
    targets = x * 0.5 + 1
    start = date(2020, 1, 1)
    return MLDataset(
        features,
        targets,
        tuple(start + timedelta(days=index) for index in range(samples)),
        tuple(f"S{index % 5}" for index in range(samples)),
        ("momentum@1", "value@1"),
    )


def test_walk_forward_has_purge_embargo_and_no_random_split() -> None:
    splitter = WalkForwardSplitter(20, 5, 5, 5, purge_size=2, embargo_size=1)
    folds = splitter.split(40)
    assert len(folds) == 2
    first = folds[0]
    assert first.train[-1] == 19
    assert first.validation[0] == 22
    assert first.test[0] == 28
    assert max(first.train) < min(first.validation) < min(first.test)
    with pytest.raises(ValueError):
        WalkForwardSplitter(0, 1, 1, 1)
    with pytest.raises(ValueError, match="not enough"):
        splitter.split(10)


def test_standardizer_uses_training_statistics_only() -> None:
    training = np.array([[1.0], [3.0]])
    standardizer = FeatureStandardizer.fit(training)
    assert standardizer.mean[0] == 2
    transformed_future = standardizer.transform(np.array([[100.0]]))
    assert transformed_future[0, 0] == 98
    with pytest.raises(ValueError):
        standardizer.transform(np.ones((2, 2)))


@pytest.mark.parametrize("kind", [ModelKind.RIDGE, ModelKind.ELASTIC_NET, ModelKind.LIGHTGBM])
def test_all_baseline_models_train_with_deterministic_temporal_fold(kind: ModelKind) -> None:
    dataset = _dataset()
    fold = WalkForwardSplitter(20, 5, 5, 5).split(40)[0]
    first = train_fold(dataset, fold, model_kind=kind, random_seed=7)
    second = train_fold(dataset, fold, model_kind=kind, random_seed=7)
    assert len(first.test_predictions) == 5
    assert first.test_predictions == pytest.approx(second.test_predictions)
    assert first.model_checksum == second.model_checksum
    assert len(first.model_checksum) == 64
    assert np.isfinite(first.validation_rank_ic)


def test_dataset_validates_shapes_and_selection() -> None:
    dataset = _dataset(10)
    selected = dataset.select(np.array([1, 3], dtype=np.int64))
    assert selected.symbols == ("S1", "S3")
    with pytest.raises(ValueError):
        MLDataset(np.ones((2, 2)), np.ones(3), (), (), ("a", "b"))
    with pytest.raises(ValueError):
        FeatureStandardizer.fit(np.array([]))


def test_model_artifact_registry_binds_lineage_and_metrics() -> None:
    dataset = _dataset()
    fold = WalkForwardSplitter(20, 5, 5, 5).split(40)[0]
    result = train_fold(dataset, fold, model_kind=ModelKind.RIDGE)
    artifact = ModelArtifact(
        "ridge-20260716",
        ModelKind.RIDGE,
        DataReleaseId("cn_equity_20260716_001"),
        dataset.feature_names,
        dataset.dates[0],
        dataset.dates[19],
        20260716,
        "af3395e",
        result.model_checksum,
        result.validation_rank_ic,
        result.test_rank_ic,
    )
    registry = ModelRegistry()
    registry.register(artifact)
    assert registry.get(artifact.run_id) == artifact
    with pytest.raises(ValueError, match="already"):
        registry.register(artifact)
    with pytest.raises(ValueError):
        ModelArtifact(
            "",
            ModelKind.RIDGE,
            DataReleaseId("cn_equity_20260716_001"),
            (),
            date(2026, 1, 2),
            date(2026, 1, 1),
            1,
            "bad",
            "bad",
            0,
            0,
        )


def test_factory_exposes_requested_kind() -> None:
    assert create_model(ModelKind.RIDGE).__class__.__name__ == "Ridge"
