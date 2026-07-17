import hashlib
import pickle
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aquant.ml.datasets.tabular import FeatureStandardizer, MLDataset
from aquant.ml.models.baselines import ModelKind, Regressor, create_model
from aquant.ml.validation.walk_forward import TemporalFold


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_kind: ModelKind
    model: Regressor
    standardizer: FeatureStandardizer
    validation_rank_ic: float
    test_rank_ic: float
    model_checksum: str
    test_predictions: tuple[float, ...]


def train_fold(
    dataset: MLDataset,
    fold: TemporalFold,
    *,
    model_kind: ModelKind,
    random_seed: int = 20260716,
) -> TrainingResult:
    training = dataset.select(fold.train)
    validation = dataset.select(fold.validation)
    test = dataset.select(fold.test)
    standardizer = FeatureStandardizer.fit(training.features)
    model = create_model(model_kind, random_seed=random_seed)
    model.fit(standardizer.transform(training.features), training.targets)
    validation_prediction = np.asarray(
        model.predict(standardizer.transform(validation.features)), dtype=np.float64
    )
    test_prediction = np.asarray(
        model.predict(standardizer.transform(test.features)), dtype=np.float64
    )
    checksum = hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()
    return TrainingResult(
        model_kind,
        model,
        standardizer,
        _rank_ic(validation_prediction, validation.targets),
        _rank_ic(test_prediction, test.targets),
        checksum,
        tuple(float(value) for value in test_prediction),
    )


def _rank_ic(prediction: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> float:
    valid = np.isfinite(prediction) & np.isfinite(target)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    x = np.argsort(np.argsort(prediction[valid], kind="stable"), kind="stable")
    y = np.argsort(np.argsort(target[valid], kind="stable"), kind="stable")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
