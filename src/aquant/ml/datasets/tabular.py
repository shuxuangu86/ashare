from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MLDataset:
    features: Array
    targets: Array
    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        targets = np.asarray(self.targets, dtype=np.float64)
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
            raise ValueError("ML features/targets must have shapes (samples, features)/(samples,)")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("feature names do not match feature columns")
        if len(self.dates) != targets.size or len(self.symbols) != targets.size:
            raise ValueError("ML sample metadata length mismatch")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "targets", targets.copy())

    def select(self, indices: npt.NDArray[np.int64]) -> "MLDataset":
        selected = np.asarray(indices, dtype=np.int64)
        return MLDataset(
            self.features[selected],
            self.targets[selected],
            tuple(self.dates[index] for index in selected),
            tuple(self.symbols[index] for index in selected),
            self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class FeatureStandardizer:
    mean: Array
    scale: Array

    @classmethod
    def fit(cls, training_features: Array) -> "FeatureStandardizer":
        values = np.asarray(training_features, dtype=np.float64)
        if values.ndim != 2 or not values.size:
            raise ValueError("standardizer requires a non-empty 2D training matrix")
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(mean, scale)

    def transform(self, features: Array) -> Array:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("standardizer feature shape mismatch")
        return (values - self.mean) / self.scale
