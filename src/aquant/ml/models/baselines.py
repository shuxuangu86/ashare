from enum import StrEnum
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge  # type: ignore[import-untyped]

Array = npt.NDArray[np.float64]


class Regressor(Protocol):
    def fit(self, features: Array, targets: Array) -> "Regressor": ...

    def predict(self, features: Array) -> Array: ...

    def get_params(self, deep: bool = True) -> dict[str, object]: ...


class ModelKind(StrEnum):
    RIDGE = "RIDGE"
    ELASTIC_NET = "ELASTIC_NET"
    LIGHTGBM = "LIGHTGBM"


def create_model(kind: ModelKind, *, random_seed: int = 20260716) -> Regressor:
    if kind is ModelKind.RIDGE:
        return cast(Regressor, Ridge(alpha=1.0))
    if kind is ModelKind.ELASTIC_NET:
        return cast(
            Regressor,
            ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_seed, max_iter=5000),
        )
    return LGBMRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        random_state=random_seed,
        n_jobs=1,
        verbosity=-1,
    )
