from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import ElasticNet, Ridge  # type: ignore[import-untyped]


class ModelKind(StrEnum):
    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHT = "ic_weight"
    ICIR_WEIGHT = "icir_weight"
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"
    LIGHTGBM = "lightgbm"


class AlphaAggregator:
    def __init__(self, kind: ModelKind, **parameters: Any) -> None:
        self.kind = kind
        self.parameters = parameters
        self._model: Any = None
        self._weights: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        features: npt.ArrayLike,
        target: npt.ArrayLike,
        *,
        historical_ic: npt.ArrayLike | None = None,
        historical_icir: npt.ArrayLike | None = None,
    ) -> "AlphaAggregator":
        x, y = _validated(features, target)
        if self.kind == ModelKind.EQUAL_WEIGHT:
            self._weights = np.full(x.shape[1], 1 / x.shape[1])
        elif self.kind in {ModelKind.IC_WEIGHT, ModelKind.ICIR_WEIGHT}:
            source = historical_ic if self.kind is ModelKind.IC_WEIGHT else historical_icir
            if source is None:
                raise ValueError(f"{self.kind.value} aggregation requires historical evidence")
            evidence = np.asarray(source, dtype=float)
            if evidence.shape != (x.shape[1],):
                raise ValueError("historical evidence must align with factors")
            denominator = np.sum(np.abs(evidence))
            self._weights = (
                evidence / denominator
                if denominator > 0
                else np.full_like(evidence, 1 / len(evidence))
            )
        elif self.kind == ModelKind.RIDGE:
            self._model = Ridge(
                alpha=float(self.parameters.get("alpha", 1.0)),
                random_state=None,
            ).fit(x, y)
        elif self.kind == ModelKind.ELASTIC_NET:
            self._model = ElasticNet(
                alpha=float(self.parameters.get("alpha", 0.01)),
                l1_ratio=float(self.parameters.get("l1_ratio", 0.5)),
                random_state=0,
                max_iter=10_000,
            ).fit(x, y)
        else:
            from lightgbm import LGBMRegressor

            self._model = LGBMRegressor(
                n_estimators=int(self.parameters.get("n_estimators", 100)),
                max_depth=int(self.parameters.get("max_depth", 3)),
                learning_rate=float(self.parameters.get("learning_rate", 0.05)),
                random_state=0,
                verbosity=-1,
            ).fit(x, y)
        return self

    def predict(self, features: npt.ArrayLike) -> npt.NDArray[np.float64]:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or np.any(~np.isfinite(x)):
            raise ValueError("prediction features must be a finite matrix")
        if self._weights is not None:
            return x @ self._weights
        if self._model is None:
            raise ValueError("aggregator must be fitted before prediction")
        return np.asarray(self._model.predict(x), dtype=float)


def _validated(
    features: npt.ArrayLike,
    target: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 1
        or len(x) != len(y)
        or not len(y)
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(y))
    ):
        raise ValueError("training data must be finite aligned features and target")
    return x, y
