from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata  # type: ignore[import-untyped]

from aquant.factors.aggregation.models import AlphaAggregator, ModelKind
from aquant.factors.evaluation.walk_forward import TimeSplit, walk_forward_splits


@dataclass(frozen=True, slots=True)
class WalkForwardPrediction:
    predictions: npt.NDArray[np.float64]
    splits: tuple[TimeSplit, ...]
    rank_correlation: float
    coverage: float


def walk_forward_predict(
    kind: ModelKind,
    features: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    train_size: int,
    validation_size: int,
    step: int,
    purge: int,
    embargo: int = 0,
    parameters: dict[str, object] | None = None,
) -> WalkForwardPrediction:
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("walk-forward training data must be finite and aligned")
    splits = walk_forward_splits(
        len(y),
        train_size=train_size,
        validation_size=validation_size,
        step=step,
        purge=purge,
        embargo=embargo,
    )
    if not splits:
        raise ValueError("walk-forward configuration produces no validation fold")
    predictions = np.full(len(y), np.nan)
    for split in splits:
        train_x, train_y = x[split.train], y[split.train]
        correlations = np.asarray(
            [
                np.corrcoef(train_x[:, column], train_y)[0, 1]
                if np.std(train_x[:, column]) and np.std(train_y)
                else 0.0
                for column in range(train_x.shape[1])
            ]
        )
        model = AlphaAggregator(kind, **(parameters or {})).fit(
            train_x,
            train_y,
            historical_ic=correlations,
            historical_icir=correlations,
        )
        predictions[split.validation] = model.predict(x[split.validation])
    valid = np.isfinite(predictions)
    rank_correlation = (
        float(np.corrcoef(rankdata(predictions[valid]), rankdata(y[valid]))[0, 1])
        if np.count_nonzero(valid) > 1
        else float("nan")
    )
    return WalkForwardPrediction(
        predictions,
        splits,
        rank_correlation,
        float(np.mean(valid)),
    )
