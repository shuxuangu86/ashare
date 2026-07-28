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
    groups: npt.ArrayLike | None = None,
    parameters: dict[str, object] | None = None,
) -> WalkForwardPrediction:
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("walk-forward training data must be finite and aligned")
    resolved_groups = None if groups is None else np.asarray(groups)
    if resolved_groups is not None and (
        resolved_groups.shape != (len(y),) or np.any(resolved_groups[1:] < resolved_groups[:-1])
    ):
        raise ValueError("walk-forward groups must be aligned and ordered")
    unique_groups = (
        np.arange(len(y), dtype=np.int64) if resolved_groups is None else np.unique(resolved_groups)
    )
    splits = walk_forward_splits(
        len(unique_groups),
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
        train_indices = _group_indices(resolved_groups, unique_groups, split.train)
        validation_indices = _group_indices(resolved_groups, unique_groups, split.validation)
        train_x, train_y = x[train_indices], y[train_indices]
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
        predictions[validation_indices] = model.predict(x[validation_indices])
    valid = np.isfinite(predictions)
    rank_correlation = _rank_correlation(predictions, y, resolved_groups)
    return WalkForwardPrediction(
        predictions,
        splits,
        rank_correlation,
        float(np.mean(valid)),
    )


def _group_indices(
    groups: npt.NDArray[np.generic] | None,
    unique_groups: npt.NDArray[np.generic],
    selected: slice,
) -> npt.NDArray[np.int64]:
    if groups is None:
        return np.arange(len(unique_groups), dtype=np.int64)[selected]
    return np.flatnonzero(np.isin(groups, unique_groups[selected]))


def _rank_correlation(
    predictions: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    groups: npt.NDArray[np.generic] | None,
) -> float:
    valid = np.isfinite(predictions)
    if groups is None:
        return (
            float(np.corrcoef(rankdata(predictions[valid]), rankdata(target[valid]))[0, 1])
            if np.count_nonzero(valid) > 1
            else float("nan")
        )
    correlations: list[float] = []
    for group in np.unique(groups[valid]):
        group_valid = valid & (groups == group)
        if np.count_nonzero(group_valid) > 1:
            left = rankdata(predictions[group_valid])
            right = rankdata(target[group_valid])
            if np.std(left) and np.std(right):
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")
