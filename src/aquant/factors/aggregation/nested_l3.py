from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aquant.factors.aggregation.models import AlphaAggregator, ModelKind
from aquant.factors.evaluation.ic import information_coefficient
from aquant.factors.operators.cross_sectional import cs_percentile, cs_zscore
from aquant.strategies.nested_l4 import select_l4_configuration

FloatArray = npt.NDArray[np.float64]

_METHOD_COMPLEXITY = {
    ModelKind.EQUAL_WEIGHT: 0,
    ModelKind.IC_WEIGHT: 1,
    ModelKind.ICIR_WEIGHT: 2,
    ModelKind.RIDGE: 3,
    ModelKind.ELASTIC_NET: 4,
}


def run_nested_l3(
    *,
    factor_values: npt.ArrayLike,
    close: npt.ArrayLike,
    trade_dates: tuple[date, ...],
    factor_ids: tuple[str, ...],
    fold_pools: dict[str, Any],
    output_scores: Path,
    output_metadata: Path,
    horizon: int = 5,
    inner_validation_dates: int = 252,
    inner_purge_dates: int = 6,
    maximum_training_rows: int = 200_000,
    methods: tuple[ModelKind, ...] = (
        ModelKind.EQUAL_WEIGHT,
        ModelKind.IC_WEIGHT,
        ModelKind.ICIR_WEIGHT,
        ModelKind.RIDGE,
        ModelKind.ELASTIC_NET,
    ),
    complexity_penalty: float = 0.001,
    neutralization: str = "SIZE_NEUTRAL",
    benchmark_returns: npt.ArrayLike | None = None,
    eligibility_mask: npt.ArrayLike | None = None,
    open_prices: npt.ArrayLike | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train family L3 models inside each outer fold and emit untouched outer predictions."""
    values = np.asarray(factor_values)
    prices = np.asarray(close, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != prices.shape:
        raise ValueError("factor values must be factor-by-date-by-security and align with close")
    if values.shape[0] != len(factor_ids) or prices.shape[0] != len(trade_dates):
        raise ValueError("factor ids and trade dates must align with values")
    benchmark = (
        None if benchmark_returns is None else np.asarray(benchmark_returns, dtype=np.float64)
    )
    eligibility = None if eligibility_mask is None else np.asarray(eligibility_mask, dtype=bool)
    opening = None if open_prices is None else np.asarray(open_prices, dtype=np.float64)
    if benchmark is not None and benchmark.shape != (len(prices),):
        raise ValueError("nested L3 benchmark returns must align with dates")
    if eligibility is not None and eligibility.shape != prices.shape:
        raise ValueError("nested L3 eligibility mask must align with close")
    if opening is not None and opening.shape != prices.shape:
        raise ValueError("nested L3 open prices must align with close")
    if tuple(sorted(set(trade_dates))) != trade_dates:
        raise ValueError("trade dates must be unique and ordered")
    if horizon <= 0 or inner_purge_dates <= horizon:
        raise ValueError("T+1 labels require a purge longer than the return horizon")
    if inner_validation_dates < 20 or maximum_training_rows < 1_000:
        raise ValueError("inner validation and training row budgets are too small")
    if not methods or any(method not in _METHOD_COMPLEXITY for method in methods):
        raise ValueError("nested L3 methods must be supported and non-empty")
    labels = _pit_labels(prices, horizon)
    factor_index = {factor_id: index for index, factor_id in enumerate(factor_ids)}
    date_index = {value: index for index, value in enumerate(trade_dates)}
    score_map = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        _temporary_score_path(output_scores),
        mode="w+",
        dtype=np.float32,
        shape=prices.shape,
    )
    score_map[:] = np.nan
    fold_results: list[dict[str, Any]] = []
    try:
        for fold in fold_pools["folds"]:
            result = _train_fold(
                values=values,
                close=prices,
                labels=labels,
                trade_dates=trade_dates,
                factor_index=factor_index,
                date_index=date_index,
                fold=fold,
                label_horizon=horizon,
                inner_validation_dates=inner_validation_dates,
                inner_purge_dates=inner_purge_dates,
                maximum_training_rows=maximum_training_rows,
                methods=methods,
                complexity_penalty=complexity_penalty,
                benchmark_returns=benchmark,
                eligibility_mask=eligibility,
                open_prices=opening,
            )
            start = date_index[date.fromisoformat(fold["test_start"])]
            end = date_index[date.fromisoformat(fold["test_end"])] + 1
            score_map[start:end] = result.pop("scores")
            fold_results.append(result)
        score_map.flush()
        temporary = Path(score_map.filename)
        score_map._mmap.close()
        output_scores.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_scores)
    except BaseException:
        temporary = Path(score_map.filename)
        temporary.unlink(missing_ok=True)
        raise
    payload: dict[str, Any] = {
        "status": "PASS",
        "research_status": "RESEARCH_ONLY",
        "stage": "NESTED_WALK_FORWARD_L3",
        "neutralization": neutralization,
        "horizon": horizon,
        "inner_validation_dates": inner_validation_dates,
        "inner_purge_dates": inner_purge_dates,
        "maximum_training_rows": maximum_training_rows,
        "methods": [method.value for method in methods],
        "complexity_penalty": complexity_penalty,
        "outer_test_used_for_model_selection": False,
        "preprocessing": "DAILY_CROSS_SECTIONAL_PERCENTILE_ZSCORE; missing_to_zero_after_fit",
        "cross_family_aggregation": "EQUAL_WEIGHT_DAILY_ZSCORE",
        "inner_l4_benchmark": (
            "PIT_ALL_A_SHARE_DAILY_EQUAL_PROXY"
            if benchmark is not None
            else "AVAILABLE_SECURITY_DAILY_EQUAL_CLOSE_RETURN_PROXY"
        ),
        "inner_l4_eligibility": (
            "PIT_LISTED_120D_NON_ST_NON_DELISTING_RISK"
            if eligibility is not None
            else "FINITE_SCORE_AND_CLOSE_ONLY"
        ),
        "inner_l4_execution": (
            "T_CLOSE_SIGNAL_T_PLUS_1_OPEN_WEIGHT_TRANSITION"
            if opening is not None
            else "DELAYED_CLOSE_PROXY"
        ),
        "fold_pool_hash": fold_pools["content_hash"],
        "score_file": str(output_scores),
        "score_file_hash": _file_hash(output_scores),
        "folds": fold_results,
        "selected_method_counts": dict(
            sorted(
                Counter(
                    family["selected_method"]
                    for fold in fold_results
                    for family in fold["families"]
                ).items()
            )
        ),
        "provenance": provenance or {},
    }
    payload["content_hash"] = _content_hash(payload)
    _write_json(output_metadata, payload)
    return payload


def _train_fold(
    *,
    values: npt.NDArray[np.generic],
    close: FloatArray,
    labels: FloatArray,
    trade_dates: tuple[date, ...],
    factor_index: dict[str, int],
    date_index: dict[date, int],
    fold: dict[str, Any],
    label_horizon: int,
    inner_validation_dates: int,
    inner_purge_dates: int,
    maximum_training_rows: int,
    methods: tuple[ModelKind, ...],
    complexity_penalty: float,
    benchmark_returns: FloatArray | None,
    eligibility_mask: npt.NDArray[np.bool_] | None,
    open_prices: FloatArray | None,
) -> dict[str, Any]:
    train_end = date_index[date.fromisoformat(fold["train_end_after_purge"])]
    test_start = date_index[date.fromisoformat(fold["test_start"])]
    test_end = date_index[date.fromisoformat(fold["test_end"])] + 1
    if test_start - train_end <= label_horizon + 1:
        raise ValueError(f"fold {fold['fold']} training labels overlap the outer test boundary")
    visible = np.arange(train_end + 1, dtype=np.int64)
    if len(visible) <= inner_validation_dates + inner_purge_dates + 20:
        raise ValueError(f"fold {fold['fold']} has insufficient inner training history")
    inner_train = visible[: -(inner_validation_dates + inner_purge_dates)]
    inner_validation = visible[-inner_validation_dates:]
    full_train = visible
    test_positions = np.arange(test_start, test_end, dtype=np.int64)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in fold["factors"].values():
        factor_id = str(item["factor_id"])
        if factor_id not in factor_index:
            raise ValueError(f"fold factor is missing from materialized cache: {factor_id}")
        by_family[str(item["family"])].append(item)
    family_outputs: list[npt.NDArray[np.float32]] = []
    validation_outputs: list[npt.NDArray[np.float32]] = []
    family_results: list[dict[str, Any]] = []
    for family, members in sorted(by_family.items()):
        ids = tuple(str(item["factor_id"]) for item in members)
        directions = np.asarray([int(item["direction"]) for item in members], dtype=np.float64)
        raw = np.asarray([values[factor_index[factor_id]] for factor_id in ids])
        normalized = _normalize(raw, directions)
        train_x, train_y = _model_rows(
            normalized, labels, inner_train, maximum_rows=maximum_training_rows
        )
        evidence = _historical_evidence(normalized, labels, inner_train)
        validation_scores: dict[ModelKind, float | None] = {}
        for method in methods:
            model = _fit(method, train_x, train_y, evidence)
            predictions = _predict_dates(model, normalized, inner_validation)
            validation_scores[method] = _finite_or_none(
                _daily_rank_ic(predictions, labels[inner_validation])
            )
        valid_methods = tuple(method for method in methods if validation_scores[method] is not None)

        selected = (
            max(
                valid_methods,
                key=partial(
                    _method_selection_key,
                    scores=validation_scores,
                    complexity_penalty=complexity_penalty,
                ),
            )
            if valid_methods
            else min(methods, key=_METHOD_COMPLEXITY.__getitem__)
        )
        selected_validation_score = validation_scores[selected]
        selected_inner_model = _fit(selected, train_x, train_y, evidence)
        validation_outputs.append(
            _predict_dates(selected_inner_model, normalized, inner_validation).astype(np.float32)
        )
        full_x, full_y = _model_rows(
            normalized, labels, full_train, maximum_rows=maximum_training_rows
        )
        full_evidence = _historical_evidence(normalized, labels, full_train)
        final_model = _fit(selected, full_x, full_y, full_evidence)
        outer_predictions = _predict_dates(final_model, normalized, test_positions).astype(
            np.float32
        )
        family_outputs.append(outer_predictions)
        family_results.append(
            {
                "family": family,
                "factor_ids": list(ids),
                "selected_method": selected.value,
                "inner_validation_rank_ic": {
                    method.value: validation_scores[method] for method in methods
                },
                "selection_score": (
                    selected_validation_score - complexity_penalty * _METHOD_COMPLEXITY[selected]
                    if selected_validation_score is not None
                    else None
                ),
                "inner_validation_status": (
                    "AVAILABLE" if valid_methods else "ALL_METHODS_UNAVAILABLE"
                ),
                "outer_rank_ic": _finite_or_none(
                    _daily_rank_ic(outer_predictions.astype(np.float64), labels[test_positions])
                ),
                "outer_used_for_selection": False,
            }
        )
        del raw, normalized
    if not family_outputs:
        raise ValueError(f"fold {fold['fold']} produced no L3 family outputs")
    cross_family = np.asarray(
        [cs_zscore(output.astype(np.float64)).astype(np.float32) for output in family_outputs]
    )
    composite = _mean_available(cross_family)
    validation_cross_family = np.asarray(
        [cs_zscore(output.astype(np.float64)) for output in validation_outputs]
    )
    validation_composite = _mean_available(validation_cross_family)
    l4_scores = np.full(close.shape, np.nan)
    l4_scores[inner_validation] = validation_composite
    l4_selection = select_l4_configuration(
        scores=l4_scores,
        close=close,
        trade_dates=trade_dates,
        validation_positions=inner_validation,
        benchmark_returns=benchmark_returns,
        eligibility_mask=eligibility_mask,
        open_prices=open_prices,
    )
    return {
        "fold": int(fold["fold"]),
        "train_start": trade_dates[0].isoformat(),
        "train_end_after_purge": trade_dates[train_end].isoformat(),
        "inner_train_end": trade_dates[int(inner_train[-1])].isoformat(),
        "inner_validation_start": trade_dates[int(inner_validation[0])].isoformat(),
        "inner_validation_end": trade_dates[int(inner_validation[-1])].isoformat(),
        "test_start": trade_dates[test_start].isoformat(),
        "test_end": trade_dates[test_end - 1].isoformat(),
        "outer_test_used_for_model_selection": False,
        "family_count": len(family_results),
        "families": family_results,
        "l4_selection": l4_selection,
        "outer_composite_rank_ic": _finite_or_none(
            _daily_rank_ic(composite.astype(np.float64), labels[test_positions])
        ),
        "scores": composite,
    }


def _normalize(values: npt.NDArray[np.generic], directions: FloatArray) -> FloatArray:
    result = np.empty(values.shape, dtype=np.float32)
    for index, direction in enumerate(directions):
        percentiles = cs_percentile(values[index]) - 0.5
        result[index] = (cs_zscore(percentiles) * direction).astype(np.float32)
    return result.astype(np.float64)


def _model_rows(
    values: FloatArray,
    labels: FloatArray,
    positions: npt.NDArray[np.int64],
    *,
    maximum_rows: int,
) -> tuple[FloatArray, FloatArray]:
    security_count = values.shape[2]
    target = cs_percentile(labels[positions]) - 0.5
    valid = np.isfinite(target)
    has_feature = np.zeros(target.shape, dtype=np.bool_)
    for factor in values:
        has_feature |= np.isfinite(factor[positions])
    eligible = valid & has_feature
    candidates = np.flatnonzero(eligible)
    if len(candidates) > maximum_rows:
        # Deterministically stratify by date, then sample securities inside each
        # cross-section. This avoids a permanent code slice and date imbalance.
        seed = np.random.SeedSequence(
            [int(positions[0]), int(positions[-1]), len(positions), security_count]
        )
        generator = np.random.default_rng(seed)
        base, remainder = divmod(maximum_rows, len(positions))
        quotas = np.full(len(positions), base, dtype=np.int64)
        if base:
            quotas[:remainder] += 1
        else:
            chosen_dates = generator.choice(len(positions), size=maximum_rows, replace=False)
            quotas[chosen_dates] = 1
        sampled: list[npt.NDArray[np.int64]] = []
        for date_offset in range(len(positions)):
            securities = np.flatnonzero(eligible[date_offset])
            quota = int(quotas[date_offset])
            if quota <= 0 or not len(securities):
                continue
            selected = (
                securities
                if len(securities) <= quota
                else np.sort(generator.choice(securities, size=quota, replace=False))
            )
            sampled.append(date_offset * security_count + selected)
        candidates = np.concatenate(sampled) if sampled else np.empty(0, dtype=np.int64)
    date_offsets, security_positions = np.divmod(candidates, security_count)
    rows = values[:, positions[date_offsets], security_positions].T
    y = target[date_offsets, security_positions]
    return np.nan_to_num(rows), y


def _historical_evidence(
    values: FloatArray, labels: FloatArray, positions: npt.NDArray[np.int64]
) -> tuple[FloatArray, FloatArray]:
    means: list[float] = []
    icirs: list[float] = []
    for factor in values:
        statistics = information_coefficient(factor[positions], labels[positions], rank=True)
        mean = statistics.mean if np.isfinite(statistics.mean) else 0.0
        icir = statistics.icir if np.isfinite(statistics.icir) else 0.0
        means.append(mean)
        icirs.append(icir)
    return np.asarray(means), np.asarray(icirs)


def _fit(
    method: ModelKind,
    features: FloatArray,
    target: FloatArray,
    evidence: tuple[FloatArray, FloatArray],
) -> AlphaAggregator:
    kwargs: dict[str, npt.ArrayLike] = {}
    if method in {ModelKind.IC_WEIGHT, ModelKind.ICIR_WEIGHT}:
        kwargs = {"historical_ic": evidence[0], "historical_icir": evidence[1]}
    return AlphaAggregator(method).fit(features, target, **kwargs)


def _predict_dates(
    model: AlphaAggregator, values: FloatArray, positions: npt.NDArray[np.int64]
) -> FloatArray:
    block = values[:, positions].transpose(1, 2, 0)
    missing = np.all(~np.isfinite(block), axis=2)
    predictions = model.predict(np.nan_to_num(block.reshape(-1, values.shape[0]))).reshape(
        len(positions), values.shape[2]
    )
    predictions[missing] = np.nan
    return predictions


def _daily_rank_ic(predictions: FloatArray, labels: FloatArray) -> float:
    return information_coefficient(predictions, labels, rank=True).mean


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _mean_available(values: FloatArray) -> FloatArray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(np.nansum(values, axis=0), counts, out=result, where=counts > 0)
    return result


def _method_selection_key(
    method: ModelKind,
    scores: dict[ModelKind, float | None],
    complexity_penalty: float,
) -> tuple[float, int]:
    score = scores[method]
    if score is None:
        return (float("-inf"), -_METHOD_COMPLEXITY[method])
    return (
        score - complexity_penalty * _METHOD_COMPLEXITY[method],
        -_METHOD_COMPLEXITY[method],
    )


def _pit_labels(close: FloatArray, horizon: int) -> FloatArray:
    labels = np.full(close.shape, np.nan)
    if horizon + 1 < len(close):
        with np.errstate(all="ignore"):
            labels[: -(horizon + 1)] = close[horizon + 1 :] / close[1:-horizon] - 1
    labels[~np.isfinite(labels)] = np.nan
    return labels


def _temporary_score_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".nested-l3-", suffix=".npy", dir=output.parent)
    os.close(descriptor)
    Path(name).unlink()
    return Path(name)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".nested-l3-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
