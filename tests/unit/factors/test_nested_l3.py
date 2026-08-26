from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from aquant.factors.aggregation.models import ModelKind
from aquant.factors.aggregation.nested_l3 import _model_rows, run_nested_l3


def test_nested_l3_outer_fold_is_not_used_for_selection(tmp_path: Path) -> None:
    generator = np.random.default_rng(9)
    date_count, security_count = 90, 24
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(date_count))
    factors = generator.normal(size=(2, date_count, security_count))
    returns = (
        0.03 * factors[0]
        - 0.02 * factors[1]
        + generator.normal(scale=0.01, size=(date_count, security_count))
    )
    close = np.full((date_count, security_count), 100.0)
    for index in range(1, date_count):
        close[index] = close[index - 1] * (1 + returns[index - 1])
    pools = _pools(dates)
    first = run_nested_l3(
        factor_values=factors,
        close=close,
        trade_dates=dates,
        factor_ids=("a", "b"),
        fold_pools=pools,
        output_scores=tmp_path / "first.npy",
        output_metadata=tmp_path / "first.json",
        horizon=2,
        inner_validation_dates=20,
        inner_purge_dates=3,
        maximum_training_rows=2_000,
        methods=(ModelKind.EQUAL_WEIGHT, ModelKind.IC_WEIGHT, ModelKind.RIDGE),
    )
    changed = close.copy()
    changed[50:70] *= np.linspace(1.0, 3.0, 20)[:, None]
    second = run_nested_l3(
        factor_values=factors,
        close=changed,
        trade_dates=dates,
        factor_ids=("a", "b"),
        fold_pools=pools,
        output_scores=tmp_path / "second.npy",
        output_metadata=tmp_path / "second.json",
        horizon=2,
        inner_validation_dates=20,
        inner_purge_dates=3,
        maximum_training_rows=2_000,
        methods=(ModelKind.EQUAL_WEIGHT, ModelKind.IC_WEIGHT, ModelKind.RIDGE),
    )

    assert (
        first["folds"][0]["families"][0]["selected_method"]
        == second["folds"][0]["families"][0]["selected_method"]
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "first.npy")[50:70],
        np.load(tmp_path / "second.npy")[50:70],
        equal_nan=True,
    )
    assert first["outer_test_used_for_model_selection"] is False


def test_nested_l3_rejects_outer_label_overlap(tmp_path: Path) -> None:
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(90))
    pools = _pools(dates)
    pools["folds"][0]["train_end_after_purge"] = dates[47].isoformat()
    with pytest.raises(ValueError, match="overlap the outer test boundary"):
        run_nested_l3(
            factor_values=np.ones((2, 90, 24)),
            close=np.full((90, 24), 100.0),
            trade_dates=dates,
            factor_ids=("a", "b"),
            fold_pools=pools,
            output_scores=tmp_path / "scores.npy",
            output_metadata=tmp_path / "metadata.json",
            horizon=2,
            inner_validation_dates=20,
            inner_purge_dates=3,
            maximum_training_rows=2_000,
            methods=(ModelKind.EQUAL_WEIGHT,),
        )


def test_nested_l3_serializes_degenerate_validation_metrics(tmp_path: Path) -> None:
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(90))
    generator = np.random.default_rng(27)
    factors = generator.normal(size=(2, 90, 24))
    factors[:, 27:47] = np.nan
    close = np.full((90, 24), 100.0)
    for index in range(1, len(close)):
        close[index] = close[index - 1] * (1 + generator.normal(scale=0.01, size=close.shape[1]))
    payload = run_nested_l3(
        factor_values=factors,
        close=close,
        trade_dates=dates,
        factor_ids=("a", "b"),
        fold_pools=_pools(dates),
        output_scores=tmp_path / "scores.npy",
        output_metadata=tmp_path / "metadata.json",
        horizon=2,
        inner_validation_dates=20,
        inner_purge_dates=3,
        maximum_training_rows=2_000,
        methods=(ModelKind.EQUAL_WEIGHT,),
    )

    family = payload["folds"][0]["families"][0]
    assert family["inner_validation_status"] == "ALL_METHODS_UNAVAILABLE"
    assert family["selection_score"] is None
    assert (tmp_path / "metadata.json").is_file()


def test_model_row_sampling_is_deterministic_date_stratified_and_not_a_code_slice() -> None:
    date_count, security_count = 10, 100
    encoded = (np.arange(date_count)[:, None] * 1_000 + np.arange(security_count)[None, :]).astype(
        float
    )
    values = encoded[None, :, :]
    labels = np.broadcast_to(np.arange(security_count, dtype=float), encoded.shape)
    positions = np.arange(date_count, dtype=np.int64)

    first, first_y = _model_rows(values, labels, positions, maximum_rows=200)
    second, second_y = _model_rows(values, labels, positions, maximum_rows=200)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_y, second_y)
    assert set((first[:, 0] // 1_000).astype(int)) == set(range(date_count))
    assert len(set((first[:, 0] % 1_000).astype(int))) > 20
    assert not np.array_equal(
        np.sort((first[:20, 0] % 1_000).astype(int)),
        np.sort((first[20:40, 0] % 1_000).astype(int)),
    )


def _pools(dates: tuple[date, ...]) -> dict[str, object]:
    factor = {
        "a": {"factor_id": "a", "family": "momentum", "direction": 1},
        "b": {"factor_id": "b", "family": "momentum", "direction": -1},
    }
    return {
        "content_hash": "a" * 64,
        "folds": [
            {
                "fold": 0,
                "train_end_after_purge": dates[46].isoformat(),
                "test_start": dates[50].isoformat(),
                "test_end": dates[69].isoformat(),
                "factors": factor,
            },
            {
                "fold": 1,
                "train_end_after_purge": dates[66].isoformat(),
                "test_start": dates[70].isoformat(),
                "test_end": dates[89].isoformat(),
                "factors": factor,
            },
        ],
    }
