from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from aquant.factors.selection.nested_walk_forward import (
    build_nested_l2_pools,
    define_outer_folds,
)


def test_outer_folds_are_contiguous_and_cover_locked_period() -> None:
    dates = tuple(date(2021, 1, 1) + timedelta(days=index) for index in range(13))
    folds = define_outer_folds(dates, fold_count=5)

    assert sum(item.test_observations for item in folds) == len(dates)
    assert folds[0].test_start == dates[0]
    assert folds[-1].test_end == dates[-1]
    assert all(left.test_end < right.test_start for left, right in pairwise(folds))


def test_nested_selection_does_not_use_outer_values(tmp_path: Path) -> None:
    dates = tuple(date(2019, 1, 1) + timedelta(days=index) for index in range(500))
    outer_dates = dates[300:]
    first = _records(len(dates))
    first_payload = build_nested_l2_pools(
        factor_records=first,
        evidence_dates=dates,
        outer_dates=outer_dates,
        output=tmp_path / "first.json",
        purge_observations=6,
        fold_count=2,
        minimum_observations=50,
    )
    second = _records(len(dates))
    for record in second.values():
        record["rank_ic"][300:] *= -100
    second_payload = build_nested_l2_pools(
        factor_records=second,
        evidence_dates=dates,
        outer_dates=outer_dates,
        output=tmp_path / "second.json",
        purge_observations=6,
        fold_count=2,
        minimum_observations=50,
    )

    assert (
        first_payload["folds"][0]["selected_factor_ids"]
        == second_payload["folds"][0]["selected_factor_ids"]
    )
    assert first_payload["folds"][0]["factors"] == second_payload["folds"][0]["factors"]
    assert first_payload["outer_test_used_for_selection"] is False


def test_nested_selection_purges_label_overlap(tmp_path: Path) -> None:
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(260))
    payload = build_nested_l2_pools(
        factor_records=_records(len(dates)),
        evidence_dates=dates,
        outer_dates=dates[200:],
        output=tmp_path / "pool.json",
        purge_observations=6,
        fold_count=2,
        minimum_observations=50,
    )

    first = payload["folds"][0]
    assert first["train_end_after_purge"] == dates[193].isoformat()
    assert first["test_start"] == dates[200].isoformat()
    assert Path(tmp_path / "pool.json").is_file()


def test_nested_selection_rejects_misaligned_series(tmp_path: Path) -> None:
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(20))
    records = _records(len(dates))
    records["a"]["rank_ic"] = records["a"]["rank_ic"][:-1]
    with pytest.raises(ValueError, match="align"):
        build_nested_l2_pools(
            factor_records=records,
            evidence_dates=dates,
            outer_dates=dates[10:],
            output=tmp_path / "pool.json",
            minimum_observations=2,
        )


def test_nested_selection_deduplicates_sign_mirrors(tmp_path: Path) -> None:
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(260))
    records = _records(len(dates))
    records["b"]["expected_direction"] = 1
    records["b"]["rank_ic"] = -np.asarray(records["a"]["rank_ic"])
    payload = build_nested_l2_pools(
        factor_records=records,
        evidence_dates=dates,
        outer_dates=dates[200:],
        output=tmp_path / "pool.json",
        purge_observations=6,
        fold_count=2,
        minimum_observations=50,
    )

    assert payload["folds"][0]["diagnostics"]["duplicate_count"] == 1


def _records(length: int) -> dict[str, dict[str, object]]:
    x = np.linspace(0, 8 * np.pi, length)
    return {
        "a": {
            "family": "momentum",
            "subfamily": "trend",
            "expected_direction": 1,
            "expression_hash": "a" * 64,
            "complexity_score": 1.0,
            "rank_ic": 0.03 + 0.01 * np.sin(x),
        },
        "b": {
            "family": "momentum",
            "subfamily": "reversal",
            "expected_direction": 0,
            "expression_hash": "b" * 64,
            "complexity_score": 2.0,
            "rank_ic": -0.02 + 0.02 * np.cos(x),
        },
    }
