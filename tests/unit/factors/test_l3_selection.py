import json
from pathlib import Path

import pytest

from aquant.factors.aggregation.l3_selection import select_l3_family_models


def test_family_selection_never_uses_holdout_fold(tmp_path: Path) -> None:
    smoke = _smoke()
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(smoke))
    first = select_l3_family_models(smoke_path=path, output=tmp_path / "selection.json")
    smoke["results"][0]["fold_rank_ic"][2] = -99.0
    path.write_text(json.dumps(smoke))
    second = select_l3_family_models(smoke_path=path, output=tmp_path / "selection-2.json")

    assert first["families"][0]["selected_method"] == "equal_weight"
    assert second["families"][0]["selected_method"] == "equal_weight"
    assert first["holdout_used_for_selection"] is False
    assert (tmp_path / "selection.md").is_file()


def test_family_selection_validates_research_artifact(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({"research_status": "PRODUCTION", "results": []}))
    with pytest.raises(ValueError, match="RESEARCH_ONLY"):
        select_l3_family_models(smoke_path=path, output=tmp_path / "selection.json")


def _smoke() -> dict[str, object]:
    return {
        "research_status": "RESEARCH_ONLY",
        "data_release_id": "release",
        "evaluation_code_version": "code",
        "content_hash": "a" * 64,
        "feature_eligible_pool_hash": "b" * 64,
        "results": [
            {
                "family": "momentum",
                "method": "equal_weight",
                "factor_ids": ["a", "b"],
                "fold_rank_ic": [0.05, 0.04, 0.03],
            },
            {
                "family": "momentum",
                "method": "ridge",
                "factor_ids": ["a", "b"],
                "fold_rank_ic": [0.01, 0.02, 0.90],
            },
        ],
    }
