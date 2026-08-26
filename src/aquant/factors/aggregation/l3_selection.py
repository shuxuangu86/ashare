from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_COMPLEXITY_ORDER = {
    "equal_weight": 0,
    "ic_weight": 1,
    "icir_weight": 2,
    "ridge": 3,
    "elastic_net": 4,
}


def select_l3_family_models(
    *,
    smoke_path: Path,
    output: Path,
    stability_penalty: float = 0.25,
    complexity_penalty: float = 0.001,
) -> dict[str, Any]:
    """Select one model per family without consulting the final smoke fold."""
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("research_status") != "RESEARCH_ONLY":
        raise ValueError("L3 family selection requires a RESEARCH_ONLY smoke artifact")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in smoke["results"]:
        folds = result["fold_rank_ic"]
        if len(folds) != 3:
            raise ValueError("L3 family model selection requires exactly three ordered folds")
        if result["method"] not in _COMPLEXITY_ORDER:
            raise ValueError(f"unsupported L3 family method: {result['method']}")
        grouped[str(result["family"])].append(result)
    if not grouped:
        raise ValueError("L3 smoke artifact contains no family results")

    decisions: list[dict[str, Any]] = []
    for family, candidates in sorted(grouped.items()):
        scored = []
        for candidate in candidates:
            development = [float(value) for value in candidate["fold_rank_ic"][:2]]
            mean = sum(development) / len(development)
            dispersion = abs(development[0] - development[1]) / 2
            score = (
                mean
                - stability_penalty * dispersion
                - complexity_penalty * _COMPLEXITY_ORDER[str(candidate["method"])]
            )
            scored.append((score, -_COMPLEXITY_ORDER[str(candidate["method"])], candidate))
        _, _, selected = max(scored, key=lambda item: (item[0], item[1], item[2]["method"]))
        development = [float(value) for value in selected["fold_rank_ic"][:2]]
        holdout = float(selected["fold_rank_ic"][2])
        development_mean = sum(development) / len(development)
        decisions.append(
            {
                "family": family,
                "selected_method": selected["method"],
                "factor_ids": selected["factor_ids"],
                "development_fold_rank_ic": development,
                "development_mean_rank_ic": development_mean,
                "holdout_rank_ic": holdout,
                "holdout_direction_consistent": holdout >= 0,
                "generalization_ratio": (
                    holdout / development_mean if abs(development_mean) > 1e-12 else 0.0
                ),
                "selection_score": max(item[0] for item in scored),
            }
        )

    holdout_values = [float(item["holdout_rank_ic"]) for item in decisions]
    payload: dict[str, Any] = {
        "status": "PASS",
        "research_status": "RESEARCH_ONLY",
        "stage": "L3_FAMILY_MODEL_SELECTION",
        "input_neutralization": "SIZE_NEUTRAL",
        "industry_neutralization": "IGNORED_BY_USER_DIRECTION",
        "data_release_id": smoke["data_release_id"],
        "evaluation_code_version": smoke["evaluation_code_version"],
        "source_smoke_hash": smoke["content_hash"],
        "feature_eligible_pool_hash": smoke["feature_eligible_pool_hash"],
        "selection_folds": [0, 1],
        "holdout_fold": 2,
        "holdout_used_for_selection": False,
        "stability_penalty": stability_penalty,
        "complexity_penalty": complexity_penalty,
        "family_count": len(decisions),
        "holdout_positive_family_count": sum(value >= 0 for value in holdout_values),
        "mean_family_holdout_rank_ic": sum(holdout_values) / len(holdout_values),
        "selected_method_counts": dict(
            sorted(Counter(item["selected_method"] for item in decisions).items())
        ),
        "families": decisions,
        "next_stage": "L3_CROSS_FAMILY_WALK_FORWARD",
    }
    payload["content_hash"] = _content_hash(payload)
    _write_json(output, payload)
    _write_markdown(output.with_suffix(".md"), payload)
    return payload


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".l3-family-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# L3 Family Model Selection",
        "",
        f"- Status: `{payload['research_status']}`",
        f"- Families: {payload['family_count']}",
        f"- Positive holdout families: {payload['holdout_positive_family_count']}",
        f"- Mean family holdout RankIC: {payload['mean_family_holdout_rank_ic']:.6f}",
        "- Industry neutralization: ignored; input is size-neutral only.",
        "",
        "| Family | Method | Development RankIC | Holdout RankIC |",
        "|---|---:|---:|---:|",
    ]
    for item in payload["families"]:
        lines.append(
            f"| {item['family']} | {item['selected_method']} | "
            f"{item['development_mean_rank_ic']:.6f} | {item['holdout_rank_ic']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
