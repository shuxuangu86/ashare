from pathlib import Path

from aquant.factors.feature_sets import FeatureSetSpec


def write_feature_set_report(spec: FeatureSetSpec, path: Path) -> None:
    members = "\n".join(
        f"- `{item.factor_id}` @ `{item.factor_version}`" for item in spec.factor_members
    )
    path.write_text(
        f"""# Feature Set: {spec.feature_set_id}

- Version: `{spec.version}`
- Status: `{spec.status.value}`
- Target horizon: `{spec.target_horizon}`
- Universe: `{spec.universe}`
- Data release: `{spec.data_release_id}`
- Effective from: `{spec.effective_from or "not set"}`
- Training window: `{spec.training_window_days or "not set"}` trading days
- Standardization: `{spec.standardization_method}`
- Winsorization: `{spec.winsorization_method}`
- Missing values: `{spec.missing_value_strategy}`
- Neutralization: `{", ".join(spec.neutralization) or "none"}`
- Selection: `{spec.selection_method}` from `{spec.created_from_experiment}`
- Code version: `{spec.code_version}`
- Configuration hash: `{spec.config_hash}`
- Content hash: `{spec.content_hash}`

## Pinned factor versions

{members}
""",
        encoding="utf-8",
    )
