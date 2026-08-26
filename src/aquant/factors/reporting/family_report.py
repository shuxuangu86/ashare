from pathlib import Path

from aquant.factors.spec import FactorSpec


def write_family_report(family: str, specs: tuple[FactorSpec, ...], path: Path) -> None:
    members = tuple(spec for spec in specs if spec.family == family)
    lines = [
        f"# Factor Family: {family}",
        "",
        f"- Factor versions: {len(members)}",
        "",
        "| Factor | Version | Status | History |",
        "|---|---:|---|---:|",
    ]
    lines.extend(
        f"| {item.factor_id} | {item.version} | {item.status.value} | {item.required_history} |"
        for item in members
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
