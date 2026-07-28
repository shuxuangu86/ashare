import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aquant.factors.evaluation.ic import ICStatistics
from aquant.factors.evaluation.quality import FactorQuality
from aquant.factors.spec import FactorSpec


@dataclass(frozen=True, slots=True)
class FactorReport:
    spec: FactorSpec
    data_release_id: str
    quality: FactorQuality | None = None
    pearson_ic: ICStatistics | None = None
    rank_ic: ICStatistics | None = None
    failure_reason: str | None = None
    related_factors: tuple[str, ...] = ()
    cluster_id: str | None = None
    extra_metrics: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        payload = {
            "factor": json.loads(self.spec.model_dump_json()),
            "data_release_id": self.data_release_id,
            "pit_rule": f"data_lag={self.spec.data_lag}; available_at <= evaluation_time",
            "quality": asdict(self.quality) if self.quality else None,
            "pearson_ic": asdict(self.pearson_ic) if self.pearson_ic else None,
            "rank_ic": asdict(self.rank_ic) if self.rank_ic else None,
            "related_factors": self.related_factors,
            "cluster_id": self.cluster_id,
            "failure_reason": self.failure_reason,
            "extra_metrics": self.extra_metrics or {},
        }
        payload["content_hash"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return payload

    def markdown(self) -> str:
        quality = self.quality
        rank_ic = self.rank_ic
        metrics = self.extra_metrics or {}
        horizons = metrics.get("horizons", {})
        rows = _evaluation_rows(horizons)
        oos_rows = _evaluation_rows(metrics.get("oos_horizons", {}))
        timing = metrics.get("timing", {}).get("definition", "not evaluated")
        oos_timing = metrics.get("oos_timing", {})
        provenance = metrics.get("provenance", {})
        return f"""# Factor Report: {self.spec.factor_id}

- Version: `{self.spec.version}`
- Family / layer: `{self.spec.family}` / `{self.spec.layer.value}`
- Status: `{self.spec.status.value}`
- Data release: `{self.data_release_id}`
- Expression hash: `{self.spec.expression_hash}`
- Required history / data lag: `{self.spec.required_history}` / `{self.spec.data_lag}`
- Inputs: `{", ".join(self.spec.input_fields)}`

## Definition and hypothesis

{self.spec.description}

{self.spec.hypothesis}

## Quality and evaluation

- Coverage: {quality.coverage if quality else "not evaluated"}
- Missing rate: {quality.missing_rate if quality else "not evaluated"}
- Quality flags: {", ".join(quality.flags) if quality else "not evaluated"}
- Rank IC mean: {rank_ic.mean if rank_ic else "not evaluated"}
- Rank ICIR: {rank_ic.icir if rank_ic else "not evaluated"}
- Failure reason: {self.failure_reason or "none"}

## Institutional evaluation

| Horizon | Rank IC | Rank ICIR | Newey-West t | Net long-short | Turnover | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|
{rows or "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |"}

- Timing: `{timing}`
- Code version: `{provenance.get("code_version", "not recorded")}`
- Configuration hash: `{provenance.get("config_hash", "not recorded")}`

## Fixed-time holdout

| Horizon | Rank IC | Rank ICIR | Newey-West t | Net long-short | Turnover | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|
{oos_rows or "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |"}

- Method: `{oos_timing.get("method", "not evaluated")}`
- Range: `{oos_timing.get("start_date", "n/a")}` to `{oos_timing.get("end_date", "n/a")}`

## Lineage and controls

- Parents: {", ".join(self.spec.parent_factor_ids) or "none"}
- Variant dimension: {self.spec.variant_dimension or "none"}
- PIT rule: `available_at <= evaluation_time`
- Related factors: {", ".join(self.related_factors) or "none"}
- Cluster: {self.cluster_id or "none"}
"""


def _evaluation_rows(horizons: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"| {horizon} | {float(values['rank_ic_mean']):.6f} | "
            f"{float(values['rank_icir']):.4f} | "
            f"{float(values.get('newey_west_t', float('nan'))):.4f} | "
            f"{float(values.get('net_long_short_return', float('nan'))):.6f} | "
            f"{float(values['turnover']):.4f} | "
            f"{float(values.get('max_drawdown', float('nan'))):.4f} |"
        )
        for horizon, values in sorted(
            horizons.items(),
            key=lambda item: int(item[0]),
        )
    )


def write_factor_report(report: FactorReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{report.spec.factor_id}-{report.spec.version}.json"
    markdown_path = directory / f"{report.spec.factor_id}-{report.spec.version}.md"
    json_path.write_text(
        json.dumps(report.payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(report.markdown(), encoding="utf-8")
    return json_path, markdown_path
