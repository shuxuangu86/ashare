from dataclasses import dataclass
from enum import StrEnum

from aquant.domain.data_release import DataReleaseId
from aquant.factors.evaluation.analyzer import FactorEvaluation


class FactorApprovalStatus(StrEnum):
    DRAFT = "DRAFT"
    COMPUTED = "COMPUTED"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"


@dataclass(frozen=True, slots=True)
class FactorCard:
    name: str
    version: str
    hypothesis: str
    expression: str
    expression_hash: str
    data_sources: tuple[str, ...]
    available_time_rule: str
    preprocessing: str
    expected_exposures: str
    failure_conditions: str
    code_version: str
    data_release_id: DataReleaseId
    approval_status: FactorApprovalStatus
    evaluation: FactorEvaluation

    def __post_init__(self) -> None:
        text_fields = (
            self.name,
            self.version,
            self.hypothesis,
            self.expression,
            self.available_time_rule,
            self.preprocessing,
            self.expected_exposures,
            self.failure_conditions,
            self.code_version,
        )
        if any(not value.strip() for value in text_fields) or not self.data_sources:
            raise ValueError("factor card required fields must not be blank")
        if len(self.expression_hash) != 64:
            raise ValueError("factor card expression_hash must be SHA-256")

    def to_markdown(self) -> str:
        metrics = self.evaluation
        sources = ", ".join(self.data_sources)
        quantiles = ", ".join(f"{value:.6f}" for value in metrics.quantile_returns)
        return f"""# Factor Card: {self.name}

- Version: `{self.version}`
- Status: `{self.approval_status.value}`
- Expression: `{self.expression}`
- Expression SHA-256: `{self.expression_hash}`
- Data release: `{self.data_release_id}`
- Code version: `{self.code_version}`
- Data sources: {sources}

## Research hypothesis

{self.hypothesis}

## Point-in-time and preprocessing

- Availability rule: {self.available_time_rule}
- Preprocessing: {self.preprocessing}
- Expected exposures: {self.expected_exposures}
- Failure conditions: {self.failure_conditions}

## Out-of-sample evaluation

- IC mean: {metrics.ic_mean:.6f}
- Rank IC mean: {metrics.rank_ic_mean:.6f}
- Rank ICIR: {metrics.rank_ic_ir:.6f}
- Long-short return: {metrics.long_short_return:.6f}
- Monotonicity: {metrics.monotonicity:.6f}
- Turnover: {metrics.turnover:.6f}
- Coverage: {metrics.coverage:.2%}
- Quantile returns: {quantiles}
"""
