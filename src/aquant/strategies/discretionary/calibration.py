from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from aquant.domain.time import require_aware
from aquant.strategies.discretionary.models import OutcomeLayer


def _probability(value: Decimal, *, field_name: str) -> Decimal:
    normalized = Decimal(value)
    if not Decimal("0") <= normalized <= Decimal("1"):
        raise ValueError(f"{field_name} must be between 0 and 1")
    return normalized


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    outcome_id: str
    evidence_id: str
    source_id: str
    method_version: str
    layer: OutcomeLayer
    predicted_probability: Decimal
    realized_outcome: Decimal
    prediction_available_at: datetime
    outcome_available_at: datetime

    def __post_init__(self) -> None:
        prediction_available_at = require_aware(
            self.prediction_available_at, field_name="prediction_available_at"
        )
        outcome_available_at = require_aware(
            self.outcome_available_at, field_name="outcome_available_at"
        )
        if outcome_available_at <= prediction_available_at:
            raise ValueError("calibration outcome must become available after its prediction")
        for field_name in ("outcome_id", "evidence_id", "source_id", "method_version"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "predicted_probability",
            _probability(self.predicted_probability, field_name="predicted_probability"),
        )
        object.__setattr__(
            self,
            "realized_outcome",
            _probability(self.realized_outcome, field_name="realized_outcome"),
        )
        object.__setattr__(self, "prediction_available_at", prediction_available_at)
        object.__setattr__(self, "outcome_available_at", outcome_available_at)


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    source_id: str
    method_version: str
    layer: OutcomeLayer
    asof_time: datetime
    sample_size: int
    mean_error: Decimal
    brier_score: Decimal
    directional_accuracy: Decimal
    reliability: Decimal


class CalibrationLedger:
    """Append-only posterior labels with layer-specific and point-in-time calibration."""

    def __init__(self, *, prior_weight: Decimal = Decimal("4")) -> None:
        normalized = Decimal(prior_weight)
        if normalized <= 0:
            raise ValueError("calibration prior_weight must be positive")
        self._prior_weight = normalized
        self._outcomes: dict[str, CalibrationOutcome] = {}

    def add(self, outcome: CalibrationOutcome) -> None:
        if outcome.outcome_id in self._outcomes:
            raise ValueError(f"calibration outcome already exists: {outcome.outcome_id}")
        self._outcomes[outcome.outcome_id] = outcome

    def outcomes_asof(self, asof_time: datetime) -> tuple[CalibrationOutcome, ...]:
        normalized_asof = require_aware(asof_time, field_name="asof_time")
        return tuple(
            sorted(
                (
                    outcome
                    for outcome in self._outcomes.values()
                    if outcome.outcome_available_at <= normalized_asof
                ),
                key=lambda outcome: (outcome.outcome_available_at, outcome.outcome_id),
            )
        )

    def summary(
        self,
        *,
        source_id: str,
        method_version: str,
        layer: OutcomeLayer,
        asof_time: datetime,
        default_reliability: Decimal,
    ) -> CalibrationSummary:
        normalized_asof = require_aware(asof_time, field_name="asof_time")
        default = _probability(default_reliability, field_name="default_reliability")
        outcomes = tuple(
            outcome
            for outcome in self.outcomes_asof(normalized_asof)
            if outcome.source_id == source_id
            and outcome.method_version == method_version
            and outcome.layer is layer
        )
        if not outcomes:
            return CalibrationSummary(
                source_id,
                method_version,
                layer,
                normalized_asof,
                0,
                Decimal("0"),
                Decimal("0"),
                Decimal("0"),
                default,
            )

        errors = tuple(
            outcome.predicted_probability - outcome.realized_outcome for outcome in outcomes
        )
        squared_errors = tuple(error * error for error in errors)
        sample_size = len(outcomes)
        divisor = Decimal(sample_size)
        mean_error = sum(errors, Decimal("0")) / divisor
        brier_score = sum(squared_errors, Decimal("0")) / divisor
        correct_directions = sum(
            1
            for outcome in outcomes
            if (outcome.predicted_probability >= Decimal("0.5"))
            == (outcome.realized_outcome >= Decimal("0.5"))
        )
        directional_accuracy = Decimal(correct_directions) / divisor

        # A 0.5 forecast on a binary outcome has Brier 0.25 and therefore zero skill.
        skill_scores = tuple(
            max(Decimal("0"), Decimal("1") - squared_error / Decimal("0.25"))
            for squared_error in squared_errors
        )
        reliability = (self._prior_weight * default + sum(skill_scores, Decimal("0"))) / (
            self._prior_weight + divisor
        )
        reliability = min(Decimal("1"), max(Decimal("0"), reliability))
        return CalibrationSummary(
            source_id,
            method_version,
            layer,
            normalized_asof,
            sample_size,
            mean_error,
            brier_score,
            directional_accuracy,
            reliability,
        )

    def reliability(
        self,
        *,
        source_id: str,
        method_version: str,
        asof_time: datetime,
        default_reliability: Decimal,
    ) -> Decimal:
        """Only measurement accuracy calibrates an evidence sensor's future weight."""
        return self.summary(
            source_id=source_id,
            method_version=method_version,
            layer=OutcomeLayer.MEASUREMENT,
            asof_time=asof_time,
            default_reliability=default_reliability,
        ).reliability
