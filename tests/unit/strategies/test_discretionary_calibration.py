from datetime import timedelta
from decimal import Decimal

import pytest
from tests.fixtures.discretionary import (
    DECISION_TIME,
    NOW,
    RELEASE_ID,
    make_case,
    make_evidence,
)

from aquant.strategies.discretionary import (
    BayesianBeliefEngine,
    CalibrationLedger,
    CalibrationOutcome,
    OutcomeLayer,
)


def _outcome(
    outcome_id: str,
    *,
    layer: OutcomeLayer = OutcomeLayer.MEASUREMENT,
    outcome_available_at=NOW + timedelta(hours=8),  # type: ignore[no-untyped-def]
) -> CalibrationOutcome:
    return CalibrationOutcome(
        outcome_id,
        "sensor-evidence",
        "panel-source",
        "v1",
        layer,
        Decimal("0.9"),
        Decimal("1"),
        NOW + timedelta(hours=4),
        outcome_available_at,
    )


def test_calibration_is_append_only_point_in_time_and_layer_specific() -> None:
    ledger = CalibrationLedger(prior_weight=Decimal("2"))
    measurement = _outcome("measurement")
    transmission = _outcome("transmission", layer=OutcomeLayer.TRANSMISSION)
    investment = _outcome("investment", layer=OutcomeLayer.INVESTMENT)
    future = _outcome("future", outcome_available_at=DECISION_TIME + timedelta(seconds=1))
    for outcome in (investment, future, measurement, transmission):
        ledger.add(outcome)

    summary = ledger.summary(
        source_id="panel-source",
        method_version="v1",
        layer=OutcomeLayer.MEASUREMENT,
        asof_time=DECISION_TIME,
        default_reliability=Decimal("0.4"),
    )

    assert summary.sample_size == 1
    assert summary.mean_error == Decimal("-0.1")
    assert summary.brier_score == Decimal("0.01")
    assert summary.directional_accuracy == Decimal("1")
    assert Decimal("0.4") < summary.reliability <= Decimal("1")
    assert [item.outcome_id for item in ledger.outcomes_asof(DECISION_TIME)] == [
        "investment",
        "measurement",
        "transmission",
    ]

    investment_summary = ledger.summary(
        source_id="panel-source",
        method_version="v1",
        layer=OutcomeLayer.INVESTMENT,
        asof_time=DECISION_TIME,
        default_reliability=Decimal("0.4"),
    )
    assert investment_summary.sample_size == 1
    assert (
        ledger.reliability(
            source_id="panel-source",
            method_version="v1",
            asof_time=DECISION_TIME,
            default_reliability=Decimal("0.4"),
        )
        == summary.reliability
    )


def test_unobserved_source_keeps_prior_reliability() -> None:
    ledger = CalibrationLedger()

    summary = ledger.summary(
        source_id="new-source",
        method_version="v1",
        layer=OutcomeLayer.MEASUREMENT,
        asof_time=DECISION_TIME,
        default_reliability=Decimal("0.3"),
    )

    assert summary.sample_size == 0
    assert summary.reliability == Decimal("0.3")


def test_calibration_rejects_invalid_or_duplicate_outcomes() -> None:
    ledger = CalibrationLedger()
    outcome = _outcome("duplicate")
    ledger.add(outcome)

    with pytest.raises(ValueError, match="already exists"):
        ledger.add(outcome)
    with pytest.raises(ValueError, match="after its prediction"):
        CalibrationOutcome(
            "bad-time",
            "evidence",
            "source",
            "v1",
            OutcomeLayer.MEASUREMENT,
            Decimal("0.5"),
            Decimal("1"),
            NOW,
            NOW,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        CalibrationOutcome(
            "bad-probability",
            "evidence",
            "source",
            "v1",
            OutcomeLayer.MEASUREMENT,
            Decimal("1.2"),
            Decimal("1"),
            NOW,
            NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="prior_weight"):
        CalibrationLedger(prior_weight=Decimal("0"))


def test_only_available_measurement_outcomes_change_future_belief_weight() -> None:
    case = make_case()
    evidence = make_evidence(
        "sensor-evidence",
        source_id="panel-source",
        reliability=Decimal("0.2"),
    )
    ledger = CalibrationLedger(prior_weight=Decimal("1"))
    ledger.add(_outcome("correct-label"))
    engine = BayesianBeliefEngine()

    before = engine.update(
        case,
        (evidence,),
        asof_time=NOW + timedelta(hours=6),
        data_release_id=RELEASE_ID,
        calibration=ledger,
    )
    after = engine.update(
        case,
        (evidence,),
        asof_time=DECISION_TIME,
        data_release_id=RELEASE_ID,
        calibration=ledger,
    )

    assert after.snapshot.probability_for("fundamental") > before.snapshot.probability_for(
        "fundamental"
    )
