from datetime import UTC, datetime

import pytest

from aquant.execution import BrokerOrderStatus, KillSwitch
from aquant.execution.order_manager import BrokerCallback, BrokerEventJournal
from aquant.monitoring import evaluate_live_readiness
from aquant.workflows import FailClosedWorkflow, StageResult

NOW = datetime(2026, 7, 17, tzinfo=UTC)


def test_duplicate_and_out_of_order_callbacks_do_not_regress_order_state() -> None:
    journal = BrokerEventJournal()
    submitted = BrokerCallback("e1", "o1", BrokerOrderStatus.SUBMITTED, NOW)
    filled = BrokerCallback("e2", "o1", BrokerOrderStatus.FILLED, NOW)
    late_partial = BrokerCallback("e3", "o1", BrokerOrderStatus.PARTIALLY_FILLED, NOW)
    assert journal.record(submitted)
    assert journal.record(filled)
    assert not journal.record(late_partial)
    assert not journal.record(filled)
    assert journal.status("o1") is BrokerOrderStatus.FILLED
    assert len(journal.callbacks) == 3


def test_daily_workflow_stops_immediately_on_data_quality_failure() -> None:
    called: list[str] = []

    def stage(name: str, passed: bool) -> StageResult:
        called.append(name)
        return StageResult(name, passed)

    workflow = FailClosedWorkflow(
        (
            ("download", lambda: stage("download", True)),
            ("quality", lambda: stage("quality", False)),
            ("publish", lambda: stage("publish", True)),
        )
    )
    result = workflow.run()
    assert not result.completed
    assert called == ["download", "quality"]


def test_live_readiness_requires_every_independent_gate() -> None:
    kill = KillSwitch()
    blocked = evaluate_live_readiness(
        configuration_allows_live=False,
        manual_confirmation=False,
        data_release_passed=True,
        account_reconciled=True,
        risk_service_healthy=True,
        kill_switch=kill.state,
        broker_connected=False,
    )
    assert not blocked.ready
    assert set(blocked.blockers) == {
        "LIVE_CONFIGURATION_DISABLED",
        "MANUAL_CONFIRMATION_MISSING",
        "BROKER_DISCONNECTED",
    }
    ready = evaluate_live_readiness(
        configuration_allows_live=True,
        manual_confirmation=True,
        data_release_passed=True,
        account_reconciled=True,
        risk_service_healthy=True,
        kill_switch=kill.state,
        broker_connected=True,
    )
    assert ready.ready


def test_workflow_rejects_stage_identity_corruption() -> None:
    with pytest.raises(ValueError, match="identity"):
        FailClosedWorkflow((("expected", lambda: StageResult("wrong", True)),)).run()
