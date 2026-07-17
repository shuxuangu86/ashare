from dataclasses import dataclass

from aquant.execution.kill_switch import KillSwitchState


@dataclass(frozen=True, slots=True)
class LiveReadiness:
    ready: bool
    blockers: tuple[str, ...]


def evaluate_live_readiness(
    *,
    configuration_allows_live: bool,
    manual_confirmation: bool,
    data_release_passed: bool,
    account_reconciled: bool,
    risk_service_healthy: bool,
    kill_switch: KillSwitchState,
    broker_connected: bool,
) -> LiveReadiness:
    checks = {
        "LIVE_CONFIGURATION_DISABLED": configuration_allows_live,
        "MANUAL_CONFIRMATION_MISSING": manual_confirmation,
        "DATA_RELEASE_FAILED": data_release_passed,
        "ACCOUNT_NOT_RECONCILED": account_reconciled,
        "RISK_SERVICE_UNHEALTHY": risk_service_healthy,
        "KILL_SWITCH_ACTIVE": not kill_switch.active,
        "BROKER_DISCONNECTED": broker_connected,
    }
    blockers = tuple(name for name, passed in checks.items() if not passed)
    return LiveReadiness(not blockers, blockers)
