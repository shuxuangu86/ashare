from dataclasses import dataclass
from datetime import datetime

from aquant.domain.time import require_aware


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    active: bool
    reason: str | None = None
    activated_at: datetime | None = None


class KillSwitch:
    def __init__(self) -> None:
        self._state = KillSwitchState(False)

    @property
    def state(self) -> KillSwitchState:
        return self._state

    def activate(self, reason: str, *, activated_at: datetime) -> KillSwitchState:
        if not reason.strip():
            raise ValueError("kill switch reason must not be blank")
        timestamp = require_aware(activated_at, field_name="activated_at")
        if not self._state.active:
            self._state = KillSwitchState(True, reason.strip(), timestamp)
        return self._state

    def reset(self, *, manual_confirmation: bool) -> None:
        if not manual_confirmation:
            raise ValueError("kill switch reset requires manual confirmation")
        self._state = KillSwitchState(False)

    def require_inactive(self) -> None:
        if self._state.active:
            raise RuntimeError(f"kill switch is active: {self._state.reason}")
