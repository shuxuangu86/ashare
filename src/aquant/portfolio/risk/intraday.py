from datetime import UTC, datetime, timedelta
from decimal import Decimal

from aquant.execution.kill_switch import KillSwitch


class IntradayRiskMonitor:
    def __init__(
        self,
        kill_switch: KillSwitch,
        *,
        heartbeat_timeout: timedelta = timedelta(seconds=60),
        maximum_drawdown: Decimal = Decimal("0.05"),
    ) -> None:
        self._kill_switch = kill_switch
        self._heartbeat_timeout = heartbeat_timeout
        self._maximum_drawdown = maximum_drawdown

    def evaluate(
        self,
        *,
        now: datetime,
        last_heartbeat: datetime,
        opening_equity: Decimal,
        current_equity: Decimal,
        unknown_order_count: int,
    ) -> None:
        reasons: list[str] = []
        if now - last_heartbeat > self._heartbeat_timeout:
            reasons.append("BROKER_HEARTBEAT_STALE")
        if opening_equity > 0 and (opening_equity - current_equity) / opening_equity > (
            self._maximum_drawdown
        ):
            reasons.append("INTRADAY_DRAWDOWN")
        if unknown_order_count > 0:
            reasons.append("UNKNOWN_ORDER_STATE")
        if reasons:
            self._kill_switch.activate(",".join(reasons), activated_at=now.astimezone(UTC))
