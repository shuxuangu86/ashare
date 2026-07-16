from datetime import UTC, date, datetime

import pytest

from aquant.domain.enums import Exchange, OrderStatus, Side
from aquant.domain.identifiers import Symbol
from aquant.domain.orders import OrderIntent, validate_order_transition


def _intent(**overrides: object) -> OrderIntent:
    values: dict[str, object] = {
        "account_id": "paper-001",
        "strategy_id": "multi-factor-v1",
        "trade_date": date(2026, 7, 17),
        "symbol": Symbol("600000", Exchange.XSHG),
        "side": Side.BUY,
        "target_quantity": 1000,
        "created_at": datetime(2026, 7, 16, 8, tzinfo=UTC),
    }
    values.update(overrides)
    return OrderIntent(**values)  # type: ignore[arg-type]


def test_idempotency_key_is_stable_for_same_logical_intent() -> None:
    first = _intent()
    second = _intent(created_at=datetime(2026, 7, 16, 9, tzinfo=UTC))

    assert first.intent_id != second.intent_id
    assert first.idempotency_key == second.idempotency_key
    assert len(first.idempotency_key) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "paper-002"),
        ("strategy_id", "multi-factor-v2"),
        ("trade_date", date(2026, 7, 18)),
        ("symbol", Symbol("000001", Exchange.XSHE)),
        ("side", Side.SELL),
        ("target_quantity", 0),
    ],
)
def test_idempotency_key_changes_when_business_identity_changes(field: str, value: object) -> None:
    assert _intent().idempotency_key != _intent(**{field: value}).idempotency_key


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("account_id", " ", ValueError),
        ("strategy_id", "", ValueError),
        ("target_quantity", -1, ValueError),
        ("target_quantity", True, ValueError),
        ("target_quantity", 1.5, TypeError),
        ("created_at", datetime(2026, 7, 16, 8), ValueError),
    ],
)
def test_order_intent_rejects_invalid_values(
    field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _intent(**{field: value})


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (OrderStatus.DRAFT, OrderStatus.PENDING_APPROVAL),
        (OrderStatus.PENDING_APPROVAL, OrderStatus.APPROVED),
        (OrderStatus.APPROVED, OrderStatus.SUBMITTING),
        (OrderStatus.SUBMITTING, OrderStatus.SUBMITTED),
        (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED),
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCEL_PENDING),
        (OrderStatus.CANCEL_PENDING, OrderStatus.CANCELLED),
        (OrderStatus.UNKNOWN, OrderStatus.FILLED),
    ],
)
def test_valid_order_transitions(current: OrderStatus, target: OrderStatus) -> None:
    validate_order_transition(current, target)


def test_order_state_cannot_skip_approval() -> None:
    with pytest.raises(ValueError, match="invalid order status transition"):
        validate_order_transition(OrderStatus.DRAFT, OrderStatus.SUBMITTED)


@pytest.mark.parametrize(
    "terminal", [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
)
def test_terminal_order_states_cannot_transition(terminal: OrderStatus) -> None:
    with pytest.raises(ValueError, match="terminal"):
        validate_order_transition(terminal, OrderStatus.UNKNOWN)
