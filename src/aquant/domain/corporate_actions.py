from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from aquant.domain.identifiers import Symbol
from aquant.domain.time import require_aware


class CorporateActionKind(StrEnum):
    CASH_DIVIDEND_ENTITLEMENT = "CASH_DIVIDEND_ENTITLEMENT"
    CASH_DIVIDEND_PAYMENT = "CASH_DIVIDEND_PAYMENT"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    CAPITALIZATION = "CAPITALIZATION"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    DELISTING = "DELISTING"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    action_id: UUID
    symbol: Symbol
    kind: CorporateActionKind
    occurred_at: datetime
    ratio: Decimal = Decimal("0")
    cash_per_share: Decimal = Decimal("0")
    subscription_price: Decimal = Decimal("0")
    settlement_price: Decimal = Decimal("0")
    cash_in_lieu_price: Decimal | None = None
    reference_action_id: UUID | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurred_at",
            require_aware(self.occurred_at, field_name="occurred_at"),
        )
        for name in (
            "ratio",
            "cash_per_share",
            "subscription_price",
            "settlement_price",
        ):
            value = Decimal(getattr(self, name))
            if not value.is_finite() or value < 0:
                raise ValueError(f"corporate action {name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.cash_in_lieu_price is not None:
            cash_in_lieu = Decimal(self.cash_in_lieu_price)
            if not cash_in_lieu.is_finite() or cash_in_lieu < 0:
                raise ValueError("cash-in-lieu price must be finite and non-negative")
            object.__setattr__(self, "cash_in_lieu_price", cash_in_lieu)
        self._validate_kind_fields()

    def _validate_kind_fields(self) -> None:
        if self.kind is CorporateActionKind.CASH_DIVIDEND_ENTITLEMENT:
            if self.cash_per_share <= 0:
                raise ValueError("cash-dividend entitlement requires cash per share")
        elif self.kind is CorporateActionKind.CASH_DIVIDEND_PAYMENT:
            if self.reference_action_id is None:
                raise ValueError("cash-dividend payment requires entitlement reference")
        elif self.kind in {
            CorporateActionKind.STOCK_DIVIDEND,
            CorporateActionKind.CAPITALIZATION,
            CorporateActionKind.RIGHTS_ISSUE,
        }:
            if self.ratio <= 0:
                raise ValueError("share action requires a positive ratio")
        elif self.kind in {CorporateActionKind.SPLIT, CorporateActionKind.REVERSE_SPLIT}:
            if self.ratio <= 0:
                raise ValueError("split action requires a positive ratio")
        elif self.kind is CorporateActionKind.DELISTING and self.settlement_price < 0:
            raise ValueError("delisting settlement price cannot be negative")
