from enum import StrEnum


class RunMode(StrEnum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class LiveTier(StrEnum):
    READ_ONLY = "READ_ONLY"
    SMALL_CAPITAL = "SMALL_CAPITAL"
    FULL = "FULL"


class Exchange(StrEnum):
    XSHG = "XSHG"
    XSHE = "XSHE"


class SecurityType(StrEnum):
    STOCK = "STOCK"


class Board(StrEnum):
    MAIN = "MAIN"
    STAR = "STAR"
    CHINEXT = "CHINEXT"
    OTHER = "OTHER"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
