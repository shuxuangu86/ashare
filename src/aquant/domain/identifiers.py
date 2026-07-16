import re
from dataclasses import dataclass

from aquant.domain.enums import Exchange

_A_SHARE_CODE = re.compile(r"^[0-9]{6}$")


@dataclass(frozen=True, slots=True, order=True)
class Symbol:
    """Canonical A-share identifier independent of any data vendor."""

    code: str
    exchange: Exchange

    def __post_init__(self) -> None:
        if not _A_SHARE_CODE.fullmatch(self.code):
            raise ValueError("symbol code must contain exactly six digits")

    @property
    def canonical(self) -> str:
        return f"{self.code}.{self.exchange.value}"

    @classmethod
    def parse(cls, value: str) -> "Symbol":
        try:
            code, exchange = value.strip().upper().split(".", maxsplit=1)
        except ValueError as exc:
            raise ValueError("symbol must use the canonical CODE.EXCHANGE form") from exc
        return cls(code=code, exchange=Exchange(exchange))

    def __str__(self) -> str:
        return self.canonical
