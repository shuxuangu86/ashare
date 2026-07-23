import pytest

from aquant.domain.enums import Exchange
from aquant.domain.identifiers import Symbol


def test_symbol_round_trip() -> None:
    symbol = Symbol.parse("000001.xshe")

    assert symbol == Symbol(code="000001", exchange=Exchange.XSHE)
    assert symbol.canonical == "000001.XSHE"
    assert str(symbol) == symbol.canonical


def test_symbol_supports_beijing_stock_exchange() -> None:
    symbol = Symbol.parse("920001.XBSE")

    assert symbol == Symbol(code="920001", exchange=Exchange.XBSE)
    assert symbol.canonical == "920001.XBSE"


@pytest.mark.parametrize("value", ["1.XSHE", "000001", "ABC001.XSHG", "000001.XNAS"])
def test_symbol_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError):
        Symbol.parse(value)


def test_symbol_constructor_rejects_bad_code() -> None:
    with pytest.raises(ValueError, match="six digits"):
        Symbol(code="60000", exchange=Exchange.XSHG)
