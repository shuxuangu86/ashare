from datetime import date
from types import SimpleNamespace

from scripts.build_nested_root_cause_eligibility import (
    _eligibility_flags,
    _exclusion_reason,
    _ts_code,
)


def _item(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "symbol": SimpleNamespace(canonical="600000.XSHG"),
        "suspended": False,
        "is_st": False,
        "is_delisting_risk": False,
        "list_date": date(2020, 1, 1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ts_code_maps_repository_symbols() -> None:
    assert _ts_code("600000.XSHG") == "600000.SH"
    assert _ts_code("000001.XSHE") == "000001.SZ"


def test_exclusion_reason_matches_production_selector_order() -> None:
    trade_date = date(2021, 1, 1)

    assert _exclusion_reason(_item(), trade_date) is None
    assert _exclusion_reason(_item(suspended=True), trade_date) == "suspended"
    assert _exclusion_reason(_item(is_st=True), trade_date) == "st"
    assert _exclusion_reason(_item(list_date=date(2020, 12, 1)), trade_date) == "new_listing"


def test_eligibility_flags_encode_independent_controls() -> None:
    trade_date = date(2021, 1, 1)

    assert _eligibility_flags(_item(), trade_date) == 31
    assert _eligibility_flags(_item(is_st=True), trade_date) == 27
    assert _eligibility_flags(_item(suspended=True), trade_date) == 23
