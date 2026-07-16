import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aquant.data.normalization import BaoStockPayloadError, normalize_baostock_daily_bars
from aquant.data.providers import BaoStockProvider, DatasetRequest, MarketDataProvider

FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "tradestatus",
    "isST",
]
ROW = [
    "2026-07-16",
    "sh.600000",
    "10.10",
    "10.50",
    "10.00",
    "10.30",
    "10.00",
    "123400",
    "1269000.25",
    "3",
    "1",
    "0",
]


class FakeResult:
    error_code = "0"
    error_msg = ""
    fields = FIELDS

    def __init__(self) -> None:
        self._position = 0

    def next(self) -> bool:
        self._position += 1
        return self._position == 1

    def get_row_data(self) -> list[str]:
        return ROW


class FakeBackend:
    def __init__(self, *, login_code: str = "0", query_code: str = "0") -> None:
        self.login_code = login_code
        self.query_code = query_code
        self.calls: list[tuple[object, ...]] = []
        self.logged_out = False

    def login(self) -> SimpleNamespace:
        return SimpleNamespace(error_code=self.login_code, error_msg="login")

    def logout(self) -> None:
        self.logged_out = True

    def query_history_k_data_plus(self, code: str, fields: str, **kwargs: str) -> FakeResult:
        self.calls.append((code, fields, kwargs))
        result = FakeResult()
        result.error_code = self.query_code
        result.error_msg = "query"
        return result


def _request(*, dataset: str = "daily_bars", symbol: str = "600000.XSHG") -> DatasetRequest:
    return DatasetRequest.create(
        dataset=dataset,
        params={"symbol": symbol, "start_date": "20260701", "end_date": "2026-07-16"},
        requested_at=datetime(2026, 7, 16, 8, tzinfo=UTC),
    )


def test_fetches_unadjusted_daily_bars_and_normalizes() -> None:
    backend = FakeBackend()
    request = _request()
    provider = BaoStockProvider(
        backend=backend,
        backend_version="0.9.3",
        clock=lambda: request.requested_at + timedelta(seconds=1),
    )

    response = provider.fetch(request)
    (bar,) = normalize_baostock_daily_bars(response.body)

    assert isinstance(provider, MarketDataProvider)
    assert response.record_count == 1
    assert bar.symbol.canonical == "600000.XSHG"
    assert str(bar.volume) == "123400"
    assert backend.calls[0][0] == "sh.600000"
    assert backend.calls[0][2] == {
        "start_date": "2026-07-01",
        "end_date": "2026-07-16",
        "frequency": "d",
        "adjustflag": "3",
    }
    assert backend.logged_out


def test_supports_shenzhen_symbol() -> None:
    backend = FakeBackend()
    request = _request(symbol="000001.XSHE")
    provider = BaoStockProvider(backend=backend, clock=lambda: request.requested_at)
    provider.fetch(request)
    assert backend.calls[0][0] == "sz.000001"


@pytest.mark.parametrize(
    ("backend", "message"),
    [(FakeBackend(login_code="1"), "login failed"), (FakeBackend(query_code="1"), "query failed")],
)
def test_provider_errors_are_fail_closed(backend: FakeBackend, message: str) -> None:
    request = _request()
    provider = BaoStockProvider(backend=backend, clock=lambda: request.requested_at)
    with pytest.raises(RuntimeError, match=message):
        provider.fetch(request)


@pytest.mark.parametrize(
    "dataset_request",
    [
        _request(dataset="financials"),
        _request(symbol="600000.UNKNOWN"),
        _request(symbol="bad"),
    ],
)
def test_rejects_unsupported_requests(dataset_request: DatasetRequest) -> None:
    provider = BaoStockProvider(backend=FakeBackend(), clock=lambda: dataset_request.requested_at)
    with pytest.raises((KeyError, ValueError)):
        provider.fetch(dataset_request)


@pytest.mark.parametrize(
    "payload",
    [
        b"bad-json",
        b"[]",
        json.dumps({"fields": ["date"], "items": []}).encode(),
        json.dumps({"fields": FIELDS, "items": [["short"]]}).encode(),
        json.dumps({"fields": FIELDS, "items": [[*ROW[:1], "xx.600000", *ROW[2:]]]}).encode(),
    ],
)
def test_normalizer_rejects_malformed_payload(payload: bytes) -> None:
    with pytest.raises((BaoStockPayloadError, ValueError)):
        normalize_baostock_daily_bars(payload)
