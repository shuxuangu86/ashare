import json
from datetime import UTC, datetime
from email.message import Message
from typing import Any
from urllib.request import Request

import pytest

from aquant.data.providers import DatasetRequest, MarketDataProvider, TushareProvider
from aquant.data.providers.tushare import (
    HttpResponse,
    UrllibHttpTransport,
)


class RecordingTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> HttpResponse:
        self.calls.append((url, payload, timeout))
        return self.response


class StubUrlResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = Message()
        self.headers["Content-Type"] = "application/json; charset=utf-8"

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "StubUrlResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _request(dataset: str = "instruments") -> DatasetRequest:
    return DatasetRequest.create(
        dataset=dataset,
        params={"exchange": "", "list_status": "L"},
        requested_at=datetime(2026, 7, 16, 8, tzinfo=UTC),
    )


def _response(body: object) -> HttpResponse:
    return HttpResponse(
        status_code=200,
        body=json.dumps(body).encode(),
        content_type="application/json",
    )


def test_tushare_provider_uses_official_http_envelope() -> None:
    transport = RecordingTransport(
        _response(
            {
                "request_id": "request-123",
                "code": 0,
                "msg": None,
                "data": {"fields": ["symbol"], "items": [["600000"]]},
            }
        )
    )
    provider = TushareProvider(
        "secret-token",
        transport=transport,
        clock=lambda: datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC),
    )

    response = provider.fetch(_request())

    assert isinstance(provider, MarketDataProvider)
    assert response.source_request_id == "request-123"
    assert response.record_count == 1
    assert response.transport_status == 200
    url, payload, timeout = transport.calls[0]
    assert url == "http://api.tushare.pro"
    assert payload["api_name"] == "stock_basic"
    assert payload["token"] == "secret-token"
    assert payload["params"] == {"exchange": "", "list_status": "L"}
    assert payload["fields"].startswith("ts_code,symbol,name")
    assert timeout == 30.0


def test_urllib_transport_sends_json_post(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> StubUrlResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return StubUrlResponse(b'{"code":0,"data":{"fields":[],"items":[]}}')

    monkeypatch.setattr("aquant.data.providers.tushare.urlopen", fake_urlopen)

    response = UrllibHttpTransport().post_json(
        "http://api.tushare.pro",
        {"api_name": "trade_cal", "token": "secret"},
        timeout=12.5,
    )

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json; charset=utf-8"
    assert json.loads(request.data or b"{}") == {"api_name": "trade_cal", "token": "secret"}
    assert captured["timeout"] == 12.5
    assert response.status_code == 200
    assert response.content_type == "application/json"


def test_tushare_provider_preserves_error_response_for_raw_archival() -> None:
    body = {"request_id": "denied-1", "code": 2002, "msg": "permission denied", "data": None}
    transport = RecordingTransport(_response(body))
    provider = TushareProvider(
        "secret-token",
        transport=transport,
        clock=lambda: datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC),
    )

    response = provider.fetch(_request())

    assert json.loads(response.body) == body
    assert response.source_request_id == "denied-1"
    assert response.record_count is None


@pytest.mark.parametrize("body", [b"not-json", b"[]", b'{"code":0}'])
def test_tushare_provider_builds_fallback_request_id(body: bytes) -> None:
    transport = RecordingTransport(HttpResponse(200, body, "application/json"))
    provider = TushareProvider(
        "secret-token",
        transport=transport,
        clock=lambda: datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC),
    )

    response = provider.fetch(_request())

    assert len(response.source_request_id) == 64
    assert response.record_count is None


def test_tushare_provider_restricts_dataset_allowlist() -> None:
    provider = TushareProvider("secret-token")
    with pytest.raises(KeyError, match="does not allow"):
        provider.fetch(_request("arbitrary_api"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token": ""},
        {"token": "token", "timeout": 0},
        {"token": "token", "endpoint": "file:///tmp/secret"},
    ],
)
def test_tushare_provider_rejects_unsafe_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TushareProvider(**kwargs)  # type: ignore[arg-type]


def test_provider_response_rejects_invalid_http_status() -> None:
    transport = RecordingTransport(HttpResponse(99, b"{}", "application/json"))
    provider = TushareProvider(
        "secret-token",
        transport=transport,
        clock=lambda: datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="HTTP status"):
        provider.fetch(_request())
