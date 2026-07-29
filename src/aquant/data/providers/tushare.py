import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.request import Request, urlopen

import tinyshare as ts  # type: ignore[import-untyped]

from aquant.data.providers.base import DatasetRequest, ProviderResponse


@dataclass(frozen=True, slots=True)
class TushareDatasetSpec:
    api_name: str
    fields: tuple[str, ...]


DATASET_SPECS: Mapping[str, TushareDatasetSpec] = {
    "instruments": TushareDatasetSpec(
        api_name="stock_basic",
        fields=(
            "ts_code",
            "symbol",
            "name",
            "market",
            "exchange",
            "list_status",
            "list_date",
            "delist_date",
        ),
    ),
    "trading_calendar": TushareDatasetSpec(
        api_name="trade_cal",
        fields=("exchange", "cal_date", "is_open", "pretrade_date"),
    ),
}


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes
    content_type: str


class HttpTransport(Protocol):
    def post_json(
        self, url: str, payload: Mapping[str, Any], *, timeout: float
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    def post_json(self, url: str, payload: Mapping[str, Any], *, timeout: float) -> HttpResponse:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "aquant/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status_code=response.status,
                body=response.read(),
                content_type=response.headers.get_content_type(),
            )


class TinyShareHttpTransport:
    """Translate TinyShare's DataFrame response into the immutable Raw envelope."""

    def __init__(self, token: str, *, client: Any | None = None) -> None:
        self._client = client if client is not None else ts.pro_api(token)

    def post_json(self, url: str, payload: Mapping[str, Any], *, timeout: float) -> HttpResponse:
        del url, timeout
        api_name = str(payload["api_name"])
        params = dict(payload.get("params") or {})
        fields = str(payload.get("fields") or "")
        frame = self._client.query(api_name, fields=fields, **params)
        split = json.loads(frame.to_json(orient="split", force_ascii=False, date_format="iso"))
        body = json.dumps(
            {
                "code": 0,
                "msg": None,
                "data": {
                    "fields": split["columns"],
                    "items": split["data"],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return HttpResponse(status_code=200, body=body, content_type="application/json")


class TushareProvider:
    """Raw Tushare Pro HTTP adapter; normalization happens after archival."""

    def __init__(
        self,
        token: str,
        *,
        endpoint: str = "http://api.tushare.pro",
        timeout: float = 30.0,
        transport: HttpTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("Tushare token must not be blank")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not endpoint.startswith(("http://", "https://", "tinyshare://")):
            raise ValueError("data endpoint must use HTTP, HTTPS, or tinyshare")
        self._token = token.strip()
        self._endpoint = endpoint
        self._timeout = timeout
        self._transport = transport or (
            TinyShareHttpTransport(self._token)
            if endpoint.startswith("tinyshare://")
            else UrllibHttpTransport()
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def name(self) -> str:
        return "tushare"

    @property
    def version(self) -> str:
        return "pro-http-v1"

    def fetch(self, request: DatasetRequest) -> ProviderResponse:
        try:
            spec = DATASET_SPECS[request.dataset]
        except KeyError as exc:
            raise KeyError(f"Tushare adapter does not allow dataset: {request.dataset}") from exc
        payload = {
            "api_name": spec.api_name,
            "token": self._token,
            "params": request.params,
            "fields": ",".join(spec.fields),
        }
        response = self._transport.post_json(self._endpoint, payload, timeout=self._timeout)
        source_request_id, record_count = self._response_metadata(response.body, request)
        return ProviderResponse(
            provider=self.name,
            provider_version=self.version,
            request=request,
            received_at=self._clock(),
            body=response.body,
            media_type=response.content_type,
            source_request_id=source_request_id,
            record_count=record_count,
            transport_status=response.status_code,
        )

    @staticmethod
    def _response_metadata(body: bytes, request: DatasetRequest) -> tuple[str, int | None]:
        fallback = hashlib.sha256(request.params_json.encode("utf-8") + b"\0" + body).hexdigest()
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return fallback, None
        if not isinstance(payload, dict):
            return fallback, None
        request_id = payload.get("request_id")
        source_request_id = request_id.strip() if isinstance(request_id, str) else fallback
        data = payload.get("data")
        items = data.get("items") if isinstance(data, dict) else None
        record_count = len(items) if isinstance(items, list) else None
        return source_request_id or fallback, record_count
