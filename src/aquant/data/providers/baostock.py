import hashlib
import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast

from aquant.data.providers.base import DatasetRequest, ProviderResponse

_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus,isST"


class BaoStockResult(Protocol):
    error_code: str
    error_msg: str
    fields: list[str]

    def next(self) -> bool: ...

    def get_row_data(self) -> list[str]: ...


class BaoStockLoginResult(Protocol):
    error_code: str
    error_msg: str


class BaoStockBackend(Protocol):
    def login(self) -> BaoStockLoginResult: ...

    def logout(self) -> object: ...

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ) -> BaoStockResult: ...


class BaoStockProvider:
    """BaoStock validation-source adapter using unadjusted daily data."""

    def __init__(
        self,
        *,
        backend: BaoStockBackend | None = None,
        backend_version: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if backend is None:
            module = importlib.import_module("baostock")
            backend = cast(BaoStockBackend, module)
            backend_version = str(getattr(module, "__version__", "unknown"))
        self._backend = backend
        self._backend_version = (backend_version or "injected").strip()
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def name(self) -> str:
        return "baostock"

    @property
    def version(self) -> str:
        return f"baostock-{self._backend_version}"

    def fetch(self, request: DatasetRequest) -> ProviderResponse:
        if request.dataset != "daily_bars":
            raise KeyError(f"BaoStock adapter does not allow dataset: {request.dataset}")
        params = request.params
        login = self._backend.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
        try:
            result = self._backend.query_history_k_data_plus(
                self._provider_symbol(self._required_param(params, "symbol")),
                _FIELDS,
                start_date=self._iso_date(self._required_param(params, "start_date")),
                end_date=self._iso_date(self._required_param(params, "end_date")),
                frequency="d",
                adjustflag="3",
            )
            if result.error_code != "0":
                raise RuntimeError(f"BaoStock query failed: {result.error_code} {result.error_msg}")
            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())
            body = json.dumps(
                {"fields": result.fields, "items": rows},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        finally:
            self._backend.logout()

        source_request_id = hashlib.sha256(request.params_json.encode() + b"\0" + body).hexdigest()
        return ProviderResponse(
            provider=self.name,
            provider_version=self.version,
            request=request,
            received_at=self._clock(),
            body=body,
            media_type="application/vnd.aquant.tabular+json",
            source_request_id=source_request_id,
            record_count=len(rows),
        )

    @staticmethod
    def _required_param(params: dict[str, object], name: str) -> str:
        value = params.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"BaoStock request requires string parameter: {name}")
        return value.strip()

    @staticmethod
    def _provider_symbol(canonical: str) -> str:
        try:
            code, exchange = canonical.upper().split(".", maxsplit=1)
        except ValueError as exc:
            raise ValueError("symbol must use CODE.EXCHANGE") from exc
        venue = {"XSHG": "sh", "XSHE": "sz"}.get(exchange)
        if venue is None or len(code) != 6 or not code.isdigit():
            raise ValueError(f"unsupported A-share symbol: {canonical}")
        return f"{venue}.{code}"

    @staticmethod
    def _iso_date(value: str) -> str:
        compact = value.replace("-", "")
        if len(compact) != 8 or not compact.isdigit():
            raise ValueError(f"invalid date: {value}")
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
