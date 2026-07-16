import hashlib
import importlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from aquant.data.providers.base import DatasetRequest, ProviderResponse


class TabularFrame(Protocol):
    def __len__(self) -> int: ...

    def to_json(
        self,
        *,
        orient: str,
        date_format: str,
        force_ascii: bool,
    ) -> str: ...


class AkshareBackend(Protocol):
    def stock_zh_a_hist(
        self,
        *,
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
        adjust: str,
        timeout: float,
    ) -> TabularFrame: ...


class AkshareProvider:
    """AKShare library adapter that archives the exact returned table before normalization."""

    def __init__(
        self,
        *,
        backend: AkshareBackend | None = None,
        backend_version: str | None = None,
        timeout: float = 30.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if backend is None:
            module = importlib.import_module("akshare")
            backend = module
            backend_version = str(getattr(module, "__version__", "unknown"))
        self._backend = backend
        self._backend_version = (backend_version or "injected").strip()
        self._timeout = timeout
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def name(self) -> str:
        return "akshare"

    @property
    def version(self) -> str:
        return f"akshare-{self._backend_version}"

    def fetch(self, request: DatasetRequest) -> ProviderResponse:
        if request.dataset != "daily_bars":
            raise KeyError(f"AKShare adapter does not allow dataset: {request.dataset}")
        params = request.params
        canonical_symbol = self._required_param(params, "symbol")
        code = canonical_symbol.split(".", maxsplit=1)[0]
        frame = self._backend.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=self._required_param(params, "start_date"),
            end_date=self._required_param(params, "end_date"),
            adjust="",
            timeout=self._timeout,
        )
        body = frame.to_json(
            orient="table",
            date_format="iso",
            force_ascii=False,
        ).encode("utf-8")
        source_request_id = hashlib.sha256(
            request.params_json.encode("utf-8") + b"\0" + body
        ).hexdigest()
        return ProviderResponse(
            provider=self.name,
            provider_version=self.version,
            request=request,
            received_at=self._clock(),
            body=body,
            media_type="application/vnd.aquant.tabular+json",
            source_request_id=source_request_id,
            record_count=len(frame),
        )

    @staticmethod
    def _required_param(params: dict[str, object], name: str) -> str:
        value = params.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AKShare request requires string parameter: {name}")
        return value.strip()
