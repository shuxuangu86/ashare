import hashlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from aquant.data.providers.base import DatasetRequest, ProviderResponse


class LocalFileProvider:
    """Deterministic offline provider used for integration and golden tests."""

    def __init__(
        self,
        fixture_paths: Mapping[str, Path],
        *,
        name: str = "local_fixture",
        version: str = "1",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._fixture_paths = {
            dataset.strip().lower(): path for dataset, path in fixture_paths.items()
        }
        self._name = name.strip().lower()
        self._version = version.strip()
        self._clock = clock or (lambda: datetime.now(UTC))
        if not self._name or not self._version:
            raise ValueError("provider name and version must not be blank")

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def fetch(self, request: DatasetRequest) -> ProviderResponse:
        try:
            path = self._fixture_paths[request.dataset]
        except KeyError as exc:
            raise KeyError(f"fixture provider does not contain dataset: {request.dataset}") from exc
        body = path.read_bytes()
        digest_input = b"\0".join(
            [
                self.name.encode(),
                self.version.encode(),
                request.dataset.encode(),
                request.params_json.encode(),
                body,
            ]
        )
        return ProviderResponse(
            provider=self.name,
            provider_version=self.version,
            request=request,
            received_at=self._clock(),
            body=body,
            media_type="application/json",
            source_request_id=hashlib.sha256(digest_input).hexdigest(),
        )
