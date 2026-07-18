import hashlib
import http.client
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from aquant.data.ingestion.raw_store import RawBatchStore
from aquant.data.providers.base import DatasetRequest, ProviderResponse
from aquant.data.providers.tushare import HttpTransport, UrllibHttpTransport


class TushareApiError(RuntimeError):
    def __init__(self, api_name: str, code: int | None, message: str) -> None:
        super().__init__(f"Tushare API {api_name} failed with code {code}: {message}")
        self.api_name = api_name
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code in {None, -1, -2, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class ArchivedPage:
    api_name: str
    params: dict[str, Any]
    row_count: int
    fields: tuple[str, ...]
    payload_path: Path
    manifest_path: Path
    resumed: bool = False


class BackfillState:
    """SQLite checkpoint store; raw files remain the immutable source of truth."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pages (
                    job_key TEXT PRIMARY KEY,
                    api_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    row_count INTEGER,
                    fields_json TEXT,
                    payload_path TEXT,
                    manifest_path TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.commit()

    def completed(self, job_key: str) -> ArchivedPage | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT api_name, params_json, row_count, fields_json, payload_path, manifest_path
                FROM pages WHERE job_key = ? AND status = 'COMPLETED'
                """,
                (job_key,),
            ).fetchone()
        if row is None:
            return None
        payload_path = Path(row[4])
        manifest_path = Path(row[5])
        if not payload_path.is_file() or not manifest_path.is_file():
            return None
        return ArchivedPage(
            api_name=row[0],
            params=json.loads(row[1]),
            row_count=row[2],
            fields=tuple(json.loads(row[3])),
            payload_path=payload_path,
            manifest_path=manifest_path,
            resumed=True,
        )

    def record_attempt(self, job_key: str, api_name: str, params_json: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO pages(job_key, api_name, params_json, status, attempts, updated_at)
                VALUES (?, ?, ?, 'RUNNING', 1, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    status = 'RUNNING', attempts = pages.attempts + 1,
                    error = NULL, updated_at = excluded.updated_at
                """,
                (job_key, api_name, params_json, now),
            )
            self._connection.commit()

    def record_success(self, job_key: str, page: ArchivedPage) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE pages SET status = 'COMPLETED', row_count = ?, fields_json = ?,
                    payload_path = ?, manifest_path = ?, error = NULL, updated_at = ?
                WHERE job_key = ?
                """,
                (
                    page.row_count,
                    json.dumps(page.fields, ensure_ascii=False),
                    str(page.payload_path.resolve()),
                    str(page.manifest_path.resolve()),
                    datetime.now(UTC).isoformat(),
                    job_key,
                ),
            )
            self._connection.commit()

    def record_failure(self, job_key: str, error: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE pages SET status = 'FAILED', error = ?, updated_at = ? WHERE job_key = ?
                """,
                (error[:2000], datetime.now(UTC).isoformat(), job_key),
            )
            self._connection.commit()

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                status: count
                for status, count in self._connection.execute(
                    "SELECT status, count(*) FROM pages GROUP BY status"
                )
            }

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class TushareBulkArchiver:
    def __init__(
        self,
        *,
        token: str,
        endpoint: str,
        raw_root: Path,
        state_path: Path,
        timeout_seconds: float = 45,
        minimum_interval_seconds: float = 0.35,
        max_attempts: int = 8,
        transport: HttpTransport | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("Tushare token must not be blank")
        if not endpoint.startswith("https://"):
            raise ValueError("bulk archive endpoint must use HTTPS")
        if timeout_seconds <= 0 or minimum_interval_seconds < 0 or max_attempts <= 0:
            raise ValueError("invalid retry or throttle configuration")
        self._token = token.strip()
        self._endpoint = endpoint
        self._store = RawBatchStore(raw_root)
        self.state = BackfillState(state_path)
        self._timeout = timeout_seconds
        self._minimum_interval = minimum_interval_seconds
        self._max_attempts = max_attempts
        self._transport = transport or UrllibHttpTransport()
        self._last_request_at = 0.0
        self._throttle_lock = threading.Lock()

    @staticmethod
    def _canonical_params(params: Mapping[str, Any]) -> str:
        return json.dumps(dict(params), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def job_key(cls, api_name: str, params: Mapping[str, Any], fields: str) -> str:
        material = f"{api_name}\0{cls._canonical_params(params)}\0{fields}".encode()
        return hashlib.sha256(material).hexdigest()

    def archive_page(
        self,
        api_name: str,
        params: Mapping[str, Any],
        *,
        fields: str = "",
    ) -> ArchivedPage:
        canonical_params = self._canonical_params(params)
        key = self.job_key(api_name, params, fields)
        completed = self.state.completed(key)
        if completed is not None:
            return completed

        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            self.state.record_attempt(key, api_name, canonical_params)
            try:
                self._throttle()
                page = self._request_and_archive(api_name, dict(params), fields)
                self.state.record_success(key, page)
                return page
            except (
                HTTPError,
                URLError,
                TimeoutError,
                TushareApiError,
                http.client.HTTPException,
                OSError,
            ) as exc:
                last_error = exc
                self.state.record_failure(key, f"{type(exc).__name__}: {exc}")
                if isinstance(exc, TushareApiError) and not exc.retryable:
                    raise
                if attempt < self._max_attempts:
                    time.sleep(min(90.0, 2.0 ** (attempt - 1)))
        assert last_error is not None
        raise RuntimeError(
            f"Tushare page failed after {self._max_attempts} attempts: {api_name}"
        ) from last_error

    def archive_paginated(
        self,
        api_name: str,
        params: Mapping[str, Any],
        *,
        fields: str = "",
        page_size: int = 6000,
    ) -> Iterator[ArchivedPage]:
        if page_size <= 0 or page_size > 6000:
            raise ValueError("page_size must be between 1 and 6000")
        offset = 0
        seen_page_signatures: set[str] = set()
        while True:
            page_params = {**params, "limit": page_size, "offset": offset}
            page = self.archive_page(api_name, page_params, fields=fields)
            signature = self._items_signature(page)
            if page.row_count and signature in seen_page_signatures:
                raise TushareApiError(
                    api_name,
                    None,
                    "provider repeated a page; refusing an incomplete/infinite pagination loop",
                )
            seen_page_signatures.add(signature)
            yield page
            if page.row_count < page_size:
                break
            offset += page_size

    def read_items(self, page: ArchivedPage) -> list[dict[str, Any]]:
        payload = json.loads(page.payload_path.read_bytes())
        data = payload.get("data") or {}
        fields = data.get("fields") or []
        return [dict(zip(fields, row, strict=True)) for row in data.get("items") or []]

    @staticmethod
    def _items_signature(page: ArchivedPage) -> str:
        payload = json.loads(page.payload_path.read_bytes())
        items = (payload.get("data") or {}).get("items") or []
        body = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(body).hexdigest()

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._minimum_interval:
                time.sleep(self._minimum_interval - elapsed)
            self._last_request_at = time.monotonic()

    def _request_and_archive(
        self, api_name: str, params: dict[str, Any], fields: str
    ) -> ArchivedPage:
        requested_at = datetime.now(UTC)
        request = DatasetRequest.create(
            dataset=api_name,
            params=params,
            requested_at=requested_at,
        )
        payload: dict[str, Any] = {
            "api_name": api_name,
            "token": self._token,
            "params": params,
        }
        if fields:
            payload["fields"] = fields
        http_response = self._transport.post_json(self._endpoint, payload, timeout=self._timeout)
        received_at = datetime.now(UTC)
        try:
            candidate = json.loads(http_response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            candidate = None
        decoded = candidate if isinstance(candidate, dict) else {}
        data = decoded.get("data") or {}
        items = data.get("items") or []
        response_fields = data.get("fields") or []
        valid_envelope = (
            isinstance(candidate, dict)
            and isinstance(data, dict)
            and isinstance(items, list)
            and isinstance(response_fields, list)
        )
        provider_response = ProviderResponse(
            provider="tushare",
            provider_version="proxy-http-v1",
            request=request,
            received_at=received_at,
            body=http_response.body,
            media_type=http_response.content_type,
            source_request_id=hashlib.sha256(http_response.body).hexdigest(),
            record_count=len(items) if valid_envelope else None,
            transport_status=http_response.status_code,
        )
        result = self._store.write_response(provider_response, payload_file="response.json")
        if not valid_envelope:
            raise TushareApiError(api_name, None, "malformed data envelope")
        code = decoded.get("code")
        if code != 0:
            raise TushareApiError(api_name, code, str(decoded.get("msg") or "unknown error"))
        return ArchivedPage(
            api_name=api_name,
            params=params,
            row_count=len(items),
            fields=tuple(str(field) for field in response_fields),
            payload_path=result.payload_path,
            manifest_path=result.manifest_path,
        )
