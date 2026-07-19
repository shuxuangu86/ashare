import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from aquant.data.ingestion import TushareApiError, TushareBulkArchiver
from aquant.data.providers.tushare import HttpResponse


def _response(rows: list[list[object]], *, code: int = 0, msg: str | None = None) -> HttpResponse:
    body = json.dumps(
        {
            "code": code,
            "msg": msg,
            "data": {"fields": ["ts_code", "value"], "items": rows},
        }
    ).encode()
    return HttpResponse(200, body, "application/json")


class SequenceTransport:
    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> HttpResponse:
        with self._lock:
            self.calls.append((url, payload))
            response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _archiver(
    tmp_path: Path, transport: SequenceTransport, *, attempts: int = 2
) -> TushareBulkArchiver:
    return TushareBulkArchiver(
        token="test-token",
        endpoint="https://example.test/",
        raw_root=tmp_path / "raw",
        state_path=tmp_path / "state.sqlite3",
        minimum_interval_seconds=0,
        max_attempts=attempts,
        transport=transport,
    )


def test_archive_is_immutable_and_resumes_without_second_request(tmp_path: Path) -> None:
    transport = SequenceTransport([_response([["000001.SZ", 1]])])
    archiver = _archiver(tmp_path, transport)

    first = archiver.archive_page("daily", {"trade_date": "20260715"})
    second = archiver.archive_page("daily", {"trade_date": "20260715"})

    assert first.payload_path.read_bytes() == _response([["000001.SZ", 1]]).body
    assert first.manifest_path.is_file()
    assert second.resumed is True
    assert len(transport.calls) == 1
    assert "test-token" not in first.manifest_path.read_text(encoding="utf-8")
    archiver.state.close()


def test_pagination_uses_stable_offsets_until_short_page(tmp_path: Path) -> None:
    transport = SequenceTransport(
        [
            _response([["A", 1], ["B", 2]]),
            _response([["C", 3]]),
        ]
    )
    archiver = _archiver(tmp_path, transport)

    pages = list(archiver.archive_paginated("daily", {}, page_size=2))

    assert [page.row_count for page in pages] == [2, 1]
    assert [call[1]["params"]["offset"] for call in transport.calls] == [0, 2]
    archiver.state.close()


@pytest.mark.parametrize(
    "failure",
    [
        URLError("temporary"),
        http.client.IncompleteRead(b"partial response", 100),
        http.client.RemoteDisconnected("remote closed connection"),
    ],
)
def test_retryable_transport_failure_is_checkpointed_then_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr("aquant.data.ingestion.tushare_bulk.time.sleep", lambda _: None)
    transport = SequenceTransport([failure, _response([])])
    archiver = _archiver(tmp_path, transport)

    page = archiver.archive_page("trade_cal", {})

    assert page.row_count == 0
    assert len(transport.calls) == 2
    assert archiver.state.counts() == {"COMPLETED": 1}
    archiver.state.close()


def test_rate_limit_response_uses_long_backoff_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("aquant.data.ingestion.tushare_bulk.time.sleep", sleeps.append)
    transport = SequenceTransport(
        [
            _response([], code=429, msg="request rate too high"),
            _response([]),
        ]
    )
    archiver = _archiver(tmp_path, transport)

    page = archiver.archive_page("suspend_d", {"trade_date": "20260718"})

    assert page.row_count == 0
    assert sleeps == [15.0]
    assert archiver.state.counts() == {"COMPLETED": 1}
    archiver.state.close()


def test_nonretryable_api_error_is_archived_but_not_marked_complete(tmp_path: Path) -> None:
    transport = SequenceTransport([_response([], code=-2001, msg="permission denied")])
    archiver = _archiver(tmp_path, transport)

    with pytest.raises(TushareApiError, match="permission denied"):
        archiver.archive_page("premium_api", {})

    assert len(list((tmp_path / "raw").rglob("response.json"))) == 1
    assert archiver.state.counts() == {"FAILED": 1}
    archiver.state.close()


def test_bulk_archiver_rejects_unsafe_or_invalid_configuration(tmp_path: Path) -> None:
    transport = SequenceTransport([])

    with pytest.raises(ValueError, match="HTTPS"):
        TushareBulkArchiver(
            token="token",
            endpoint="http://example.test",
            raw_root=tmp_path,
            state_path=tmp_path / "state.sqlite3",
            transport=transport,
        )
    with pytest.raises(ValueError, match="retry or throttle"):
        TushareBulkArchiver(
            token="token",
            endpoint="https://example.test",
            raw_root=tmp_path,
            state_path=tmp_path / "state.sqlite3",
            max_attempts=0,
            transport=transport,
        )


def test_page_size_guard_rejects_values_above_provider_limit(tmp_path: Path) -> None:
    archiver = _archiver(tmp_path, SequenceTransport([]))

    with pytest.raises(ValueError, match="between 1 and 6000"):
        list(archiver.archive_paginated("daily", {}, page_size=6001))
    archiver.state.close()


def test_checkpoint_store_supports_concurrent_unique_pages(tmp_path: Path) -> None:
    page_count = 24
    transport = SequenceTransport(
        [_response([[f"S{number}", number]]) for number in range(page_count)]
    )
    archiver = _archiver(tmp_path, transport)

    with ThreadPoolExecutor(max_workers=8) as executor:
        pages = list(
            executor.map(
                lambda number: archiver.archive_page("daily", {"trade_date": f"D{number}"}),
                range(page_count),
            )
        )

    assert len(pages) == page_count
    assert archiver.state.counts() == {"COMPLETED": page_count}
    assert len(list((tmp_path / "raw").rglob("response.json"))) == page_count
    archiver.state.close()
