import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from aquant.data.ingestion import RawBatchStore
from aquant.data.providers import DatasetRequest, ProviderResponse

BATCH_ID = UUID("12345678-1234-5678-1234-567812345678")


def _response(**overrides: object) -> ProviderResponse:
    request = DatasetRequest.create(
        dataset="daily",
        params={"trade_date": "2026-07-16"},
        requested_at=datetime(2026, 7, 16, 8, tzinfo=UTC),
    )
    values: dict[str, object] = {
        "provider": "tushare_fixture",
        "provider_version": "1",
        "request": request,
        "received_at": datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC),
        "body": b"original-provider-response",
        "media_type": "application/octet-stream",
        "source_request_id": "provider-request-001",
        "record_count": 2,
    }
    values.update(overrides)
    return ProviderResponse(**values)  # type: ignore[arg-type]


def test_raw_batch_store_writes_partitioned_immutable_batch(tmp_path: Path) -> None:
    result = RawBatchStore(tmp_path).write_response(_response(), batch_id=BATCH_ID)

    assert result.directory == (
        tmp_path
        / "provider=tushare_fixture"
        / "dataset=daily"
        / "ingest_date=2026-07-16"
        / f"batch_id={BATCH_ID}"
    )
    assert result.payload_path.read_bytes() == b"original-provider-response"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "aquant.raw-batch.v1"
    assert manifest["request_params"] == {"trade_date": "2026-07-16"}
    assert manifest["sha256"] == hashlib.sha256(b"original-provider-response").hexdigest()
    assert manifest["byte_count"] == len(b"original-provider-response")
    assert manifest["record_count"] == 2
    assert not list(result.directory.parent.glob(".*.tmp-*"))


def test_raw_batch_store_never_overwrites_existing_batch(tmp_path: Path) -> None:
    store = RawBatchStore(tmp_path)
    first = store.write_response(_response(), batch_id=BATCH_ID)

    with pytest.raises(FileExistsError, match="already exists"):
        store.write_response(_response(body=b"revised"), batch_id=BATCH_ID)

    assert first.payload_path.read_bytes() == b"original-provider-response"


@pytest.mark.parametrize(
    ("field", "value"),
    [("provider", "../escape"), ("provider", "UPPER/unsafe")],
)
def test_raw_batch_store_rejects_unsafe_partitions(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        RawBatchStore(tmp_path).write_response(_response(**{field: value}))


@pytest.mark.parametrize("payload_file", ["../response.bin", "", "nested/file.json"])
def test_raw_batch_store_rejects_unsafe_payload_name(tmp_path: Path, payload_file: str) -> None:
    with pytest.raises(ValueError, match="plain filename"):
        RawBatchStore(tmp_path).write_response(_response(), payload_file=payload_file)
