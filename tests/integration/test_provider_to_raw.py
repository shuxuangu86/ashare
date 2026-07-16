import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from aquant.data.ingestion import RawBatchStore
from aquant.data.providers import DatasetRequest, LocalFileProvider

PROJECT_ROOT = Path(__file__).parents[2]


def test_local_provider_response_is_archived_without_transformation(tmp_path: Path) -> None:
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "providers" / "demo" / "instruments.json"
    requested_at = datetime(2026, 7, 16, 8, tzinfo=UTC)
    received_at = datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC)
    provider = LocalFileProvider(
        {"instruments": fixture},
        clock=lambda: received_at,
    )
    request = DatasetRequest.create(
        dataset="instruments",
        params={"asof": "2026-07-16"},
        requested_at=requested_at,
    )

    response = provider.fetch(request)
    archived = RawBatchStore(tmp_path).write_response(
        response,
        batch_id=UUID("11111111-1111-1111-1111-111111111111"),
        payload_file="response.json",
    )

    assert archived.payload_path.read_bytes() == fixture.read_bytes()
    assert len(json.loads(archived.payload_path.read_text(encoding="utf-8"))) == 2
    assert archived.manifest.source_request_id == response.source_request_id
