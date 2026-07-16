import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from aquant.data.providers.base import ProviderResponse

_PARTITION_VALUE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class RawBatchManifest:
    schema_version: str
    batch_id: str
    provider: str
    provider_version: str
    dataset: str
    request_params: dict[str, Any]
    source_request_id: str
    response_started_at: str
    response_completed_at: str
    payload_file: str
    media_type: str
    transport_status: int | None
    sha256: str
    byte_count: int
    record_count: int | None

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class RawBatchResult:
    directory: Path
    payload_path: Path
    manifest_path: Path
    manifest: RawBatchManifest


class RawBatchStore:
    """Append-only filesystem storage for original provider responses."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write_response(
        self,
        response: ProviderResponse,
        *,
        batch_id: UUID | None = None,
        payload_file: str = "response.bin",
    ) -> RawBatchResult:
        provider = self._partition_value(response.provider, field_name="provider")
        dataset = self._partition_value(response.request.dataset, field_name="dataset")
        if Path(payload_file).name != payload_file or not payload_file:
            raise ValueError("payload_file must be a plain filename")

        resolved_batch_id = batch_id or uuid4()
        ingest_date = response.received_at.date().isoformat()
        partition = (
            self._root
            / f"provider={provider}"
            / f"dataset={dataset}"
            / f"ingest_date={ingest_date}"
        )
        final_directory = partition / f"batch_id={resolved_batch_id}"
        if final_directory.exists():
            raise FileExistsError(f"raw batch already exists: {final_directory}")

        partition.mkdir(parents=True, exist_ok=True)
        temporary_directory = partition / f".{resolved_batch_id}.tmp-{uuid4().hex}"
        temporary_directory.mkdir()
        checksum = hashlib.sha256(response.body).hexdigest()
        manifest = RawBatchManifest(
            schema_version="aquant.raw-batch.v1",
            batch_id=str(resolved_batch_id),
            provider=provider,
            provider_version=response.provider_version,
            dataset=dataset,
            request_params=response.request.params,
            source_request_id=response.source_request_id,
            response_started_at=self._isoformat(response.request.requested_at),
            response_completed_at=self._isoformat(response.received_at),
            payload_file=payload_file,
            media_type=response.media_type,
            transport_status=response.transport_status,
            sha256=checksum,
            byte_count=len(response.body),
            record_count=response.record_count,
        )

        try:
            self._write_new_file(temporary_directory / payload_file, response.body)
            self._write_new_file(
                temporary_directory / "manifest.json", manifest.to_json().encode("utf-8")
            )
            temporary_directory.rename(final_directory)
        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

        return RawBatchResult(
            directory=final_directory,
            payload_path=final_directory / payload_file,
            manifest_path=final_directory / "manifest.json",
            manifest=manifest,
        )

    @staticmethod
    def _partition_value(value: str, *, field_name: str) -> str:
        normalized = value.strip().lower()
        if not _PARTITION_VALUE.fullmatch(normalized):
            raise ValueError(f"unsafe {field_name} partition value: {value!r}")
        return normalized

    @staticmethod
    def _isoformat(value: datetime) -> str:
        return value.isoformat(timespec="microseconds")

    @staticmethod
    def _write_new_file(path: Path, body: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
