import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aquant.data.quality.report import QualityReport
from aquant.domain.data_release import DataReleaseId, DataReleaseStatus


@dataclass(frozen=True, slots=True)
class DataReleaseManifest:
    schema_version: str
    release_id: str
    status: DataReleaseStatus
    created_at: str
    data_files: tuple[tuple[str, str], ...]
    quality_report_hashes: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


class DataReleasePublisher:
    """Publishes immutable release manifests only after every quality gate passes."""

    def __init__(self, root: Path, *, quarantine_root: Path | None = None) -> None:
        self._root = root
        self._quarantine_root = quarantine_root or root.parent / "quarantine"

    def publish(
        self,
        release_id: DataReleaseId,
        *,
        data_files: tuple[Path, ...],
        reports: tuple[QualityReport, ...],
        created_at: datetime | None = None,
    ) -> Path:
        if not data_files or not reports:
            raise ValueError("data release requires data files and quality reports")
        failures = tuple(report for report in reports if not report.passed)
        if failures:
            self._write_quarantine(release_id, failures)
            raise ValueError(f"quality gate blocked release: {release_id}")
        files = tuple(sorted((path.name, self._sha256(path)) for path in data_files))
        manifest = DataReleaseManifest(
            schema_version="aquant.data-release.v1",
            release_id=str(release_id),
            status=DataReleaseStatus.PUBLISHED,
            created_at=(created_at or datetime.now(UTC)).isoformat(timespec="microseconds"),
            data_files=files,
            quality_report_hashes=tuple(sorted(report.sha256 for report in reports)),
        )
        final = self._root / str(release_id)
        if final.exists():
            raise FileExistsError(f"data release already exists: {release_id}")
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = self._root / f".{release_id}.tmp-{uuid4().hex}"
        temporary.mkdir()
        try:
            self._write_new(temporary / "manifest.json", manifest.to_json().encode())
            temporary.rename(final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final

    def _write_quarantine(
        self, release_id: DataReleaseId, reports: tuple[QualityReport, ...]
    ) -> None:
        self._quarantine_root.mkdir(parents=True, exist_ok=True)
        path = self._quarantine_root / f"{release_id}-{uuid4().hex}.json"
        body = json.dumps(
            {
                "release_id": str(release_id),
                "reports": [json.loads(item.to_json()) for item in reports],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        self._write_new(path, body)

    @staticmethod
    def _sha256(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_new(path: Path, body: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
