import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MaterializedFile:
    path: str
    rows: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MaterializationManifest:
    materialization_id: str
    data_release_id: str
    factor_versions: tuple[tuple[str, str], ...]
    start_date: str
    end_date: str
    universe: str
    code_version: str
    config_hash: str
    computed_at: str
    files: tuple[MaterializedFile, ...]
    total_rows: int
    content_hash: str
    status: str = "PUBLISHED"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, ensure_ascii=False)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "MaterializationManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["factor_versions"] = tuple(tuple(item) for item in payload["factor_versions"])
        payload["files"] = tuple(MaterializedFile(**item) for item in payload["files"])
        return cls(**payload)


def manifest_content_hash(files: tuple[MaterializedFile, ...]) -> str:
    payload = [(item.path, item.rows, item.sha256) for item in files]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("manifest time must be timezone-aware")
    return value.isoformat()
