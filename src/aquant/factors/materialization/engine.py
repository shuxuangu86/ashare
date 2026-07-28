import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from aquant.domain.data_release import DataReleaseId
from aquant.factors.atomic.models import AtomicFactor, FactorPanelInput
from aquant.factors.materialization.manifest import (
    MaterializationManifest,
    iso_utc,
    manifest_content_hash,
)
from aquant.factors.materialization.planner import plan_dependencies
from aquant.factors.materialization.storage import write_factor_array
from aquant.factors.types import FactorInputLoader


@dataclass(frozen=True, slots=True)
class MaterializationRequest:
    data_release_id: DataReleaseId
    factors: tuple[AtomicFactor, ...]
    start_date: date
    end_date: date
    as_of_time: datetime
    universe: str
    input_loader: FactorInputLoader
    calendar: Any
    config: dict[str, Any]
    code_version: str
    config_hash: str

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("materialization date range is inverted")
        if not self.factors or not self.universe.strip() or not self.code_version.strip():
            raise ValueError("materialization factors/universe/code version are required")
        for digest in (self.config_hash,):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("materialization config hash must be lowercase SHA-256")
        if self.as_of_time.tzinfo is None or self.as_of_time.utcoffset() is None:
            raise ValueError("materialization as_of_time must be timezone-aware")

    @property
    def materialization_id(self) -> str:
        payload = {
            "data_release_id": str(self.data_release_id),
            "factor_versions": sorted(
                (item.spec.factor_id, item.spec.version) for item in self.factors
            ),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "universe": self.universe,
            "code_version": self.code_version,
            "config_hash": self.config_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class FactorMaterializationEngine:
    def __init__(self, root: Path) -> None:
        self.root = root

    def materialize(self, request: MaterializationRequest) -> MaterializationManifest:
        target = self.root / f"materialization={request.materialization_id}"
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            return MaterializationManifest.read(manifest_path)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".factor-tmp-", dir=self.root))
        try:
            ordered_factors = plan_dependencies(request.factors)
            input_fields = tuple(
                sorted({field for factor in ordered_factors for field in factor.spec.input_fields})
            )
            panel = request.input_loader.load(
                fields=input_fields,
                start_date=request.start_date,
                end_date=request.end_date,
                as_of_time=request.as_of_time,
                universe_id=request.universe,
                data_release_id=request.data_release_id,
            )
            if not isinstance(panel, FactorPanelInput):
                raise TypeError("factor loader must return FactorPanelInput")
            files = tuple(
                file
                for factor in ordered_factors
                for file in write_factor_array(
                    temporary,
                    spec=factor.spec,
                    panel=panel,
                    values=factor.compute_array(panel),
                    data_release_id=request.data_release_id,
                    computed_at=request.as_of_time,
                )
            )
            content_hash = manifest_content_hash(files)
            manifest = MaterializationManifest(
                materialization_id=request.materialization_id,
                data_release_id=str(request.data_release_id),
                factor_versions=tuple(
                    sorted(
                        (factor.spec.factor_id, factor.spec.version) for factor in request.factors
                    )
                ),
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
                universe=request.universe,
                code_version=request.code_version,
                config_hash=request.config_hash,
                computed_at=iso_utc(request.as_of_time),
                files=files,
                total_rows=sum(item.rows for item in files),
                content_hash=content_hash,
            )
            manifest.write(temporary / "manifest.json")
            try:
                os.replace(temporary, target)
            except OSError as exc:
                if not manifest_path.exists():
                    raise
                existing = MaterializationManifest.read(manifest_path)
                if existing.content_hash != manifest.content_hash:
                    raise ValueError(
                        "idempotent materialization produced conflicting content"
                    ) from exc
                shutil.rmtree(temporary)
                return existing
            return manifest
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
