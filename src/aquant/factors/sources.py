from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from aquant.factors.spec import ImplementationStatus, SourceType


class ResearchSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    title: str
    authors_or_institution: str
    publication_date: date
    source_type: SourceType
    identifier: str
    public_access_status: str
    local_file: str | None = None
    content_hash: str | None = None
    factor_families: tuple[str, ...]
    required_data: tuple[str, ...]
    implementation_status: ImplementationStatus
    notes: str = ""

    @field_validator(
        "source_id",
        "title",
        "authors_or_institution",
        "identifier",
        "public_access_status",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source metadata must not be blank")
        return value.strip()

    @field_validator("content_hash")
    @classmethod
    def _valid_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("source content_hash must be a lowercase SHA-256")
        return value


class SourceRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    sources: tuple[ResearchSource, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> SourceRegistry:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        registry = cls.model_validate(payload)
        ids = [source.source_id for source in registry.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source registry contains duplicate source_id")
        return registry

    @property
    def content_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, source_id: str) -> ResearchSource:
        try:
            return next(source for source in self.sources if source.source_id == source_id)
        except StopIteration as exc:
            raise KeyError(source_id) from exc
