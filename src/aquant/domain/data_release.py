import re
from dataclasses import dataclass
from enum import StrEnum

_DATA_RELEASE_PATTERN = re.compile(r"^cn_equity_[0-9]{8}_[0-9]{3}$")


@dataclass(frozen=True, slots=True, order=True)
class DataReleaseId:
    value: str

    def __post_init__(self) -> None:
        if not _DATA_RELEASE_PATTERN.fullmatch(self.value):
            raise ValueError("data release id must match cn_equity_YYYYMMDD_NNN")

    def __str__(self) -> str:
        return self.value


class DataReleaseStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    PUBLISHED = "PUBLISHED"
    QUARANTINED = "QUARANTINED"
