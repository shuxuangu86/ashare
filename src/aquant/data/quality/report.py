import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class Severity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True, order=True)
class QualityIssue:
    severity: Severity
    code: str
    key: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.key.strip() or not self.message.strip():
            raise ValueError("quality issue fields must not be blank")


@dataclass(frozen=True, slots=True)
class QualityReport:
    dataset: str
    checked_records: int
    issues: tuple[QualityIssue, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset.strip() or self.checked_records < 0:
            raise ValueError("quality report dataset/count is invalid")
        object.__setattr__(self, "issues", tuple(sorted(self.issues)))

    @property
    def passed(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()
