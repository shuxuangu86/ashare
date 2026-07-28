from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class IndustryClassification:
    classification_system: str
    classification_version: str
    industry_code: str
    native_industry_code: str
    industry_name: str
    industry_level: str
    parent_industry_code: str | None
    source: str
    source_record_id: str
    ingested_at: datetime
    data_release_id: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class IndustryMembershipRecord:
    classification_system: str
    classification_version: str
    industry_code: str
    industry_name: str
    industry_level: str
    parent_industry_code: str | None
    instrument_id: str
    ts_code: str
    effective_from: date
    effective_to: date | None
    announced_at: datetime | None
    available_at: datetime
    ingested_at: datetime
    source: str
    source_record_id: str
    revision_no: int
    data_release_id: str
    content_hash: str

    def contains(self, trade_date: date) -> bool:
        return self.effective_from <= trade_date and (
            self.effective_to is None or trade_date < self.effective_to
        )


@dataclass(frozen=True, slots=True)
class IndustryPITPanel:
    trade_dates: tuple[date, ...]
    ts_codes: tuple[str, ...]
    industries: npt.NDArray[np.object_]
    available_dates: npt.NDArray[np.object_]

    def __post_init__(self) -> None:
        shape = (len(self.trade_dates), len(self.ts_codes))
        if self.industries.shape != shape or self.available_dates.shape != shape:
            raise ValueError("industry PIT panel fields must align")
