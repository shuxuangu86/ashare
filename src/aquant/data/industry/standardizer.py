import hashlib
import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from aquant.data.history.tushare import RawHistoryPage, TushareHistoryCatalog
from aquant.data.industry.models import IndustryClassification, IndustryMembershipRecord

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_VERSION_WINDOWS = {
    "SW2014": (date(2014, 2, 21), date(2021, 12, 13)),
    "SW2021": (date(2021, 12, 13), None),
}


@dataclass(frozen=True, slots=True)
class IndustryStandardization:
    classifications: tuple[IndustryClassification, ...]
    memberships: tuple[IndustryMembershipRecord, ...]
    quarantine: tuple[dict[str, object], ...]
    raw_classification_rows: int
    raw_membership_rows: int
    duplicate_membership_rows: int
    source_fingerprint: str


@dataclass(frozen=True, slots=True)
class _SourceRow:
    value: dict[str, Any]
    source_hash: str
    ingested_at: datetime


def standardize_industry_history(
    catalog: TushareHistoryCatalog,
    *,
    data_release_id: str,
    trading_days: tuple[date, ...],
    systems: tuple[str, ...] = ("SW2014", "SW2021"),
    start_date: date | None = None,
    end_date: date | None = None,
) -> IndustryStandardization:
    if not data_release_id.strip() or not trading_days:
        raise ValueError("industry standardization requires release id and trading calendar")
    resolved_systems = tuple(dict.fromkeys(system.upper() for system in systems))
    unknown = set(resolved_systems) - set(_VERSION_WINDOWS)
    if unknown:
        raise ValueError(f"unsupported industry systems: {sorted(unknown)}")
    classification_rows = _source_rows(catalog.pages("index_classify"))
    membership_rows = _source_rows(catalog.pages("index_member"))
    classifications, classification_quarantine = _classifications(
        classification_rows,
        data_release_id=data_release_id,
        systems=resolved_systems,
    )
    memberships, membership_quarantine, duplicates = _memberships(
        membership_rows,
        classifications=classifications,
        data_release_id=data_release_id,
        trading_days=trading_days,
        start_date=start_date,
        end_date=end_date,
    )
    memberships, overlap_quarantine = _remove_overlaps(memberships)
    fingerprint = hashlib.sha256(
        "".join(
            sorted(row.source_hash for row in (*classification_rows, *membership_rows))
        ).encode()
    ).hexdigest()
    return IndustryStandardization(
        classifications=classifications,
        memberships=memberships,
        quarantine=tuple(
            sorted(
                (*classification_quarantine, *membership_quarantine, *overlap_quarantine),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
        ),
        raw_classification_rows=len(classification_rows),
        raw_membership_rows=len(membership_rows),
        duplicate_membership_rows=duplicates,
        source_fingerprint=fingerprint,
    )


def _source_rows(pages: tuple[RawHistoryPage, ...]) -> tuple[_SourceRow, ...]:
    output: list[_SourceRow] = []
    for page in pages:
        manifest = json.loads(page.manifest_path.read_text(encoding="utf-8"))
        source_hash = str(manifest["sha256"])
        payload = json.loads(page.payload_path.read_text(encoding="utf-8"))
        data = payload.get("data") or {}
        fields = data.get("fields") or []
        for item in data.get("items") or []:
            output.append(
                _SourceRow(
                    dict(zip(fields, item, strict=True)),
                    source_hash,
                    page.updated_at.astimezone(UTC),
                )
            )
    return tuple(output)


def _classifications(
    rows: tuple[_SourceRow, ...],
    *,
    data_release_id: str,
    systems: tuple[str, ...],
) -> tuple[tuple[IndustryClassification, ...], tuple[dict[str, object], ...]]:
    accepted: dict[tuple[str, str, str], IndustryClassification] = {}
    quarantine: list[dict[str, object]] = []
    for row in rows:
        value = row.value
        system = str(value.get("src", "")).upper()
        level = str(value.get("level", "")).upper()
        index_code = str(value.get("index_code", "")).upper()
        industry_code = str(value.get("industry_code", "")).upper()
        name = str(value.get("industry_name", "")).strip()
        if (
            system not in systems
            or level not in {"L1", "L2", "L3"}
            or not index_code
            or not industry_code
            or not name
        ):
            quarantine.append(_quarantine("INVALID_CLASSIFICATION", value))
            continue
        parent = str(value.get("parent_code") or "").upper() or None
        identity = (system, level, index_code)
        source_record_id = _hash(
            {"dataset": "index_classify", "identity": identity, "source": row.source_hash}
        )
        material = {
            "classification_system": "SW",
            "classification_version": system,
            "industry_code": index_code,
            "native_industry_code": industry_code,
            "industry_name": name,
            "industry_level": level,
            "parent_industry_code": parent,
            "source": "tushare",
            "source_record_id": source_record_id,
            "ingested_at": row.ingested_at.isoformat(),
            "data_release_id": data_release_id,
        }
        record = IndustryClassification(
            classification_system="SW",
            classification_version=system,
            industry_code=index_code,
            native_industry_code=industry_code,
            industry_name=name,
            industry_level=level,
            parent_industry_code=parent,
            source="tushare",
            source_record_id=source_record_id,
            ingested_at=row.ingested_at,
            data_release_id=data_release_id,
            content_hash=_hash(material),
        )
        existing = accepted.get(identity)
        if existing is not None and existing.content_hash != record.content_hash:
            quarantine.append(_quarantine("CONFLICTING_CLASSIFICATION", value))
            accepted.pop(identity, None)
        elif existing is None:
            accepted[identity] = record
    valid_industry_codes = {
        (item.classification_version, item.industry_level, item.native_industry_code)
        for item in accepted.values()
    }
    for identity, record in tuple(accepted.items()):
        if record.industry_level == "L1":
            continue
        parent_level = "L1" if record.industry_level == "L2" else "L2"
        if (
            record.classification_version,
            parent_level,
            record.parent_industry_code,
        ) not in valid_industry_codes:
            quarantine.append(
                _quarantine(
                    "ORPHAN_PARENT",
                    {
                        "classification_version": record.classification_version,
                        "industry_code": record.industry_code,
                        "parent_industry_code": record.parent_industry_code,
                    },
                )
            )
            accepted.pop(identity)
    return tuple(sorted(accepted.values(), key=_classification_key)), tuple(quarantine)


def _memberships(
    rows: tuple[_SourceRow, ...],
    *,
    classifications: tuple[IndustryClassification, ...],
    data_release_id: str,
    trading_days: tuple[date, ...],
    start_date: date | None,
    end_date: date | None,
) -> tuple[
    tuple[IndustryMembershipRecord, ...],
    tuple[dict[str, object], ...],
    int,
]:
    by_index: dict[str, list[IndustryClassification]] = defaultdict(list)
    for classification in classifications:
        by_index[classification.industry_code].append(classification)
    accepted: dict[tuple[str, str, str, date, date | None], IndustryMembershipRecord] = {}
    quarantine: list[dict[str, object]] = []
    duplicates = 0
    for row in rows:
        value = row.value
        index_code = str(value.get("index_code", "")).upper()
        ts_code = str(value.get("con_code", "")).upper()
        try:
            raw_from = _yyyymmdd(value.get("in_date"))
            raw_out = _optional_yyyymmdd(value.get("out_date"))
        except ValueError:
            quarantine.append(_quarantine("INVALID_MEMBERSHIP_DATE", value))
            continue
        matching = by_index.get(index_code, [])
        if not matching:
            quarantine.append(_quarantine("UNKNOWN_INDUSTRY_CODE", value))
            continue
        for classification in matching:
            version_start, version_end = _VERSION_WINDOWS[classification.classification_version]
            effective_from = max(raw_from, version_start)
            effective_to = _next_trading_day(raw_out, trading_days) if raw_out else None
            if version_end is not None:
                effective_to = min(effective_to, version_end) if effective_to else version_end
            if effective_to is not None and effective_from >= effective_to:
                continue
            if start_date is not None and effective_to is not None and effective_to <= start_date:
                continue
            if end_date is not None and effective_from > end_date:
                continue
            available_date = max(effective_from, version_start)
            available_at = datetime.combine(available_date, time(15), _SHANGHAI)
            identity = (
                classification.classification_version,
                classification.industry_level,
                ts_code,
                effective_from,
                effective_to,
            )
            source_record_id = _hash(
                {
                    "dataset": "index_member",
                    "index_code": index_code,
                    "ts_code": ts_code,
                    "in_date": str(value.get("in_date", "")),
                    "out_date": str(value.get("out_date", "")),
                }
            )
            material = {
                "classification_system": classification.classification_system,
                "classification_version": classification.classification_version,
                "industry_code": classification.industry_code,
                "industry_name": classification.industry_name,
                "industry_level": classification.industry_level,
                "parent_industry_code": classification.parent_industry_code,
                "instrument_id": ts_code,
                "ts_code": ts_code,
                "effective_from": effective_from.isoformat(),
                "effective_to": effective_to.isoformat() if effective_to else None,
                "announced_at": None,
                "available_at": available_at.isoformat(),
                "ingested_at": row.ingested_at.isoformat(),
                "source": "tushare",
                "source_record_id": source_record_id,
                "revision_no": 0,
                "data_release_id": data_release_id,
            }
            record = IndustryMembershipRecord(
                classification_system=classification.classification_system,
                classification_version=classification.classification_version,
                industry_code=classification.industry_code,
                industry_name=classification.industry_name,
                industry_level=classification.industry_level,
                parent_industry_code=classification.parent_industry_code,
                instrument_id=ts_code,
                ts_code=ts_code,
                effective_from=effective_from,
                effective_to=effective_to,
                announced_at=None,
                available_at=available_at,
                ingested_at=row.ingested_at,
                source="tushare",
                source_record_id=source_record_id,
                revision_no=0,
                data_release_id=data_release_id,
                content_hash=_hash(material),
            )
            existing = accepted.get(identity)
            if existing is None:
                accepted[identity] = record
            elif _membership_semantics(existing) == _membership_semantics(record):
                duplicates += 1
            else:
                quarantine.append(_quarantine("CONFLICTING_MEMBERSHIP", value))
                accepted.pop(identity, None)
    return tuple(sorted(accepted.values(), key=_membership_key)), tuple(quarantine), duplicates


def _remove_overlaps(
    memberships: tuple[IndustryMembershipRecord, ...],
) -> tuple[tuple[IndustryMembershipRecord, ...], tuple[dict[str, object], ...]]:
    grouped: dict[tuple[str, str, str], list[IndustryMembershipRecord]] = defaultdict(list)
    for record in memberships:
        grouped[(record.classification_version, record.industry_level, record.ts_code)].append(
            record
        )
    rejected: set[str] = set()
    quarantine: list[dict[str, object]] = []
    for records in grouped.values():
        records.sort(key=_membership_key)
        active: IndustryMembershipRecord | None = None
        for record in records:
            if (
                active is not None
                and (active.effective_to is None or record.effective_from < active.effective_to)
                and active.industry_code != record.industry_code
            ):
                rejected.update((active.content_hash, record.content_hash))
                quarantine.append(
                    _quarantine(
                        "OVERLAPPING_MEMBERSHIP",
                        {
                            "left": active.content_hash,
                            "right": record.content_hash,
                            "ts_code": record.ts_code,
                        },
                    )
                )
            if active is None or (
                active.effective_to is not None
                and (record.effective_to is None or record.effective_to > active.effective_to)
            ):
                active = record
    return (
        tuple(record for record in memberships if record.content_hash not in rejected),
        tuple(quarantine),
    )


def _next_trading_day(value: date, trading_days: tuple[date, ...]) -> date:
    position = bisect_right(trading_days, value)
    if position >= len(trading_days):
        return value.fromordinal(value.toordinal() + 1)
    return trading_days[position]


def _yyyymmdd(value: object) -> date:
    text = str(value or "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError("invalid YYYYMMDD")
    return datetime.strptime(text, "%Y%m%d").date()


def _optional_yyyymmdd(value: object) -> date | None:
    return None if value in {None, ""} else _yyyymmdd(value)


def _membership_semantics(record: IndustryMembershipRecord) -> tuple[object, ...]:
    return (
        record.classification_version,
        record.industry_code,
        record.industry_level,
        record.ts_code,
        record.effective_from,
        record.effective_to,
    )


def _classification_key(record: IndustryClassification) -> tuple[str, str, str]:
    return record.classification_version, record.industry_level, record.industry_code


def _membership_key(record: IndustryMembershipRecord) -> tuple[object, ...]:
    return (
        record.classification_version,
        record.industry_level,
        record.ts_code,
        record.effective_from,
        record.industry_code,
    )


def _quarantine(reason: str, value: object) -> dict[str, object]:
    return {"reason_code": reason, "record": value}


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
