import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from aquant.data.daily_update import parse_tushare_symbol
from aquant.data.history.tushare import RawHistoryPage, TushareHistoryCatalog
from aquant.domain.identifiers import Symbol


@dataclass(frozen=True, slots=True)
class IndexWeightSnapshot:
    trade_date: date
    weights: tuple[tuple[Symbol, Decimal], ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len({symbol for symbol, _ in self.weights}):
            raise ValueError("index weight snapshot contains duplicate constituents")
        total = sum((weight for _, weight in self.weights), Decimal("0"))
        if not self.weights or any(weight <= 0 for _, weight in self.weights):
            raise ValueError("index weight snapshot requires positive weights")
        if abs(total - Decimal("100")) > Decimal("0.1"):
            raise ValueError(f"index weight snapshot sum is invalid: {total}")


@dataclass(frozen=True, slots=True)
class IndexHistory:
    index_code: str
    weight_snapshots: tuple[IndexWeightSnapshot, ...]
    closes: tuple[tuple[date, Decimal], ...]
    source_fingerprint: str

    def __post_init__(self) -> None:
        if not self.index_code.strip() or not self.weight_snapshots or not self.closes:
            raise ValueError("index history is incomplete")
        if tuple(sorted(self.weight_snapshots, key=lambda item: item.trade_date)) != (
            self.weight_snapshots
        ):
            raise ValueError("index weight snapshots must be ordered")
        if tuple(sorted(self.closes)) != self.closes:
            raise ValueError("index closes must be ordered")


def load_index_history(
    catalog: TushareHistoryCatalog,
    *,
    index_code: str,
    start_date: date,
    end_date: date,
) -> IndexHistory:
    if start_date > end_date:
        raise ValueError("index history date range is invalid")
    code = index_code.strip().upper()
    weight_rows, weight_hashes = _rows(
        catalog.pages("index_weight"),
        index_code=code,
        start_date=start_date,
        end_date=end_date,
    )
    daily_rows, daily_hashes = _rows(
        catalog.pages("index_daily"),
        index_code=code,
        start_date=start_date,
        end_date=end_date,
    )
    weights_by_date: dict[date, dict[Symbol, Decimal]] = {}
    for row in weight_rows:
        trade_date = _date(row["trade_date"])
        symbol = parse_tushare_symbol(str(row["con_code"]))
        weight = _decimal(row["weight"])
        existing = weights_by_date.setdefault(trade_date, {}).get(symbol)
        if existing is not None and existing != weight:
            raise ValueError(f"conflicting index weight for {trade_date} {symbol}")
        weights_by_date[trade_date][symbol] = weight
    closes: dict[date, Decimal] = {}
    for row in daily_rows:
        trade_date = _date(row["trade_date"])
        close = _decimal(row["close"])
        existing = closes.get(trade_date)
        if existing is not None and existing != close:
            raise ValueError(f"conflicting index close for {trade_date}")
        closes[trade_date] = close
    fingerprint = hashlib.sha256("".join(sorted(weight_hashes | daily_hashes)).encode()).hexdigest()
    return IndexHistory(
        index_code=code,
        weight_snapshots=tuple(
            IndexWeightSnapshot(day, tuple(sorted(weights.items())))
            for day, weights in sorted(weights_by_date.items())
        ),
        closes=tuple(sorted(closes.items())),
        source_fingerprint=fingerprint,
    )


def _rows(
    pages: tuple[RawHistoryPage, ...],
    *,
    index_code: str,
    start_date: date,
    end_date: date,
) -> tuple[tuple[dict[str, object], ...], set[str]]:
    output: list[dict[str, object]] = []
    hashes: set[str] = set()
    for page in pages:
        requested_code = str(page.params.get("index_code") or page.params.get("ts_code") or "")
        if requested_code.upper() != index_code or not _overlaps(page, start_date, end_date):
            continue
        body = page.payload_path.read_bytes()
        manifest = json.loads(page.manifest_path.read_text(encoding="utf-8"))
        source_hash = hashlib.sha256(body).hexdigest()
        if source_hash != manifest.get("sha256"):
            raise ValueError(f"raw index page hash mismatch: {page.payload_path}")
        payload = json.loads(body)
        data = payload.get("data") or {}
        fields = data.get("fields") or []
        for values in data.get("items") or []:
            row = dict(zip(fields, values, strict=True))
            trade_date = _date(row["trade_date"])
            if start_date <= trade_date <= end_date:
                output.append(row)
        hashes.add(source_hash)
    return tuple(output), hashes


def _overlaps(page: RawHistoryPage, start_date: date, end_date: date) -> bool:
    requested_start = _optional_date(page.params.get("start_date")) or date.min
    requested_end = _optional_date(page.params.get("end_date")) or date.max
    return requested_start <= end_date and requested_end >= start_date


def _optional_date(value: object) -> date | None:
    return _date(value) if value not in {None, ""} else None


def _date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y%m%d").date()


def _decimal(value: object) -> Decimal:
    resolved = Decimal(str(value))
    if not resolved.is_finite() or resolved <= 0:
        raise ValueError("index history numeric value must be finite and positive")
    return resolved
