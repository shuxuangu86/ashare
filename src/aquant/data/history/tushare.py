import hashlib
import json
import os
import shutil
import sqlite3
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import pstdev
from typing import Any
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

DEFAULT_HISTORY_DATASETS: tuple[str, ...] = (
    "daily",
    "daily_basic",
    "adj_factor",
    "stk_limit",
    "suspend_d",
    "stock_basic",
    "namechange",
    "trade_cal",
    "fina_indicator",
    "dividend",
)

_MARKET_DATASETS = frozenset({"daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d"})
_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "daily": frozenset({"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}),
    "daily_basic": frozenset(
        {"ts_code", "trade_date", "total_mv", "circ_mv", "pb", "turnover_rate", "dv_ttm"}
    ),
    "adj_factor": frozenset({"ts_code", "trade_date", "adj_factor"}),
    "stk_limit": frozenset({"ts_code", "trade_date", "up_limit", "down_limit"}),
    "suspend_d": frozenset({"ts_code", "trade_date"}),
    "stock_basic": frozenset({"ts_code", "symbol", "name", "market", "list_date"}),
    "namechange": frozenset({"ts_code", "name", "start_date", "end_date"}),
    "trade_cal": frozenset({"exchange", "cal_date", "is_open"}),
    "fina_indicator": frozenset(
        {"ts_code", "ann_date", "end_date", "netprofit_yoy", "debt_to_assets"}
    ),
    "dividend": frozenset(
        {"ts_code", "stk_div", "cash_div_tax", "record_date", "ex_date", "pay_date"}
    ),
}


@dataclass(frozen=True, slots=True)
class RawHistoryPage:
    api_name: str
    params: Mapping[str, Any]
    row_count: int
    fields: tuple[str, ...]
    payload_path: Path
    manifest_path: Path
    updated_at: datetime

    @property
    def offset(self) -> int:
        return int(self.params.get("offset", 0))

    @property
    def limit(self) -> int:
        return int(self.params.get("limit", 0))

    @property
    def base_params(self) -> dict[str, Any]:
        return {key: value for key, value in self.params.items() if key not in {"limit", "offset"}}


@dataclass(frozen=True, slots=True)
class HistoryDatasetManifest:
    schema_version: str
    dataset: str
    row_count: int
    min_date: str | None
    max_date: str | None
    files: tuple[tuple[str, str, int], ...]
    source_page_count: int
    source_fingerprint: str
    transformations: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "HistoryDatasetManifest":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("history dataset manifest root must be an object")
        payload["files"] = tuple(tuple(item) for item in payload["files"])
        payload["transformations"] = tuple(payload["transformations"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class HistoryReleaseManifest:
    schema_version: str
    release_id: str
    provider: str
    start_date: str
    end_date: str
    created_at: str
    datasets: tuple[tuple[str, str, int], ...]
    raw_state_path: str
    status: str
    caveats: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "HistoryReleaseManifest":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("history release manifest root must be an object")
        payload["datasets"] = tuple(tuple(item) for item in payload["datasets"])
        payload["caveats"] = tuple(payload["caveats"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class HistoryMaterializationResult:
    release_directory: Path
    manifest_path: Path
    manifest: HistoryReleaseManifest


class TushareHistoryCatalog:
    """Read-only resolver for coherent completed page families in the backfill state."""

    def __init__(self, state_path: Path) -> None:
        if not state_path.is_file():
            raise FileNotFoundError(f"Tushare backfill state not found: {state_path}")
        self.state_path = state_path.resolve()

    def pages(self, api_name: str, *, include_empty: bool = False) -> tuple[RawHistoryPage, ...]:
        uri = f"file:{self.state_path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                """
                SELECT params_json, COALESCE(row_count, 0), fields_json,
                       payload_path, manifest_path, updated_at
                FROM pages
                WHERE api_name = ? AND status = 'COMPLETED'
                ORDER BY params_json, updated_at
                """,
                (api_name,),
            ).fetchall()
        finally:
            connection.close()
        candidates: list[RawHistoryPage] = []
        for params_json, row_count, fields_json, payload, manifest, updated_at in rows:
            page = RawHistoryPage(
                api_name=api_name,
                params=json.loads(params_json),
                row_count=int(row_count),
                fields=tuple(json.loads(fields_json or "[]")),
                payload_path=Path(payload),
                manifest_path=Path(manifest),
                updated_at=datetime.fromisoformat(updated_at),
            )
            if not page.payload_path.is_file() or not page.manifest_path.is_file():
                raise FileNotFoundError(
                    f"completed {api_name} checkpoint points to a missing raw file: "
                    f"{page.payload_path}"
                )
            candidates.append(page)
        selected = self._select_coherent_families(candidates)
        return selected if include_empty else tuple(page for page in selected if page.row_count)

    def audit_required_fields(self, datasets: Iterable[str]) -> None:
        for dataset in datasets:
            required = _REQUIRED_FIELDS.get(dataset)
            if required is None:
                raise ValueError(f"unsupported history dataset: {dataset}")
            pages = self.pages(dataset)
            if not pages:
                raise ValueError(f"history dataset has no completed non-empty pages: {dataset}")
            available = set().union(*(set(page.fields) for page in pages))
            missing = required - available
            if missing:
                raise ValueError(
                    f"history dataset {dataset} is missing required fields: {sorted(missing)}"
                )

    @staticmethod
    def _select_coherent_families(
        candidates: Sequence[RawHistoryPage],
    ) -> tuple[RawHistoryPage, ...]:
        by_request: dict[str, list[RawHistoryPage]] = defaultdict(list)
        for page in candidates:
            key = json.dumps(
                page.base_params,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            by_request[key].append(page)

        selected: list[RawHistoryPage] = []
        for request_pages in by_request.values():
            by_shape: dict[tuple[int, tuple[str, ...]], list[RawHistoryPage]] = defaultdict(list)
            for page in request_pages:
                by_shape[(page.limit, page.fields)].append(page)
            valid: list[list[RawHistoryPage]] = []
            for (limit, _fields), family in by_shape.items():
                if limit <= 0:
                    valid.append(
                        [
                            max(
                                family,
                                key=lambda item: (
                                    item.updated_at,
                                    str(item.payload_path),
                                ),
                            )
                        ]
                    )
                    continue
                by_offset: dict[int, list[RawHistoryPage]] = defaultdict(list)
                for page in family:
                    if page.offset >= 0 and page.offset % limit == 0:
                        by_offset[page.offset].append(page)
                terminals = sorted(
                    (page for page in family if page.row_count < limit),
                    key=lambda item: (
                        item.updated_at,
                        item.offset,
                        str(item.payload_path),
                    ),
                )
                for terminal in terminals:
                    if terminal.offset < 0 or terminal.offset % limit:
                        continue
                    reconstructed: list[RawHistoryPage] = []
                    for offset in range(0, terminal.offset, limit):
                        eligible = [
                            page
                            for page in by_offset.get(offset, ())
                            if page.row_count == limit and page.updated_at <= terminal.updated_at
                        ]
                        if not eligible:
                            break
                        reconstructed.append(
                            max(
                                eligible,
                                key=lambda item: (
                                    item.updated_at,
                                    str(item.payload_path),
                                ),
                            )
                        )
                    else:
                        reconstructed.append(terminal)
                        valid.append(reconstructed)
            if not valid:
                label = json.dumps(request_pages[0].base_params, ensure_ascii=False)
                raise ValueError(f"no complete pagination family for {label}")
            winner = max(
                valid,
                key=lambda family: (
                    max(page.updated_at for page in family),
                    family[0].limit,
                ),
            )
            selected.extend(winner)
        return tuple(
            sorted(
                selected,
                key=lambda page: (
                    str(page.base_params.get("trade_date", "")),
                    str(page.base_params.get("ts_code", "")),
                    json.dumps(page.base_params, sort_keys=True),
                    page.offset,
                ),
            )
        )


class TushareHistoryMaterializer:
    """Build an immutable, unit-correct historical Parquet release from archived Raw."""

    SCHEMA_VERSION = "aquant.history-release.v1"

    def __init__(
        self,
        *,
        catalog: TushareHistoryCatalog,
        standard_root: Path,
        release_id: str,
        start_date: date,
        end_date: date,
        verify_raw_checksums: bool = True,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        if not release_id.strip() or "/" in release_id or "\\" in release_id:
            raise ValueError("release_id must be a non-empty plain path component")
        if start_date > end_date:
            raise ValueError("history start_date cannot be after end_date")
        self._catalog = catalog
        self._standard_root = standard_root
        self._release_id = release_id.strip()
        self._start = start_date
        self._end = end_date
        self._verify_raw_checksums = verify_raw_checksums
        self._progress = progress

    def materialize(
        self,
        datasets: Sequence[str] = DEFAULT_HISTORY_DATASETS,
    ) -> HistoryMaterializationResult:
        requested = tuple(dict.fromkeys(datasets))
        if not requested:
            raise ValueError("at least one history dataset is required")
        unknown = set(requested) - set(DEFAULT_HISTORY_DATASETS)
        if unknown:
            raise ValueError(f"unsupported history datasets: {sorted(unknown)}")
        self._catalog.audit_required_fields(requested)

        final_directory = self._standard_root / f"history-release={self._release_id}"
        if final_directory.exists():
            self._emit("release_reused", release_id=self._release_id)
            return self._load_existing(final_directory)
        building = self._standard_root / f".history-release={self._release_id}.building"
        building.mkdir(parents=True, exist_ok=True)

        ordered = tuple(
            dataset
            for dataset in DEFAULT_HISTORY_DATASETS
            if dataset in requested and dataset != "stock_basic"
        )
        manifests: dict[str, HistoryDatasetManifest] = {}
        for dataset in ordered:
            manifests[dataset] = self._materialize_dataset(building, dataset)
        if "stock_basic" in requested:
            manifests["stock_basic"] = self._materialize_dataset(building, "stock_basic")

        release_manifest = HistoryReleaseManifest(
            schema_version=self.SCHEMA_VERSION,
            release_id=self._release_id,
            provider="tushare",
            start_date=self._start.isoformat(),
            end_date=self._end.isoformat(),
            created_at=datetime.now(UTC).isoformat(timespec="microseconds"),
            datasets=tuple(
                (
                    dataset,
                    self._checksum(building / f"dataset={dataset}" / "manifest.json"),
                    manifest.row_count,
                )
                for dataset, manifest in sorted(manifests.items())
            ),
            raw_state_path=str(self._catalog.state_path),
            status="MATERIALIZED_NOT_BACKTEST_APPROVED",
            caveats=(
                "stock_basic exchange/list_status are recovered from ts_code/request parameters",
                (
                    "delist_date is inferred only for D-status securities "
                    "with a last daily bar in the materialized window"
                ),
                (
                    "fundamentals retain announcement dates; PIT consumers "
                    "must require ann_date < signal date"
                ),
                "unadjusted prices, adjustment factors, and dividends are stored separately",
                (
                    "returns are not backtest-approved until corporate actions "
                    "are applied by the ledger"
                ),
            ),
        )
        self._write_new(building / "manifest.json", release_manifest.to_json().encode())
        self._fsync_directory(building)
        building.rename(final_directory)
        self._fsync_directory(self._standard_root)
        self._emit(
            "release_complete",
            release_id=self._release_id,
            datasets=len(manifests),
            rows=sum(manifest.row_count for manifest in manifests.values()),
        )
        return HistoryMaterializationResult(
            final_directory,
            final_directory / "manifest.json",
            release_manifest,
        )

    def _materialize_dataset(
        self,
        building: Path,
        dataset: str,
    ) -> HistoryDatasetManifest:
        final = building / f"dataset={dataset}"
        manifest_path = final / "manifest.json"
        if manifest_path.is_file():
            manifest = HistoryDatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
            self._verify_dataset_files(final, manifest)
            self._emit(
                "dataset_reused",
                dataset=dataset,
                rows=manifest.row_count,
                source_pages=manifest.source_page_count,
            )
            return manifest
        temporary = building / f".dataset={dataset}.tmp-{uuid4().hex}"
        temporary.mkdir()
        self._emit("dataset_started", dataset=dataset)
        try:
            pages = self._catalog.pages(dataset)
            if dataset == "daily":
                manifest = self._materialize_daily(temporary, pages)
            elif dataset == "daily_basic":
                manifest = self._materialize_daily_basic(temporary, pages)
            elif dataset in _MARKET_DATASETS:
                manifest = self._materialize_market_dataset(temporary, dataset, pages)
            elif dataset == "stock_basic":
                manifest = self._materialize_stock_basic(temporary, pages, building)
            else:
                manifest = self._materialize_reference_dataset(temporary, dataset, pages)
            self._write_new(temporary / "manifest.json", manifest.to_json().encode())
            temporary.rename(final)
            self._emit(
                "dataset_complete",
                dataset=dataset,
                rows=manifest.row_count,
                source_pages=manifest.source_page_count,
                min_date=manifest.min_date,
                max_date=manifest.max_date,
            )
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _materialize_daily(
        self,
        directory: Path,
        pages: Sequence[RawHistoryPage],
    ) -> HistoryDatasetManifest:
        schema = pa.schema(
            [
                pa.field("ts_code", pa.string(), nullable=False),
                pa.field("exchange", pa.string(), nullable=False),
                pa.field("trade_date", pa.date32(), nullable=False),
                pa.field("open", pa.decimal128(20, 6), nullable=False),
                pa.field("high", pa.decimal128(20, 6), nullable=False),
                pa.field("low", pa.decimal128(20, 6), nullable=False),
                pa.field("close", pa.decimal128(20, 6), nullable=False),
                pa.field("volume", pa.decimal128(28, 4), nullable=False),
                pa.field("amount", pa.decimal128(28, 4), nullable=False),
                pa.field(
                    "prior_20d_average_volume",
                    pa.decimal128(28, 4),
                    nullable=True,
                ),
            ]
        )
        windows: dict[str, deque[Decimal]] = defaultdict(lambda: deque(maxlen=20))
        target_pages, warmup_pages = self._market_pages_with_warmup(pages, sessions=20)
        target_paths = {page.payload_path for page in target_pages}
        last_trade: dict[str, date] = {}

        def normalized_pages() -> Iterator[tuple[RawHistoryPage, list[dict[str, Any]]]]:
            for page in (*warmup_pages, *target_pages):
                output: list[dict[str, Any]] = []
                for record in self._records(page):
                    code = _required_text(record, "ts_code")
                    trade_date = _yyyymmdd(record.get("trade_date"), field="trade_date")
                    volume = _decimal(record.get("vol"), field="vol") * Decimal("100")
                    history = windows[code]
                    prior = (
                        sum(history, Decimal("0")) / len(history) if len(history) == 20 else None
                    )
                    history.append(volume)
                    if page.payload_path not in target_paths:
                        continue
                    output.append(
                        {
                            "ts_code": code,
                            "exchange": _exchange_from_ts_code(code),
                            "trade_date": trade_date,
                            "open": _quantize_price(record.get("open"), field="open"),
                            "high": _quantize_price(record.get("high"), field="high"),
                            "low": _quantize_price(record.get("low"), field="low"),
                            "close": _quantize_price(record.get("close"), field="close"),
                            "volume": volume.quantize(Decimal("0.0001")),
                            "amount": (
                                _decimal(record.get("amount"), field="amount") * Decimal("1000")
                            ).quantize(Decimal("0.0001")),
                            "prior_20d_average_volume": (
                                prior.quantize(Decimal("0.0001")) if prior is not None else None
                            ),
                        }
                    )
                    last_trade[code] = max(last_trade.get(code, trade_date), trade_date)
                yield page, output

        manifest = self._write_date_partitioned(
            directory,
            "daily",
            schema,
            normalized_pages(),
            transformations=(
                "Tushare vol lots multiplied by 100 to shares",
                "Tushare amount thousand-CNY multiplied by 1000 to CNY",
                "prior_20d_average_volume uses only T-20..T-1 observations",
                "exchange recovered from ts_code suffix",
            ),
        )
        self._write_new(
            directory / "symbol-last-trade.json",
            json.dumps(
                {code: value.isoformat() for code, value in sorted(last_trade.items())},
                ensure_ascii=False,
                sort_keys=True,
            ).encode(),
        )
        return manifest

    def _materialize_daily_basic(
        self,
        directory: Path,
        pages: Sequence[RawHistoryPage],
    ) -> HistoryDatasetManifest:
        schema = pa.schema(
            [
                pa.field("ts_code", pa.string(), nullable=False),
                pa.field("trade_date", pa.date32(), nullable=False),
                pa.field("total_market_cap", pa.decimal128(28, 4), nullable=False),
                pa.field("float_market_cap", pa.decimal128(28, 4), nullable=True),
                pa.field("pb", pa.decimal128(20, 6), nullable=True),
                pa.field("turnover_rate", pa.decimal128(20, 6), nullable=True),
                pa.field("dividend_yield", pa.decimal128(20, 6), nullable=True),
                pa.field(
                    "turnover_volatility_20d",
                    pa.decimal128(20, 6),
                    nullable=True,
                ),
            ]
        )
        windows: dict[str, deque[Decimal]] = defaultdict(lambda: deque(maxlen=20))
        target_pages, warmup_pages = self._market_pages_with_warmup(pages, sessions=19)
        target_paths = {page.payload_path for page in target_pages}

        def normalized_pages() -> Iterator[tuple[RawHistoryPage, list[dict[str, Any]]]]:
            for page in (*warmup_pages, *target_pages):
                output: list[dict[str, Any]] = []
                for record in self._records(page):
                    code = _required_text(record, "ts_code")
                    turnover = _optional_decimal(record.get("turnover_rate"))
                    history = windows[code]
                    if turnover is not None:
                        history.append(turnover)
                    volatility = (
                        Decimal(str(pstdev(history))).quantize(Decimal("0.000001"))
                        if len(history) == 20
                        else None
                    )
                    if page.payload_path not in target_paths:
                        continue
                    total_mv = _decimal(record.get("total_mv"), field="total_mv")
                    if total_mv <= 0:
                        raise ValueError(f"non-positive total_mv for {code}")
                    output.append(
                        {
                            "ts_code": code,
                            "trade_date": _yyyymmdd(record.get("trade_date"), field="trade_date"),
                            "total_market_cap": (total_mv * Decimal("10000")).quantize(
                                Decimal("0.0001")
                            ),
                            "float_market_cap": _scaled_optional(
                                record.get("circ_mv"),
                                Decimal("10000"),
                                Decimal("0.0001"),
                            ),
                            "pb": _quantized_optional(record.get("pb"), Decimal("0.000001")),
                            "turnover_rate": (
                                turnover.quantize(Decimal("0.000001"))
                                if turnover is not None
                                else None
                            ),
                            "dividend_yield": _quantized_optional(
                                record.get("dv_ttm"), Decimal("0.000001")
                            ),
                            "turnover_volatility_20d": volatility,
                        }
                    )
                yield page, output

        return self._write_date_partitioned(
            directory,
            "daily_basic",
            schema,
            normalized_pages(),
            transformations=(
                "Tushare total_mv/circ_mv ten-thousand-CNY multiplied by 10000 to CNY",
                "turnover_volatility_20d is population standard deviation over T-19..T",
                "dv_ttm retained in provider percentage units",
            ),
        )

    def _materialize_market_dataset(
        self,
        directory: Path,
        dataset: str,
        pages: Sequence[RawHistoryPage],
    ) -> HistoryDatasetManifest:
        schemas: dict[str, pa.Schema] = {
            "adj_factor": pa.schema(
                [
                    pa.field("ts_code", pa.string(), nullable=False),
                    pa.field("trade_date", pa.date32(), nullable=False),
                    pa.field("adj_factor", pa.decimal128(28, 10), nullable=False),
                ]
            ),
            "stk_limit": pa.schema(
                [
                    pa.field("ts_code", pa.string(), nullable=False),
                    pa.field("trade_date", pa.date32(), nullable=False),
                    pa.field("up_limit", pa.decimal128(20, 6), nullable=False),
                    pa.field("down_limit", pa.decimal128(20, 6), nullable=False),
                ]
            ),
            "suspend_d": pa.schema(
                [
                    pa.field("ts_code", pa.string(), nullable=False),
                    pa.field("trade_date", pa.date32(), nullable=False),
                ]
            ),
        }
        transformations = {
            "adj_factor": ("cumulative adjustment factor retained without rebasing",),
            "stk_limit": ("provider price limits retained in unadjusted CNY/share",),
            "suspend_d": ("presence of row means suspended on trade_date",),
        }

        def transform(record: Mapping[str, Any]) -> dict[str, Any]:
            base: dict[str, Any] = {
                "ts_code": _required_text(record, "ts_code"),
                "trade_date": _yyyymmdd(record.get("trade_date"), field="trade_date"),
            }
            if dataset == "adj_factor":
                base["adj_factor"] = _decimal(
                    record.get("adj_factor"), field="adj_factor"
                ).quantize(Decimal("0.0000000001"))
            elif dataset == "stk_limit":
                base["up_limit"] = _quantize_price(record.get("up_limit"), field="up_limit")
                base["down_limit"] = _quantize_price(record.get("down_limit"), field="down_limit")
            return base

        rows = (
            (page, [transform(record) for record in self._records(page)])
            for page in self._filter_market_pages(pages)
        )
        return self._write_date_partitioned(
            directory,
            dataset,
            schemas[dataset],
            rows,
            transformations=transformations[dataset],
        )

    def _materialize_stock_basic(
        self,
        directory: Path,
        pages: Sequence[RawHistoryPage],
        building: Path,
    ) -> HistoryDatasetManifest:
        last_trade_path = building / "dataset=daily" / "symbol-last-trade.json"
        if not last_trade_path.is_file():
            raise ValueError("stock_basic materialization requires the daily dataset")
        last_trade = json.loads(last_trade_path.read_text(encoding="utf-8"))
        records_by_code: dict[str, dict[str, Any]] = {}
        source_pages: list[RawHistoryPage] = []
        for page in pages:
            request_status = str(page.base_params.get("list_status", "")).upper()
            if request_status not in {"L", "D", "P"}:
                raise ValueError(
                    "stock_basic page has no recoverable list_status request parameter"
                )
            source_pages.append(page)
            for record in self._records(page):
                code = _required_text(record, "ts_code")
                official_delist = _optional_yyyymmdd(record.get("delist_date"))
                inferred = (
                    _optional_iso_date(last_trade.get(code))
                    if request_status == "D" and official_delist is None
                    else None
                )
                normalized = {
                    "ts_code": code,
                    "symbol": _required_text(record, "symbol"),
                    "name": _required_text(record, "name"),
                    "market": _required_text(record, "market"),
                    "exchange": _exchange_from_ts_code(code),
                    "list_status": request_status,
                    "list_date": _yyyymmdd(record.get("list_date"), field="list_date"),
                    "delist_date": official_delist or inferred,
                    "delist_date_inferred": official_delist is None and inferred is not None,
                }
                existing = records_by_code.get(code)
                if existing is not None and existing != normalized:
                    raise ValueError(f"conflicting stock_basic rows for {code}")
                records_by_code[code] = normalized
            if len(source_pages) % 500 == 0:
                self._emit(
                    "dataset_progress",
                    dataset="stock_basic",
                    pages=len(source_pages),
                    rows=len(records_by_code),
                )
        schema = pa.schema(
            [
                pa.field("ts_code", pa.string(), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("name", pa.string(), nullable=False),
                pa.field("market", pa.string(), nullable=False),
                pa.field("exchange", pa.string(), nullable=False),
                pa.field("list_status", pa.string(), nullable=False),
                pa.field("list_date", pa.date32(), nullable=False),
                pa.field("delist_date", pa.date32(), nullable=True),
                pa.field("delist_date_inferred", pa.bool_(), nullable=False),
            ]
        )
        rows = sorted(records_by_code.values(), key=lambda item: str(item["ts_code"]))
        return self._write_single_file(
            directory,
            "stock_basic",
            schema,
            rows,
            source_pages,
            date_field="list_date",
            transformations=(
                "exchange recovered from ts_code suffix because provider field was absent",
                "list_status recovered from the L/D/P request parameter",
                (
                    "missing D-status delist_date inferred as last daily trade_date "
                    "when present in the materialized window"
                ),
            ),
        )

    def _materialize_reference_dataset(
        self,
        directory: Path,
        dataset: str,
        pages: Sequence[RawHistoryPage],
    ) -> HistoryDatasetManifest:
        schema_by_dataset: dict[str, pa.Schema] = {
            "namechange": pa.schema(
                [
                    pa.field("ts_code", pa.string(), nullable=False),
                    pa.field("name", pa.string(), nullable=False),
                    pa.field("start_date", pa.date32(), nullable=False),
                    pa.field("end_date", pa.date32(), nullable=True),
                    pa.field("ann_date", pa.date32(), nullable=True),
                ]
            ),
            "trade_cal": pa.schema(
                [
                    pa.field("exchange", pa.string(), nullable=False),
                    pa.field("cal_date", pa.date32(), nullable=False),
                    pa.field("is_open", pa.bool_(), nullable=False),
                ]
            ),
            "fina_indicator": pa.schema(
                [
                    pa.field("ts_code", pa.string(), nullable=False),
                    pa.field("ann_date", pa.date32(), nullable=False),
                    pa.field("end_date", pa.date32(), nullable=False),
                    pa.field("netprofit_yoy", pa.decimal128(28, 8), nullable=True),
                    pa.field("debt_to_assets", pa.decimal128(28, 8), nullable=True),
                    pa.field("update_flag", pa.string(), nullable=True),
                ]
            ),
            "dividend": pa.schema(
                [
                    pa.field("ts_code", pa.string(), nullable=False),
                    pa.field("div_proc", pa.string(), nullable=True),
                    pa.field("stk_div", pa.decimal128(28, 10), nullable=True),
                    pa.field("cash_div_tax", pa.decimal128(28, 10), nullable=True),
                    pa.field("record_date", pa.date32(), nullable=True),
                    pa.field("ex_date", pa.date32(), nullable=True),
                    pa.field("pay_date", pa.date32(), nullable=True),
                ]
            ),
        }
        rows: list[dict[str, Any]] = []
        selected_pages: list[RawHistoryPage] = []
        seen: set[tuple[object, ...]] = set()
        for page in pages:
            selected_pages.append(page)
            for record in self._records(page):
                normalized = self._normalize_reference_record(dataset, record)
                if normalized is None:
                    continue
                key = tuple(normalized.values())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(normalized)
            if len(selected_pages) % 500 == 0:
                self._emit(
                    "dataset_progress",
                    dataset=dataset,
                    pages=len(selected_pages),
                    rows=len(rows),
                )
        date_field = {
            "namechange": "start_date",
            "trade_cal": "cal_date",
            "fina_indicator": "ann_date",
            "dividend": "ex_date",
        }[dataset]
        transformations = {
            "namechange": ("historical name intervals retained for PIT ST detection",),
            "trade_cal": ("SSE/SZSE calendars retained; consumers use union of open dates",),
            "fina_indicator": (
                "announcement and report period dates retained",
                "PIT consumers must require ann_date strictly before signal trade_date",
            ),
            "dividend": (
                "cash/stock dividend values retained per share",
                "rows without ex_date are retained for audit but cannot be applied as actions",
            ),
        }[dataset]
        return self._write_single_file(
            directory,
            dataset,
            schema_by_dataset[dataset],
            rows,
            selected_pages,
            date_field=date_field,
            transformations=transformations,
        )

    def _normalize_reference_record(
        self,
        dataset: str,
        record: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if dataset == "namechange":
            start = _optional_yyyymmdd(record.get("start_date"))
            if start is None:
                return None
            return {
                "ts_code": _required_text(record, "ts_code"),
                "name": _required_text(record, "name"),
                "start_date": start,
                "end_date": _optional_yyyymmdd(record.get("end_date")),
                "ann_date": _optional_yyyymmdd(record.get("ann_date")),
            }
        if dataset == "trade_cal":
            value = _yyyymmdd(record.get("cal_date"), field="cal_date")
            if not self._start <= value <= self._end:
                return None
            return {
                "exchange": _required_text(record, "exchange"),
                "cal_date": value,
                "is_open": str(record.get("is_open")) == "1",
            }
        if dataset == "fina_indicator":
            announced = _optional_yyyymmdd(record.get("ann_date"))
            period = _optional_yyyymmdd(record.get("end_date"))
            if announced is None or period is None or announced > self._end:
                return None
            return {
                "ts_code": _required_text(record, "ts_code"),
                "ann_date": announced,
                "end_date": period,
                "netprofit_yoy": _quantized_optional(
                    record.get("netprofit_yoy"), Decimal("0.00000001")
                ),
                "debt_to_assets": _quantized_optional(
                    record.get("debt_to_assets"), Decimal("0.00000001")
                ),
                "update_flag": _optional_text(record.get("update_flag")),
            }
        if dataset == "dividend":
            ex_date = _optional_yyyymmdd(record.get("ex_date"))
            pay_date = _optional_yyyymmdd(record.get("pay_date"))
            record_date = _optional_yyyymmdd(record.get("record_date"))
            if all(value is None for value in (ex_date, pay_date, record_date)):
                return None
            if ex_date is not None and ex_date > self._end:
                return None
            return {
                "ts_code": _required_text(record, "ts_code"),
                "div_proc": _optional_text(record.get("div_proc")),
                "stk_div": _quantized_optional(record.get("stk_div"), Decimal("0.0000000001")),
                "cash_div_tax": _quantized_optional(
                    record.get("cash_div_tax"), Decimal("0.0000000001")
                ),
                "record_date": record_date,
                "ex_date": ex_date,
                "pay_date": pay_date,
            }
        raise AssertionError(f"unsupported reference dataset: {dataset}")

    def _write_date_partitioned(
        self,
        directory: Path,
        dataset: str,
        schema: pa.Schema,
        page_rows: Iterable[tuple[RawHistoryPage, list[dict[str, Any]]]],
        *,
        transformations: tuple[str, ...],
    ) -> HistoryDatasetManifest:
        writers: dict[int, pq.ParquetWriter] = {}
        file_paths: dict[int, Path] = {}
        counts: dict[int, int] = defaultdict(int)
        page_ids: list[str] = []
        minimum: date | None = None
        maximum: date | None = None
        seen_by_date: dict[date, set[str]] = {}
        processed_pages = 0
        processed_rows = 0
        try:
            for page, rows in page_rows:
                processed_pages += 1
                page_ids.append(self._raw_batch_id(page))
                if not rows:
                    continue
                deduplicated: list[dict[str, Any]] = []
                for row in rows:
                    row_date = row["trade_date"]
                    code = str(row["ts_code"])
                    date_seen = seen_by_date.setdefault(row_date, set())
                    if code in date_seen:
                        raise ValueError(f"duplicate {dataset} primary key: {code}|{row_date}")
                    date_seen.add(code)
                    deduplicated.append(row)
                for year, year_rows in _group_rows_by_year(deduplicated, "trade_date").items():
                    if year not in writers:
                        partition = directory / f"year={year:04d}"
                        partition.mkdir()
                        path = partition / "data.parquet"
                        writers[year] = pq.ParquetWriter(path, schema, compression="zstd")
                        file_paths[year] = path
                    table = pa.Table.from_pylist(year_rows, schema=schema)
                    writers[year].write_table(table)
                    counts[year] += len(year_rows)
                    processed_rows += len(year_rows)
                dates = [row["trade_date"] for row in deduplicated]
                minimum = min(minimum or min(dates), min(dates))
                maximum = max(maximum or max(dates), max(dates))
                if len(seen_by_date) > 3:
                    for old_date in sorted(seen_by_date)[:-2]:
                        del seen_by_date[old_date]
                if processed_pages % 250 == 0:
                    self._emit(
                        "dataset_progress",
                        dataset=dataset,
                        pages=processed_pages,
                        rows=processed_rows,
                        trade_date=maximum.isoformat() if maximum else None,
                    )
        finally:
            for writer in writers.values():
                writer.close()
        if not counts:
            raise ValueError(f"history dataset has no rows in requested window: {dataset}")
        files = tuple(
            (
                str(path.relative_to(directory)),
                self._checksum(path),
                counts[year],
            )
            for year, path in sorted(file_paths.items())
        )
        return HistoryDatasetManifest(
            schema_version=f"aquant.history.{dataset}.v1",
            dataset=dataset,
            row_count=sum(counts.values()),
            min_date=minimum.isoformat() if minimum else None,
            max_date=maximum.isoformat() if maximum else None,
            files=files,
            source_page_count=len(page_ids),
            source_fingerprint=_fingerprint(page_ids),
            transformations=transformations,
        )

    def _write_single_file(
        self,
        directory: Path,
        dataset: str,
        schema: pa.Schema,
        rows: Sequence[dict[str, Any]],
        pages: Sequence[RawHistoryPage],
        *,
        date_field: str,
        transformations: tuple[str, ...],
    ) -> HistoryDatasetManifest:
        if not rows:
            raise ValueError(f"history dataset has no usable rows: {dataset}")
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                row.get(date_field) or date.min,
                str(row.get("ts_code") or row.get("exchange") or ""),
            ),
        )
        path = directory / "data.parquet"
        pq.write_table(pa.Table.from_pylist(sorted_rows, schema=schema), path, compression="zstd")
        values = [row[date_field] for row in sorted_rows if row.get(date_field) is not None]
        page_ids = [self._raw_batch_id(page) for page in pages]
        return HistoryDatasetManifest(
            schema_version=f"aquant.history.{dataset}.v1",
            dataset=dataset,
            row_count=len(sorted_rows),
            min_date=min(values).isoformat() if values else None,
            max_date=max(values).isoformat() if values else None,
            files=((path.name, self._checksum(path), len(sorted_rows)),),
            source_page_count=len(page_ids),
            source_fingerprint=_fingerprint(page_ids),
            transformations=transformations,
        )

    def _records(self, page: RawHistoryPage) -> Iterator[dict[str, Any]]:
        body = page.payload_path.read_bytes()
        manifest = json.loads(page.manifest_path.read_text(encoding="utf-8"))
        if self._verify_raw_checksums and hashlib.sha256(body).hexdigest() != manifest.get(
            "sha256"
        ):
            raise ValueError(f"raw payload checksum mismatch: {page.payload_path}")
        payload = json.loads(body)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"raw payload has no data object: {page.payload_path}")
        fields = data.get("fields")
        items = data.get("items")
        if not isinstance(fields, list) or not isinstance(items, list):
            raise ValueError(f"raw payload has invalid tabular data: {page.payload_path}")
        if tuple(fields) != page.fields:
            raise ValueError(f"checkpoint/raw field mismatch: {page.payload_path}")
        if len(items) != page.row_count:
            raise ValueError(f"checkpoint/raw row count mismatch: {page.payload_path}")
        for item in items:
            if not isinstance(item, list) or len(item) != len(fields):
                raise ValueError(f"raw row does not match fields: {page.payload_path}")
            yield dict(zip(fields, item, strict=True))

    def _raw_batch_id(self, page: RawHistoryPage) -> str:
        manifest = json.loads(page.manifest_path.read_text(encoding="utf-8"))
        batch_id = manifest.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError(f"raw manifest has no batch_id: {page.manifest_path}")
        return batch_id

    def _market_pages_with_warmup(
        self,
        pages: Sequence[RawHistoryPage],
        *,
        sessions: int,
    ) -> tuple[tuple[RawHistoryPage, ...], tuple[RawHistoryPage, ...]]:
        by_date: dict[date, list[RawHistoryPage]] = defaultdict(list)
        for page in pages:
            raw = page.base_params.get("trade_date")
            if raw is not None:
                by_date[_yyyymmdd(raw, field="trade_date")].append(page)
        dates = sorted(by_date)
        target_dates = [value for value in dates if self._start <= value <= self._end]
        if not target_dates:
            return (), ()
        first_index = dates.index(target_dates[0])
        warmup_dates = dates[max(0, first_index - sessions) : first_index]
        target = tuple(page for value in target_dates for page in by_date[value])
        warmup = tuple(page for value in warmup_dates for page in by_date[value])
        return target, warmup

    def _filter_market_pages(
        self,
        pages: Sequence[RawHistoryPage],
    ) -> tuple[RawHistoryPage, ...]:
        selected: list[RawHistoryPage] = []
        for page in pages:
            raw = page.base_params.get("trade_date")
            if raw is None:
                continue
            value = _yyyymmdd(raw, field="trade_date")
            if self._start <= value <= self._end:
                selected.append(page)
        return tuple(selected)

    def _load_existing(self, directory: Path) -> HistoryMaterializationResult:
        manifest_path = directory / "manifest.json"
        manifest = HistoryReleaseManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.release_id != self._release_id:
            raise ValueError("existing release identity does not match requested release")
        if (
            manifest.start_date != self._start.isoformat()
            or manifest.end_date != self._end.isoformat()
        ):
            raise ValueError("existing release date window does not match requested window")
        for dataset, checksum, _row_count in manifest.datasets:
            dataset_manifest = directory / f"dataset={dataset}" / "manifest.json"
            if self._checksum(dataset_manifest) != checksum:
                raise ValueError(f"existing history dataset manifest was modified: {dataset}")
            parsed = HistoryDatasetManifest.from_json(dataset_manifest.read_text(encoding="utf-8"))
            self._verify_dataset_files(dataset_manifest.parent, parsed)
        return HistoryMaterializationResult(directory, manifest_path, manifest)

    def _verify_dataset_files(
        self,
        directory: Path,
        manifest: HistoryDatasetManifest,
    ) -> None:
        for relative, checksum, expected_rows in manifest.files:
            path = directory / relative
            if self._checksum(path) != checksum:
                raise ValueError(f"history Parquet checksum mismatch: {path}")
            actual_rows = pq.ParquetFile(path).metadata.num_rows
            if actual_rows != expected_rows:
                raise ValueError(f"history Parquet row count mismatch: {path}")

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_new(path: Path, body: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _emit(self, event: str, **payload: object) -> None:
        if self._progress is not None:
            self._progress(event, payload)


def _group_rows_by_year(
    rows: Sequence[dict[str, Any]],
    date_field: str,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row[date_field]
        if not isinstance(value, date):
            raise ValueError(f"{date_field} must be a date")
        result[value.year].append(row)
    return result


def _fingerprint(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required text field: {field}")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid decimal field {field}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite decimal field {field}: {value!r}")
    return result


def _optional_decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    result = _decimal(value, field="optional")
    return result


def _quantize_price(value: object, *, field: str) -> Decimal:
    result = _decimal(value, field=field)
    if result <= 0:
        raise ValueError(f"non-positive price field {field}: {value!r}")
    return result.quantize(Decimal("0.000001"))


def _quantized_optional(value: object, quantum: Decimal) -> Decimal | None:
    parsed = _optional_decimal(value)
    return parsed.quantize(quantum) if parsed is not None else None


def _scaled_optional(
    value: object,
    multiplier: Decimal,
    quantum: Decimal,
) -> Decimal | None:
    parsed = _optional_decimal(value)
    return (parsed * multiplier).quantize(quantum) if parsed is not None else None


def _yyyymmdd(value: object, *, field: str) -> date:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError(f"invalid {field}: {value!r}")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def _optional_yyyymmdd(value: object) -> date | None:
    if value is None or value == "":
        return None
    return _yyyymmdd(value, field="optional_date")


def _optional_iso_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc


def _exchange_from_ts_code(value: str) -> str:
    try:
        suffix = value.strip().upper().rsplit(".", maxsplit=1)[1]
        return {"SH": "XSHG", "SZ": "XSHE", "BJ": "XBSE"}[suffix]
    except (IndexError, KeyError) as exc:
        raise ValueError(f"unsupported Tushare ts_code: {value!r}") from exc
