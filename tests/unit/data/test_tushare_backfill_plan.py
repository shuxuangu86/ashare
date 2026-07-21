import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aquant.data.ingestion import ArchivedPage


def _load_backfill_script() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts/tushare_backfill.py"
    spec = importlib.util.spec_from_file_location("tushare_backfill_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Tushare backfill script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tushare_backfill = _load_backfill_script()


def _page(api_name: str, rows: int = 1) -> ArchivedPage:
    return ArchivedPage(
        api_name=api_name,
        params={},
        row_count=rows,
        fields=(),
        payload_path=Path("response.json"),
        manifest_path=Path("manifest.json"),
    )


class _State:
    def counts(self) -> dict[str, int]:
        return {"COMPLETED": 1}


class _Archiver:
    state = _State()

    def read_items(self, page: ArchivedPage) -> list[dict[str, Any]]:
        if page.api_name == "stock_basic":
            return [{"ts_code": "000001.SZ"}]
        if page.api_name == "trade_cal":
            return [
                {"cal_date": "20260102", "is_open": 1},
                {"cal_date": "20260103", "is_open": 0},
            ]
        return []


def test_reference_collects_unique_open_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_archive(
        _archiver: object,
        api_name: str,
        _params: dict[str, Any],
        **_kwargs: object,
    ) -> list[ArchivedPage]:
        return [_page(api_name)]

    monkeypatch.setattr(tushare_backfill, "_archive_pages", fake_archive)

    stocks, sessions = tushare_backfill._reference(
        _Archiver(),
        "20260101",
        "20261231",
    )

    assert stocks == ("000001.SZ",)
    assert sessions == ("20260102",)


def test_market_partitions_requests_by_trade_date_not_global_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_archive(
        _archiver: object,
        api_name: str,
        params: dict[str, Any],
        **_kwargs: object,
    ) -> list[ArchivedPage]:
        calls.append((api_name, params))
        return [_page(api_name, 10)]

    monkeypatch.setattr(tushare_backfill, "_archive_pages", fake_archive)
    monkeypatch.setattr(tushare_backfill, "_log", lambda *_args, **_kwargs: None)

    tushare_backfill._market(
        _Archiver(),
        "20260102",
        "20260105",
        ("20260101", "20260102", "20260105", "20260106"),
        3,
    )

    assert len(calls) == len(tushare_backfill.MARKET_APIS) * 2
    assert {tuple(params) for _, params in calls} == {("trade_date",)}
    assert {params["trade_date"] for _, params in calls} == {"20260102", "20260105"}


def test_financials_partition_requests_by_stock_not_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_archive(
        _archiver: object,
        api_name: str,
        params: dict[str, Any],
        **_kwargs: object,
    ) -> list[ArchivedPage]:
        calls.append((api_name, params))
        return [_page(api_name, 10)]

    monkeypatch.setattr(tushare_backfill, "_archive_pages", fake_archive)
    monkeypatch.setattr(tushare_backfill, "_log", lambda *_args, **_kwargs: None)

    tushare_backfill._financials(
        _Archiver(),
        ("000001.SZ", "600000.SH"),
        2,
    )

    assert len(calls) == len(tushare_backfill.FINANCIAL_APIS) * 2
    assert {tuple(params) for _, params in calls} == {("ts_code",)}
    assert {params["ts_code"] for _, params in calls} == {"000001.SZ", "600000.SH"}


def test_financials_require_stock_universe() -> None:
    with pytest.raises(RuntimeError, match="at least one stock code"):
        tushare_backfill._financials(_Archiver(), (), 2)


def test_progress_checkpoint_contains_finish_time_and_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tushare_backfill.time, "monotonic", lambda: 120.0)

    timing = tushare_backfill._progress_timing(started_at=60.0, completed=100, total=200)

    assert timing["elapsed_minutes"] == 1.0
    assert timing["sessions_per_minute"] == 100.0
    assert "checkpoint_finished_at" in timing
    assert "estimated_api_finish_at" in timing
