from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from aquant.data.history.tushare import TushareHistoryCatalog
from aquant.domain.data_release import DataReleaseId
from aquant.factors.data_loader import StandardPITFactorLoader
from aquant.regime.panel import MarketStatePanelInput
from aquant.regime.universe import build_dynamic_universes, build_snapshot_membership

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_INDEX_CODES = {
    "HS300": ("000300.SH", 280),
    "CSI500": ("000905.SH", 450),
    "CSI1000": ("000852.SH", 800),
}
_INDEX_LEVEL_CODES = {
    "HS300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
    "CHINEXT": "399006.SZ",
}


def load_market_state_panel(
    release_directory: Path,
    raw_state_path: Path,
    *,
    start_date: date,
    end_date: date,
    max_workers: int = 2,
) -> MarketStatePanelInput:
    fields = (
        "close",
        "amount",
        "total_market_cap",
        "float_market_cap",
        "pb",
        "turnover_rate",
        "dividend_yield",
        "netprofit_yoy",
        "debt_to_assets",
        "up_limit",
        "down_limit",
    )
    loader = StandardPITFactorLoader(
        release_directory,
        dtype=np.float32,
        duckdb_memory_limit="2GB",
        duckdb_threads=max_workers,
    )
    panel = loader.load(
        fields=fields,
        start_date=start_date,
        end_date=end_date,
        as_of_time=datetime.combine(end_date, time(23, 59), _SHANGHAI),
        universe_id="all_a_share",
        data_release_id=DataReleaseId(f"cn_equity_{end_date:%Y%m%d}_001"),
    )
    arrays = panel.fields
    universes = dict(
        build_dynamic_universes(
            panel.trade_dates,
            close=arrays["close"],
            total_market_cap=arrays["total_market_cap"],
            pb=arrays["pb"],
            turnover_rate=arrays["turnover_rate"],
            dividend_yield=arrays["dividend_yield"],
            netprofit_yoy=arrays["netprofit_yoy"],
        )
    )
    catalog = TushareHistoryCatalog(raw_state_path)
    snapshots = _index_snapshots(catalog, start_date=start_date, end_date=end_date)
    for name, (code, minimum) in _INDEX_CODES.items():
        universes[name] = build_snapshot_membership(
            panel.trade_dates,
            panel.ts_codes,
            snapshots.get(code, {}),
            minimum_constituents=minimum,
        )
    levels = _index_levels(
        catalog,
        panel.trade_dates,
        start_date=start_date,
        end_date=end_date,
    )
    return MarketStatePanelInput(
        dates=panel.trade_dates,
        ts_codes=panel.ts_codes,
        close=arrays["close"],
        amount=arrays["amount"],
        total_market_cap=arrays["total_market_cap"],
        float_market_cap=arrays["float_market_cap"],
        pb=arrays["pb"],
        turnover_rate=arrays["turnover_rate"],
        dividend_yield=arrays["dividend_yield"],
        netprofit_yoy=arrays["netprofit_yoy"],
        debt_to_assets=arrays["debt_to_assets"],
        up_limit=arrays["up_limit"],
        down_limit=arrays["down_limit"],
        universes=universes,
        index_levels=levels,
    )


def _index_snapshots(
    catalog: TushareHistoryCatalog,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, dict[date, tuple[str, ...]]]:
    grouped: dict[str, dict[date, set[str]]] = defaultdict(lambda: defaultdict(set))
    for page in catalog.pages("index_weight"):
        code = str(page.params.get("index_code", "")).upper()
        if code not in {value[0] for value in _INDEX_CODES.values()}:
            continue
        for row in _page_rows(page.payload_path):
            trade_date = _yyyymmdd(row["trade_date"])
            if trade_date <= end_date:
                grouped[code][trade_date].add(str(row["con_code"]).upper())
    return {
        code: {trade_date: tuple(sorted(codes)) for trade_date, codes in sorted(snapshots.items())}
        for code, snapshots in grouped.items()
    }


def _index_levels(
    catalog: TushareHistoryCatalog,
    dates: tuple[date, ...],
    *,
    start_date: date,
    end_date: date,
) -> dict[str, npt.NDArray[np.float64]]:
    by_code: dict[str, dict[date, float]] = defaultdict(dict)
    wanted = set(_INDEX_LEVEL_CODES.values())
    for page in catalog.pages("index_daily"):
        code = str(page.params.get("ts_code", "")).upper()
        if code not in wanted:
            continue
        for row in _page_rows(page.payload_path):
            trade_date = _yyyymmdd(row["trade_date"])
            if start_date <= trade_date <= end_date:
                by_code[code][trade_date] = float(str(row["close"]))
    output: dict[str, npt.NDArray[np.float64]] = {}
    for name, code in _INDEX_LEVEL_CODES.items():
        values = np.asarray([by_code[code].get(value, np.nan) for value in dates])
        if np.count_nonzero(np.isfinite(values)) >= 20:
            output[name] = values
    return output


def _page_rows(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data") or {}
    fields = data.get("fields") or []
    return tuple(dict(zip(fields, values, strict=True)) for values in data.get("items") or [])


def _yyyymmdd(value: object) -> date:
    return datetime.strptime(str(value), "%Y%m%d").date()
