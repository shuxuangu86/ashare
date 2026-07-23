import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from aquant.data.daily_update import (
    BENCHMARKS,
    DailyUpdateMode,
    UpdateJob,
    UpdateJobResult,
    announcement_dates,
    build_update_plan,
    next_release_id,
    parse_daily_bar,
    parse_tushare_symbol,
    raw_batch_ids,
    select_market_dates,
    validate_update_results,
)
from aquant.data.ingestion import ArchivedPage
from aquant.domain.enums import Exchange

AS_OF = date(2026, 7, 23)


def _calendar_rows() -> tuple[dict[str, Any], ...]:
    return (
        {"cal_date": "20260717", "is_open": 1},
        {"cal_date": "20260718", "is_open": 0},
        {"cal_date": "20260720", "is_open": 1},
        {"cal_date": "20260721", "is_open": 1},
        {"cal_date": "20260722", "is_open": 1},
        {"cal_date": "20260723", "is_open": 1},
    )


def _page(
    tmp_path: Path,
    *,
    api_name: str,
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    batch_id: str = "11111111-1111-1111-1111-111111111111",
) -> ArchivedPage:
    fields = tuple(rows[0]) if rows else ("ts_code", "trade_date")
    items = [[row.get(field) for field in fields] for row in rows]
    payload = tmp_path / f"{api_name}-response.json"
    manifest = tmp_path / f"{api_name}-manifest.json"
    payload.write_text(
        json.dumps({"code": 0, "data": {"fields": fields, "items": items}}),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"batch_id": batch_id}), encoding="utf-8")
    return ArchivedPage(
        api_name=api_name,
        params=params,
        row_count=len(rows),
        fields=fields,
        payload_path=payload,
        manifest_path=manifest,
    )


class _Archiver:
    @staticmethod
    def read_items(page: ArchivedPage) -> list[dict[str, Any]]:
        payload = json.loads(page.payload_path.read_text(encoding="utf-8"))
        fields = payload["data"]["fields"]
        return [dict(zip(fields, row, strict=True)) for row in payload["data"]["items"]]


def test_close_selects_only_current_open_session() -> None:
    assert select_market_dates(
        _calendar_rows(),
        as_of_date=AS_OF,
        mode=DailyUpdateMode.CLOSE,
    ) == (AS_OF,)
    assert (
        select_market_dates(
            _calendar_rows(),
            as_of_date=date(2026, 7, 18),
            mode=DailyUpdateMode.CLOSE,
        )
        == ()
    )


def test_morning_rechecks_three_completed_sessions_and_not_current_day() -> None:
    selected = select_market_dates(
        _calendar_rows(),
        as_of_date=AS_OF,
        mode=DailyUpdateMode.MORNING,
    )

    assert selected == (
        date(2026, 7, 20),
        date(2026, 7, 21),
        date(2026, 7, 22),
    )
    assert (
        announcement_dates(
            as_of_date=AS_OF,
            market_dates=selected,
            mode=DailyUpdateMode.MORNING,
        )
        == selected
    )


def test_plan_merges_close_datasets_and_keeps_morning_recheck_explicit() -> None:
    plan = build_update_plan(market_dates=(AS_OF,), announcement_check_dates=(AS_OF,))

    assert len(plan) == 4 + 7 + len(BENCHMARKS) * 2 + 9
    assert {job.api_name for job in plan if job.group == "market" and job.required} == {
        "daily",
        "adj_factor",
        "daily_basic",
        "suspend_d",
        "stk_limit",
    }
    assert any(job.api_name == "income" and job.params == {"ann_date": "20260723"} for job in plan)
    assert next(job for job in plan if job.api_name == "income").required
    assert any(
        job.api_name == "disclosure_date" and job.params == {"actual_date": "20260723"}
        for job in plan
    )


def test_tushare_daily_bar_supports_shenzhen_shanghai_and_beijing() -> None:
    assert parse_tushare_symbol("600000.SH").exchange is Exchange.XSHG
    assert parse_tushare_symbol("000001.SZ").exchange is Exchange.XSHE
    assert parse_tushare_symbol("920001.BJ").exchange is Exchange.XBSE
    with pytest.raises(ValueError, match="unsupported"):
        parse_tushare_symbol("AAPL.US")

    bar = parse_daily_bar(
        {
            "ts_code": "920001.BJ",
            "trade_date": "20260723",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "vol": 1000,
            "amount": 10500,
        }
    )
    assert bar.symbol.canonical == "920001.XBSE"
    assert bar.trade_date == AS_OF


def test_validation_warns_for_optional_failure_but_keeps_release_eligible(
    tmp_path: Path,
) -> None:
    daily_job = UpdateJob(
        "daily",
        {"trade_date": "20260723"},
        "market",
        required=True,
        expect_rows=True,
    )
    daily_page = _page(
        tmp_path,
        api_name="daily",
        params=dict(daily_job.params),
        rows=[
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260723",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "vol": 1000,
                "amount": 10500,
            }
        ],
    )
    optional = UpdateJob("moneyflow", {"trade_date": "20260723"}, "market", required=False)

    validation = validate_update_results(
        _Archiver(),  # type: ignore[arg-type]
        (
            UpdateJobResult(daily_job, (daily_page,)),
            UpdateJobResult(optional, error="TushareApiError: permission denied"),
        ),
        market_dates=(AS_OF,),
    )

    assert validation.report.passed
    assert len(validation.bars) == 1
    assert {issue.code for issue in validation.report.issues} == {"OPTIONAL_API_FAILED"}


def test_validation_blocks_required_failure_and_wrong_session(tmp_path: Path) -> None:
    required = UpdateJob(
        "adj_factor",
        {"trade_date": "20260723"},
        "market",
        required=True,
        expect_rows=True,
    )
    wrong_date_page = _page(
        tmp_path,
        api_name="adj_factor",
        params=dict(required.params),
        rows=[{"ts_code": "000001.SZ", "trade_date": "20260722"}],
    )
    failed = UpdateJob(
        "daily",
        {"trade_date": "20260723"},
        "market",
        required=True,
        expect_rows=True,
    )

    validation = validate_update_results(
        _Archiver(),  # type: ignore[arg-type]
        (
            UpdateJobResult(required, (wrong_date_page,)),
            UpdateJobResult(failed, error="network unavailable"),
        ),
        market_dates=(AS_OF,),
    )

    assert not validation.report.passed
    assert {issue.code for issue in validation.report.issues} == {
        "REQUIRED_API_FAILED",
        "UNEXPECTED_TRADE_DATE",
    }


def test_release_sequence_and_raw_lineage_are_deterministic(tmp_path: Path) -> None:
    releases = tmp_path / "releases"
    (releases / "cn_equity_20260723_001").mkdir(parents=True)
    (releases / "cn_equity_20260723_003").mkdir()
    (releases / "unrelated").mkdir()

    assert str(next_release_id(releases, AS_OF)) == "cn_equity_20260723_004"

    page = _page(tmp_path, api_name="daily", params={}, rows=[])
    assert raw_batch_ids((page, page)) == ("11111111-1111-1111-1111-111111111111",)


def test_job_invariants_reject_blank_identity_and_mixed_failure_result() -> None:
    with pytest.raises(ValueError, match="identity"):
        UpdateJob("", {}, "market", required=True)
    page = ArchivedPage("daily", {}, 0, (), Path("x"), Path("y"))
    job = UpdateJob("daily", {}, "market", required=True)
    with pytest.raises(ValueError, match="failed job"):
        UpdateJobResult(job, (page,), error="failed")
