from datetime import UTC, datetime, timedelta

import pytest

from aquant.data.providers import AkshareProvider, DatasetRequest, MarketDataProvider


class FakeFrame:
    def __init__(self, body: str, *, rows: int = 1) -> None:
        self.body = body
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def __len__(self) -> int:
        return self.rows

    def to_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.body


class FakeAkshare:
    def __init__(self, frame: FakeFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, object]] = []

    def stock_zh_a_hist(self, **kwargs: object) -> FakeFrame:
        self.calls.append(kwargs)
        return self.frame


def _request(**params: object) -> DatasetRequest:
    now = datetime(2026, 7, 16, 8, tzinfo=UTC)
    values: dict[str, object] = {
        "symbol": "600000.XSHG",
        "start_date": "20260701",
        "end_date": "20260716",
    }
    values.update(params)
    return DatasetRequest.create(dataset="daily_bars", params=values, requested_at=now)


def test_fetches_unadjusted_daily_bars_and_archives_returned_table() -> None:
    body = '{"schema":{},"data":[{"股票代码":"600000"}]}'
    frame = FakeFrame(body)
    backend = FakeAkshare(frame)
    completed_at = _request().requested_at + timedelta(seconds=2)
    provider = AkshareProvider(
        backend=backend,
        backend_version="1.18.0",
        timeout=12,
        clock=lambda: completed_at,
    )

    response = provider.fetch(_request())

    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "akshare"
    assert provider.version == "akshare-1.18.0"
    assert backend.calls == [
        {
            "symbol": "600000",
            "period": "daily",
            "start_date": "20260701",
            "end_date": "20260716",
            "adjust": "",
            "timeout": 12,
        }
    ]
    assert frame.calls == [{"orient": "table", "date_format": "iso", "force_ascii": False}]
    assert response.body == body.encode()
    assert response.record_count == 1
    assert response.media_type == "application/vnd.aquant.tabular+json"
    assert len(response.source_request_id) == 64


def test_source_request_id_is_deterministic() -> None:
    frame = FakeFrame('{"data":[]} ', rows=0)
    backend = FakeAkshare(frame)
    completed_at = _request().requested_at + timedelta(seconds=1)
    provider = AkshareProvider(backend=backend, clock=lambda: completed_at)

    first = provider.fetch(_request())
    second = provider.fetch(_request())

    assert first.source_request_id == second.source_request_id


@pytest.mark.parametrize(
    ("dataset", "params", "error"),
    [
        ("financials", {}, KeyError),
        ("daily_bars", {"symbol": ""}, ValueError),
        ("daily_bars", {"start_date": 20260701}, ValueError),
        ("daily_bars", {"end_date": None}, ValueError),
    ],
)
def test_rejects_unsupported_or_incomplete_requests(
    dataset: str, params: dict[str, object], error: type[Exception]
) -> None:
    request = _request(**params)
    if dataset != "daily_bars":
        request = DatasetRequest.create(
            dataset=dataset,
            params=request.params,
            requested_at=request.requested_at,
        )
    provider = AkshareProvider(
        backend=FakeAkshare(FakeFrame('{"data":[]}')),
        clock=lambda: request.requested_at,
    )

    with pytest.raises(error):
        provider.fetch(request)


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        AkshareProvider(backend=FakeAkshare(FakeFrame("{}")), timeout=0)
