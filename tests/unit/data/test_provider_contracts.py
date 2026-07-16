from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aquant.data.providers import (
    DatasetRequest,
    LocalFileProvider,
    MarketDataProvider,
    ProviderResponse,
)


def _request(**overrides: object) -> DatasetRequest:
    values: dict[str, object] = {
        "dataset": "instruments",
        "params": {"asof": "2026-07-16", "fields": ["symbol", "name"]},
        "requested_at": datetime(2026, 7, 16, 8, tzinfo=UTC),
    }
    values.update(overrides)
    return DatasetRequest.create(**values)  # type: ignore[arg-type]


def test_dataset_request_canonicalizes_parameters() -> None:
    request = _request()

    assert request.dataset == "instruments"
    assert request.params_json == '{"asof":"2026-07-16","fields":["symbol","name"]}'
    assert request.params == {"asof": "2026-07-16", "fields": ["symbol", "name"]}


def test_dataset_request_returns_fresh_parameter_copy() -> None:
    request = _request()
    first = request.params
    first["asof"] = "changed"

    assert request.params["asof"] == "2026-07-16"


@pytest.mark.parametrize(
    ("params_json", "error"),
    [("[]", "JSON object"), ("not-json", "Expecting value")],
)
def test_dataset_request_rejects_invalid_json(params_json: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        DatasetRequest("instruments", params_json, datetime(2026, 7, 16, tzinfo=UTC))


def test_dataset_request_rejects_blank_dataset_and_naive_time() -> None:
    with pytest.raises(ValueError, match="dataset"):
        _request(dataset=" ")
    with pytest.raises(ValueError, match="timezone"):
        _request(requested_at=datetime(2026, 7, 16))


def test_local_file_provider_returns_original_bytes(tmp_path: Path) -> None:
    fixture = tmp_path / "instruments.json"
    fixture.write_bytes(b'[{"symbol":"600000.XSHG"}]')
    received_at = datetime(2026, 7, 16, 8, 0, 1, tzinfo=UTC)
    provider = LocalFileProvider(
        {"instruments": fixture},
        name=" Fixture ",
        version=" v1 ",
        clock=lambda: received_at,
    )

    response = provider.fetch(_request())

    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "fixture"
    assert provider.version == "v1"
    assert response.body == fixture.read_bytes()
    assert response.received_at == received_at
    assert len(response.source_request_id) == 64


def test_local_provider_request_id_is_deterministic(tmp_path: Path) -> None:
    fixture = tmp_path / "data.json"
    fixture.write_text("[]", encoding="utf-8")
    provider = LocalFileProvider({"instruments": fixture})
    request = _request()

    assert provider.fetch(request).source_request_id == provider.fetch(request).source_request_id


def test_local_provider_rejects_unknown_dataset(tmp_path: Path) -> None:
    provider = LocalFileProvider({})
    with pytest.raises(KeyError, match="does not contain"):
        provider.fetch(_request())


def test_local_provider_rejects_blank_identity() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        LocalFileProvider({}, name=" ")


def test_provider_response_rejects_invalid_metadata() -> None:
    request = _request()
    valid = {
        "provider": "fixture",
        "provider_version": "1",
        "request": request,
        "received_at": request.requested_at,
        "body": b"[]",
        "media_type": "application/json",
        "source_request_id": "request-1",
    }
    with pytest.raises(ValueError, match="provider"):
        ProviderResponse(**(valid | {"provider": ""}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_request_id"):
        ProviderResponse(**(valid | {"source_request_id": ""}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="negative"):
        ProviderResponse(**(valid | {"record_count": -1}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="earlier"):
        ProviderResponse(**(valid | {"received_at": request.requested_at - timedelta(seconds=1)}))  # type: ignore[arg-type]
