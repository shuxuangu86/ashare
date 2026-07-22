import pytest
from fastapi.testclient import TestClient

from aquant.api import create_app
from aquant.knowledge import KnowledgeDocument, KnowledgeStore
from aquant.monitoring import LiveReadiness


def test_api_health_version_and_fail_closed_readiness() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {
        "status": "healthy",
        "service": "aquant",
    }
    assert client.get("/version").json() == {
        "version": "0.2.0",
        "live_default": "disabled",
    }
    assert client.get("/ready").json() == {
        "ready": False,
        "blockers": ["READINESS_PROVIDER_NOT_CONFIGURED"],
    }


def test_api_exposes_supplied_readiness_without_mutating_it() -> None:
    expected = LiveReadiness(True, ())
    client = TestClient(create_app(lambda: expected))

    assert client.get("/ready").json() == {"ready": True, "blockers": []}
    assert expected == LiveReadiness(True, ())


def test_knowledge_search_is_deterministic_and_keeps_source_citation() -> None:
    store = KnowledgeStore()
    store.add(
        KnowledgeDocument(
            "factor-momentum",
            "Momentum factor card",
            "docs/factors/momentum.md",
            "Momentum uses only point-in-time prices. Momentum turnover is monitored.",
        )
    )
    store.add(
        KnowledgeDocument(
            "pit-policy",
            "Point-in-time policy",
            "docs/data/point-in-time.md",
            "Point-in-time queries enforce available_at before every research timestamp.",
        )
    )

    hits = store.search("momentum point-in-time")

    assert [hit.document_id for hit in hits] == ["factor-momentum", "pit-policy"]
    assert hits[0].source_uri == "docs/factors/momentum.md"
    assert hits[0].score == 3


def test_knowledge_store_rejects_duplicate_and_invalid_search() -> None:
    store = KnowledgeStore()
    document = KnowledgeDocument("doc", "Title", "docs/source.md", "audited content")
    store.add(document)

    with pytest.raises(ValueError, match="already exists"):
        store.add(document)
    with pytest.raises(ValueError, match="query/limit"):
        store.search("   ")
    with pytest.raises(ValueError, match="query/limit"):
        store.search("content", limit=0)


@pytest.mark.parametrize("field", ["document_id", "title", "source_uri", "content"])
def test_knowledge_document_rejects_blank_fields(field: str) -> None:
    values = {
        "document_id": "doc",
        "title": "Title",
        "source_uri": "docs/source.md",
        "content": "content",
    }
    values[field] = " "

    with pytest.raises(ValueError, match="must not be blank"):
        KnowledgeDocument(**values)
