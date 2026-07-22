from datetime import UTC, datetime

import pytest

from aquant.knowledge import (
    KnowledgeCandidate,
    KnowledgeLifecycleRegistry,
    KnowledgeLifecycleStatus,
    KnowledgeSourceKind,
)

NOW = datetime(2026, 7, 22, tzinfo=UTC)
EVIDENCE_HASH = "a" * 64


def _candidate(
    candidate_id: str = "paper-001",
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.ACADEMIC_PAPER,
) -> KnowledgeCandidate:
    return KnowledgeCandidate(
        candidate_id,
        "Candidate title",
        source_kind,
        "https://example.test/source",
        "A falsifiable investment hypothesis",
        "A-share daily strategies",
        NOW,
    )


def test_knowledge_funnel_requires_reproduction_shadow_and_human_approval() -> None:
    registry = KnowledgeLifecycleRegistry()
    registry.discover(_candidate())

    with pytest.raises(ValueError, match="evidence SHA-256"):
        registry.transition("paper-001", KnowledgeLifecycleStatus.REPRODUCED)
    reproduced = registry.transition(
        "paper-001",
        KnowledgeLifecycleStatus.REPRODUCED,
        evidence_hash=EVIDENCE_HASH,
    )
    assert reproduced.status is KnowledgeLifecycleStatus.REPRODUCED
    registry.transition("paper-001", KnowledgeLifecycleStatus.VALIDATED)
    registry.transition("paper-001", KnowledgeLifecycleStatus.SHADOW)

    with pytest.raises(ValueError, match="human confirmation"):
        registry.transition(
            "paper-001",
            KnowledgeLifecycleStatus.APPROVED,
            approved_by="researcher",
        )
    approved = registry.transition(
        "paper-001",
        KnowledgeLifecycleStatus.APPROVED,
        approved_by="researcher",
        manual_confirmation=True,
    )
    production = registry.transition("paper-001", KnowledgeLifecycleStatus.PRODUCTION)

    assert approved.approved_by == "researcher"
    assert production.status is KnowledgeLifecycleStatus.PRODUCTION
    assert registry.get("paper-001") == production


def test_rejected_candidate_enters_failure_library_with_reason() -> None:
    registry = KnowledgeLifecycleRegistry()
    registry.discover(_candidate("failed-github", KnowledgeSourceKind.GITHUB_TOOL))

    with pytest.raises(ValueError, match="failure reason"):
        registry.transition("failed-github", KnowledgeLifecycleStatus.REJECTED)
    rejected = registry.transition(
        "failed-github",
        KnowledgeLifecycleStatus.REJECTED,
        failure_reason="PIT reproduction exposed future data leakage",
    )

    assert registry.failure_library() == (rejected,)
    assert "future data leakage" in (rejected.failure_reason or "")


def test_explicit_failure_source_is_retained_even_before_rejection() -> None:
    registry = KnowledgeLifecycleRegistry()
    failure = _candidate("known-failure", KnowledgeSourceKind.FAILURE_CASE)
    registry.discover(failure)

    assert registry.failure_library() == (failure,)


def test_knowledge_registry_rejects_duplicates_invalid_transitions_and_hashes() -> None:
    registry = KnowledgeLifecycleRegistry()
    candidate = _candidate()
    registry.discover(candidate)

    with pytest.raises(ValueError, match="already exists"):
        registry.discover(candidate)
    with pytest.raises(ValueError, match="invalid knowledge lifecycle"):
        registry.transition("paper-001", KnowledgeLifecycleStatus.VALIDATED)
    with pytest.raises(ValueError, match="SHA-256"):
        KnowledgeCandidate(
            "bad-hash",
            "Title",
            KnowledgeSourceKind.REGULATION,
            "https://example.test",
            "hypothesis",
            "applicability",
            NOW,
            evidence_hash="invalid",
        )
    with pytest.raises(ValueError, match="promoted knowledge"):
        KnowledgeCandidate(
            "missing-promotion-hash",
            "Title",
            KnowledgeSourceKind.REGULATION,
            "https://example.test",
            "hypothesis",
            "applicability",
            NOW,
            status=KnowledgeLifecycleStatus.REPRODUCED,
        )
    with pytest.raises(ValueError, match="start at DISCOVERED"):
        registry.discover(
            KnowledgeCandidate(
                "already-reproduced",
                "Title",
                KnowledgeSourceKind.BROKER_RESEARCH,
                "https://example.test",
                "hypothesis",
                "applicability",
                NOW,
                status=KnowledgeLifecycleStatus.REPRODUCED,
                evidence_hash=EVIDENCE_HASH,
            )
        )
