from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    title: str
    source_uri: str
    content: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.document_id, self.title, self.source_uri, self.content)
        ):
            raise ValueError("knowledge document fields must not be blank")


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    document_id: str
    title: str
    source_uri: str
    excerpt: str
    score: int


class KnowledgeStore:
    """Deterministic lexical baseline; pgvector can replace retrieval without changing citations."""

    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}

    def add(self, document: KnowledgeDocument) -> None:
        if document.document_id in self._documents:
            raise ValueError(f"knowledge document already exists: {document.document_id}")
        self._documents[document.document_id] = document

    def search(self, query: str, *, limit: int = 5) -> tuple[KnowledgeHit, ...]:
        terms = {term.lower() for term in query.split() if term.strip()}
        if not terms or limit <= 0:
            raise ValueError("knowledge search query/limit is invalid")
        hits: list[KnowledgeHit] = []
        for document in self._documents.values():
            content = document.content.lower()
            score = sum(content.count(term) for term in terms)
            if score:
                hits.append(
                    KnowledgeHit(
                        document.document_id,
                        document.title,
                        document.source_uri,
                        document.content[:240],
                        score,
                    )
                )
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.document_id))[:limit])
