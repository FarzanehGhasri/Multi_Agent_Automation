"""Vector search and SOP extraction - the machinery behind Node 1 (RAG).

Pipeline: ``KnowledgeBase`` -> chunks -> embeddings -> cosine search ->
:class:`StructuredSopParser` -> :class:`StandardProcess`.

SOLID notes
-----------
* SRP - :class:`InMemoryVectorStore` only *finds* text; :class:`StructuredSopParser`
        only *interprets* it. Neither knows about agents or graphs.
* DIP - the store depends on the :class:`EmbeddingModel` port, so the same
        class works with the offline TF-IDF embedder or a hosted one.
* OCP - swapping in FAISS/Chroma means writing another ``DocumentRetriever``;
        no existing class changes.
"""

from __future__ import annotations

from typing import Sequence

from ..domain.models import RetrievedChunk, StandardProcess
from ..interfaces import EmbeddingModel
from .embeddings import cosine_similarity
from .knowledge_base import KnowledgeBase, TextChunk


class InMemoryVectorStore:
    """A small, transparent vector store: embeddings + cosine similarity.

    Purpose-built for a corpus of a few dozen SOP chunks. It implements the
    :class:`~process_automation.interfaces.DocumentRetriever` port, which is
    the only thing the RAG agent ever sees - replacing it with FAISS, Chroma
    or pgvector requires no change anywhere else.
    """

    def __init__(self, embeddings: EmbeddingModel, *, min_score: float = 0.0) -> None:
        self._embeddings = embeddings
        self._min_score = min_score
        self._chunks: list[TextChunk] = []
        self._vectors: list[list[float]] = []

    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: KnowledgeBase,
        embeddings: EmbeddingModel,
        *,
        min_score: float = 0.0,
    ) -> "InMemoryVectorStore":
        """Convenience constructor: build the index in one call."""
        store = cls(embeddings, min_score=min_score)
        store.index(knowledge_base.all_chunks())
        return store

    def index(self, chunks: Sequence[TextChunk]) -> "InMemoryVectorStore":
        """Embed and store ``chunks`` (idempotent per call, additive overall)."""
        if not chunks:
            return self
        self._chunks.extend(chunks)
        # Embedding the *whole* corpus at once lets a TF-IDF embedder learn its
        # IDF statistics; batch APIs of hosted providers also prefer this.
        self._vectors = self._embeddings.embed_documents([chunk.text for chunk in self._chunks])
        return self

    @property
    def size(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        if not self._chunks:
            return []
        query_vector = self._embeddings.embed_query(query)
        scored = [
            (cosine_similarity(query_vector, vector), chunk)
            for vector, chunk in zip(self._vectors, self._chunks)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                document_id=chunk.document_id,
                text=chunk.text,
                score=round(score, 4),
                metadata={"chunk_id": chunk.chunk_id, "section": chunk.section},
            )
            for score, chunk in scored[:top_k]
            if score >= self._min_score
        ]


class StructuredSopParser:
    """Turns retrieved chunks into the structured *should-be* standard.

    Retrieval alone gives loose text; a process analysis needs numbers. This
    parser therefore performs the "agentic" second hop of the RAG pipeline:

    1. score every *document* by the chunks retrieved from it;
    2. select the winning document;
    3. read its already-parsed SLA and step table from the knowledge base.

    Doing the voting on documents rather than on the single best chunk makes
    the extraction robust when, say, only the policy-notes section matched.
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    def parse(
        self, process_name: str, chunks: Sequence[RetrievedChunk]
    ) -> StandardProcess:
        document_id = self.select_document(chunks)
        if document_id is None:
            # Nothing relevant was found: return an explicitly empty standard
            # rather than inventing one. The engineer node will then report
            # "no documented standard" instead of comparing against a guess.
            return StandardProcess(process_name=process_name, sla_hours=0.0)

        document = self._knowledge_base.get(document_id)
        if document is None:  # defensive: index and corpus out of sync
            return StandardProcess(process_name=process_name, sla_hours=0.0)
        return document.to_standard_process()

    @staticmethod
    def select_document(chunks: Sequence[RetrievedChunk]) -> str | None:
        """Vote on the source document using the retrieval scores."""
        if not chunks:
            return None
        totals: dict[str, float] = {}
        for chunk in chunks:
            totals[chunk.document_id] = totals.get(chunk.document_id, 0.0) + chunk.score
        return max(totals, key=totals.__getitem__)


__all__ = ["InMemoryVectorStore", "StructuredSopParser"]
