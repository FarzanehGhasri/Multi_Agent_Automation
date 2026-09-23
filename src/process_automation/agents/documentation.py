"""Node 1 - Documentation Agent (Agentic RAG).

Role: quality-assurance and process expert.
Task: given a process name, search the vector store built from the company's
flowcharts and written procedures, and extract the SLA and the approved steps.
Output: the ``should-be`` picture (``standard_sop`` + ``standard_profile``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..interfaces import DocumentRetriever, SopParser
from ..state import ProcessAnalysisState
from .base import AgentNode


class DocumentationAgent(AgentNode):
    """Retrieves and structures the approved standard for a process."""

    name = "documentation_agent"

    def __init__(
        self,
        retriever: DocumentRetriever,
        parser: SopParser,
        *,
        top_k: int = 4,
    ) -> None:
        # Both collaborators are *ports*: this agent works unchanged with an
        # in-memory TF-IDF store or with a hosted FAISS/Chroma index.
        super().__init__()
        self._retriever = retriever
        self._parser = parser
        self._top_k = top_k

    def run(self, state: ProcessAnalysisState) -> Mapping[str, Any]:
        query = state["process_name"]
        trace = [self.log(f"RAG lookup for: '{query}' (top_k={self._top_k})")]

        chunks = self._retriever.retrieve(query, top_k=self._top_k)
        for chunk in chunks:
            trace.append(
                self.log(
                    f"  hit score={chunk.score:.3f} doc={chunk.document_id} "
                    f"section='{chunk.metadata.get('section', '?')}'"
                )
            )
        if not chunks:
            trace.append(self.log("  no chunk passed the relevance threshold"))

        standard = self._parser.parse(query, chunks)
        document_id = chunks[0].document_id if chunks else ""

        if standard.sla_hours > 0:
            trace.append(
                self.log(
                    f"extracted SLA={standard.sla_hours:g}h and "
                    f"{len(standard.steps)} approved step(s) from {document_id}"
                )
            )
            controls = standard.mandatory_controls
            if controls:
                trace.append(
                    self.log(
                        "mandatory control(s) found: "
                        + ", ".join(
                            f"{step.title}"
                            f"{' / ' + step.control_reference if step.control_reference else ''}"
                            for step in controls
                        )
                    )
                )
        else:
            trace.append(
                self.log("no approved standard could be extracted for this process")
            )

        return {
            # The plain-text field required by the project brief...
            "standard_sop": standard.to_text(),
            # ...plus the structured mirror Node 3 needs to compute numbers.
            "standard_profile": standard.to_payload(),
            "resolved_process": standard.process_name or query,
            "sop_document_id": document_id,
            "retrieved_documents": [
                {
                    "document_id": chunk.document_id,
                    "section": chunk.metadata.get("section", ""),
                    "score": chunk.score,
                    "excerpt": _excerpt(chunk.text),
                }
                for chunk in chunks
            ],
            "trace": trace,
        }


def _excerpt(text: str, limit: int = 220) -> str:
    """Shorten a chunk for the audit trail without losing its beginning."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


__all__ = ["DocumentationAgent"]
