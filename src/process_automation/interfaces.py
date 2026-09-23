"""Ports (abstract interfaces) of the system - the heart of the DIP.

High-level policy (the agents) depends *only* on the small protocols declared
here. Low-level detail (vector stores, LangChain tools, chat models, console
prompts) implements them in :mod:`process_automation.infrastructure`.

Every protocol is intentionally tiny (Interface Segregation Principle): an
implementer of :class:`MetricsProvider` is never forced to know about
embeddings, and a vector store never has to know what an LLM is.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .domain.models import (
    ActionPlan,
    ApprovalDecision,
    Constraint,
    Gap,
    ProcessMetrics,
    RetrievedChunk,
    StandardProcess,
)


# ---------------------------------------------------------------------------
# Retrieval side (Node 1 - Agentic RAG)
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turns text into vectors. Deliberately provider-agnostic."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a corpus (may also fit internal statistics such as IDF)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


@runtime_checkable
class DocumentRetriever(Protocol):
    """Semantic search over the company's documentation."""

    def retrieve(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant chunks for ``query``."""


@runtime_checkable
class SopParser(Protocol):
    """Extracts the structured standard (SLA + steps) from raw chunks."""

    def parse(
        self, process_name: str, chunks: Sequence[RetrievedChunk]
    ) -> StandardProcess:
        """Build the ``should-be`` picture out of retrieved documentation."""


# ---------------------------------------------------------------------------
# Measurement side (Node 2 - Tool Calling)
# ---------------------------------------------------------------------------


@runtime_checkable
class MetricsProvider(Protocol):
    """Reads the real performance of a process from the operational systems."""

    def fetch(self, process_name: str) -> ProcessMetrics:
        """Return the measured metrics for ``process_name``."""

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Names of the underlying tools, for the execution trace."""


# ---------------------------------------------------------------------------
# Reasoning side (Node 3 - Process Engineer)
# ---------------------------------------------------------------------------


@runtime_checkable
class LanguageModel(Protocol):
    """Minimal chat-completion port, so no agent imports a vendor SDK."""

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's answer to ``prompt``."""

    @property
    def name(self) -> str:
        """Human-readable identifier shown in the execution trace."""


@runtime_checkable
class SolutionComposer(Protocol):
    """Writes (and rewrites) the Action Plan."""

    def compose(
        self,
        standard: StandardProcess,
        metrics: ProcessMetrics,
        gaps: Sequence[Gap],
        constraints: Sequence[Constraint] = (),
        revision: int = 1,
    ) -> ActionPlan:
        """Produce the plan that closes ``gaps``, honouring ``constraints``."""


# ---------------------------------------------------------------------------
# Control side (Node 4 - Human in the Loop)
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalGateway(Protocol):
    """Asks the systems manager whether the current plan may be executed."""

    def review(self, plan: ActionPlan, *, revision: int) -> ApprovalDecision:
        """Return the reviewer's decision for ``plan``."""


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphVisualizer(Protocol):
    """Renders the compiled LangGraph for the report."""

    def render(self, compiled_graph: Any, output_path: str | None = None) -> Mapping[str, Any]:
        """Render the graph and describe what was produced."""


__all__ = [
    "ApprovalGateway",
    "DocumentRetriever",
    "EmbeddingModel",
    "GraphVisualizer",
    "LanguageModel",
    "MetricsProvider",
    "SolutionComposer",
    "SopParser",
]
