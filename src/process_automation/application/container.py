"""Composition root - the only place where concrete classes are chosen.

Every other module in this project depends on abstractions. Here, and only
here, those abstractions are bound to implementations: TF-IDF embeddings or a
hosted embedder, a JSON-backed ServiceDesk or a real API, a rule-based
composer or one decorated with a language model.

That is the practical pay-off of the Dependency Inversion Principle: to run
the whole system against a different stack you edit this file, and nothing
else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..agents import (
    DiagnosticAgent,
    DocumentationAgent,
    HumanReviewNode,
    ProcessEngineerAgent,
)
from ..config import Settings
from ..domain.feedback import RuleBasedFeedbackInterpreter
from ..domain.gap_analysis import GapAnalyzer
from ..domain.remediation import RemediationRegistry
from ..graph import WorkflowGraphBuilder
from ..infrastructure.approval import AutoApproveGateway
from ..infrastructure.composition import LlmNarrativeComposer, RuleBasedSolutionComposer
from ..infrastructure.embeddings import HashingTfidfEmbeddings
from ..infrastructure.knowledge_base import KnowledgeBase, MarkdownKnowledgeBaseLoader
from ..infrastructure.llm import build_language_model
from ..infrastructure.rendering import MermaidGraphVisualizer
from ..infrastructure.retrieval import InMemoryVectorStore, StructuredSopParser
from ..infrastructure.tools import (
    ServiceDeskDataset,
    ServiceDeskMetricsProvider,
    ServiceDeskToolFactory,
)
from ..interfaces import ApprovalGateway
from .session import WorkflowSession

_logger = logging.getLogger("process_automation.container")


@dataclass
class MultiAgentSystem:
    """The assembled system, with its parts exposed for inspection.

    The notebook needs to show the vector store, the tools and the graph
    separately, so the container returns them instead of hiding everything
    behind the session.
    """

    settings: Settings
    knowledge_base: KnowledgeBase
    vector_store: InMemoryVectorStore
    tools: list[Any]
    metrics_provider: ServiceDeskMetricsProvider
    graph_builder: WorkflowGraphBuilder
    compiled_graph: Any
    session: WorkflowSession
    visualizer: MermaidGraphVisualizer
    language_model_name: str = "none (deterministic mode)"

    def describe(self) -> str:
        """One-screen summary of how the system was wired for this run."""
        return "\n".join(
            [
                "Multi-agent process analysis system",
                f"  knowledge base : {len(self.knowledge_base)} SOP document(s), "
                f"{self.vector_store.size} indexed chunk(s)",
                f"  tools          : {', '.join(self.metrics_provider.tool_names)}",
                f"  graph nodes    : {' -> '.join(self.graph_builder.node_names)}",
                f"  reasoning      : {self.language_model_name}",
                f"  max revisions  : {self.settings.review.max_revisions}",
            ]
        )


def build_system(
    settings: Settings | None = None,
    *,
    approval_gateway: ApprovalGateway | None = None,
    remediation_registry: RemediationRegistry | None = None,
) -> MultiAgentSystem:
    """Wire the whole system together and return it ready to run."""
    settings = settings or Settings()

    # -- Node 1 dependencies: knowledge base -> embeddings -> vector store --
    knowledge_base = MarkdownKnowledgeBaseLoader(
        settings.retrieval.knowledge_base_dir
    ).load()
    embeddings = HashingTfidfEmbeddings(settings.retrieval.embedding_dimensions)
    vector_store = InMemoryVectorStore.from_knowledge_base(
        knowledge_base, embeddings, min_score=settings.retrieval.min_score
    )
    documentation_agent = DocumentationAgent(
        retriever=vector_store,
        parser=StructuredSopParser(knowledge_base),
        top_k=settings.retrieval.top_k,
    )

    # -- Node 2 dependencies: the mock ServiceDesk tools --------------------
    dataset = ServiceDeskDataset.from_json(settings.diagnostics.dataset_path)
    tools = ServiceDeskToolFactory(dataset).build()
    metrics_provider = ServiceDeskMetricsProvider(tools)
    diagnostic_agent = DiagnosticAgent(metrics_provider)

    # -- Node 3 dependencies: analysis + composition ------------------------
    composer: Any = RuleBasedSolutionComposer(
        remediation_registry or RemediationRegistry(),
        max_proposals=settings.engineering.max_proposals,
        allow_control_removal=settings.engineering.allow_control_removal,
    )
    language_model = build_language_model(settings)
    language_model_name = "none (deterministic mode)"
    if language_model is not None:
        # Decorator, not replacement: the numbers stay deterministic and only
        # the management summary is written by the model.
        composer = LlmNarrativeComposer(composer, language_model)
        language_model_name = language_model.name
        _logger.info("language model enabled for the plan narrative: %s", language_model_name)

    process_engineer = ProcessEngineerAgent(
        analyzer=GapAnalyzer(),
        composer=composer,
        feedback_interpreter=RuleBasedFeedbackInterpreter(),
    )

    # -- Node 4 and the graph ----------------------------------------------
    human_review = HumanReviewNode()
    graph_builder = WorkflowGraphBuilder(
        documentation_agent,
        diagnostic_agent,
        process_engineer,
        human_review,
        max_revisions=settings.review.max_revisions,
    )
    compiled_graph = graph_builder.compile()

    session = WorkflowSession(
        compiled_graph,
        approval_gateway or AutoApproveGateway(),
        human_node=human_review.name,
        default_thread_id=settings.review.default_thread_id,
        max_interrupts=settings.review.max_revisions + 2,
    )

    return MultiAgentSystem(
        settings=settings,
        knowledge_base=knowledge_base,
        vector_store=vector_store,
        tools=tools,
        metrics_provider=metrics_provider,
        graph_builder=graph_builder,
        compiled_graph=compiled_graph,
        session=session,
        visualizer=MermaidGraphVisualizer(),
        language_model_name=language_model_name,
    )


__all__ = ["MultiAgentSystem", "build_system"]
