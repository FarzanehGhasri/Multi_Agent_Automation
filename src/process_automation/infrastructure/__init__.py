"""Infrastructure layer: concrete adapters for the ports in ``interfaces``.

Nothing in :mod:`process_automation.domain` or :mod:`process_automation.agents`
imports this package directly - only the composition root does. That is what
keeps the dependency arrows pointing inwards (Dependency Inversion Principle).
"""

from .approval import (
    AutoApproveGateway,
    CallbackApprovalGateway,
    ConsoleApprovalGateway,
    ScriptedApprovalGateway,
    approve,
    reject_with,
)
from .composition import LlmNarrativeComposer, RuleBasedSolutionComposer
from .embeddings import HashingTfidfEmbeddings, LangChainEmbeddingsAdapter
from .knowledge_base import KnowledgeBase, MarkdownKnowledgeBaseLoader, SopDocument
from .llm import ChatModelAdapter, EchoLanguageModel, build_language_model
from .rendering import MermaidGraphVisualizer
from .retrieval import InMemoryVectorStore, StructuredSopParser
from .tools import (
    ProcessNotMonitored,
    ServiceDeskDataset,
    ServiceDeskMetricsProvider,
    ServiceDeskToolFactory,
)

__all__ = [
    "AutoApproveGateway",
    "CallbackApprovalGateway",
    "ChatModelAdapter",
    "ConsoleApprovalGateway",
    "EchoLanguageModel",
    "HashingTfidfEmbeddings",
    "InMemoryVectorStore",
    "KnowledgeBase",
    "LangChainEmbeddingsAdapter",
    "LlmNarrativeComposer",
    "MarkdownKnowledgeBaseLoader",
    "MermaidGraphVisualizer",
    "ProcessNotMonitored",
    "RuleBasedSolutionComposer",
    "ScriptedApprovalGateway",
    "ServiceDeskDataset",
    "ServiceDeskMetricsProvider",
    "ServiceDeskToolFactory",
    "SopDocument",
    "StructuredSopParser",
    "approve",
    "build_language_model",
    "reject_with",
]
