"""Configuration objects - one place to tune the system's behaviour.

Passing a :class:`Settings` instance around (instead of reading environment
variables deep inside a class) keeps the components pure and testable, and it
is what lets the composition root swap an offline implementation for a real
LLM without touching a single agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Directory that ships the demo SOP corpus and the mock ServiceDesk dataset.
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class RetrievalSettings:
    """Knobs of the RAG stage."""

    knowledge_base_dir: Path = RESOURCES_DIR / "knowledge_base"
    top_k: int = 4
    #: Chunks scoring below this cosine similarity are ignored.
    min_score: float = 0.05
    embedding_dimensions: int = 512


@dataclass(frozen=True)
class DiagnosticSettings:
    """Knobs of the tool-calling stage."""

    dataset_path: Path = RESOURCES_DIR / "service_desk" / "metrics.json"


@dataclass(frozen=True)
class EngineeringSettings:
    """Knobs of the Process Engineer."""

    #: Maximum number of proposals kept in one plan (focus over volume).
    max_proposals: int = 3
    #: When True (the default) the engineer optimises purely for cycle time on
    #: its first pass and may suggest dropping a documented control - the
    #: situation the Human-in-the-Loop node is designed to catch. Set to False
    #: for a policy-compliant engineer that never proposes such a removal.
    allow_control_removal: bool = True


@dataclass(frozen=True)
class HumanReviewSettings:
    """Knobs of the Human-in-the-Loop stage."""

    #: Safety valve: after this many rejected revisions the graph stops asking.
    max_revisions: int = 3
    #: Thread id used by the LangGraph checkpointer when none is supplied.
    default_thread_id: str = "sm-session"


@dataclass(frozen=True)
class Settings:
    """Root configuration object."""

    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    diagnostics: DiagnosticSettings = field(default_factory=DiagnosticSettings)
    engineering: EngineeringSettings = field(default_factory=EngineeringSettings)
    review: HumanReviewSettings = field(default_factory=HumanReviewSettings)
    #: "auto" -> use a hosted chat model when an API key is present, otherwise
    #: fall back to the deterministic offline components. Can be forced to
    #: "offline" (fully reproducible) or "llm" (fail loudly without a key).
    reasoning_mode: str = "auto"
    #: Chat model used when a hosted provider is selected.
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Read the few values it makes sense to override from the shell."""
        return cls(
            reasoning_mode=os.getenv("PA_REASONING_MODE", "auto"),
            llm_model=os.getenv("PA_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.getenv("PA_LLM_TEMPERATURE", "0")),
        )


__all__ = [
    "DiagnosticSettings",
    "EngineeringSettings",
    "HumanReviewSettings",
    "RESOURCES_DIR",
    "RetrievalSettings",
    "Settings",
]
