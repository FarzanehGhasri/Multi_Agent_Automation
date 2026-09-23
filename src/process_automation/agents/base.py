"""Common scaffolding shared by the four graph nodes.

Every node is a *class* rather than a bare function for three reasons:

1. it can hold its injected collaborators (retriever, tools, composer) instead
   of reaching for globals - Dependency Inversion;
2. it can be instantiated twice with different configuration in the same
   process, which a module-level function cannot;
3. it keeps ``__call__`` thin, so LangGraph still receives a plain callable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..state import ProcessAnalysisState


class AgentNode(ABC):
    """Base class of every node in the LangGraph workflow."""

    #: Node name used in the graph, the logs and the Mermaid diagram.
    name: str = "agent"

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"process_automation.{self.name}")

    # LangGraph only needs a callable; delegating keeps ``run`` easy to test
    # directly, without constructing a graph.
    def __call__(self, state: ProcessAnalysisState) -> Mapping[str, Any]:
        return self.run(state)

    @abstractmethod
    def run(self, state: ProcessAnalysisState) -> Mapping[str, Any]:
        """Return the *partial* state update produced by this node.

        Returning a partial mapping (instead of a whole state) is what lets
        LangGraph merge concurrent updates and apply the reducers declared on
        :class:`~process_automation.state.ProcessAnalysisState`.
        """

    # -- shared helpers ----------------------------------------------------

    def log(self, message: str) -> str:
        """Write one line to the logger *and* to the state's audit trail."""
        self._logger.info(message)
        return f"[{self.name}] {message}"


__all__ = ["AgentNode"]
