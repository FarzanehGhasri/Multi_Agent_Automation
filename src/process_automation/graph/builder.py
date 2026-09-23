"""Assembles the four agents into a LangGraph state machine.

    START -> documentation_agent -> diagnostic_agent -> process_engineer
          -> [PAUSE: checkpointer] -> human_review -> (revise? -> process_engineer | END)

SOLID notes
-----------
* DIP - the builder receives *node objects*; it never constructs an agent, so
        the graph topology and the agents' dependencies evolve independently.
* SRP - this class only wires nodes and edges. Running the graph, asking the
        human and rendering the diagram all live elsewhere.
* OCP - inserting a fifth node (e.g. a cost estimator) means one extra
        ``add_node``/``add_edge`` pair here and nothing anywhere else.
"""

from __future__ import annotations

from typing import Any, Sequence

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from ..agents.base import AgentNode
from ..state import ProcessAnalysisState
from .routing import build_review_router, review_route_map


class WorkflowGraphBuilder:
    """Builds and compiles the multi-agent workflow graph."""

    def __init__(
        self,
        documentation_agent: AgentNode,
        diagnostic_agent: AgentNode,
        process_engineer: AgentNode,
        human_review: AgentNode,
        *,
        max_revisions: int = 3,
    ) -> None:
        self._documentation = documentation_agent
        self._diagnostic = diagnostic_agent
        self._engineer = process_engineer
        self._human_review = human_review
        self._max_revisions = max_revisions

    @property
    def node_names(self) -> tuple[str, ...]:
        return (
            self._documentation.name,
            self._diagnostic.name,
            self._engineer.name,
            self._human_review.name,
        )

    def build(self) -> StateGraph:
        """Declare nodes and edges (no compilation, no checkpointer yet)."""
        graph = StateGraph(ProcessAnalysisState)

        # -- nodes: one per agent, registered under the agent's own name -----
        graph.add_node(self._documentation.name, self._documentation)
        graph.add_node(self._diagnostic.name, self._diagnostic)
        graph.add_node(self._engineer.name, self._engineer)
        graph.add_node(self._human_review.name, self._human_review)

        # -- linear part: understand the standard, then measure reality ------
        graph.add_edge(START, self._documentation.name)
        graph.add_edge(self._documentation.name, self._diagnostic.name)
        graph.add_edge(self._diagnostic.name, self._engineer.name)

        # -- the plan always goes to the reviewer; the interrupt is applied
        #    at compile time, right before this node executes ----------------
        graph.add_edge(self._engineer.name, self._human_review.name)

        # -- conditional edge: approve and stop, or re-plan with feedback ----
        graph.add_conditional_edges(
            self._human_review.name,
            build_review_router(self._max_revisions),
            review_route_map(self._engineer.name),
        )
        return graph

    def compile(
        self,
        checkpointer: Any | None = None,
        *,
        interrupt_before: Sequence[str] | None = None,
    ) -> Any:
        """Compile the graph with a checkpointer and the human interrupt.

        The checkpointer is what makes the Human-in-the-Loop possible: it
        persists the state at the interrupt so the run can be resumed - even
        minutes later, or from a different process - once the systems manager
        has answered.
        """
        return self.build().compile(
            checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
            interrupt_before=list(
                interrupt_before if interrupt_before is not None else [self._human_review.name]
            ),
        )


__all__ = ["WorkflowGraphBuilder"]
