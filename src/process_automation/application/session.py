"""Driving the compiled graph across the Human-in-the-Loop interrupt.

LangGraph stops the run *before* the ``human_review`` node and persists the
state through the checkpointer. Somebody then has to:

1. notice that the run is paused,
2. show the current Action Plan to the systems manager,
3. write the decision back into the checkpointed state, and
4. resume the graph.

That orchestration is this module's single responsibility. Keeping it out of
the graph is what allows the very same compiled graph to be driven by a
notebook, a CLI, an automated test or an HTTP endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..domain.models import ActionPlan, ApprovalDecision
from ..interfaces import ApprovalGateway
from ..state import initial_state

_logger = logging.getLogger("process_automation.session")


@dataclass
class RunResult:
    """Everything a caller needs after one analysis run."""

    process_name: str
    thread_id: str
    state: Mapping[str, Any]
    decisions: tuple[ApprovalDecision, ...] = ()
    interrupts: int = 0
    stopped_reason: str = "completed"
    checkpoints: tuple[str, ...] = field(default_factory=tuple)

    @property
    def approved(self) -> bool:
        return bool(self.state.get("approved", False))

    @property
    def revisions(self) -> int:
        return int(self.state.get("revision", 0))

    @property
    def plan(self) -> ActionPlan | None:
        """The final Action Plan, rebuilt from the persisted state."""
        payload = self.state.get("action_plan") or {}
        return ActionPlan.from_payload(payload) if payload else None

    @property
    def trace(self) -> tuple[str, ...]:
        return tuple(self.state.get("trace", ()))

    def print_trace(self) -> None:
        """Print the execution log (RAG hits, tool calls, routing decisions)."""
        for line in self.trace:
            print(line)


class WorkflowSession:
    """Runs one process analysis from start to approval."""

    def __init__(
        self,
        compiled_graph: Any,
        approval_gateway: ApprovalGateway,
        *,
        human_node: str = "human_review",
        default_thread_id: str = "sm-session",
        max_interrupts: int = 6,
    ) -> None:
        self._graph = compiled_graph
        self._gateway = approval_gateway
        self._human_node = human_node
        self._default_thread_id = default_thread_id
        # Hard stop independent of the router's budget: a defective gateway
        # must never be able to spin this loop forever.
        self._max_interrupts = max_interrupts

    @property
    def graph(self) -> Any:
        return self._graph

    def run(self, process_name: str, *, thread_id: str | None = None) -> RunResult:
        """Analyse ``process_name``, pausing for approval as often as needed."""
        thread = thread_id or self._default_thread_id
        config = {"configurable": {"thread_id": thread}}

        _logger.info("starting analysis of '%s' on thread '%s'", process_name, thread)
        # First leg: RAG -> tools -> engineer, then the graph interrupts.
        state: Mapping[str, Any] = self._graph.invoke(initial_state(process_name), config)

        decisions: list[ApprovalDecision] = []
        checkpoints: list[str] = []
        interrupts = 0
        reason = "completed"

        while self._is_paused(config):
            interrupts += 1
            checkpoints.append(f"interrupt #{interrupts} before '{self._human_node}'")
            _logger.info(
                "graph interrupted before '%s' (revision %s) - waiting for the human",
                self._human_node,
                state.get("revision"),
            )
            if interrupts > self._max_interrupts:
                reason = "interrupt budget exhausted"
                _logger.warning("stopping: %s", reason)
                break

            plan = self._current_plan(state)
            decision = self._gateway.review(plan, revision=int(state.get("revision", 1)))
            decisions.append(decision)
            _logger.info(
                "reviewer decision: %s%s",
                "APPROVED" if decision.approved else "REJECTED",
                f" - '{decision.feedback}'" if decision.feedback else "",
            )

            # Write the decision into the checkpointed state, then resume.
            # ``invoke(None, config)`` means "continue from the checkpoint".
            self._graph.update_state(
                config,
                {"approved": decision.approved, "human_feedback": decision.feedback},
            )
            state = self._graph.invoke(None, config)

        if reason == "completed" and not state.get("approved"):
            reason = "ended without approval"

        return RunResult(
            process_name=process_name,
            thread_id=thread,
            state=state,
            decisions=tuple(decisions),
            interrupts=interrupts,
            stopped_reason=reason,
            checkpoints=tuple(checkpoints),
        )

    # -- helpers -----------------------------------------------------------

    def _is_paused(self, config: Mapping[str, Any]) -> bool:
        """True while the checkpointer holds the run before the human node."""
        snapshot = self._graph.get_state(config)
        return self._human_node in (snapshot.next or ())

    def snapshot(self, thread_id: str | None = None) -> Any:
        """Expose the raw checkpoint - handy for inspecting a paused run."""
        thread = thread_id or self._default_thread_id
        return self._graph.get_state({"configurable": {"thread_id": thread}})

    @staticmethod
    def _current_plan(state: Mapping[str, Any]) -> ActionPlan:
        """Rebuild the plan the reviewer has to judge."""
        payload = state.get("action_plan") or {}
        if payload:
            return ActionPlan.from_payload(payload)
        # Defensive fallback: the engineer produced prose but no structure.
        return ActionPlan(
            process_name=str(state.get("process_name", "")),
            gaps=(),
            proposals=(),
            summary=str(state.get("proposed_solution", "")),
            revision=int(state.get("revision", 1)),
        )


__all__ = ["RunResult", "WorkflowSession"]
