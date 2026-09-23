"""Human-in-the-Loop gateways - how the graph asks the systems manager.

The graph itself is *paused* by LangGraph's checkpointer; these classes only
answer the question "what did the reviewer say?". Separating the two means the
same compiled graph serves an interactive console session, an automated test
and, later, a web UI - without a single change to the nodes.

SOLID notes
-----------
* LSP - all gateways are interchangeable; the runner calls ``review`` and never
        inspects the concrete type.
* SRP - a gateway collects a decision. It does not render the plan, route the
        graph or persist anything.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Sequence

from ..domain.models import ActionPlan, ApprovalDecision


class AutoApproveGateway:
    """Approves everything - useful for smoke tests and CI."""

    @property
    def name(self) -> str:
        return "auto-approve"

    def review(self, plan: ActionPlan, *, revision: int) -> ApprovalDecision:
        return ApprovalDecision(approved=True, feedback="", reviewer="auto")


class ScriptedApprovalGateway:
    """Replays a pre-recorded list of decisions.

    This is what makes the notebook's second scenario reproducible: the
    reviewer's exact sentence is scripted, so the run is identical every time
    it is executed by the grader.
    """

    def __init__(self, decisions: Sequence[ApprovalDecision]) -> None:
        self._queue: deque[ApprovalDecision] = deque(decisions)
        self._history: list[ApprovalDecision] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def history(self) -> tuple[ApprovalDecision, ...]:
        return tuple(self._history)

    def review(self, plan: ActionPlan, *, revision: int) -> ApprovalDecision:
        # Once the script is exhausted the reviewer approves, so a run can
        # never loop forever because of a missing line in the script.
        decision = (
            self._queue.popleft()
            if self._queue
            else ApprovalDecision(approved=True, feedback="", reviewer="scripted")
        )
        self._history.append(decision)
        return decision


class ConsoleApprovalGateway:
    """Asks the reviewer in the terminal / notebook (real HITL)."""

    def __init__(
        self,
        prompt_fn: Callable[[str], str] = input,
        writer: Callable[[str], None] = print,
    ) -> None:
        # Injecting ``input``/``print`` keeps this class testable without
        # monkey-patching builtins.
        self._prompt = prompt_fn
        self._write = writer

    @property
    def name(self) -> str:
        return "console"

    def review(self, plan: ActionPlan, *, revision: int) -> ApprovalDecision:
        self._write("\n" + "=" * 72)
        self._write(f"HUMAN-IN-THE-LOOP - review of revision {revision}")
        self._write("=" * 72)
        self._write(plan.to_markdown())
        self._write("-" * 72)
        answer = self._prompt(
            "Approve this action plan? [y = approve / anything else = feedback]: "
        ).strip()
        if answer.lower() in {"y", "yes", "approve", "ok", "بله", "تایید"}:
            return ApprovalDecision(approved=True)
        feedback = answer if answer else self._prompt("Your feedback: ").strip()
        return ApprovalDecision(approved=False, feedback=feedback)


class CallbackApprovalGateway:
    """Delegates the decision to any callable - e.g. a web request handler."""

    def __init__(self, callback: Callable[[ActionPlan, int], ApprovalDecision]) -> None:
        self._callback = callback

    @property
    def name(self) -> str:
        return "callback"

    def review(self, plan: ActionPlan, *, revision: int) -> ApprovalDecision:
        return self._callback(plan, revision)


def reject_with(feedback: str) -> ApprovalDecision:
    """Small helper that makes scripted scenarios read like plain English."""
    return ApprovalDecision(approved=False, feedback=feedback)


def approve() -> ApprovalDecision:
    return ApprovalDecision(approved=True)


def scripted(decisions: Iterable[ApprovalDecision]) -> ScriptedApprovalGateway:
    return ScriptedApprovalGateway(list(decisions))


__all__ = [
    "AutoApproveGateway",
    "CallbackApprovalGateway",
    "ConsoleApprovalGateway",
    "ScriptedApprovalGateway",
    "approve",
    "reject_with",
    "scripted",
]
