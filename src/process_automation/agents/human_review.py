"""Node 4 - Human-in-the-Loop (systems manager approval).

The *pause* itself is not implemented here: the graph is compiled with
``interrupt_before=["human_review"]``, so LangGraph's checkpointer stops the
run before this node executes and persists the state. The runner then collects
the reviewer's decision and writes it back with ``update_state``.

This node therefore does exactly one thing: record the decision that is now in
the state, and make it visible in the audit trail. Keeping the *waiting* in the
graph runtime and the *asking* in an
:class:`~process_automation.interfaces.ApprovalGateway` is what lets the same
graph serve a notebook, a CLI and a web UI unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..state import ProcessAnalysisState
from .base import AgentNode


class HumanReviewNode(AgentNode):
    """Records the systems manager's verdict on the current Action Plan."""

    name = "human_review"

    def run(self, state: ProcessAnalysisState) -> Mapping[str, Any]:
        feedback = (state.get("human_feedback") or "").strip()
        revision = int(state.get("revision", 1))

        # Feedback always wins over a stale ``approved`` flag: a reviewer who
        # writes a correction has not approved the plan, whatever else is set.
        approved = bool(state.get("approved", False)) and not feedback

        if approved:
            trace = [self.log(f"revision {revision} APPROVED by the systems manager")]
            return {"approved": True, "human_feedback": "", "trace": trace}

        trace = [
            self.log(f"revision {revision} REJECTED - feedback: '{feedback}'")
            if feedback
            else self.log(f"revision {revision} not approved and no feedback given")
        ]
        return {
            "approved": False,
            "human_feedback": feedback,
            "feedback_log": [f"revision {revision}: {feedback}"] if feedback else [],
            "trace": trace,
        }


__all__ = ["HumanReviewNode"]
