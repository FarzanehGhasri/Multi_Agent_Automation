"""Step 3 of the brief - the conditional edges of the graph.

Routing is deliberately kept out of the nodes. A node answers "what did I
find?"; a router answers "where does the flow go next?". Separating the two
(SRP) means the revision loop can be re-tuned - or replaced by an escalation
to a second reviewer - without reopening a single agent.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END

from ..state import ProcessAnalysisState

#: Destination names used by the conditional edge map.
REVISE = "revise"
FINISH = "finish"


def build_review_router(max_revisions: int = 3) -> Callable[[ProcessAnalysisState], str]:
    """Return the router used after the Human-in-the-Loop node.

    The decision table:

    ======================================  ==========================
    state                                   next
    ======================================  ==========================
    approved                                END
    feedback given, revisions left          back to ``process_engineer``
    feedback given, revision budget spent   END (with a warning)
    no feedback and not approved            END (reviewer abstained)
    ======================================  ==========================

    ``max_revisions`` is a safety valve: without it, a reviewer who keeps
    rejecting would spin the graph forever.
    """

    def route_after_review(state: ProcessAnalysisState) -> str:
        if state.get("approved"):
            return FINISH
        feedback = (state.get("human_feedback") or "").strip()
        if not feedback:
            # Neither approval nor feedback: nothing actionable to re-plan.
            return FINISH
        if int(state.get("revision", 0)) >= max_revisions:
            return FINISH
        return REVISE

    return route_after_review


#: Mapping from the router's return value to the actual graph destination.
#: Declaring it next to the router keeps the two definitions in sync.
def review_route_map(engineer_node: str) -> dict[str, str]:
    return {REVISE: engineer_node, FINISH: END}


__all__ = ["FINISH", "REVISE", "build_review_router", "review_route_map"]
