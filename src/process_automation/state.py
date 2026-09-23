"""Step 1 of the brief - the state that travels between the graph nodes.

``WorkflowState`` below is *verbatim* the ``TypedDict`` required by the
project specification. Everything the implementation needs on top of it lives
in :class:`ProcessAnalysisState`, a subclass with ``total=False`` optional
keys: the required contract stays untouched and is simply extended.
That is the Open/Closed Principle applied to a data structure, and it is also
why the notebook can show the mandated class exactly as the brief writes it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

# ---------------------------------------------------------------------------
# The state required by the project brief (unchanged, on purpose)
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict):
    """Shared memory of the multi-agent graph."""

    process_name: str       # نام فرآیند درخواستی - the requested process
    standard_sop: str       # استاندارد استخراج شده (خروجی RAG) - Node 1 output
    actual_metrics: dict    # دیتای واقعی استخراج شده (خروجی Tools) - Node 2 output
    identified_gaps: list   # شکاف‌ها و گلوگاه‌های شناسایی شده - Node 3 finding
    proposed_solution: str  # راهکار نهایی اتوماسیون - Node 3 output
    human_feedback: str     # بازخورد کاربر انسانی برای اصلاح - Node 4 input


# ---------------------------------------------------------------------------
# The extended state actually compiled into the graph
# ---------------------------------------------------------------------------


class ProcessAnalysisState(WorkflowState, total=False):
    """``WorkflowState`` + the bookkeeping the runtime needs.

    ``total=False`` marks every key below as optional, so a run can still be
    started with nothing but ``process_name``.

    ``Annotated[list, operator.add]`` installs a *reducer*: when a node
    returns ``{"trace": ["..."]}`` LangGraph appends instead of overwriting,
    which is what turns these two fields into an append-only audit log.
    """

    #: True once the systems manager has approved the current plan.
    approved: bool
    #: 1 for the first plan, 2 after the first round of human feedback, ...
    revision: int
    #: Machine-readable constraints distilled from the human feedback.
    constraints: list[dict[str, Any]]
    #: Structured mirror of ``proposed_solution`` (gaps + proposals + code).
    action_plan: dict[str, Any]
    #: Chunks the RAG node actually used, for traceability of the extraction.
    retrieved_documents: list[dict[str, Any]]
    #: Canonical process name resolved by the RAG node from the user's wording.
    resolved_process: str
    #: Identifier of the SOP document the standard was extracted from.
    sop_document_id: str
    #: Structured mirror of ``standard_sop`` (SLA + typed steps), so Node 3 can
    #: compare numbers instead of re-parsing prose.
    standard_profile: dict[str, Any]
    #: Append-only execution log (RAG lookups, tool calls, routing decisions).
    trace: Annotated[list[str], operator.add]
    #: Append-only history of everything the human reviewer said.
    feedback_log: Annotated[list[str], operator.add]


def initial_state(process_name: str) -> ProcessAnalysisState:
    """Build a fully-populated starting state for one analysis run.

    Initialising every key up front keeps the nodes free of ``state.get(...)``
    defensive noise and makes a run reproducible.
    """
    return ProcessAnalysisState(
        process_name=process_name,
        standard_sop="",
        actual_metrics={},
        identified_gaps=[],
        proposed_solution="",
        human_feedback="",
        approved=False,
        revision=0,
        constraints=[],
        action_plan={},
        retrieved_documents=[],
        resolved_process="",
        sop_document_id="",
        standard_profile={},
        trace=[],
        feedback_log=[],
    )


__all__ = ["ProcessAnalysisState", "WorkflowState", "initial_state"]
