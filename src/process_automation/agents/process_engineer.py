"""Node 3 - Process Engineer Agent.

Role: systems & methods architect.
Task: compare Node 1 (what *should* be) with Node 2 (what *is*), identify the
deviations and bottlenecks, and produce an Action Plan with concrete
automation measures - including code where a script is the right answer.
On a second pass it re-plans under the constraints the human reviewer stated.
Output: ``identified_gaps`` + ``proposed_solution`` (+ structured ``action_plan``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..domain.feedback import FeedbackInterpreter, merge_constraints
from ..domain.gap_analysis import GapAnalyzer
from ..domain.models import Constraint, ProcessMetrics, StandardProcess
from ..interfaces import SolutionComposer
from ..state import ProcessAnalysisState
from .base import AgentNode


class ProcessEngineerAgent(AgentNode):
    """Produces - and, after feedback, re-produces - the Action Plan.

    The agent orchestrates three injected collaborators and owns none of the
    logic itself:

    * :class:`~process_automation.domain.gap_analysis.GapAnalyzer` finds the
      deviations;
    * a :class:`~process_automation.interfaces.FeedbackInterpreter` turns the
      reviewer's sentence into machine-checkable constraints;
    * a :class:`~process_automation.interfaces.SolutionComposer` writes the plan.

    That is why the second scenario of the brief needs no special code path:
    revision 2 is the *same* call with a longer constraint list.
    """

    name = "process_engineer"

    def __init__(
        self,
        analyzer: GapAnalyzer,
        composer: SolutionComposer,
        feedback_interpreter: FeedbackInterpreter,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._composer = composer
        self._interpreter = feedback_interpreter

    def run(self, state: ProcessAnalysisState) -> Mapping[str, Any]:
        standard = StandardProcess.from_payload(state.get("standard_profile") or {})
        metrics_payload = state.get("actual_metrics") or {}
        revision = int(state.get("revision", 0)) + 1
        trace: list[str] = []

        if not metrics_payload:
            # Without measurements there is nothing to compare against; say so
            # explicitly instead of producing a plausible-looking plan.
            trace.append(self.log("no metrics available - cannot compare"))
            return {
                "identified_gaps": [],
                "proposed_solution": (
                    f"# Action Plan - {standard.process_name or state['process_name']}\n\n"
                    "The monitoring tools returned no data for this process, so the "
                    "documented standard cannot be compared with reality. "
                    "Onboard the process into the ServiceDesk analytics before "
                    "requesting an automation analysis."
                ),
                "action_plan": {},
                "revision": revision,
                "approved": False,
                "human_feedback": "",
                "trace": trace,
            }

        metrics = ProcessMetrics.from_payload(metrics_payload)

        # 1) Carry forward every constraint stated in previous revisions, then
        #    translate any new feedback into additional ones. Constraints are
        #    cumulative, which is what makes the revision loop converge.
        constraints = tuple(
            Constraint.from_payload(item) for item in state.get("constraints", [])
        )
        feedback = (state.get("human_feedback") or "").strip()
        if feedback:
            new_constraints = self._interpreter.interpret(feedback, standard)
            constraints = merge_constraints(constraints, new_constraints)
            trace.append(self.log(f"re-planning after human feedback: '{feedback}'"))
            for constraint in new_constraints:
                trace.append(self.log(f"  derived constraint -> {constraint.describe()}"))

        # 2) Compare should-be with as-is.
        gaps = self._analyzer.analyse(standard, metrics)
        trace.append(
            self.log(
                f"revision {revision}: {len(gaps)} gap(s) identified "
                f"({', '.join(gap.severity.value for gap in gaps) or 'none'})"
            )
        )
        for gap in gaps:
            trace.append(self.log(f"  [{gap.severity.value}] {gap.title}"))

        # 3) Compose the plan under the current constraints.
        plan = self._composer.compose(standard, metrics, gaps, constraints, revision)
        for proposal in plan.proposals:
            trace.append(
                self.log(
                    f"  proposal '{proposal.title}' "
                    f"(~{proposal.expected_gain_hours:.1f}h, {proposal.gain_kind})"
                )
            )
            if proposal.removed_controls:
                trace.append(
                    self.log(
                        "  ! this proposal removes the control(s): "
                        + ", ".join(proposal.removed_controls)
                    )
                )
        trace.append(
            self.log(
                f"projected duration {plan.baseline_hours:g}h -> "
                f"{plan.projected_hours:.1f}h (SLA {plan.sla_hours:g}h)"
            )
        )

        return {
            "identified_gaps": [gap.to_payload() for gap in gaps],
            "proposed_solution": plan.to_markdown(),
            "action_plan": plan.to_payload(),
            "constraints": [constraint.to_payload() for constraint in constraints],
            "revision": revision,
            # The feedback has now been consumed and turned into constraints;
            # clearing it prevents the router from looping on a stale value.
            "human_feedback": "",
            "approved": False,
            "trace": trace,
        }


__all__ = ["ProcessEngineerAgent"]
