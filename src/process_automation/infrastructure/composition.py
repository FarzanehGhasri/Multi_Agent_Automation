"""Writing the Action Plan - the reasoning behind Node 3.

Two implementations of the :class:`~process_automation.interfaces.SolutionComposer`
port live here:

* :class:`RuleBasedSolutionComposer` - deterministic, offline, auditable. It
  joins each gap with a remediation strategy and renders the plan.
* :class:`LlmNarrativeComposer` - a **decorator** that keeps the deterministic
  gaps, proposals and numbers untouched and asks a language model to write the
  management summary in better prose.

The decorator is the Open/Closed Principle in action: adding LLM prose did not
require a single edit to the rule-based composer, and removing the LLM again is
a one-line change in the composition root.
"""

from __future__ import annotations

from typing import Sequence

from ..domain.models import (
    ActionPlan,
    AutomationProposal,
    Constraint,
    Gap,
    ProcessMetrics,
    StandardProcess,
)
from ..domain.remediation import RemediationRegistry, build_contexts
from ..interfaces import LanguageModel, SolutionComposer


class RuleBasedSolutionComposer:
    """Builds the Action Plan from gaps + remediation strategies."""

    def __init__(
        self,
        registry: RemediationRegistry | None = None,
        *,
        max_proposals: int = 3,
        allow_control_removal: bool = True,
    ) -> None:
        self._registry = registry or RemediationRegistry()
        self._max_proposals = max_proposals
        self._allow_control_removal = allow_control_removal

    @property
    def name(self) -> str:
        return "rule-based-composer"

    def compose(
        self,
        standard: StandardProcess,
        metrics: ProcessMetrics,
        gaps: Sequence[Gap],
        constraints: Sequence[Constraint] = (),
        revision: int = 1,
    ) -> ActionPlan:
        """Turn the gap list into a prioritised, constraint-compliant plan."""
        contexts = build_contexts(gaps, standard, metrics, constraints)

        # ``selected`` is keyed by strategy id so the same remedy proposed for
        # two different gaps appears once, with its best estimated gain.
        selected: dict[str, AutomationProposal] = {}
        for context in contexts:
            proposal = self._registry.propose(
                context, allow_control_removal=self._allow_control_removal
            )
            if proposal is None:
                continue  # no legal remedy for this gap in this revision
            previous = selected.get(proposal.identifier)
            if previous is None or proposal.expected_gain_hours > previous.expected_gain_hours:
                selected[proposal.identifier] = proposal

        # Gaps arrive worst-first, so insertion order already encodes priority;
        # only the top ``max_proposals`` are kept to keep the plan executable.
        proposals = tuple(selected.values())[: self._max_proposals]

        return ActionPlan(
            process_name=standard.process_name or metrics.process_name,
            gaps=tuple(gaps),
            proposals=proposals,
            summary=self._summarise(standard, metrics, gaps, proposals, constraints, revision),
            revision=revision,
            honoured_constraints=tuple(constraints),
            baseline_hours=metrics.average_resolution_hours,
            sla_hours=standard.sla_hours,
        )

    # -- narrative ---------------------------------------------------------

    @staticmethod
    def _summarise(
        standard: StandardProcess,
        metrics: ProcessMetrics,
        gaps: Sequence[Gap],
        proposals: Sequence[AutomationProposal],
        constraints: Sequence[Constraint],
        revision: int,
    ) -> str:
        """One paragraph a systems manager can read in fifteen seconds."""
        if standard.sla_hours <= 0:
            return (
                f"No approved standard could be retrieved for "
                f"'{metrics.process_name}', so the measured behaviour "
                f"({metrics.average_resolution_hours:g}h average) cannot be "
                f"compared. Documenting the process is the prerequisite for "
                f"any automation."
            )

        ratio = metrics.average_resolution_hours / standard.sla_hours
        bottleneck = metrics.bottleneck
        sentences = [
            f"The documented SLA for '{standard.process_name}' is "
            f"{standard.sla_hours:g}h, but the {metrics.sample_period} sample shows an "
            f"average of {metrics.average_resolution_hours:g}h - {ratio:.1f}x the standard."
        ]
        if bottleneck is not None:
            share = metrics.share_of_total(bottleneck)
            sentences.append(
                f"The dominant bottleneck is '{bottleneck.stage}' with "
                f"{bottleneck.average_hours:g}h ({share:.0%} of the cycle, "
                f"{bottleneck.waiting_share:.0%} of it pure waiting)."
            )
        sentences.append(
            f"{len(gaps)} gap(s) were identified and {len(proposals)} automation "
            f"measure(s) are proposed below."
        )
        if revision > 1:
            sentences.append(
                f"This is revision {revision}, rewritten after the systems "
                f"manager's feedback."
            )
        if constraints:
            sentences.append(
                "Constraints taken into account: "
                + "; ".join(constraint.describe() for constraint in constraints)
                + "."
            )
        return " ".join(sentences)


class LlmNarrativeComposer:
    """Decorator adding LLM-written prose on top of a deterministic plan.

    Only the *summary* is delegated to the model. Gaps, proposals, hours and
    code stay exactly as the rule engine computed them, so an LLM outage - or
    a hallucination - can never alter a number in a compliance document.
    """

    _SYSTEM_PROMPT = (
        "You are a senior Systems & Methods (business process re-engineering) "
        "consultant writing for an IT operations director. Be concrete, "
        "quantitative and neutral. Never invent numbers that are not given to "
        "you. Answer with one paragraph of at most 120 words, no bullet points."
    )

    def __init__(self, inner: SolutionComposer, language_model: LanguageModel) -> None:
        self._inner = inner
        self._llm = language_model

    @property
    def name(self) -> str:
        return f"llm-narrative({self._llm.name})"

    def compose(
        self,
        standard: StandardProcess,
        metrics: ProcessMetrics,
        gaps: Sequence[Gap],
        constraints: Sequence[Constraint] = (),
        revision: int = 1,
    ) -> ActionPlan:
        plan = self._inner.compose(standard, metrics, gaps, constraints, revision)
        try:
            summary = self._llm.complete(
                self._build_prompt(plan, standard, metrics),
                system=self._SYSTEM_PROMPT,
            ).strip()
        except Exception:  # noqa: BLE001 - never fail the run because of the LLM
            # Graceful degradation: the deterministic summary is always valid.
            return plan
        return ActionPlan(
            process_name=plan.process_name,
            gaps=plan.gaps,
            proposals=plan.proposals,
            summary=summary or plan.summary,
            revision=plan.revision,
            honoured_constraints=plan.honoured_constraints,
            baseline_hours=plan.baseline_hours,
            sla_hours=plan.sla_hours,
        )

    @staticmethod
    def _build_prompt(
        plan: ActionPlan, standard: StandardProcess, metrics: ProcessMetrics
    ) -> str:
        """Feed the model facts only - it is asked to phrase, not to decide."""
        gap_lines = "\n".join(
            f"- [{gap.severity.value}] {gap.title}: {gap.evidence}" for gap in plan.gaps
        ) or "- none"
        proposal_lines = "\n".join(
            f"- {proposal.title} (saves ~{proposal.expected_gain_hours:g}h)"
            for proposal in plan.proposals
        ) or "- none"
        constraint_lines = "\n".join(
            f"- {constraint.describe()}" for constraint in plan.honoured_constraints
        ) or "- none"
        return (
            f"Process: {standard.process_name}\n"
            f"Documented SLA: {standard.sla_hours:g} hours\n"
            f"Measured average: {metrics.average_resolution_hours:g} hours "
            f"({metrics.sample_period})\n"
            f"Revision: {plan.revision}\n\n"
            f"Identified gaps:\n{gap_lines}\n\n"
            f"Proposed measures:\n{proposal_lines}\n\n"
            f"Constraints stated by the reviewer:\n{constraint_lines}\n\n"
            "Write the management summary of this action plan."
        )


__all__ = ["LlmNarrativeComposer", "RuleBasedSolutionComposer"]
