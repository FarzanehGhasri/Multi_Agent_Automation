"""Domain models (Entities & Value Objects).

This module is the *innermost* layer of the architecture: it knows nothing
about LangGraph, LangChain, vector stores, LLM providers or notebooks.
Everything else may depend on this module, but this module depends on nothing.

SOLID notes
-----------
* SRP  - every class here models exactly one business concept.
* OCP  - the models are frozen value objects; behaviour is extended by adding
         new services around them, never by mutating them in place.
* DIP  - because this layer has zero infrastructure imports, the concrete
         adapters (LangChain tools, vector stores, chat models) depend on the
         domain and not the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Severity(str, Enum):
    """Business impact of an identified gap, ordered from low to critical."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank so gaps can be sorted by importance."""
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class AutomationClass(str, Enum):
    """How a single SOP step behaves with respect to automation.

    The value is read from the SOP document metadata, which means a new
    process can be on-boarded by *adding a document* rather than by editing
    Python code (Open/Closed Principle).
    """

    AUTOMATED = "automated"               # already machine executed
    MANUAL_APPROVAL = "manual_approval"   # a human presses "approve"
    MANUAL_PROVISIONING = "manual_provisioning"  # a human configures a system
    SECURITY_REVIEW = "security_review"   # a security control, usually mandatory
    DATA_ENTRY = "data_entry"             # a human re-types data between systems
    WAITING_QUEUE = "waiting_queue"       # the request waits in a backlog


# ---------------------------------------------------------------------------
# "What should be" - the standard extracted from company documentation (RAG)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessStep:
    """One approved step of a documented process (a node of the flowchart)."""

    order: int
    title: str
    owner: str
    automation_class: AutomationClass
    target_hours: float | None = None
    #: A mandatory control cannot be deleted by an automation proposal
    #: (e.g. a security sign-off required by the information-security policy).
    is_mandatory_control: bool = False
    control_reference: str | None = None

    @property
    def is_manual(self) -> bool:
        """True when a human has to act for this step to complete."""
        return self.automation_class is not AutomationClass.AUTOMATED

    def to_payload(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "title": self.title,
            "owner": self.owner,
            "automation_class": self.automation_class.value,
            "target_hours": self.target_hours,
            "is_mandatory_control": self.is_mandatory_control,
            "control_reference": self.control_reference,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProcessStep":
        return cls(
            order=int(payload["order"]),
            title=str(payload["title"]),
            owner=str(payload.get("owner", "unknown")),
            automation_class=AutomationClass(payload.get("automation_class", "data_entry")),
            target_hours=payload.get("target_hours"),
            is_mandatory_control=bool(payload.get("is_mandatory_control", False)),
            control_reference=payload.get("control_reference"),
        )


@dataclass(frozen=True)
class StandardProcess:
    """The *should-be* picture: SLA + approved steps, extracted from the SOP.

    This is the expected output of Node 1 (Documentation Agent / RAG).
    """

    process_name: str
    sla_hours: float
    steps: tuple[ProcessStep, ...] = ()
    policy_notes: tuple[str, ...] = ()
    source_documents: tuple[str, ...] = ()

    def step_by_title(self, title: str) -> ProcessStep | None:
        """Best-effort lookup used to map a measured stage back to the SOP."""
        needle = _normalise(title)
        for step in self.steps:
            haystack = _normalise(step.title)
            if needle == haystack or needle in haystack or haystack in needle:
                return step
        return None

    @property
    def mandatory_controls(self) -> tuple[ProcessStep, ...]:
        return tuple(step for step in self.steps if step.is_mandatory_control)

    def to_payload(self) -> dict[str, Any]:
        """Serialise the standard so it can travel inside the graph state."""
        return {
            "process_name": self.process_name,
            "sla_hours": self.sla_hours,
            "steps": [step.to_payload() for step in self.steps],
            "policy_notes": list(self.policy_notes),
            "source_documents": list(self.source_documents),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StandardProcess":
        return cls(
            process_name=str(payload.get("process_name", "")),
            sla_hours=float(payload.get("sla_hours", 0.0)),
            steps=tuple(ProcessStep.from_payload(item) for item in payload.get("steps", ())),
            policy_notes=tuple(payload.get("policy_notes", ())),
            source_documents=tuple(payload.get("source_documents", ())),
        )

    def to_text(self) -> str:
        """Render the SOP as the plain-text ``standard_sop`` field of the state."""
        lines = [
            f"Process: {self.process_name}",
            f"Documented SLA: {self.sla_hours:g} hour(s) end-to-end",
            "Approved steps:",
        ]
        for step in self.steps:
            target = f"{step.target_hours:g}h" if step.target_hours is not None else "n/a"
            flags = " [MANDATORY CONTROL]" if step.is_mandatory_control else ""
            lines.append(
                f"  {step.order}. {step.title} - owner: {step.owner}"
                f" - target: {target} - type: {step.automation_class.value}{flags}"
            )
        if self.policy_notes:
            lines.append("Policy notes:")
            lines.extend(f"  - {note}" for note in self.policy_notes)
        if self.source_documents:
            lines.append(f"Sources: {', '.join(self.source_documents)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# "What is" - the real picture measured through the monitoring tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageTiming:
    """How long a single stage of the real process actually takes."""

    stage: str
    average_hours: float
    waiting_share: float = 0.0  # fraction of the stage that is pure waiting


@dataclass(frozen=True)
class ProcessMetrics:
    """The *as-is* picture returned by the Diagnostic Agent's tools.

    Mirrors what a real Jira / ServiceDesk analytics endpoint would expose.
    """

    process_name: str
    average_resolution_hours: float
    ticket_volume: int
    reopened_tickets: int
    common_errors: tuple[str, ...] = ()
    stage_timings: tuple[StageTiming, ...] = ()
    sample_period: str = "unknown"

    @property
    def reopen_rate(self) -> float:
        """Share of tickets that came back (rework indicator)."""
        if self.ticket_volume <= 0:
            return 0.0
        return self.reopened_tickets / self.ticket_volume

    @property
    def bottleneck(self) -> StageTiming | None:
        """The single slowest stage - the primary automation candidate."""
        if not self.stage_timings:
            return None
        return max(self.stage_timings, key=lambda timing: timing.average_hours)

    def share_of_total(self, timing: StageTiming) -> float:
        """Fraction of the end-to-end duration consumed by one stage."""
        total = sum(item.average_hours for item in self.stage_timings)
        return timing.average_hours / total if total else 0.0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProcessMetrics":
        """Build the value object from the raw dictionary returned by a tool.

        Keeping the parsing here (instead of inside the agent) means the agent
        does not have to change when the tool payload evolves - SRP again.
        """
        timings = tuple(
            StageTiming(
                stage=str(item["stage"]),
                average_hours=float(item["average_hours"]),
                waiting_share=float(item.get("waiting_share", 0.0)),
            )
            for item in payload.get("stage_timings", ())
        )
        return cls(
            process_name=str(payload.get("process_name", "unknown")),
            average_resolution_hours=float(payload.get("average_resolution_hours", 0.0)),
            ticket_volume=int(payload.get("ticket_volume", 0)),
            reopened_tickets=int(payload.get("reopened_tickets", 0)),
            common_errors=tuple(payload.get("common_errors", ())),
            stage_timings=timings,
            sample_period=str(payload.get("sample_period", "unknown")),
        )

    def to_payload(self) -> dict[str, Any]:
        """Inverse of :meth:`from_payload`; used to fill ``actual_metrics``."""
        return {
            "process_name": self.process_name,
            "average_resolution_hours": self.average_resolution_hours,
            "ticket_volume": self.ticket_volume,
            "reopened_tickets": self.reopened_tickets,
            "reopen_rate": round(self.reopen_rate, 4),
            "common_errors": list(self.common_errors),
            "stage_timings": [
                {
                    "stage": timing.stage,
                    "average_hours": timing.average_hours,
                    "waiting_share": timing.waiting_share,
                }
                for timing in self.stage_timings
            ],
            "sample_period": self.sample_period,
        }


# ---------------------------------------------------------------------------
# The delta between the two pictures, and the plan that closes it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    """A single deviation between the documented standard and reality."""

    code: str
    title: str
    severity: Severity
    evidence: str
    stage: str | None = None
    lost_hours: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "stage": self.stage,
            "lost_hours": round(self.lost_hours, 2),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Gap":
        return cls(
            code=str(payload.get("code", "GAP")),
            title=str(payload.get("title", "")),
            severity=Severity(payload.get("severity", "medium")),
            evidence=str(payload.get("evidence", "")),
            stage=payload.get("stage"),
            lost_hours=float(payload.get("lost_hours", 0.0)),
        )


@dataclass(frozen=True)
class Constraint:
    """An infrastructure/policy limitation expressed by the human reviewer.

    Node 4 (Human-in-the-Loop) turns free-text feedback into these objects so
    that the next revision of the plan is *provably* compatible with them.
    """

    kind: str          # e.g. "preserve_control", "no_new_vendor", "generic"
    target: str = ""   # the step or capability the constraint applies to
    directive: str = ""  # what the human asked for instead
    source_feedback: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "directive": self.directive,
            "source_feedback": self.source_feedback,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Constraint":
        return cls(
            kind=str(payload.get("kind", "generic")),
            target=str(payload.get("target", "")),
            directive=str(payload.get("directive", "")),
            source_feedback=str(payload.get("source_feedback", "")),
        )

    def describe(self) -> str:
        parts = [self.kind.replace("_", " ")]
        if self.target:
            parts.append(f"-> {self.target}")
        if self.directive:
            parts.append(f"({self.directive})")
        return " ".join(parts)


@dataclass(frozen=True)
class AutomationProposal:
    """One concrete automation idea, optionally with runnable Python code."""

    identifier: str
    title: str
    rationale: str
    expected_gain_hours: float
    implementation_notes: tuple[str, ...] = ()
    #: ``cycle_time`` - the saving is subtracted from the end-to-end duration of
    #: every request. ``amortised`` - the saving (e.g. avoided rework) is spread
    #: over many requests and must NOT be subtracted from a single cycle.
    gain_kind: str = "cycle_time"
    code: str | None = None
    #: Controls this proposal explicitly keeps in place. Used to prove to the
    #: reviewer that a mandatory sign-off was *not* removed.
    preserved_controls: tuple[str, ...] = ()
    #: Controls this proposal would delete. A proposal is rejected when one of
    #: these is protected by a :class:`Constraint`.
    removed_controls: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "title": self.title,
            "rationale": self.rationale,
            "expected_gain_hours": round(self.expected_gain_hours, 2),
            "gain_kind": self.gain_kind,
            "implementation_notes": list(self.implementation_notes),
            "preserved_controls": list(self.preserved_controls),
            "removed_controls": list(self.removed_controls),
            "has_code": self.code is not None,
            "code": self.code,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AutomationProposal":
        return cls(
            identifier=str(payload.get("id", "proposal")),
            title=str(payload.get("title", "")),
            rationale=str(payload.get("rationale", "")),
            expected_gain_hours=float(payload.get("expected_gain_hours", 0.0)),
            implementation_notes=tuple(payload.get("implementation_notes", ())),
            gain_kind=str(payload.get("gain_kind", "cycle_time")),
            code=payload.get("code"),
            preserved_controls=tuple(payload.get("preserved_controls", ())),
            removed_controls=tuple(payload.get("removed_controls", ())),
        )


@dataclass(frozen=True)
class ActionPlan:
    """Expected output of Node 3 (Process Engineer): the full Action Plan."""

    process_name: str
    gaps: tuple[Gap, ...]
    proposals: tuple[AutomationProposal, ...]
    summary: str
    revision: int = 1
    honoured_constraints: tuple[Constraint, ...] = ()
    #: Measured end-to-end duration before the plan is implemented.
    baseline_hours: float = 0.0
    #: Documented SLA, kept next to the baseline so the plan is self-contained.
    sla_hours: float = 0.0

    @property
    def expected_gain_hours(self) -> float:
        """Every saving the plan promises, whatever its nature."""
        return sum(proposal.expected_gain_hours for proposal in self.proposals)

    @property
    def cycle_time_gain_hours(self) -> float:
        """Only the savings that shorten a single request end-to-end."""
        return sum(
            proposal.expected_gain_hours
            for proposal in self.proposals
            if proposal.gain_kind == "cycle_time"
        )

    @property
    def projected_hours(self) -> float:
        """Expected end-to-end duration once the plan is implemented."""
        return max(0.0, self.baseline_hours - self.cycle_time_gain_hours)

    @property
    def meets_sla(self) -> bool:
        return self.sla_hours > 0 and self.projected_hours <= self.sla_hours

    def to_payload(self) -> dict[str, Any]:
        """Machine-readable mirror of the plan, stored in ``action_plan``."""
        return {
            "process_name": self.process_name,
            "revision": self.revision,
            "summary": self.summary,
            "gaps": [gap.to_payload() for gap in self.gaps],
            "proposals": [proposal.to_payload() for proposal in self.proposals],
            "constraints": [c.to_payload() for c in self.honoured_constraints],
            "baseline_hours": self.baseline_hours,
            "sla_hours": self.sla_hours,
            "projected_hours": round(self.projected_hours, 2),
            "expected_gain_hours": round(self.expected_gain_hours, 2),
            "cycle_time_gain_hours": round(self.cycle_time_gain_hours, 2),
            "meets_sla": self.meets_sla,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActionPlan":
        return cls(
            process_name=str(payload.get("process_name", "")),
            gaps=tuple(Gap.from_payload(item) for item in payload.get("gaps", ())),
            proposals=tuple(
                AutomationProposal.from_payload(item)
                for item in payload.get("proposals", ())
            ),
            summary=str(payload.get("summary", "")),
            revision=int(payload.get("revision", 1)),
            honoured_constraints=tuple(
                Constraint.from_payload(item) for item in payload.get("constraints", ())
            ),
            baseline_hours=float(payload.get("baseline_hours", 0.0)),
            sla_hours=float(payload.get("sla_hours", 0.0)),
        )

    def to_markdown(self) -> str:
        """Human-readable rendering shown to the reviewer before approval."""
        lines = [
            f"# Action Plan - {self.process_name} (revision {self.revision})",
            "",
            self.summary,
            "",
            "## 1. Identified gaps",
        ]
        if self.gaps:
            for gap in self.gaps:
                stage = f" [stage: {gap.stage}]" if gap.stage else ""
                lines.append(
                    f"- **{gap.title}** ({gap.severity.value}){stage}\n"
                    f"  - evidence: {gap.evidence}"
                )
        else:
            lines.append("- No material deviation from the documented standard.")

        lines += ["", "## 2. Proposed automation"]
        if self.proposals:
            for proposal in self.proposals:
                lines.append(
                    f"### {proposal.title}\n"
                    f"{proposal.rationale}\n\n"
                    f"*Expected saving:* ~{proposal.expected_gain_hours:g} hour(s) per request."
                )
                for note in proposal.implementation_notes:
                    lines.append(f"- {note}")
                if proposal.preserved_controls:
                    lines.append(
                        "- Controls kept intact: "
                        + ", ".join(proposal.preserved_controls)
                    )
                if proposal.code:
                    lines.append(f"\n```python\n{proposal.code.strip()}\n```")
        else:
            lines.append("- No automation required.")

        if self.honoured_constraints:
            lines += ["", "## 3. Constraints honoured in this revision"]
            lines += [f"- {c.describe()}" for c in self.honoured_constraints]

        lines += ["", "## 4. Expected effect"]
        lines.append(
            f"- Cycle-time saving: ~{self.cycle_time_gain_hours:.1f} h per request."
        )
        amortised = self.expected_gain_hours - self.cycle_time_gain_hours
        if amortised > 0:
            lines.append(
                f"- Additional amortised saving (avoided rework): ~{amortised:.1f} h "
                f"per request on average."
            )
        if self.baseline_hours:
            verdict = "inside" if self.meets_sla else "still above"
            lines.append(
                f"- Projected end-to-end duration: **{self.baseline_hours:.1f} h -> "
                f"{self.projected_hours:.1f} h** ({verdict} the {self.sla_hours:g} h SLA)."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ApprovalDecision:
    """What the human reviewer (Node 4) decided about the current plan."""

    approved: bool
    feedback: str = ""
    reviewer: str = "systems-manager"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store, with its similarity score."""

    document_id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _normalise(value: str) -> str:
    """Lower-case and squeeze whitespace/punctuation for fuzzy title matching."""
    return " ".join(
        "".join(char if char.isalnum() or char.isspace() else " " for char in value)
        .lower()
        .split()
    )


__all__ = [
    "ActionPlan",
    "ApprovalDecision",
    "AutomationClass",
    "AutomationProposal",
    "Constraint",
    "Gap",
    "ProcessMetrics",
    "ProcessStep",
    "RetrievedChunk",
    "Severity",
    "StageTiming",
    "StandardProcess",
]
