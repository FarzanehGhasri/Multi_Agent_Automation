"""Catalogue of automation remedies (the Process Engineer's toolbox).

Every remedy is a small, independently testable *strategy*. The Process
Engineer agent never contains an ``if process == "VPN"`` branch: it asks the
registry which strategies apply to the gap it is looking at, filters out the
ones a human constraint forbids, and renders the survivors into the plan.

SOLID notes
-----------
* SRP - one strategy = one way of removing one kind of manual work.
* OCP - new remedies are *registered*, never wired with ``if/elif``. Adding
        support for, say, RPA bots means adding a class and one ``register``
        call; :class:`RemediationRegistry` and the agent stay untouched.
* LSP - every strategy can be substituted for any other: same ``applies`` /
        ``build`` contract, same immutable :class:`AutomationProposal` output.
* ISP - the strategy interface has exactly two methods, so implementers are
        never forced to provide behaviour they do not need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import (
    AutomationClass,
    AutomationProposal,
    Constraint,
    Gap,
    ProcessMetrics,
    ProcessStep,
    StageTiming,
    StandardProcess,
)

#: Constraint kind meaning "this control must stay in the flow".
PRESERVE_CONTROL = "preserve_control"


@dataclass(frozen=True)
class RemediationContext:
    """Everything a strategy needs to turn one gap into one proposal."""

    gap: Gap
    standard: StandardProcess
    metrics: ProcessMetrics
    step: ProcessStep | None = None       # the matching SOP step, when known
    timing: StageTiming | None = None     # the measured timing of that stage
    constraints: tuple[Constraint, ...] = ()

    @property
    def automation_class(self) -> AutomationClass | None:
        return self.step.automation_class if self.step else None

    @property
    def stage_hours(self) -> float:
        return self.timing.average_hours if self.timing else self.gap.lost_hours

    @property
    def target_hours(self) -> float:
        if self.step and self.step.target_hours is not None:
            return self.step.target_hours
        return 0.0

    def is_protected(self, label: str) -> bool:
        """True when the human reviewer forbade touching ``label``."""
        needle = label.lower()
        for constraint in self.constraints:
            if constraint.kind != PRESERVE_CONTROL:
                continue
            target = constraint.target.lower()
            if not target or target in needle or needle in target:
                return True
        return False


class RemediationStrategy(ABC):
    """Strategy interface: one reusable way of closing a gap."""

    identifier: str = "remedy"
    #: Higher priority wins when several strategies fit the same gap.
    priority: int = 0

    @abstractmethod
    def applies(self, context: RemediationContext) -> bool:
        """Can this strategy address the gap described by ``context``?"""

    @abstractmethod
    def build(self, context: RemediationContext) -> AutomationProposal:
        """Render the concrete proposal (text + code) for this context."""


# ---------------------------------------------------------------------------
# Strategies for manual approval steps
# ---------------------------------------------------------------------------


class MessengerApprovalStrategy(RemediationStrategy):
    """Keep the human decision, delete the human *waiting*.

    The approver is pushed an actionable card in the corporate messenger the
    moment the request is filed, and the downstream system API is called
    automatically as soon as the button is pressed. This is the remedy the
    project brief expects for the VPN scenario.
    """

    identifier = "messenger-approval-bot"
    priority = 60

    def applies(self, context: RemediationContext) -> bool:
        return context.automation_class is AutomationClass.MANUAL_APPROVAL

    def build(self, context: RemediationContext) -> AutomationProposal:
        step = context.step
        assert step is not None  # guaranteed by ``applies``
        # We keep the SOP target for the step and assume the bot can reach it;
        # everything above the target is the saving.
        gain = max(0.0, context.stage_hours - max(context.target_hours, 0.25))
        return AutomationProposal(
            identifier=self.identifier,
            title=f"Push-based approval bot for '{step.title}'",
            rationale=(
                f"'{step.title}' currently takes {context.stage_hours:g}h, almost all of "
                f"it queue time while the request waits in the approver's inbox. "
                f"The approval *decision* stays with {step.owner}; only the way the "
                f"request reaches them and the way the result is applied become "
                f"automatic."
            ),
            expected_gain_hours=gain,
            implementation_notes=(
                "On ticket creation, post an interactive approve/reject card to the "
                f"corporate messenger account of {step.owner}.",
                "Start an SLA timer; auto-escalate to the line manager's deputy when "
                f"no answer arrives within {max(context.target_hours, 1):g}h.",
                "On approval, call the downstream provisioning API directly instead of "
                "creating a second manual task.",
                "Write the approver identity, timestamp and payload to the audit log so "
                "the control remains fully traceable.",
            ),
            preserved_controls=(step.title,),
            code=_MESSENGER_APPROVAL_CODE,
        )


class ControlRemovalStrategy(RemediationStrategy):
    """Pure cycle-time optimisation: delete the sign-off altogether.

    This is intentionally the *highest priority* remedy for a review step: the
    engineer agent optimises for lead time and, on its first pass, does not
    know the organisation's infrastructure and compliance limits. That is
    precisely the situation the Human-in-the-Loop node exists for - the
    reviewer rejects it and the next revision picks a control-preserving
    strategy instead (see the second test scenario of the brief).
    """

    identifier = "control-removal"
    priority = 90

    def applies(self, context: RemediationContext) -> bool:
        if context.automation_class is not AutomationClass.SECURITY_REVIEW:
            return False
        # Once the reviewer protects this control, the strategy removes itself
        # from the candidate list - no special-casing needed in the agent.
        label = context.step.title if context.step else (context.gap.stage or "")
        return not context.is_protected(label)

    def build(self, context: RemediationContext) -> AutomationProposal:
        step = context.step
        assert step is not None
        return AutomationProposal(
            identifier=self.identifier,
            title=f"Replace '{step.title}' with an automated policy check",
            rationale=(
                f"'{step.title}' costs {context.stage_hours:g}h per request and is the "
                f"single largest contributor to the SLA breach. Encoding the review "
                f"criteria as machine-readable rules would let the request be granted "
                f"without waiting for {step.owner}."
            ),
            expected_gain_hours=max(0.0, context.stage_hours - 0.25),
            implementation_notes=(
                "Translate the review checklist into policy-as-code rules.",
                "Grant access automatically when every rule passes.",
                f"Retire the manual '{step.title}' queue.",
            ),
            removed_controls=(step.title,),
            code=_POLICY_AS_CODE,
        )


class SecurityFastTrackStrategy(RemediationStrategy):
    """Keep the security sign-off, make it fast.

    Selected automatically once the reviewer declares the control mandatory.
    Matches the alternative the brief expects: a dedicated dashboard that lets
    the security officer decide in minutes instead of days.
    """

    identifier = "security-fast-track"
    priority = 50

    def applies(self, context: RemediationContext) -> bool:
        return context.automation_class is AutomationClass.SECURITY_REVIEW

    def build(self, context: RemediationContext) -> AutomationProposal:
        step = context.step
        assert step is not None
        target = max(context.target_hours, 0.5)
        return AutomationProposal(
            identifier=self.identifier,
            title=f"Decision-support fast track for '{step.title}'",
            rationale=(
                f"The control stays exactly where the policy puts it"
                f"{' (' + step.control_reference + ')' if step.control_reference else ''}; "
                f"what is removed is the {context.stage_hours:g}h the officer currently "
                f"spends *collecting* evidence by hand. Every input the reviewer needs "
                f"is pre-assembled and pushed to a single queue."
            ),
            expected_gain_hours=max(0.0, context.stage_hours - target),
            implementation_notes=(
                "Compute a risk score at intake (requester role, data classification, "
                "requested privilege level, historical incidents).",
                "Pre-assemble the evidence pack: current entitlements, peer-group "
                "comparison, last access review, open incidents.",
                "Render one real-time dashboard queue for the security officer, ordered "
                "by risk and remaining SLA, with one-click approve/reject and a "
                "mandatory justification field.",
                "Auto-expire the granted privilege so that low-risk requests are "
                "time-boxed rather than permanent.",
                f"The {step.title} decision itself remains 100% human.",
            ),
            preserved_controls=(step.title,),
            code=_SECURITY_FAST_TRACK_CODE,
        )


# ---------------------------------------------------------------------------
# Strategies for provisioning / data-entry / queueing steps
# ---------------------------------------------------------------------------


class ProvisioningApiStrategy(RemediationStrategy):
    """Replace hand configuration of a target system with an API call."""

    identifier = "provisioning-api"
    priority = 55

    def applies(self, context: RemediationContext) -> bool:
        return context.automation_class is AutomationClass.MANUAL_PROVISIONING

    def build(self, context: RemediationContext) -> AutomationProposal:
        step = context.step
        assert step is not None
        return AutomationProposal(
            identifier=self.identifier,
            title=f"Direct API provisioning for '{step.title}'",
            rationale=(
                f"'{step.title}' is executed by hand by {step.owner}, which costs "
                f"{context.stage_hours:g}h and is the source of the recurring "
                f"configuration errors visible in the ticket data."
            ),
            expected_gain_hours=max(0.0, context.stage_hours - max(context.target_hours, 0.1)),
            implementation_notes=(
                "Model the target configuration as a versioned template.",
                "Apply it through the vendor API inside a transaction, with an "
                "automatic rollback when validation fails.",
                "Re-read the applied configuration and attach the diff to the ticket.",
            ),
            preserved_controls=(),
            code=_PROVISIONING_API_CODE,
        )


class DataEntrySyncStrategy(RemediationStrategy):
    """Stop humans from retyping the same data between two systems."""

    identifier = "system-to-system-sync"
    priority = 40

    def applies(self, context: RemediationContext) -> bool:
        return context.automation_class is AutomationClass.DATA_ENTRY

    def build(self, context: RemediationContext) -> AutomationProposal:
        step = context.step
        assert step is not None
        return AutomationProposal(
            identifier=self.identifier,
            title=f"System-to-system synchronisation for '{step.title}'",
            rationale=(
                f"'{step.title}' only moves data that both systems already hold. "
                f"Re-keying costs {context.stage_hours:g}h and injects typing errors."
            ),
            expected_gain_hours=max(0.0, context.stage_hours * 0.9),
            implementation_notes=(
                "Expose the source record through an event/webhook.",
                "Map the fields once, declaratively, and validate on write.",
                "Keep a reconciliation job for records that fail the mapping.",
            ),
            code=None,
        )


class QueueTriageStrategy(RemediationStrategy):
    """Kill dead time in a shared backlog with automatic triage."""

    identifier = "auto-triage"
    priority = 35

    def applies(self, context: RemediationContext) -> bool:
        return context.automation_class is AutomationClass.WAITING_QUEUE

    def build(self, context: RemediationContext) -> AutomationProposal:
        step = context.step
        assert step is not None
        return AutomationProposal(
            identifier=self.identifier,
            title=f"Automatic triage and assignment for '{step.title}'",
            rationale=(
                f"Requests sit {context.stage_hours:g}h in the '{step.title}' backlog "
                f"before anyone owns them."
            ),
            expected_gain_hours=max(0.0, context.stage_hours * 0.7),
            implementation_notes=(
                "Classify the request at intake and route it to the competent group.",
                "Assign a named owner immediately (round-robin on availability).",
                "Alert on SLA burn-down instead of relying on manual inspection.",
            ),
            code=None,
        )


class IntakeValidationStrategy(RemediationStrategy):
    """Fix rework and recurring errors at the source: the intake form."""

    identifier = "intake-validation"
    priority = 30

    def applies(self, context: RemediationContext) -> bool:
        return context.gap.code.startswith(("HIGH_REWORK", "RECURRING_ERROR"))

    def build(self, context: RemediationContext) -> AutomationProposal:
        errors = ", ".join(context.metrics.common_errors[:3]) or "incomplete requests"
        # One reopened ticket is assumed to cost a full extra cycle.
        gain = context.metrics.reopen_rate * context.metrics.average_resolution_hours
        return AutomationProposal(
            identifier=self.identifier,
            title="Validate the request at intake instead of after execution",
            rationale=(
                f"Rework is driven by defects that are only discovered downstream "
                f"({errors}). Each reopened ticket re-runs the whole process."
            ),
            expected_gain_hours=max(0.0, gain),
            gain_kind="amortised",
            implementation_notes=(
                "Make the intake form dynamic: required fields depend on the request "
                "type, and free text is replaced by pick-lists wherever possible.",
                "Validate against the source of truth (HR system, CMDB) before the "
                "ticket can be submitted.",
                "Block submission - rather than the whole downstream process - when a "
                "prerequisite is missing.",
            ),
            code=None,
        )


class GenericStreamliningStrategy(RemediationStrategy):
    """Fallback so that *every* gap gets at least one actionable answer."""

    identifier = "generic-streamlining"
    priority = 1

    def applies(self, context: RemediationContext) -> bool:
        # Everything except the aggregate SLA breach, which has no single
        # owner: it disappears when the stage-level gaps are fixed.
        return not context.gap.code.startswith("SLA_BREACH")

    def build(self, context: RemediationContext) -> AutomationProposal:
        stage = context.gap.stage or context.standard.process_name
        return AutomationProposal(
            identifier=self.identifier,
            title=f"Instrument and streamline '{stage}'",
            rationale=(
                f"{context.gap.title}. {context.gap.evidence} No dedicated automation "
                f"pattern matches this gap yet, so the first step is measurement."
            ),
            expected_gain_hours=max(0.0, context.gap.lost_hours * 0.3),
            implementation_notes=(
                "Add per-stage timestamps so the loss can be attributed precisely.",
                "Review the step with its owner and delete any activity that does not "
                "change the outcome of the request.",
                "Re-assess after one measurement period.",
            ),
            code=None,
        )


class RemediationRegistry:
    """Holds the available strategies and selects the best legal one.

    The registry is the single extension point of the Process Engineer: the
    agent asks it for a proposal and does not know which strategies exist.
    """

    def __init__(self, strategies: Sequence[RemediationStrategy] | None = None) -> None:
        self._strategies: list[RemediationStrategy] = list(
            strategies if strategies is not None else DEFAULT_STRATEGIES
        )

    def register(self, strategy: RemediationStrategy) -> "RemediationRegistry":
        """Add a strategy at runtime. Returns ``self`` so calls can be chained."""
        self._strategies.append(strategy)
        return self

    @property
    def strategies(self) -> tuple[RemediationStrategy, ...]:
        return tuple(self._strategies)

    def candidates(self, context: RemediationContext) -> tuple[RemediationStrategy, ...]:
        """All applicable strategies, most preferred first."""
        applicable = [s for s in self._strategies if s.applies(context)]
        applicable.sort(key=lambda strategy: strategy.priority, reverse=True)
        return tuple(applicable)

    def propose(
        self,
        context: RemediationContext,
        *,
        allow_control_removal: bool = True,
    ) -> AutomationProposal | None:
        """Return the best proposal that the current constraints permit.

        ``allow_control_removal=False`` makes the engineer policy-compliant
        from the very first revision; the default ``True`` reproduces the
        brief's second scenario, where the engineer first suggests dropping a
        mandatory sign-off and the human reviewer corrects it.
        """
        for strategy in self.candidates(context):
            proposal = strategy.build(context)
            if self._violates_constraints(proposal, context, allow_control_removal):
                continue
            return proposal
        return None

    @staticmethod
    def _violates_constraints(
        proposal: AutomationProposal,
        context: RemediationContext,
        allow_control_removal: bool,
    ) -> bool:
        """A proposal is illegal when it deletes a protected control."""
        for control in proposal.removed_controls:
            if context.is_protected(control):
                return True
            if not allow_control_removal:
                return True
            step = context.standard.step_by_title(control)
            if step is not None and step.is_mandatory_control and not allow_control_removal:
                return True
        return False


#: Default catalogue, ordered by declaration for readability only - the actual
#: selection order comes from each strategy's ``priority``.
DEFAULT_STRATEGIES: tuple[RemediationStrategy, ...] = (
    ControlRemovalStrategy(),
    MessengerApprovalStrategy(),
    ProvisioningApiStrategy(),
    SecurityFastTrackStrategy(),
    DataEntrySyncStrategy(),
    QueueTriageStrategy(),
    IntakeValidationStrategy(),
    GenericStreamliningStrategy(),
)


def build_contexts(
    gaps: Iterable[Gap],
    standard: StandardProcess,
    metrics: ProcessMetrics,
    constraints: Sequence[Constraint] = (),
) -> tuple[RemediationContext, ...]:
    """Join each gap with its SOP step and measured timing.

    Factored out of the agent so that the joining logic can be unit-tested on
    its own and reused by an LLM-backed composer.
    """
    timings = {timing.stage: timing for timing in metrics.stage_timings}
    contexts: list[RemediationContext] = []
    for gap in gaps:
        timing = timings.get(gap.stage) if gap.stage else None
        step = standard.step_by_title(gap.stage) if gap.stage else None
        contexts.append(
            RemediationContext(
                gap=gap,
                standard=standard,
                metrics=metrics,
                step=step,
                timing=timing,
                constraints=tuple(constraints),
            )
        )
    return tuple(contexts)


# ---------------------------------------------------------------------------
# Code templates attached to the proposals. They are deliberately runnable,
# self-contained sketches: the brief asks the engineer node to deliver
# "automation scripts", not prose.
# ---------------------------------------------------------------------------

_MESSENGER_APPROVAL_CODE = '''
"""Push-based approval: ask the manager in the messenger, provision on YES."""

import datetime as dt
import requests

MESSENGER_API = "https://messenger.corp.local/api/v1"
SERVICE_DESK_API = "https://servicedesk.corp.local/api/v2"
ESCALATION_AFTER = dt.timedelta(hours=1)


def request_approval(ticket: dict) -> dict:
    """Post an interactive approve/reject card to the approver."""
    response = requests.post(
        f"{MESSENGER_API}/cards",
        json={
            "to": ticket["approver_id"],
            "title": f"Access request {ticket['id']} - {ticket['requester']}",
            "body": ticket["justification"],
            "actions": [
                {"id": "approve", "label": "Approve", "style": "primary"},
                {"id": "reject", "label": "Reject", "style": "danger"},
            ],
            "callback_url": f"{SERVICE_DESK_API}/tickets/{ticket['id']}/decision",
            "expires_at": (dt.datetime.utcnow() + ESCALATION_AFTER).isoformat(),
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def on_decision(ticket: dict, decision: dict) -> None:
    """Callback invoked by the messenger when the approver clicks a button."""
    audit(ticket, decision)                       # the control stays traceable
    if decision["action"] != "approve":
        close_ticket(ticket, status="rejected", note=decision.get("comment", ""))
        return
    provision(ticket)                             # call the target system API
    close_ticket(ticket, status="done")


def escalate_if_silent(ticket: dict) -> None:
    """Re-route to the deputy when the SLA timer expires (no lost requests)."""
    if ticket["status"] == "awaiting_approval" and _age(ticket) > ESCALATION_AFTER:
        ticket["approver_id"] = ticket["deputy_approver_id"]
        request_approval(ticket)
'''.strip()

_PROVISIONING_API_CODE = '''
"""Provision the target system from a versioned template instead of by hand."""

import requests

FIREWALL_API = "https://firewall.corp.local/api/v3"


def provision(ticket: dict) -> dict:
    """Apply the access template, verify it, and attach the diff to the ticket."""
    payload = {
        "user": ticket["requester_uid"],
        "profile": ticket["access_profile"],       # template, not free-form rules
        "valid_until": ticket["valid_until"],      # every grant is time-boxed
        "ticket": ticket["id"],
    }
    applied = requests.post(f"{FIREWALL_API}/vpn-users", json=payload, timeout=15)
    applied.raise_for_status()

    # Read back and compare: configuration drift is detected immediately.
    current = requests.get(
        f"{FIREWALL_API}/vpn-users/{ticket['requester_uid']}", timeout=15
    ).json()
    if current["profile"] != payload["profile"]:
        rollback(ticket)
        raise RuntimeError(f"provisioning drift for ticket {ticket['id']}")
    return current
'''.strip()

_SECURITY_FAST_TRACK_CODE = '''
"""Fast-track the security review WITHOUT removing it.

The officer still decides; the script only removes the manual evidence
gathering and orders the queue by risk and remaining SLA.
"""

from dataclasses import dataclass

WEIGHTS = {"privilege": 0.4, "data_class": 0.3, "role_fit": 0.2, "history": 0.1}


@dataclass(frozen=True)
class ReviewItem:
    ticket_id: str
    risk_score: float
    sla_left_hours: float
    evidence: dict


def score(request: dict) -> float:
    """Cheap, explainable risk score - never an automatic approval."""
    signals = {
        "privilege": request["privilege_level"] / 5,
        "data_class": request["data_classification"] / 4,
        "role_fit": 1 - request["peer_group_match"],
        "history": min(request["past_incidents"], 3) / 3,
    }
    return round(sum(WEIGHTS[key] * value for key, value in signals.items()), 3)


def build_queue(requests_: list[dict]) -> list[ReviewItem]:
    """Everything the officer needs, pre-assembled, highest risk first."""
    items = [
        ReviewItem(
            ticket_id=request["id"],
            risk_score=score(request),
            sla_left_hours=request["sla_left_hours"],
            evidence={
                "current_entitlements": fetch_entitlements(request["requester_uid"]),
                "peer_comparison": fetch_peer_group(request["requester_uid"]),
                "last_access_review": fetch_last_review(request["requester_uid"]),
                "open_incidents": fetch_incidents(request["requester_uid"]),
            },
        )
        for request in requests_
    ]
    items.sort(key=lambda item: (-item.risk_score, item.sla_left_hours))
    return items


def record_decision(item: ReviewItem, officer_id: str, approved: bool, note: str) -> None:
    """The human decision is mandatory and always audited."""
    assert note, "a justification is required by the information-security policy"
    audit_log.write(
        ticket=item.ticket_id, officer=officer_id, approved=approved,
        risk_score=item.risk_score, note=note,
    )
'''.strip()

_POLICY_AS_CODE = '''
"""Encode the review checklist as rules and grant access automatically."""

RULES = (
    lambda request: request["privilege_level"] <= 2,
    lambda request: request["data_classification"] <= 2,
    lambda request: request["peer_group_match"] >= 0.8,
    lambda request: request["past_incidents"] == 0,
)


def auto_decide(request: dict) -> bool:
    """Grant when every rule passes - no human queue involved."""
    return all(rule(request) for rule in RULES)
'''.strip()


__all__ = [
    "ControlRemovalStrategy",
    "DEFAULT_STRATEGIES",
    "DataEntrySyncStrategy",
    "GenericStreamliningStrategy",
    "IntakeValidationStrategy",
    "MessengerApprovalStrategy",
    "PRESERVE_CONTROL",
    "ProvisioningApiStrategy",
    "QueueTriageStrategy",
    "RemediationContext",
    "RemediationRegistry",
    "RemediationStrategy",
    "SecurityFastTrackStrategy",
    "build_contexts",
]
