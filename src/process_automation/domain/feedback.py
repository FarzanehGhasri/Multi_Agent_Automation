"""Turn free-text reviewer feedback into machine-checkable constraints.

Node 4 of the brief lets the systems manager reject a plan with a sentence
such as *"the security approval cannot be removed, propose a way to speed the
security review up instead"*. That sentence has to become something the
Process Engineer can actually obey, otherwise the next revision is a coin
flip. This module performs that translation.

SOLID notes
-----------
* ISP - :class:`FeedbackInterpreter` exposes a single method, so an LLM-backed
        implementation is trivial to plug in beside the rule-based one.
* OCP - new vocabulary is added to the keyword tables, not to the algorithm.
* DIP - the Process Engineer depends on the abstract interpreter, never on the
        keyword tables below.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from .models import Constraint, StandardProcess
from .remediation import PRESERVE_CONTROL

#: Phrases meaning "you may not delete this step" (English + Persian).
_PRESERVE_PATTERNS: tuple[str, ...] = (
    r"can(?:no|')?t be (?:removed|deleted|eliminated|skipped|bypassed)",
    r"cannot be (?:removed|deleted|eliminated|skipped|bypassed)",
    r"must (?:stay|remain|be kept|not be removed)",
    r"(?:is|are) mandatory",
    r"non[- ]?removable",
    r"do(?:n't| not) (?:remove|delete|skip|bypass)",
    r"keep the .* (?:approval|review|sign[- ]?off|control)",
    r"غیر\s*قابل\s*حذف",
    r"نباید\s*حذف",
    r"نمی\s*توان\s*حذف",
    r"الزامی\s*(?:است|می\s*باشد)",
    r"حذف\s*نشود",
)

#: Phrases meaning "make it faster instead of deleting it".
_ACCELERATE_PATTERNS: tuple[str, ...] = (
    r"speed (?:it |the .* )?up",
    r"accelerat",
    r"faster",
    r"reduce the (?:time|duration|wait)",
    r"تسریع",
    r"سریع\s*تر",
    r"کاهش\s*زمان",
)

#: Words that identify *which* control the reviewer is protecting.
_TARGET_HINTS: dict[str, tuple[str, ...]] = {
    "security": ("security", "infosec", "امنیت", "امنیتی"),
    "manager": ("manager", "line manager", "مدیر", "سرپرست"),
    "legal": ("legal", "compliance", "حقوقی", "انطباق"),
    "finance": ("finance", "budget", "مالی", "بودجه"),
}


class FeedbackInterpreter(ABC):
    """Abstraction: free text in, constraints out."""

    @abstractmethod
    def interpret(self, feedback: str, standard: StandardProcess) -> tuple[Constraint, ...]:
        """Extract the constraints expressed by ``feedback``."""


class RuleBasedFeedbackInterpreter(FeedbackInterpreter):
    """Deterministic interpreter driven by keyword patterns.

    Deterministic on purpose: a compliance constraint must never depend on the
    sampling temperature of a language model. The LLM-backed interpreter (see
    :mod:`process_automation.infrastructure.llm`) is offered as an optional
    enrichment on top of - not instead of - these rules.
    """

    def interpret(self, feedback: str, standard: StandardProcess) -> tuple[Constraint, ...]:
        text = (feedback or "").strip()
        if not text:
            return ()

        lowered = text.lower()
        constraints: list[Constraint] = []

        if _matches_any(lowered, _PRESERVE_PATTERNS):
            target = self._resolve_target(lowered, standard)
            directive = (
                "accelerate the control instead of removing it"
                if _matches_any(lowered, _ACCELERATE_PATTERNS)
                else "keep the control in the flow"
            )
            constraints.append(
                Constraint(
                    kind=PRESERVE_CONTROL,
                    target=target,
                    directive=directive,
                    source_feedback=text,
                )
            )

        if not constraints:
            # Unrecognised feedback is still carried forward so that the next
            # revision - and any LLM-backed composer - can take it into
            # account, and so nothing the human said is silently dropped.
            constraints.append(
                Constraint(kind="generic", directive=text, source_feedback=text)
            )
        return tuple(constraints)

    @staticmethod
    def _resolve_target(lowered: str, standard: StandardProcess) -> str:
        """Find the SOP step the reviewer is protecting.

        Three levels of precision, best first:
        1. an explicit SOP step title mentioned in the feedback;
        2. a domain hint word ("security", "امنیت", ...) mapped to a step;
        3. the process' first mandatory control, as a safe default.
        """
        for step in standard.steps:
            if step.title.lower() in lowered:
                return step.title

        for hint_key, words in _TARGET_HINTS.items():
            if not any(word in lowered for word in words):
                continue
            for step in standard.steps:
                haystack = f"{step.title} {step.owner}".lower()
                if hint_key in haystack or any(word in haystack for word in words):
                    return step.title
            return hint_key

        controls = standard.mandatory_controls
        return controls[0].title if controls else ""


def merge_constraints(
    existing: Sequence[Constraint], new: Sequence[Constraint]
) -> tuple[Constraint, ...]:
    """Accumulate constraints across revisions without duplicating them.

    Constraints are *sticky*: a limitation the reviewer stated in revision 1
    still holds in revision 3, which is what makes the feedback loop converge.
    """
    merged: list[Constraint] = list(existing)
    seen = {(c.kind, c.target.lower()) for c in existing}
    for constraint in new:
        key = (constraint.kind, constraint.target.lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(constraint)
    return tuple(merged)


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


__all__ = [
    "FeedbackInterpreter",
    "RuleBasedFeedbackInterpreter",
    "merge_constraints",
]
