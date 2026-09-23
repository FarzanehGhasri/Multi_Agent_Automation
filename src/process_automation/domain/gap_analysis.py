"""Gap analysis: compare the *should-be* SOP with the *as-is* metrics.

This is deterministic business logic, deliberately kept out of the agents and
out of any LLM prompt: an auditable "systems & methods" calculation must give
the same answer every time it is run on the same data.

SOLID notes
-----------
* SRP - :class:`GapAnalyzer` only *detects* deviations. It never proposes a
        remedy (that is :mod:`remediation`'s job) and never talks to a graph.
* OCP - detection logic lives in small ``GapRule`` objects. A new kind of
        deviation is added by writing a new rule and registering it, without
        touching :class:`GapAnalyzer`.
* LSP - every rule honours the same contract: given the comparison context it
        returns zero or more :class:`~.models.Gap` objects and mutates nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Gap, ProcessMetrics, Severity, StandardProcess


@dataclass(frozen=True)
class ComparisonContext:
    """Everything a rule needs to judge one process. Immutable on purpose."""

    standard: StandardProcess
    metrics: ProcessMetrics

    @property
    def sla_overrun_hours(self) -> float:
        """Hours spent beyond the documented SLA (0 when inside the SLA)."""
        return max(0.0, self.metrics.average_resolution_hours - self.standard.sla_hours)

    @property
    def sla_ratio(self) -> float:
        """How many times the SLA the process actually consumes."""
        if self.standard.sla_hours <= 0:
            return 0.0
        return self.metrics.average_resolution_hours / self.standard.sla_hours


class GapRule(ABC):
    """Strategy interface: one detectable class of deviation."""

    #: Stable identifier, also used as the prefix of the produced gap codes.
    code: str = "GAP"

    @abstractmethod
    def evaluate(self, context: ComparisonContext) -> Iterable[Gap]:
        """Return the gaps this rule finds in ``context`` (possibly none)."""
        raise NotImplementedError


class SlaBreachRule(GapRule):
    """The end-to-end duration exceeds the SLA written in the SOP."""

    code = "SLA_BREACH"

    def __init__(self, tolerance: float = 1.1) -> None:
        # ``tolerance`` absorbs normal statistical noise: only a breach above
        # 110% of the SLA (by default) is reported as a real gap.
        self._tolerance = tolerance

    def evaluate(self, context: ComparisonContext) -> Iterable[Gap]:
        if context.sla_ratio <= self._tolerance:
            return ()
        ratio = context.sla_ratio
        severity = (
            Severity.CRITICAL if ratio >= 5
            else Severity.HIGH if ratio >= 2
            else Severity.MEDIUM
        )
        return (
            Gap(
                code=self.code,
                title="End-to-end duration breaches the documented SLA",
                severity=severity,
                evidence=(
                    f"SOP target is {context.standard.sla_hours:g}h but the measured "
                    f"average is {context.metrics.average_resolution_hours:g}h "
                    f"({ratio:.1f}x the SLA, +{context.sla_overrun_hours:g}h)."
                ),
                lost_hours=context.sla_overrun_hours,
            ),
        )


class BottleneckRule(GapRule):
    """A single stage swallows a disproportionate share of the cycle time."""

    code = "BOTTLENECK"

    def __init__(self, share_threshold: float = 0.10) -> None:
        self._share_threshold = share_threshold

    def evaluate(self, context: ComparisonContext) -> Iterable[Gap]:
        gaps: list[Gap] = []
        for timing in context.metrics.stage_timings:
            share = context.metrics.share_of_total(timing)
            if share < self._share_threshold:
                continue
            step = context.standard.step_by_title(timing.stage)
            target = step.target_hours if step and step.target_hours else None
            target_text = (
                f" SOP target for this step is {target:g}h."
                if target is not None
                else ""
            )
            gaps.append(
                Gap(
                    code=f"{self.code}::{timing.stage}",
                    title=f"Bottleneck at stage '{timing.stage}'",
                    severity=Severity.HIGH if share >= 0.5 else Severity.MEDIUM,
                    evidence=(
                        f"'{timing.stage}' consumes {timing.average_hours:g}h "
                        f"({share:.0%} of the total cycle time), of which "
                        f"{timing.waiting_share:.0%} is pure waiting.{target_text}"
                    ),
                    stage=timing.stage,
                    # Hours that could realistically be recovered: everything
                    # above the documented target for that step.
                    lost_hours=max(0.0, timing.average_hours - (target or 0.0)),
                )
            )
        return gaps


class ReworkRule(GapRule):
    """Too many tickets come back - a quality / first-time-right problem."""

    code = "HIGH_REWORK"

    def __init__(self, max_reopen_rate: float = 0.10) -> None:
        self._max_reopen_rate = max_reopen_rate

    def evaluate(self, context: ComparisonContext) -> Iterable[Gap]:
        rate = context.metrics.reopen_rate
        if rate <= self._max_reopen_rate:
            return ()
        return (
            Gap(
                code=self.code,
                title="Rework rate above the acceptable threshold",
                severity=Severity.HIGH if rate >= 0.2 else Severity.MEDIUM,
                evidence=(
                    f"{context.metrics.reopened_tickets} of "
                    f"{context.metrics.ticket_volume} tickets were reopened "
                    f"({rate:.0%}); the accepted ceiling is "
                    f"{self._max_reopen_rate:.0%}."
                ),
            ),
        )


class RecurringErrorRule(GapRule):
    """Operators keep hitting the same defects - a candidate for validation."""

    code = "RECURRING_ERROR"

    def __init__(self, max_reported: int = 3) -> None:
        self._max_reported = max_reported

    def evaluate(self, context: ComparisonContext) -> Iterable[Gap]:
        errors = context.metrics.common_errors[: self._max_reported]
        return tuple(
            Gap(
                code=f"{self.code}::{index}",
                title=f"Recurring operational error: {error}",
                severity=Severity.MEDIUM,
                evidence=f"Reported repeatedly during {context.metrics.sample_period}.",
            )
            for index, error in enumerate(errors, start=1)
        )


class UndocumentedStageRule(GapRule):
    """A stage exists in reality but not in the approved flowchart."""

    code = "UNDOCUMENTED_STAGE"

    def evaluate(self, context: ComparisonContext) -> Iterable[Gap]:
        return tuple(
            Gap(
                code=f"{self.code}::{timing.stage}",
                title=f"Stage '{timing.stage}' is executed but not documented",
                severity=Severity.MEDIUM,
                evidence=(
                    f"The monitoring data reports {timing.average_hours:g}h spent in "
                    f"'{timing.stage}', which does not appear in the approved SOP."
                ),
                stage=timing.stage,
                lost_hours=timing.average_hours,
            )
            for timing in context.metrics.stage_timings
            if context.standard.step_by_title(timing.stage) is None
        )


#: The rule set used by default. Extending the analysis = appending here or
#: passing a custom sequence to :class:`GapAnalyzer` (Open/Closed Principle).
DEFAULT_RULES: tuple[GapRule, ...] = (
    SlaBreachRule(),
    BottleneckRule(),
    ReworkRule(),
    RecurringErrorRule(),
    UndocumentedStageRule(),
)


class GapAnalyzer:
    """Runs a set of :class:`GapRule` strategies and ranks the findings."""

    def __init__(self, rules: Sequence[GapRule] | None = None) -> None:
        # Dependency Injection: the caller decides which rules apply. The
        # default keeps the common case a one-liner.
        self._rules: tuple[GapRule, ...] = tuple(rules if rules is not None else DEFAULT_RULES)

    @property
    def rules(self) -> tuple[GapRule, ...]:
        return self._rules

    def analyse(self, standard: StandardProcess, metrics: ProcessMetrics) -> tuple[Gap, ...]:
        """Compare the two pictures and return the gaps, worst one first."""
        context = ComparisonContext(standard=standard, metrics=metrics)
        found: list[Gap] = []
        for rule in self._rules:
            found.extend(rule.evaluate(context))
        # Sort by severity, then by the number of hours the gap costs, so that
        # the Process Engineer always works on the most valuable item first.
        found.sort(key=lambda gap: (gap.severity.rank, gap.lost_hours), reverse=True)
        return tuple(found)


__all__ = [
    "BottleneckRule",
    "ComparisonContext",
    "DEFAULT_RULES",
    "GapAnalyzer",
    "GapRule",
    "RecurringErrorRule",
    "ReworkRule",
    "SlaBreachRule",
    "UndocumentedStageRule",
]
