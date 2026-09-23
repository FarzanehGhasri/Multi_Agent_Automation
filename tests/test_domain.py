"""Unit tests for the framework-free domain layer."""

from __future__ import annotations

import pytest

from process_automation.domain.feedback import (
    RuleBasedFeedbackInterpreter,
    merge_constraints,
)
from process_automation.domain.gap_analysis import GapAnalyzer, GapRule
from process_automation.domain.models import (
    AutomationClass,
    Constraint,
    Gap,
    ProcessMetrics,
    ProcessStep,
    Severity,
    StageTiming,
    StandardProcess,
)
from process_automation.domain.remediation import (
    PRESERVE_CONTROL,
    RemediationRegistry,
    build_contexts,
)


@pytest.fixture()
def standard() -> StandardProcess:
    return StandardProcess(
        process_name="Privileged database access provisioning",
        sla_hours=8.0,
        steps=(
            ProcessStep(1, "Request registration", "Bot", AutomationClass.AUTOMATED, 0.2),
            ProcessStep(2, "Direct manager approval", "Manager",
                        AutomationClass.MANUAL_APPROVAL, 1.0),
            ProcessStep(3, "Security officer review", "ISO",
                        AutomationClass.SECURITY_REVIEW, 4.0,
                        is_mandatory_control=True, control_reference="ISMS-POL-07"),
        ),
    )


@pytest.fixture()
def metrics() -> ProcessMetrics:
    return ProcessMetrics(
        process_name="Privileged database access provisioning",
        average_resolution_hours=26.0,
        ticket_volume=85,
        reopened_tickets=9,
        common_errors=("vague justification",),
        stage_timings=(
            StageTiming("Request registration", 0.5),
            StageTiming("Direct manager approval", 3.5, 0.8),
            StageTiming("Security officer review", 18.0, 0.8),
        ),
        sample_period="2024-Q4",
    )


class TestModels:
    def test_reopen_rate_and_bottleneck(self, metrics: ProcessMetrics) -> None:
        assert metrics.reopen_rate == pytest.approx(9 / 85)
        assert metrics.bottleneck is not None
        assert metrics.bottleneck.stage == "Security officer review"

    def test_metrics_payload_round_trip(self, metrics: ProcessMetrics) -> None:
        assert ProcessMetrics.from_payload(metrics.to_payload()) == metrics

    def test_standard_payload_round_trip(self, standard: StandardProcess) -> None:
        assert StandardProcess.from_payload(standard.to_payload()) == standard

    def test_step_lookup_is_fuzzy(self, standard: StandardProcess) -> None:
        assert standard.step_by_title("security officer review") is not None
        assert standard.step_by_title("nonexistent stage") is None

    def test_mandatory_controls(self, standard: StandardProcess) -> None:
        titles = [step.title for step in standard.mandatory_controls]
        assert titles == ["Security officer review"]


class TestGapAnalyzer:
    def test_detects_sla_breach_and_bottleneck(
        self, standard: StandardProcess, metrics: ProcessMetrics
    ) -> None:
        gaps = GapAnalyzer().analyse(standard, metrics)
        codes = {gap.code for gap in gaps}
        assert "SLA_BREACH" in codes
        assert "BOTTLENECK::Security officer review" in codes

    def test_gaps_are_sorted_worst_first(
        self, standard: StandardProcess, metrics: ProcessMetrics
    ) -> None:
        gaps = GapAnalyzer().analyse(standard, metrics)
        ranks = [gap.severity.rank for gap in gaps]
        assert ranks == sorted(ranks, reverse=True)

    def test_compliant_process_yields_no_breach(self, standard: StandardProcess) -> None:
        healthy = ProcessMetrics(
            process_name=standard.process_name,
            average_resolution_hours=7.0,   # inside the 8h SLA
            ticket_volume=50,
            reopened_tickets=1,
        )
        codes = {gap.code for gap in GapAnalyzer().analyse(standard, healthy)}
        assert "SLA_BREACH" not in codes

    def test_rule_set_is_injectable(
        self, standard: StandardProcess, metrics: ProcessMetrics
    ) -> None:
        """Open/Closed: a custom rule set changes the analysis, not the class."""

        class AlwaysOneGap(GapRule):
            code = "CUSTOM"

            def evaluate(self, context):  # noqa: ANN001 - signature fixed by the ABC
                return (Gap("CUSTOM", "custom", Severity.LOW, "injected"),)

        gaps = GapAnalyzer([AlwaysOneGap()]).analyse(standard, metrics)
        assert [gap.code for gap in gaps] == ["CUSTOM"]


class TestFeedbackInterpreter:
    @pytest.mark.parametrize(
        "feedback",
        [
            "The security approval cannot be removed. Please accelerate it instead.",
            "تاییدیه امنیتی غیرقابل حذف است. راهکاری برای تسریع بررسی امنیتی پیشنهاد بده.",
        ],
    )
    def test_detects_protected_control_in_both_languages(
        self, feedback: str, standard: StandardProcess
    ) -> None:
        constraints = RuleBasedFeedbackInterpreter().interpret(feedback, standard)
        assert constraints[0].kind == PRESERVE_CONTROL
        assert constraints[0].target == "Security officer review"

    def test_unrecognised_feedback_is_kept_as_generic(
        self, standard: StandardProcess
    ) -> None:
        constraints = RuleBasedFeedbackInterpreter().interpret("do it cheaper", standard)
        assert constraints[0].kind == "generic"
        assert constraints[0].directive == "do it cheaper"

    def test_empty_feedback_yields_nothing(self, standard: StandardProcess) -> None:
        assert RuleBasedFeedbackInterpreter().interpret("   ", standard) == ()

    def test_constraints_accumulate_without_duplicates(self) -> None:
        first = (Constraint(PRESERVE_CONTROL, "Security officer review"),)
        merged = merge_constraints(first, first + (Constraint("generic", "", "cheaper"),))
        assert len(merged) == 2


class TestRemediationRegistry:
    def _context_for_security_stage(self, standard, metrics, constraints=()):
        gaps = GapAnalyzer().analyse(standard, metrics)
        security_gap = next(
            gap for gap in gaps if gap.stage == "Security officer review"
        )
        return build_contexts([security_gap], standard, metrics, constraints)[0]

    def test_without_constraints_it_proposes_removing_the_control(
        self, standard: StandardProcess, metrics: ProcessMetrics
    ) -> None:
        context = self._context_for_security_stage(standard, metrics)
        proposal = RemediationRegistry().propose(context)
        assert proposal is not None
        assert proposal.removed_controls == ("Security officer review",)

    def test_a_protected_control_forces_the_fast_track_alternative(
        self, standard: StandardProcess, metrics: ProcessMetrics
    ) -> None:
        constraints = (Constraint(PRESERVE_CONTROL, "Security officer review"),)
        context = self._context_for_security_stage(standard, metrics, constraints)
        proposal = RemediationRegistry().propose(context)
        assert proposal is not None
        assert proposal.removed_controls == ()
        assert proposal.preserved_controls == ("Security officer review",)
        assert proposal.identifier == "security-fast-track"

    def test_policy_compliant_engineer_never_removes_a_control(
        self, standard: StandardProcess, metrics: ProcessMetrics
    ) -> None:
        context = self._context_for_security_stage(standard, metrics)
        proposal = RemediationRegistry().propose(context, allow_control_removal=False)
        assert proposal is not None
        assert proposal.removed_controls == ()
