"""End-to-end tests of the graph, including the Human-in-the-Loop loop.

These are the automated equivalents of the two scenarios the project brief
asks to demonstrate in the notebook.
"""

from __future__ import annotations

import pytest

from process_automation import Settings, build_system
from process_automation.graph.routing import (
    FINISH,
    REVISE,
    build_review_router,
)
from process_automation.infrastructure.approval import (
    AutoApproveGateway,
    ScriptedApprovalGateway,
    approve,
    reject_with,
)
from process_automation.state import WorkflowState, initial_state


class TestState:
    def test_brief_contract_is_preserved(self) -> None:
        """The TypedDict required by the brief must keep its exact keys."""
        assert set(WorkflowState.__annotations__) == {
            "process_name",
            "standard_sop",
            "actual_metrics",
            "identified_gaps",
            "proposed_solution",
            "human_feedback",
        }

    def test_initial_state_populates_every_key(self) -> None:
        state = initial_state("x")
        assert state["process_name"] == "x"
        assert state["revision"] == 0 and state["approved"] is False


class TestRouter:
    def test_approved_plan_finishes(self) -> None:
        route = build_review_router(max_revisions=3)
        assert route({"approved": True, "revision": 1}) == FINISH

    def test_feedback_triggers_a_revision(self) -> None:
        route = build_review_router(max_revisions=3)
        assert route({"approved": False, "human_feedback": "no", "revision": 1}) == REVISE

    def test_revision_budget_stops_the_loop(self) -> None:
        route = build_review_router(max_revisions=2)
        assert route({"approved": False, "human_feedback": "no", "revision": 2}) == FINISH

    def test_silence_is_not_a_revision_request(self) -> None:
        route = build_review_router(max_revisions=3)
        assert route({"approved": False, "human_feedback": "", "revision": 1}) == FINISH


@pytest.fixture(scope="module")
def result():
    """Scenario 1 is executed once and asserted on from several angles."""
    system = build_system(approval_gateway=AutoApproveGateway())
    return system.session.run(
        "VPN access provisioning for new personnel", thread_id="test-vpn"
    )


@pytest.fixture(scope="module")
def run():
    """Scenario 2: the reviewer rejects revision 1, then approves revision 2."""
    gateway = ScriptedApprovalGateway(
        [
            reject_with(
                "The security approval cannot be removed. Propose a way to "
                "accelerate the security review instead."
            ),
            approve(),
        ]
    )
    system = build_system(approval_gateway=gateway)
    return system.session.run(
        "privileged access to a production database", thread_id="test-db"
    )


class TestScenarioOne:
    """VPN access: find the bottleneck and propose a successful automation."""

    def test_rag_extracted_the_documented_sla(self, result) -> None:
        assert "2 hour(s)" in result.state["standard_sop"]
        assert result.state["sop_document_id"] == "SOP-IT-014"

    def test_tools_returned_the_real_metrics(self, result) -> None:
        assert result.state["actual_metrics"]["average_resolution_hours"] == 42.0

    def test_the_manager_approval_stage_is_identified_as_the_bottleneck(
        self, result
    ) -> None:
        bottlenecks = [
            gap for gap in result.state["identified_gaps"]
            if gap["code"].startswith("BOTTLENECK")
        ]
        assert bottlenecks[0]["stage"] == "Direct manager approval"

    def test_the_plan_automates_the_approval_and_ships_code(self, result) -> None:
        proposals = result.state["action_plan"]["proposals"]
        approval_proposal = next(
            p for p in proposals if p["id"] == "messenger-approval-bot"
        )
        assert approval_proposal["has_code"] is True
        # The decision stays human - only the waiting is automated.
        assert approval_proposal["preserved_controls"] == ["Direct manager approval"]
        assert approval_proposal["removed_controls"] == []

    def test_the_firewall_api_call_is_proposed(self, result) -> None:
        ids = {p["id"] for p in result.state["action_plan"]["proposals"]}
        assert "provisioning-api" in ids

    def test_it_was_approved_in_a_single_pass(self, result) -> None:
        assert result.approved and result.revisions == 1 and result.interrupts == 1

    def test_the_graph_really_paused_for_the_human(self, result) -> None:
        assert result.checkpoints == ("interrupt #1 before 'human_review'",)

    def test_the_trace_records_rag_and_tool_calls(self, result) -> None:
        trace = "\n".join(result.trace)
        assert "RAG lookup" in trace
        assert "calling tool(s) fetch_process_metrics" in trace


class TestScenarioTwo:
    """Privileged DB access: the human overrules an unsafe proposal."""

    def test_the_first_revision_proposed_removing_the_control(self, run) -> None:
        """The engineer optimises for lead time before it is told the limits."""
        assert run.decisions[0].approved is False
        assert "cannot be removed" in run.decisions[0].feedback

    def test_the_feedback_became_a_machine_checkable_constraint(self, run) -> None:
        constraints = run.state["constraints"]
        assert constraints[0]["kind"] == "preserve_control"
        assert constraints[0]["target"] == "Security officer review"

    def test_the_second_revision_keeps_the_control(self, run) -> None:
        proposals = run.state["action_plan"]["proposals"]
        assert all(p["removed_controls"] == [] for p in proposals)
        fast_track = next(p for p in proposals if p["id"] == "security-fast-track")
        assert fast_track["preserved_controls"] == ["Security officer review"]

    def test_the_alternative_is_the_security_officer_dashboard(self, run) -> None:
        fast_track = next(
            p for p in run.state["action_plan"]["proposals"]
            if p["id"] == "security-fast-track"
        )
        notes = " ".join(fast_track["implementation_notes"]).lower()
        assert "dashboard" in notes and "risk score" in notes

    def test_the_graph_paused_twice_and_ended_approved(self, run) -> None:
        assert run.interrupts == 2
        assert run.revisions == 2
        assert run.approved is True

    def test_feedback_is_logged_for_the_audit_trail(self, run) -> None:
        assert any("revision 1:" in entry for entry in run.state["feedback_log"])


class TestSystemAssembly:
    def test_the_container_wires_every_component(self) -> None:
        system = build_system(Settings())
        assert len(system.knowledge_base) == 3
        assert system.vector_store.size > 0
        assert system.metrics_provider.tool_names == (
            "fetch_process_metrics",
            "fetch_stage_timings",
        )
        assert system.graph_builder.node_names == (
            "documentation_agent",
            "diagnostic_agent",
            "process_engineer",
            "human_review",
        )

    def test_an_unmonitored_process_degrades_gracefully(self) -> None:
        """No data must produce an honest answer, never an invented plan."""
        system = build_system(approval_gateway=AutoApproveGateway())
        result = system.session.run("payroll reconciliation", thread_id="test-unknown")
        assert result.state["identified_gaps"] == []
        assert "cannot be compared" in result.state["proposed_solution"]

    def test_a_policy_compliant_engineer_never_proposes_a_removal(self) -> None:
        """Flipping one setting changes the behaviour - no code change."""
        settings = Settings()
        settings = settings.__class__(
            retrieval=settings.retrieval,
            diagnostics=settings.diagnostics,
            engineering=settings.engineering.__class__(allow_control_removal=False),
            review=settings.review,
        )
        system = build_system(settings, approval_gateway=AutoApproveGateway())
        result = system.session.run(
            "privileged access to a production database", thread_id="test-strict"
        )
        proposals = result.state["action_plan"]["proposals"]
        assert all(p["removed_controls"] == [] for p in proposals)
        assert result.revisions == 1  # approved on the first pass
