"""Tests for the adapters: retrieval, tools and approval gateways."""

from __future__ import annotations

import pytest

from process_automation.config import Settings
from process_automation.domain.models import ActionPlan, AutomationClass
from process_automation.infrastructure.approval import (
    ConsoleApprovalGateway,
    ScriptedApprovalGateway,
    approve,
    reject_with,
)
from process_automation.infrastructure.embeddings import (
    HashingTfidfEmbeddings,
    cosine_similarity,
    tokenize,
)
from process_automation.infrastructure.knowledge_base import MarkdownKnowledgeBaseLoader
from process_automation.infrastructure.retrieval import (
    InMemoryVectorStore,
    StructuredSopParser,
)
from process_automation.infrastructure.tools import (
    ProcessNotMonitored,
    ServiceDeskDataset,
    ServiceDeskMetricsProvider,
    ServiceDeskToolFactory,
)

SETTINGS = Settings()


@pytest.fixture(scope="module")
def knowledge_base():
    return MarkdownKnowledgeBaseLoader(SETTINGS.retrieval.knowledge_base_dir).load()


@pytest.fixture(scope="module")
def tools():
    dataset = ServiceDeskDataset.from_json(SETTINGS.diagnostics.dataset_path)
    return ServiceDeskToolFactory(dataset).build()


@pytest.fixture(scope="module")
def vector_store(knowledge_base):
    return InMemoryVectorStore.from_knowledge_base(
        knowledge_base, HashingTfidfEmbeddings(SETTINGS.retrieval.embedding_dimensions)
    )


class TestEmbeddings:
    def test_tokenizer_drops_stopwords_and_keeps_persian(self) -> None:
        tokens = tokenize("The VPN access برای پرسنل جدید")
        assert "vpn" in tokens and "the" not in tokens
        assert "پرسنل" in tokens

    def test_vectors_are_deterministic_across_instances(self) -> None:
        corpus = ["vpn access provisioning", "database privileged access"]
        first = HashingTfidfEmbeddings(64).embed_documents(corpus)
        second = HashingTfidfEmbeddings(64).embed_documents(corpus)
        assert first == second

    def test_similar_text_scores_higher_than_unrelated_text(self) -> None:
        model = HashingTfidfEmbeddings(256)
        model.fit(["vpn remote access firewall", "database privilege grant dba"])
        query = model.embed_query("vpn firewall access")
        close = model.embed_query("vpn remote access firewall")
        far = model.embed_query("database privilege grant dba")
        assert cosine_similarity(query, close) > cosine_similarity(query, far)


class TestKnowledgeBase:
    def test_loads_every_sop_document(self, knowledge_base) -> None:
        assert len(knowledge_base) == 3
        assert knowledge_base.get("SOP-IT-014") is not None

    def test_parses_sla_steps_and_controls(self, knowledge_base) -> None:
        document = knowledge_base.get("SOP-SEC-021")
        assert document.sla_hours == 8.0
        assert len(document.steps) == 5
        control = document.steps[2]
        assert control.automation_class is AutomationClass.SECURITY_REVIEW
        assert control.is_mandatory_control
        assert control.control_reference == "ISMS-POL-07"

    def test_step_targets_sum_to_the_documented_sla(self, knowledge_base) -> None:
        """A consistency check on the corpus itself, not only on the parser."""
        for document in knowledge_base.documents:
            total = sum(step.target_hours or 0.0 for step in document.steps)
            assert total == pytest.approx(document.sla_hours)

    def test_policy_notes_are_extracted(self, knowledge_base) -> None:
        notes = knowledge_base.get("SOP-SEC-021").policy_notes
        assert any("ISMS-POL-07" in note for note in notes)


class TestRetrieval:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("VPN access for a new employee", "SOP-IT-014"),
            ("privileged access to a production database", "SOP-SEC-021"),
            ("level 2 support ticket handling", "SOP-IT-008"),
            ("تخصیص دسترسی VPN به پرسنل جدید", "SOP-IT-014"),
        ],
    )
    def test_retrieves_the_right_document(self, vector_store, query, expected) -> None:
        chunks = vector_store.retrieve(query, top_k=4)
        assert StructuredSopParser.select_document(chunks) == expected

    def test_parser_returns_the_structured_standard(
        self, vector_store, knowledge_base
    ) -> None:
        parser = StructuredSopParser(knowledge_base)
        chunks = vector_store.retrieve("VPN access for a new employee", top_k=4)
        standard = parser.parse("vpn", chunks)
        assert standard.sla_hours == 2.0
        assert standard.steps[1].automation_class is AutomationClass.MANUAL_APPROVAL

    def test_no_result_yields_an_empty_standard_not_a_guess(
        self, knowledge_base
    ) -> None:
        standard = StructuredSopParser(knowledge_base).parse("payroll", [])
        assert standard.sla_hours == 0.0
        assert standard.steps == ()


class TestTools:
    def test_tools_expose_llm_readable_contracts(self, tools) -> None:
        names = {tool.name for tool in tools}
        assert names == {"fetch_process_metrics", "fetch_stage_timings"}
        assert all(tool.description for tool in tools)

    def test_metrics_tool_returns_the_expected_payload(self, tools) -> None:
        payload = tools[0].invoke({"process_name": "SOP-IT-014"})
        assert payload["average_resolution_hours"] == 42.0
        assert payload["reopened_tickets"] == 18

    def test_dataset_resolves_a_free_form_process_name(self) -> None:
        dataset = ServiceDeskDataset.from_json(SETTINGS.diagnostics.dataset_path)
        assert dataset.resolve("VPN access provisioning for new personnel") == "SOP-IT-014"

    def test_unknown_process_raises_a_domain_error(self) -> None:
        dataset = ServiceDeskDataset.from_json(SETTINGS.diagnostics.dataset_path)
        with pytest.raises(ProcessNotMonitored):
            dataset.resolve("payroll reconciliation")

    def test_provider_merges_both_tool_payloads(self, tools) -> None:
        metrics = ServiceDeskMetricsProvider(tools).fetch("SOP-IT-014")
        assert metrics.average_resolution_hours == 42.0
        assert metrics.bottleneck.stage == "Direct manager approval"

    def test_provider_rejects_an_incomplete_toolset(self, tools) -> None:
        with pytest.raises(ValueError):
            ServiceDeskMetricsProvider(tools[:1])


class TestApprovalGateways:
    _PLAN = ActionPlan("p", (), (), "summary")

    def test_scripted_gateway_replays_then_approves(self) -> None:
        gateway = ScriptedApprovalGateway([reject_with("no")])
        assert gateway.review(self._PLAN, revision=1).approved is False
        assert gateway.review(self._PLAN, revision=2).approved is True

    def test_console_gateway_reads_the_answer(self) -> None:
        """Injected IO: no monkey-patching of builtins required."""
        written: list[str] = []
        gateway = ConsoleApprovalGateway(
            prompt_fn=lambda _: "y", writer=written.append
        )
        assert gateway.review(self._PLAN, revision=1).approved is True
        assert any("HUMAN-IN-THE-LOOP" in line for line in written)

    def test_console_gateway_treats_free_text_as_feedback(self) -> None:
        gateway = ConsoleApprovalGateway(prompt_fn=lambda _: "keep the sign-off",
                                         writer=lambda _: None)
        decision = gateway.review(self._PLAN, revision=1)
        assert decision.approved is False
        assert decision.feedback == "keep the sign-off"

    def test_approve_helper(self) -> None:
        assert approve().approved is True
