"""Node 2 - Diagnostic Agent (Tool Caller).

Role: systems data analyst.
Task: call the monitoring tools (mock Jira / ServiceDesk API) and return the
real performance of the process - average resolution time, reopened tickets,
common errors and the per-stage duration breakdown.
Output: the ``as-is`` picture (``actual_metrics``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..interfaces import MetricsProvider
from ..state import ProcessAnalysisState
from .base import AgentNode


class DiagnosticAgent(AgentNode):
    """Measures reality by calling tools - it never reads a file itself."""

    name = "diagnostic_agent"

    def __init__(self, metrics_provider: MetricsProvider) -> None:
        super().__init__()
        self._provider = metrics_provider

    def run(self, state: ProcessAnalysisState) -> Mapping[str, Any]:
        # Prefer the canonical name resolved by the RAG node: it matches the
        # monitoring system's vocabulary better than the user's free wording.
        target = state.get("resolved_process") or state["process_name"]
        trace = [
            self.log(
                f"calling tool(s) {', '.join(self._provider.tool_names)} "
                f"for process '{target}'"
            )
        ]

        try:
            metrics = self._provider.fetch(target)
        except LookupError as error:
            # A missing process is a *business* outcome, not a crash: the
            # engineer node will report that reality could not be measured.
            trace.append(self.log(f"tool call failed: {error}"))
            return {"actual_metrics": {}, "trace": trace}

        trace.append(
            self.log(
                f"measured average={metrics.average_resolution_hours:g}h over "
                f"{metrics.ticket_volume} ticket(s) [{metrics.sample_period}], "
                f"reopened={metrics.reopened_tickets} ({metrics.reopen_rate:.0%})"
            )
        )
        for timing in metrics.stage_timings:
            trace.append(
                self.log(
                    f"  stage '{timing.stage}': {timing.average_hours:g}h "
                    f"({metrics.share_of_total(timing):.0%} of cycle, "
                    f"{timing.waiting_share:.0%} waiting)"
                )
            )
        bottleneck = metrics.bottleneck
        if bottleneck is not None:
            trace.append(
                self.log(f"slowest stage is '{bottleneck.stage}'")
            )

        return {"actual_metrics": metrics.to_payload(), "trace": trace}


__all__ = ["DiagnosticAgent"]
