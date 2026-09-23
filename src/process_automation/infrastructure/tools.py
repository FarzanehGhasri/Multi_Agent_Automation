"""Step 2 of the brief - monitoring tools (mock ServiceDesk / Jira API).

The Diagnostic Agent must not read a JSON file: it must *call a tool*, exactly
as it would call a real analytics endpoint in production. This module provides

* :class:`ServiceDeskDataset` - the fake backend (swap for ``requests.get``);
* :class:`ServiceDeskToolFactory` - builds the ``@tool``-decorated callables;
* :class:`ServiceDeskMetricsProvider` - the adapter that lets the agent depend
  on the small :class:`~process_automation.interfaces.MetricsProvider` port
  instead of on LangChain.

SOLID notes
-----------
* DIP - tools are *built* from an injected dataset instead of reaching for a
        module-level global, so a test can feed them a two-line fixture.
* SRP - the dataset resolves and serves data; the factory declares the tool
        contracts; the provider converts payloads into domain objects.
* OCP - a real backend is added by writing another ``MetricsProvider``; the
        Diagnostic Agent is not touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.tools import BaseTool, tool

from ..domain.models import ProcessMetrics
from .embeddings import tokenize


class ProcessNotMonitored(LookupError):
    """Raised when no monitored process matches the requested name."""


class ServiceDeskDataset:
    """Stand-in for the corporate ServiceDesk/Jira analytics API.

    In production, :meth:`payload` would issue an authenticated HTTP request.
    Everything above this class - tools, agent, graph - stays identical.
    """

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self._records = dict(records)

    @classmethod
    def from_json(cls, path: Path) -> "ServiceDeskDataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw.get("processes", {}))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._records)

    def resolve(self, process_name: str) -> str:
        """Map a free-form process name onto a monitored process key.

        Three passes, cheapest first: exact key, exact name, then token
        overlap. The last one is what makes the tool tolerant of the way a
        user actually types a request ("vpn access for new staff").
        """
        needle = process_name.strip()
        if needle in self._records:
            return needle

        lowered = needle.lower()
        for key, record in self._records.items():
            if str(record.get("process_name", "")).lower() == lowered:
                return key

        query_tokens = set(tokenize(needle))
        best_key, best_score = "", 0.0
        for key, record in self._records.items():
            candidate = f"{key} {record.get('process_name', '')}"
            candidate_tokens = set(tokenize(candidate))
            if not candidate_tokens:
                continue
            overlap = len(query_tokens & candidate_tokens) / len(candidate_tokens)
            if overlap > best_score:
                best_key, best_score = key, overlap
        if best_score >= 0.3:
            return best_key
        raise ProcessNotMonitored(
            f"'{process_name}' is not monitored. Known processes: {', '.join(self._records)}"
        )

    def payload(self, process_name: str) -> Mapping[str, Any]:
        """Full record of the resolved process."""
        return self._records[self.resolve(process_name)]


class ServiceDeskToolFactory:
    """Builds the LangChain tools the Diagnostic Agent is allowed to call.

    A factory (rather than module-level tool functions) keeps the dataset
    injected instead of global, which is what makes the tools unit-testable
    and lets several graphs run side by side against different backends.
    """

    def __init__(self, dataset: ServiceDeskDataset) -> None:
        self._dataset = dataset

    def build(self) -> list[BaseTool]:
        """Return the tool objects, each with an LLM-readable docstring."""
        dataset = self._dataset  # captured by the closures below

        @tool
        def fetch_process_metrics(process_name: str) -> dict:
            """Return the headline KPIs measured for an organisational process.

            Use this tool to learn how a process performs *in reality*: the
            average end-to-end resolution time in hours, the ticket volume of
            the sample, how many tickets had to be reopened, and the errors
            operators hit most often.

            Args:
                process_name: Name of the process, e.g. "VPN access
                    provisioning for new personnel", or its SOP identifier.
            """
            record = dataset.payload(process_name)
            return {
                "process_name": record["process_name"],
                "sample_period": record["sample_period"],
                "average_resolution_hours": record["average_resolution_hours"],
                "ticket_volume": record["ticket_volume"],
                "reopened_tickets": record["reopened_tickets"],
                "common_errors": list(record["common_errors"]),
            }

        @tool
        def fetch_stage_timings(process_name: str) -> dict:
            """Return the per-stage duration breakdown of a process.

            Use this tool to find *where* the time goes: it reports, for each
            stage of the real workflow, the average hours spent there and the
            share of that time which is pure waiting (queue time).

            Args:
                process_name: Name of the process, or its SOP identifier.
            """
            record = dataset.payload(process_name)
            return {
                "process_name": record["process_name"],
                "stage_timings": [dict(item) for item in record["stage_timings"]],
            }

        return [fetch_process_metrics, fetch_stage_timings]


class ServiceDeskMetricsProvider:
    """Adapter: calls the tools and returns a domain :class:`ProcessMetrics`.

    This is the object the Diagnostic Agent actually holds. It satisfies the
    :class:`~process_automation.interfaces.MetricsProvider` port, so the agent
    never imports LangChain and can be tested with a three-line stub.
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        # Index by name so the provider does not depend on list ordering.
        self._tools: dict[str, BaseTool] = {tool_.name: tool_ for tool_ in tools}
        missing = {"fetch_process_metrics", "fetch_stage_timings"} - set(self._tools)
        if missing:
            raise ValueError(f"missing required tools: {', '.join(sorted(missing))}")

    @classmethod
    def from_dataset_path(cls, path: Path) -> "ServiceDeskMetricsProvider":
        """Default wiring used by the composition root."""
        return cls(ServiceDeskToolFactory(ServiceDeskDataset.from_json(path)).build())

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def fetch(self, process_name: str) -> ProcessMetrics:
        """Invoke both tools and merge their payloads into one value object."""
        # ``.invoke`` is the standard LangChain tool entry point: the exact
        # call an LLM-driven agent would make after emitting a tool call.
        headline = self._tools["fetch_process_metrics"].invoke(
            {"process_name": process_name}
        )
        timings = self._tools["fetch_stage_timings"].invoke({"process_name": process_name})
        return ProcessMetrics.from_payload({**headline, **timings})


__all__ = [
    "ProcessNotMonitored",
    "ServiceDeskDataset",
    "ServiceDeskMetricsProvider",
    "ServiceDeskToolFactory",
]
