"""Multi-agent system for the analysis and automation of business processes.

Generative AI final project - a LangGraph workflow in which four agents

1. read the company's documentation (Agentic RAG),
2. measure the real process through tools (Tool Calling),
3. engineer an automation Action Plan, and
4. wait for the systems manager's approval (Human-in-the-Loop),

are wired together following the SOLID principles. Start from
:func:`process_automation.build_system`.
"""

from .application import MultiAgentSystem, RunResult, WorkflowSession, build_system
from .config import Settings
from .state import ProcessAnalysisState, WorkflowState, initial_state

__version__ = "1.0.0"

__all__ = [
    "MultiAgentSystem",
    "ProcessAnalysisState",
    "RunResult",
    "Settings",
    "WorkflowSession",
    "WorkflowState",
    "__version__",
    "build_system",
    "initial_state",
]
