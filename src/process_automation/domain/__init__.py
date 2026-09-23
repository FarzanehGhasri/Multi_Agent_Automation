"""Domain layer: framework-free business rules of the S&M analysis system."""

from .feedback import FeedbackInterpreter, RuleBasedFeedbackInterpreter, merge_constraints
from .gap_analysis import ComparisonContext, GapAnalyzer, GapRule
from .models import (
    ActionPlan,
    ApprovalDecision,
    AutomationClass,
    AutomationProposal,
    Constraint,
    Gap,
    ProcessMetrics,
    ProcessStep,
    RetrievedChunk,
    Severity,
    StageTiming,
    StandardProcess,
)
from .remediation import (
    RemediationContext,
    RemediationRegistry,
    RemediationStrategy,
    build_contexts,
)

__all__ = [
    "ActionPlan",
    "ApprovalDecision",
    "AutomationClass",
    "AutomationProposal",
    "ComparisonContext",
    "Constraint",
    "FeedbackInterpreter",
    "Gap",
    "GapAnalyzer",
    "GapRule",
    "ProcessMetrics",
    "ProcessStep",
    "RemediationContext",
    "RemediationRegistry",
    "RemediationStrategy",
    "RetrievedChunk",
    "RuleBasedFeedbackInterpreter",
    "Severity",
    "StageTiming",
    "StandardProcess",
    "build_contexts",
    "merge_constraints",
]
