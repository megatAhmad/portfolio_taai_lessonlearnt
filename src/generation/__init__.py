"""Generation modules for relevance analysis and applicability checking using LLM."""

from .relevance_analyzer import (
    RelevanceAnalyzer,
    RelevanceAnalysis,
    RelevanceOutput,
    create_relevance_analyzer,
    format_analysis_for_display,
)

from .applicability_checker import (
    ApplicabilityChecker,
    ApplicabilityResult,
    ApplicabilityOutput,
    create_applicability_checker,
    format_applicability_for_display,
)

__all__ = [
    # Relevance analysis
    "RelevanceAnalyzer",
    "RelevanceAnalysis",
    "RelevanceOutput",
    "create_relevance_analyzer",
    "format_analysis_for_display",
    # Applicability checking
    "ApplicabilityChecker",
    "ApplicabilityResult",
    "ApplicabilityOutput",
    "create_applicability_checker",
    "format_applicability_for_display",
]
