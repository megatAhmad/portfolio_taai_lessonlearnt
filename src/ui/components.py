"""Reusable Streamlit UI components."""

from typing import List, Dict, Any, Optional, Callable
import streamlit as st

from .utils import (
    truncate_text,
    format_confidence_badge,
    format_relevance_badge,
    format_tier_badge,
    format_severity_badge,
)


def render_lesson_card(
    lesson: Dict[str, Any],
    show_enrichment: bool = True,
    expanded: bool = False,
    on_select: Optional[Callable] = None,
) -> None:
    """
    Render a lesson learned card.

    Args:
        lesson: Lesson dictionary
        show_enrichment: Whether to show enrichment data
        expanded: Whether to show expanded view
        on_select: Optional callback when lesson is selected
    """
    lesson_id = lesson.get("lesson_id", "Unknown")
    title = lesson.get("title", "No title")
    description = lesson.get("description", "No description")
    category = lesson.get("category", "N/A")
    severity = lesson.get("severity", "medium")
    equipment_tag = lesson.get("equipment_tag", "N/A")

    with st.container():
        # Header row
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**{lesson_id}**: {title}")

        with col2:
            st.markdown(format_severity_badge(severity))

        with col3:
            if show_enrichment and lesson.get("enrichment_confidence"):
                st.markdown(format_confidence_badge(lesson["enrichment_confidence"]))

        # Preview
        st.markdown(f"*{truncate_text(description, 200)}*")

        # Metadata row
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.caption(f"Category: {category}")
        with meta_cols[1]:
            st.caption(f"Equipment: {equipment_tag or 'N/A'}")

        if show_enrichment:
            with meta_cols[2]:
                scope = lesson.get("lesson_scope", "N/A")
                st.caption(f"Scope: {scope}")
            with meta_cols[3]:
                eq_type = lesson.get("equipment_type", "N/A")
                st.caption(f"Type: {eq_type or 'N/A'}")

        # Expanded view
        if expanded:
            with st.expander("View Details", expanded=True):
                st.markdown("**Description:**")
                st.markdown(description)

                if lesson.get("root_cause"):
                    st.markdown("**Root Cause:**")
                    st.markdown(lesson["root_cause"])

                if lesson.get("corrective_action"):
                    st.markdown("**Corrective Action:**")
                    st.markdown(lesson["corrective_action"])

                if show_enrichment:
                    st.markdown("---")
                    st.markdown("**Enrichment Data:**")

                    enrich_cols = st.columns(2)
                    with enrich_cols[0]:
                        st.markdown(f"- Specificity: {lesson.get('specificity_level', 'N/A')}")
                        st.markdown(f"- Equipment Family: {lesson.get('equipment_family', 'N/A')}")

                        applicable = lesson.get("applicable_to", [])
                        if applicable:
                            if isinstance(applicable, str):
                                applicable = applicable.split(",")
                            st.markdown(f"- Applicable to: {', '.join(applicable)}")

                    with enrich_cols[1]:
                        procedures = lesson.get("procedure_tags", [])
                        if procedures:
                            if isinstance(procedures, str):
                                procedures = procedures.split(",")
                            st.markdown(f"- Procedures: {', '.join(procedures)}")

                        safety = lesson.get("safety_categories", [])
                        if safety:
                            if isinstance(safety, str):
                                safety = safety.split(",")
                            st.markdown(f"- Safety: {', '.join(safety)}")

        st.divider()


def render_job_card(
    job: Dict[str, Any],
    selected: bool = False,
    on_select: Optional[Callable] = None,
) -> None:
    """
    Render a job description card.

    Args:
        job: Job dictionary
        selected: Whether this job is selected
        on_select: Optional callback when job is selected
    """
    job_id = job.get("job_id", "Unknown")
    title = job.get("job_title", "No title")
    description = job.get("job_description", "No description")
    equipment_tag = job.get("equipment_tag", "N/A")
    job_type = job.get("job_type", "N/A")

    container_style = "border: 2px solid #4CAF50;" if selected else ""

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**{job_id}**: {title}")

        with col2:
            if on_select:
                if st.button("Select", key=f"select_{job_id}", type="primary" if selected else "secondary"):
                    on_select(job)

        st.markdown(f"*{truncate_text(description, 150)}*")

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.caption(f"Equipment: {equipment_tag or 'N/A'}")
        with meta_cols[1]:
            st.caption(f"Type: {job_type or 'N/A'}")
        with meta_cols[2]:
            planned = job.get("planned_date", "N/A")
            st.caption(f"Planned: {planned or 'N/A'}")

        st.divider()


def render_match_result(
    result: Dict[str, Any],
    lesson: Dict[str, Any],
    rank: int = 0,
    show_details: bool = True,
) -> None:
    """
    Render a matching result with relevance analysis.

    Args:
        result: Relevance analysis result
        lesson: Original lesson dictionary
        rank: Rank position
        show_details: Whether to show detailed analysis
    """
    lesson_id = result.get("lesson_id", "Unknown")
    relevance_score = result.get("relevance_score", 0)
    match_tier = result.get("match_tier", "semantic")
    title = lesson.get("title", "No title")

    with st.container():
        # Header with rank, score, and tier
        header_cols = st.columns([0.5, 3, 1, 1.5])

        with header_cols[0]:
            st.markdown(f"**#{rank}**")

        with header_cols[1]:
            st.markdown(f"**{lesson_id}**: {title}")

        with header_cols[2]:
            st.markdown(format_relevance_badge(relevance_score))

        with header_cols[3]:
            st.markdown(format_tier_badge(match_tier))

        # Match reasoning
        if result.get("match_reasoning"):
            st.markdown(f"*{truncate_text(result['match_reasoning'], 200)}*")

        # Detailed analysis
        if show_details:
            with st.expander("View Analysis", expanded=False):
                # Technical links
                if result.get("technical_links"):
                    st.markdown("**Technical Links:**")
                    for link in result["technical_links"]:
                        st.markdown(f"- {link}")

                # Safety considerations
                if result.get("safety_considerations"):
                    st.markdown("**Safety Considerations:**")
                    st.warning(result["safety_considerations"])

                # Recommended actions
                if result.get("recommended_actions"):
                    st.markdown("**Recommended Actions:**")
                    for action in result["recommended_actions"]:
                        st.markdown(f"- {action}")

                st.markdown("---")

                # Original lesson details
                st.markdown("**Original Lesson:**")
                render_lesson_card(lesson, show_enrichment=True, expanded=False)

        st.divider()


def render_progress_bar(
    current: int,
    total: int,
    label: str = "Progress",
    show_stats: bool = True,
) -> None:
    """
    Render a progress bar with optional statistics.

    Args:
        current: Current count
        total: Total count
        label: Progress label
        show_stats: Whether to show statistics
    """
    progress = current / total if total > 0 else 0

    st.progress(progress, text=f"{label}: {current}/{total}")

    if show_stats:
        st.caption(f"{progress:.1%} complete")


def render_enrichment_stats(stats: Dict[str, Any]) -> None:
    """
    Render enrichment statistics.

    Args:
        stats: Statistics dictionary
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Lessons", stats.get("total", 0))

    with col2:
        enriched = stats.get("enriched", 0)
        total = stats.get("total", 1)
        pct = enriched / total * 100 if total > 0 else 0
        st.metric("Enriched", f"{enriched} ({pct:.0f}%)")

    with col3:
        high = stats.get("high_confidence", 0)
        st.metric("High Confidence", high)

    with col4:
        avg = stats.get("avg_confidence", 0)
        st.metric("Avg Confidence", f"{avg:.1%}")

    # Flags breakdown
    if stats.get("flags"):
        st.markdown("**Review Flags:**")
        flag_cols = st.columns(len(stats["flags"]))
        for i, (flag, count) in enumerate(stats["flags"].items()):
            with flag_cols[i]:
                st.caption(f"{flag}: {count}")


def render_filter_sidebar(
    categories: List[str],
    equipment_types: List[str],
    severities: List[str] = ["low", "medium", "high", "critical"],
) -> Dict[str, Any]:
    """
    Render filter controls in sidebar.

    Args:
        categories: List of available categories
        equipment_types: List of available equipment types
        severities: List of severity levels

    Returns:
        Dictionary of selected filters
    """
    st.sidebar.markdown("### Filters")

    filters = {}

    # Category filter
    if categories:
        selected_categories = st.sidebar.multiselect(
            "Category",
            options=categories,
            default=[],
        )
        if selected_categories:
            filters["category"] = selected_categories

    # Equipment type filter
    if equipment_types:
        selected_types = st.sidebar.multiselect(
            "Equipment Type",
            options=equipment_types,
            default=[],
        )
        if selected_types:
            filters["equipment_type"] = selected_types

    # Severity filter
    selected_severities = st.sidebar.multiselect(
        "Severity",
        options=severities,
        default=[],
    )
    if selected_severities:
        filters["severity"] = selected_severities

    # Confidence threshold
    min_confidence = st.sidebar.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )
    if min_confidence > 0:
        filters["min_confidence"] = min_confidence

    # Review status
    review_status = st.sidebar.radio(
        "Review Status",
        options=["All", "Reviewed", "Pending Review", "Flagged"],
        index=0,
    )
    if review_status != "All":
        filters["review_status"] = review_status

    return filters


def render_error_message(message: str, details: Optional[str] = None) -> None:
    """
    Render an error message.

    Args:
        message: Error message
        details: Optional detailed error information
    """
    st.error(message)

    if details:
        with st.expander("Error Details"):
            st.code(details)


def render_success_message(message: str) -> None:
    """
    Render a success message.

    Args:
        message: Success message
    """
    st.success(message)


def render_info_message(message: str) -> None:
    """
    Render an info message.

    Args:
        message: Info message
    """
    st.info(message)


def render_warning_message(message: str) -> None:
    """
    Render a warning message.

    Args:
        message: Warning message
    """
    st.warning(message)
