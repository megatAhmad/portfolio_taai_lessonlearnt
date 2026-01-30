"""Review & Edit tab for Streamlit application."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import logging

from src.data_processing.enrichment import get_enrichment_summary

from .utils import (
    format_confidence_badge,
    format_severity_badge,
    truncate_text,
)
from .components import (
    render_lesson_card,
    render_enrichment_stats,
    render_filter_sidebar,
    render_error_message,
    render_success_message,
    render_info_message,
    render_warning_message,
)

logger = logging.getLogger(__name__)


def render_review_tab(settings) -> None:
    """
    Render the Review & Edit tab.

    Args:
        settings: Application settings
    """
    st.header("Review & Edit Enrichment")

    # Check if lessons are loaded
    if not st.session_state.get("lessons_uploaded") or st.session_state.get("lessons_df") is None:
        render_info_message("Please upload lessons in the 'Upload & Enrich' tab first.")
        return

    df = st.session_state.lessons_df

    # Check if enriched
    if not st.session_state.get("lessons_enriched"):
        render_warning_message(
            "Lessons have not been enriched yet. "
            "Go to 'Upload & Enrich' tab to run enrichment."
        )

    # Render sidebar filters
    categories = df["category"].dropna().unique().tolist() if "category" in df.columns else []
    equipment_types = (
        df["equipment_type"].dropna().unique().tolist()
        if "equipment_type" in df.columns
        else []
    )

    filters = render_filter_sidebar(categories, equipment_types)

    # Main content
    col1, col2 = st.columns([1, 3])

    with col1:
        render_review_summary(df, filters)

    with col2:
        render_lessons_list(df, filters, settings)


def render_review_summary(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """
    Render the review summary section.

    Args:
        df: DataFrame with lessons
        filters: Active filters
    """
    st.subheader("Summary")

    # Get enrichment summary
    if "enrichment_confidence" in df.columns:
        stats = get_enrichment_summary(df)
        render_enrichment_stats(stats)
    else:
        st.metric("Total Lessons", len(df))

    st.divider()

    # Quick filters for flags
    st.markdown("**Quick Filters:**")

    if "enrichment_flag" in df.columns:
        flags = df["enrichment_flag"].value_counts()

        for flag, count in flags.items():
            if flag and pd.notna(flag):
                if st.button(f"{flag} ({count})", key=f"filter_{flag}"):
                    st.session_state.active_flag_filter = flag

    # Show flagged count
    if "enrichment_flag" in df.columns:
        flagged = df["enrichment_flag"].notna().sum()
        st.metric("Flagged for Review", flagged)

    # Reviewed count
    if "enrichment_reviewed" in df.columns:
        reviewed = df["enrichment_reviewed"].sum()
        st.metric("Reviewed", f"{reviewed}/{len(df)}")


def render_lessons_list(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    settings,
) -> None:
    """
    Render the lessons list with filtering.

    Args:
        df: DataFrame with lessons
        filters: Active filters
        settings: Application settings
    """
    st.subheader("Lessons")

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Apply flag filter from quick filters
    if st.session_state.get("active_flag_filter"):
        flag = st.session_state.active_flag_filter
        filtered_df = filtered_df[filtered_df["enrichment_flag"] == flag]
        st.info(f"Showing lessons with flag: {flag}")
        if st.button("Clear flag filter"):
            st.session_state.active_flag_filter = None
            st.rerun()

    # Show count
    st.caption(f"Showing {len(filtered_df)} of {len(df)} lessons")

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        options=["lesson_id", "enrichment_confidence", "severity", "category"],
        index=0,
    )

    sort_order = st.radio("Order", options=["Ascending", "Descending"], horizontal=True)
    ascending = sort_order == "Ascending"

    if sort_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_col, ascending=ascending)

    # Pagination
    page_size = st.selectbox("Items per page", options=[10, 25, 50], index=0)
    total_pages = (len(filtered_df) - 1) // page_size + 1 if len(filtered_df) > 0 else 1

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))

    st.caption(f"Page {page} of {total_pages}")

    # Render lessons
    for idx in range(start_idx, end_idx):
        row = filtered_df.iloc[idx]
        lesson = row.to_dict()

        # Handle NaN values
        for key, value in lesson.items():
            if pd.isna(value):
                lesson[key] = None

        render_lesson_review_card(lesson, idx, settings)


def render_lesson_review_card(
    lesson: Dict[str, Any],
    idx: int,
    settings,
) -> None:
    """
    Render a lesson card with review controls.

    Args:
        lesson: Lesson dictionary
        idx: Index for unique keys
        settings: Application settings
    """
    lesson_id = lesson.get("lesson_id", "Unknown")
    title = lesson.get("title", "No title")
    confidence = lesson.get("enrichment_confidence", 0)
    flag = lesson.get("enrichment_flag")
    reviewed = lesson.get("enrichment_reviewed", False)

    with st.container():
        # Header row
        header_cols = st.columns([3, 1, 1, 1])

        with header_cols[0]:
            st.markdown(f"**{lesson_id}**: {title}")

        with header_cols[1]:
            if confidence:
                st.markdown(format_confidence_badge(confidence))

        with header_cols[2]:
            if flag:
                st.markdown(f":warning: {flag}")

        with header_cols[3]:
            if reviewed:
                st.markdown(":white_check_mark: Reviewed")
            else:
                st.markdown(":hourglass: Pending")

        # Expandable details
        with st.expander("View/Edit Details", expanded=False):
            render_lesson_details(lesson, idx, settings)

        st.divider()


def render_lesson_details(
    lesson: Dict[str, Any],
    idx: int,
    settings,
) -> None:
    """
    Render lesson details with edit controls.

    Args:
        lesson: Lesson dictionary
        idx: Index for unique keys
        settings: Application settings
    """
    lesson_id = lesson.get("lesson_id", "Unknown")

    # Two columns: original data and enrichment
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Data:**")

        st.markdown(f"**Description:** {lesson.get('description', 'N/A')}")
        st.markdown(f"**Root Cause:** {lesson.get('root_cause', 'N/A')}")
        st.markdown(f"**Corrective Action:** {lesson.get('corrective_action', 'N/A')}")
        st.markdown(f"**Category:** {lesson.get('category', 'N/A')}")
        st.markdown(f"**Equipment Tag:** {lesson.get('equipment_tag', 'N/A')}")
        st.markdown(f"**Severity:** {lesson.get('severity', 'N/A')}")

    with col2:
        st.markdown("**Enrichment Data:**")

        # Editable enrichment fields
        specificity = st.selectbox(
            "Specificity Level",
            options=["equipment_id", "equipment_type", "generic", "unknown"],
            index=["equipment_id", "equipment_type", "generic", "unknown"].index(
                lesson.get("specificity_level", "unknown")
            )
            if lesson.get("specificity_level") in ["equipment_id", "equipment_type", "generic", "unknown"]
            else 3,
            key=f"specificity_{idx}",
        )

        scope = st.selectbox(
            "Lesson Scope",
            options=["specific", "general", "universal", "unknown"],
            index=["specific", "general", "universal", "unknown"].index(
                lesson.get("lesson_scope", "unknown")
            )
            if lesson.get("lesson_scope") in ["specific", "general", "universal", "unknown"]
            else 3,
            key=f"scope_{idx}",
        )

        equipment_type = st.text_input(
            "Equipment Type",
            value=lesson.get("equipment_type", "") or "",
            key=f"eq_type_{idx}",
        )

        equipment_family = st.selectbox(
            "Equipment Family",
            options=["", "rotating_equipment", "static_equipment", "instrumentation", "electrical", "piping"],
            index=["", "rotating_equipment", "static_equipment", "instrumentation", "electrical", "piping"].index(
                lesson.get("equipment_family", "") or ""
            )
            if lesson.get("equipment_family") in ["", "rotating_equipment", "static_equipment", "instrumentation", "electrical", "piping"]
            else 0,
            key=f"eq_family_{idx}",
        )

        # Multi-select for list fields
        applicable_to_str = lesson.get("applicable_to", "") or ""
        if isinstance(applicable_to_str, list):
            applicable_to = applicable_to_str
        else:
            applicable_to = [x.strip() for x in applicable_to_str.split(",") if x.strip()]

        applicable_options = settings.APPLICABLE_TO_OPTIONS if hasattr(settings, "APPLICABLE_TO_OPTIONS") else [
            "all_pumps", "all_seals", "all_rotating_equipment",
            "all_heat_exchangers", "all_valves", "all_equipment"
        ]

        new_applicable = st.multiselect(
            "Applicable To",
            options=applicable_options,
            default=[a for a in applicable_to if a in applicable_options],
            key=f"applicable_{idx}",
        )

        confidence = st.slider(
            "Confidence Score",
            min_value=0.0,
            max_value=1.0,
            value=float(lesson.get("enrichment_confidence", 0.5) or 0.5),
            step=0.05,
            key=f"confidence_{idx}",
        )

    # Action buttons
    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button("Save Changes", key=f"save_{idx}"):
            save_lesson_changes(
                lesson_id,
                {
                    "specificity_level": specificity,
                    "lesson_scope": scope,
                    "equipment_type": equipment_type if equipment_type else None,
                    "equipment_family": equipment_family if equipment_family else None,
                    "applicable_to": ",".join(new_applicable),
                    "enrichment_confidence": confidence,
                },
            )

    with action_cols[1]:
        if st.button("Mark Reviewed", key=f"review_{idx}"):
            mark_lesson_reviewed(lesson_id)

    with action_cols[2]:
        if st.button("Clear Flag", key=f"clear_flag_{idx}"):
            clear_lesson_flag(lesson_id)

    with action_cols[3]:
        if st.button("Flag for Review", key=f"flag_{idx}"):
            flag_lesson_for_review(lesson_id, "MANUAL_FLAG")


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to DataFrame.

    Args:
        df: DataFrame to filter
        filters: Filter dictionary

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if filters.get("category"):
        filtered = filtered[filtered["category"].isin(filters["category"])]

    if filters.get("equipment_type") and "equipment_type" in filtered.columns:
        filtered = filtered[filtered["equipment_type"].isin(filters["equipment_type"])]

    if filters.get("severity"):
        filtered = filtered[filtered["severity"].isin(filters["severity"])]

    if filters.get("min_confidence") and "enrichment_confidence" in filtered.columns:
        filtered = filtered[filtered["enrichment_confidence"] >= filters["min_confidence"]]

    if filters.get("review_status"):
        status = filters["review_status"]
        if status == "Reviewed" and "enrichment_reviewed" in filtered.columns:
            filtered = filtered[filtered["enrichment_reviewed"] == True]
        elif status == "Pending Review" and "enrichment_reviewed" in filtered.columns:
            filtered = filtered[filtered["enrichment_reviewed"] != True]
        elif status == "Flagged" and "enrichment_flag" in filtered.columns:
            filtered = filtered[filtered["enrichment_flag"].notna()]

    return filtered


def save_lesson_changes(lesson_id: str, changes: Dict[str, Any]) -> None:
    """
    Save changes to a lesson.

    Args:
        lesson_id: Lesson ID
        changes: Dictionary of changes
    """
    df = st.session_state.lessons_df

    # Find the lesson and update
    mask = df["lesson_id"] == lesson_id

    for field, value in changes.items():
        if field in df.columns:
            df.loc[mask, field] = value

    st.session_state.lessons_df = df
    st.success(f"Changes saved for {lesson_id}")
    st.rerun()


def mark_lesson_reviewed(lesson_id: str) -> None:
    """
    Mark a lesson as reviewed.

    Args:
        lesson_id: Lesson ID
    """
    df = st.session_state.lessons_df
    mask = df["lesson_id"] == lesson_id

    if "enrichment_reviewed" in df.columns:
        df.loc[mask, "enrichment_reviewed"] = True

    st.session_state.lessons_df = df
    st.success(f"Marked {lesson_id} as reviewed")
    st.rerun()


def clear_lesson_flag(lesson_id: str) -> None:
    """
    Clear the flag for a lesson.

    Args:
        lesson_id: Lesson ID
    """
    df = st.session_state.lessons_df
    mask = df["lesson_id"] == lesson_id

    if "enrichment_flag" in df.columns:
        df.loc[mask, "enrichment_flag"] = None

    st.session_state.lessons_df = df
    st.success(f"Cleared flag for {lesson_id}")
    st.rerun()


def flag_lesson_for_review(lesson_id: str, flag: str) -> None:
    """
    Flag a lesson for review.

    Args:
        lesson_id: Lesson ID
        flag: Flag type
    """
    df = st.session_state.lessons_df
    mask = df["lesson_id"] == lesson_id

    if "enrichment_flag" not in df.columns:
        df["enrichment_flag"] = None

    df.loc[mask, "enrichment_flag"] = flag

    st.session_state.lessons_df = df
    st.warning(f"Flagged {lesson_id} as {flag}")
    st.rerun()
