"""UI utility functions for Streamlit application."""

import io
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults = {
        # Data state
        "lessons_df": None,
        "jobs_df": None,
        "lessons_uploaded": False,
        "jobs_uploaded": False,
        "lessons_enriched": False,

        # Enrichment state
        "enrichment_results": None,
        "enrichment_progress": None,

        # Vector store state
        "vector_store_initialized": False,
        "documents_indexed": 0,

        # Matching state
        "matching_results": None,
        "selected_job": None,

        # UI state
        "current_tab": 0,
        "show_advanced": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state() -> None:
    """Reset session state to defaults."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def format_timestamp(timestamp: Optional[str]) -> str:
    """
    Format a timestamp for display.

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Formatted timestamp string
    """
    if not timestamp:
        return "N/A"

    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(timestamp)


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if not text:
        return ""

    text = str(text)
    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def format_confidence_badge(confidence: float) -> str:
    """
    Format confidence score as a colored badge.

    Args:
        confidence: Confidence score (0-1)

    Returns:
        HTML/markdown for colored badge
    """
    if confidence >= 0.85:
        return f":green[{confidence:.0%}]"
    elif confidence >= 0.70:
        return f":orange[{confidence:.0%}]"
    else:
        return f":red[{confidence:.0%}]"


def format_relevance_badge(score: int) -> str:
    """
    Format relevance score as a colored badge.

    Args:
        score: Relevance score (0-100)

    Returns:
        HTML/markdown for colored badge
    """
    if score >= 80:
        return f":green[{score}%]"
    elif score >= 50:
        return f":orange[{score}%]"
    else:
        return f":red[{score}%]"


def format_tier_badge(tier: str) -> str:
    """
    Format match tier as a colored badge.

    Args:
        tier: Match tier string

    Returns:
        Formatted tier badge
    """
    tier_colors = {
        "equipment_specific": ":green[Equipment-Specific]",
        "equipment_type": ":blue[Equipment-Type]",
        "generic": ":violet[Generic/Universal]",
        "semantic": ":gray[Semantic]",
    }
    return tier_colors.get(tier, f":gray[{tier}]")


def format_severity_badge(severity: str) -> str:
    """
    Format severity level as a colored badge.

    Args:
        severity: Severity level

    Returns:
        Formatted severity badge
    """
    severity_colors = {
        "critical": ":red[Critical]",
        "high": ":orange[High]",
        "medium": ":blue[Medium]",
        "low": ":green[Low]",
    }
    return severity_colors.get(severity.lower() if severity else "medium", ":blue[Medium]")


def create_export_excel(
    results: List[Dict[str, Any]],
    job: Dict[str, Any],
    include_details: bool = True,
) -> bytes:
    """
    Create an Excel file for export.

    Args:
        results: List of matching results
        job: Job dictionary
        include_details: Whether to include detailed analysis

    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary sheet
        summary_data = {
            "Job ID": [job.get("job_id", "N/A")],
            "Job Title": [job.get("job_title", "N/A")],
            "Equipment Tag": [job.get("equipment_tag", "N/A")],
            "Job Type": [job.get("job_type", "N/A")],
            "Total Matches": [len(results)],
            "Export Date": [datetime.now().strftime("%Y-%m-%d %H:%M")],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Results sheet
        if results:
            results_data = []
            for r in results:
                row = {
                    "Lesson ID": r.get("lesson_id", ""),
                    "Relevance Score": r.get("relevance_score", 0),
                    "Match Tier": r.get("match_tier", ""),
                    "Title": r.get("title", ""),
                    "Category": r.get("category", ""),
                    "Severity": r.get("severity", ""),
                    "Equipment Tag": r.get("equipment_tag", ""),
                }

                if include_details:
                    row["Technical Links"] = "; ".join(r.get("technical_links", []))
                    row["Safety Considerations"] = r.get("safety_considerations", "")
                    row["Recommended Actions"] = "; ".join(r.get("recommended_actions", []))
                    row["Match Reasoning"] = r.get("match_reasoning", "")

                results_data.append(row)

            results_df = pd.DataFrame(results_data)
            results_df.to_excel(writer, sheet_name="Matched Lessons", index=False)

    output.seek(0)
    return output.getvalue()


def dataframe_to_lessons_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame to a list of lesson dictionaries.

    Args:
        df: DataFrame with lessons data

    Returns:
        List of lesson dictionaries
    """
    lessons = []

    for _, row in df.iterrows():
        lesson = row.to_dict()

        # Handle NaN values
        for key, value in lesson.items():
            if pd.isna(value):
                lesson[key] = None

        # Convert date to string if present
        if "date" in lesson and lesson["date"] is not None:
            try:
                lesson["date"] = str(lesson["date"])
            except (ValueError, TypeError):
                pass

        lessons.append(lesson)

    return lessons


def dataframe_to_jobs_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame to a list of job dictionaries.

    Args:
        df: DataFrame with jobs data

    Returns:
        List of job dictionaries
    """
    jobs = []

    for _, row in df.iterrows():
        job = row.to_dict()

        # Handle NaN values
        for key, value in job.items():
            if pd.isna(value):
                job[key] = None

        # Convert date to string if present
        if "planned_date" in job and job["planned_date"] is not None:
            try:
                job["planned_date"] = str(job["planned_date"])
            except (ValueError, TypeError):
                pass

        jobs.append(job)

    return jobs


def get_equipment_type_from_tag(tag: Optional[str]) -> Optional[str]:
    """
    Extract equipment type from an equipment tag.

    Args:
        tag: Equipment tag (e.g., "P-101", "HX-205")

    Returns:
        Equipment type or None
    """
    if not tag:
        return None

    tag = tag.upper().strip()

    # Common prefixes and their types
    prefix_types = {
        "P-": "pump",
        "HX-": "heat_exchanger",
        "E-": "exchanger",
        "V-": "valve",
        "C-": "compressor",
        "T-": "tank",
        "TK-": "tank",
        "M-": "motor",
        "R-": "reactor",
        "COL-": "column",
        "FAN-": "fan",
        "BLW-": "blower",
    }

    for prefix, eq_type in prefix_types.items():
        if tag.startswith(prefix):
            return eq_type

    return None


def calculate_progress_percentage(current: int, total: int) -> float:
    """
    Calculate progress percentage.

    Args:
        current: Current count
        total: Total count

    Returns:
        Progress percentage (0-100)
    """
    if total <= 0:
        return 0.0
    return min(100.0, (current / total) * 100)
