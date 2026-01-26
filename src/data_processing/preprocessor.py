"""Text preprocessing for maintenance lessons and job descriptions."""

import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Common maintenance abbreviations and their expansions
ABBREVIATIONS = {
    # Equipment
    "hx": "heat exchanger",
    "p-": "pump ",
    "v-": "valve ",
    "c-": "compressor ",
    "t-": "tank ",
    "tk-": "tank ",
    "m-": "motor ",
    "e-": "exchanger ",
    "r-": "reactor ",
    "col-": "column ",

    # Maintenance terms
    "pm": "preventive maintenance",
    "cm": "corrective maintenance",
    "tar": "turnaround",
    "ta": "turnaround",
    "moc": "management of change",
    "sop": "standard operating procedure",
    "wps": "work permit system",
    "loto": "lockout tagout",
    "ppe": "personal protective equipment",
    "jsea": "job safety and environmental analysis",
    "jha": "job hazard analysis",
    "ptw": "permit to work",

    # Technical terms
    "rpm": "revolutions per minute",
    "psi": "pounds per square inch",
    "psig": "pounds per square inch gauge",
    "gpm": "gallons per minute",
    "cfm": "cubic feet per minute",
    "hp": "horsepower",
    "kw": "kilowatt",
    "vfd": "variable frequency drive",
    "plc": "programmable logic controller",
    "dcs": "distributed control system",
    "p&id": "piping and instrumentation diagram",
    "nde": "non-destructive examination",
    "ndt": "non-destructive testing",
    "qc": "quality control",
    "qa": "quality assurance",

    # Safety
    "h2s": "hydrogen sulfide",
    "leo": "lower explosive limit",
    "uel": "upper explosive limit",
    "msds": "material safety data sheet",
    "sds": "safety data sheet",
}


def preprocess_text(text: str, expand_abbr: bool = True, normalize: bool = True) -> str:
    """
    Preprocess text for RAG system.

    Args:
        text: Input text to preprocess
        expand_abbr: Whether to expand abbreviations
        normalize: Whether to normalize whitespace and patterns

    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Expand abbreviations if requested
    if expand_abbr:
        text = expand_abbreviations(text)

    if normalize:
        # Standardize technical patterns
        text = normalize_technical_patterns(text)

    return text


def expand_abbreviations(text: str) -> str:
    """
    Expand common maintenance abbreviations.

    Args:
        text: Input text

    Returns:
        Text with abbreviations expanded
    """
    result = text

    # Process abbreviations (case-insensitive)
    for abbr, expansion in ABBREVIATIONS.items():
        # Use word boundaries for most abbreviations
        if abbr.endswith("-"):
            # For prefixes like "P-", "V-", etc.
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            result = pattern.sub(expansion, result)
        else:
            # For standalone abbreviations
            pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
            result = pattern.sub(expansion, result)

    return result


def normalize_technical_patterns(text: str) -> str:
    """
    Normalize technical patterns like measurements, equipment tags, etc.

    Args:
        text: Input text

    Returns:
        Text with normalized patterns
    """
    result = text

    # Normalize pressure values (e.g., "100psi" -> "100 psi")
    result = re.sub(r"(\d+)\s*(psi|psig|bar|kpa|mpa)", r"\1 \2", result, flags=re.IGNORECASE)

    # Normalize temperature values
    result = re.sub(r"(\d+)\s*(°?[cf]|celsius|fahrenheit)", r"\1 \2", result, flags=re.IGNORECASE)

    # Normalize flow values
    result = re.sub(r"(\d+)\s*(gpm|lpm|cfm|m3/h)", r"\1 \2", result, flags=re.IGNORECASE)

    # Normalize speed values
    result = re.sub(r"(\d+)\s*(rpm)", r"\1 \2", result, flags=re.IGNORECASE)

    # Normalize power values
    result = re.sub(r"(\d+)\s*(hp|kw|mw)", r"\1 \2", result, flags=re.IGNORECASE)

    # Preserve equipment tags (e.g., P-101, HX-205)
    # Already handled by not modifying alphanumeric-hyphen patterns

    return result


def combine_lesson_text(lesson: dict, include_metadata: bool = True) -> str:
    """
    Combine lesson fields into a single text for embedding.

    Args:
        lesson: Dictionary containing lesson data
        include_metadata: Whether to include metadata in the combined text

    Returns:
        Combined text
    """
    parts = []

    # Title
    if lesson.get("title"):
        parts.append(f"Title: {lesson['title']}")

    # Description
    if lesson.get("description"):
        parts.append(f"Description: {lesson['description']}")

    # Root cause
    if lesson.get("root_cause"):
        parts.append(f"Root Cause: {lesson['root_cause']}")

    # Corrective action
    if lesson.get("corrective_action"):
        parts.append(f"Corrective Action: {lesson['corrective_action']}")

    if include_metadata:
        # Category
        if lesson.get("category"):
            parts.append(f"Category: {lesson['category']}")

        # Equipment
        if lesson.get("equipment_tag"):
            parts.append(f"Equipment: {lesson['equipment_tag']}")

        # Severity
        if lesson.get("severity"):
            parts.append(f"Severity: {lesson['severity']}")

    return "\n\n".join(parts)


def combine_job_text(job: dict, include_metadata: bool = True) -> str:
    """
    Combine job fields into a single text for querying.

    Args:
        job: Dictionary containing job data
        include_metadata: Whether to include metadata in the combined text

    Returns:
        Combined text
    """
    parts = []

    # Title
    if job.get("job_title"):
        parts.append(f"Job Title: {job['job_title']}")

    # Description
    if job.get("job_description"):
        parts.append(f"Job Description: {job['job_description']}")

    if include_metadata:
        # Equipment
        if job.get("equipment_tag"):
            parts.append(f"Equipment: {job['equipment_tag']}")

        # Job type
        if job.get("job_type"):
            parts.append(f"Job Type: {job['job_type']}")

    return "\n\n".join(parts)


def extract_equipment_info(text: str) -> dict:
    """
    Extract equipment information from text.

    Args:
        text: Input text

    Returns:
        Dictionary with extracted equipment info
    """
    result = {
        "equipment_tags": [],
        "equipment_types": [],
        "procedures": [],
    }

    # Extract equipment tags (e.g., P-101, HX-205, V-100)
    tag_pattern = r"\b([A-Z]{1,3}-\d{2,4}[A-Z]?)\b"
    tags = re.findall(tag_pattern, text, re.IGNORECASE)
    result["equipment_tags"] = list(set(tags))

    # Extract equipment types
    equipment_keywords = [
        "pump", "compressor", "heat exchanger", "valve", "tank", "vessel",
        "motor", "generator", "transformer", "column", "tower", "reactor",
        "pipe", "seal", "bearing", "coupling", "gearbox", "filter",
        "instrument", "sensor", "transmitter", "controller",
    ]
    for keyword in equipment_keywords:
        if keyword.lower() in text.lower():
            result["equipment_types"].append(keyword)

    # Extract procedures
    procedure_keywords = [
        "installation", "commissioning", "startup", "shutdown",
        "inspection", "maintenance", "repair", "replacement",
        "calibration", "testing", "alignment", "balancing",
        "lubrication", "cleaning", "flushing", "isolation",
        "lockout", "tagout", "permit", "verification", "training",
    ]
    for keyword in procedure_keywords:
        if keyword.lower() in text.lower():
            result["procedures"].append(keyword)

    return result
