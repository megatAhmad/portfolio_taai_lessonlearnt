"""Excel file loading and validation for lessons learned and job descriptions."""

import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Required columns for lessons learned
LESSONS_REQUIRED_COLUMNS = [
    "lesson_id", "title", "description", "root_cause",
    "corrective_action", "category"
]

LESSONS_OPTIONAL_COLUMNS = [
    "equipment_tag", "date", "severity"
]

# Required columns for job descriptions
JOBS_REQUIRED_COLUMNS = [
    "job_id", "job_title", "job_description"
]

JOBS_OPTIONAL_COLUMNS = [
    "equipment_tag", "job_type", "planned_date"
]


def load_lessons_excel(file_path_or_buffer) -> Tuple[pd.DataFrame, list[str]]:
    """
    Load lessons learned from an Excel file.

    Args:
        file_path_or_buffer: Path to Excel file or file-like object

    Returns:
        Tuple of (DataFrame, list of error messages)
    """
    errors = []

    try:
        # Try to read the Excel file
        df = pd.read_excel(file_path_or_buffer)

        # Normalize column names (lowercase, strip whitespace)
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        # Validate schema
        is_valid, schema_errors = validate_lessons_schema(df)
        errors.extend(schema_errors)

        if not is_valid:
            return pd.DataFrame(), errors

        # Clean data
        df = clean_lessons_data(df)

        logger.info(f"Successfully loaded {len(df)} lessons from Excel")
        return df, errors

    except Exception as e:
        error_msg = f"Error loading Excel file: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return pd.DataFrame(), errors


def load_jobs_excel(file_path_or_buffer) -> Tuple[pd.DataFrame, list[str]]:
    """
    Load job descriptions from an Excel file.

    Args:
        file_path_or_buffer: Path to Excel file or file-like object

    Returns:
        Tuple of (DataFrame, list of error messages)
    """
    errors = []

    try:
        # Try to read the Excel file
        df = pd.read_excel(file_path_or_buffer)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        # Validate schema
        is_valid, schema_errors = validate_jobs_schema(df)
        errors.extend(schema_errors)

        if not is_valid:
            return pd.DataFrame(), errors

        # Clean data
        df = clean_jobs_data(df)

        logger.info(f"Successfully loaded {len(df)} jobs from Excel")
        return df, errors

    except Exception as e:
        error_msg = f"Error loading Excel file: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return pd.DataFrame(), errors


def validate_lessons_schema(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Validate the schema of a lessons learned DataFrame.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if df.empty:
        errors.append("Excel file is empty")
        return False, errors

    # Check for required columns
    missing_columns = []
    for col in LESSONS_REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors

    # Check for duplicate lesson IDs
    if df["lesson_id"].duplicated().any():
        duplicates = df[df["lesson_id"].duplicated()]["lesson_id"].tolist()
        errors.append(f"Duplicate lesson IDs found: {duplicates[:5]}...")

    # Check for minimum description length
    short_descriptions = df[df["description"].str.len() < 10]
    if not short_descriptions.empty:
        errors.append(f"{len(short_descriptions)} lessons have descriptions shorter than 10 characters")

    # Check for null required fields
    for col in LESSONS_REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} missing values")

    is_valid = len([e for e in errors if "Missing required columns" in e or "Duplicate lesson IDs" in e]) == 0
    return is_valid, errors


def validate_jobs_schema(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Validate the schema of a job descriptions DataFrame.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if df.empty:
        errors.append("Excel file is empty")
        return False, errors

    # Check for required columns
    missing_columns = []
    for col in JOBS_REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors

    # Check for duplicate job IDs
    if df["job_id"].duplicated().any():
        duplicates = df[df["job_id"].duplicated()]["job_id"].tolist()
        errors.append(f"Duplicate job IDs found: {duplicates[:5]}...")

    # Check for null required fields
    for col in JOBS_REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} missing values")

    is_valid = len([e for e in errors if "Missing required columns" in e or "Duplicate job IDs" in e]) == 0
    return is_valid, errors


def clean_lessons_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize lessons learned data.

    Args:
        df: DataFrame to clean

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Convert all text columns to string and strip whitespace
    text_columns = ["lesson_id", "title", "description", "root_cause", "corrective_action", "category"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", "")

    # Handle optional columns
    if "equipment_tag" in df.columns:
        df["equipment_tag"] = df["equipment_tag"].astype(str).str.strip()
        df["equipment_tag"] = df["equipment_tag"].replace(["nan", "None", ""], None)
    else:
        df["equipment_tag"] = None

    if "severity" in df.columns:
        df["severity"] = df["severity"].astype(str).str.lower().str.strip()
        df["severity"] = df["severity"].replace(["nan", "None", ""], "medium")
    else:
        df["severity"] = "medium"

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = None

    # Normalize category
    df["category"] = df["category"].str.lower().str.strip()

    return df


def clean_jobs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize job descriptions data.

    Args:
        df: DataFrame to clean

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Convert all text columns to string and strip whitespace
    text_columns = ["job_id", "job_title", "job_description"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", "")

    # Handle optional columns
    if "equipment_tag" in df.columns:
        df["equipment_tag"] = df["equipment_tag"].astype(str).str.strip()
        df["equipment_tag"] = df["equipment_tag"].replace(["nan", "None", ""], None)
    else:
        df["equipment_tag"] = None

    if "job_type" in df.columns:
        df["job_type"] = df["job_type"].astype(str).str.lower().str.strip()
        df["job_type"] = df["job_type"].replace(["nan", "None", ""], None)
    else:
        df["job_type"] = None

    if "planned_date" in df.columns:
        df["planned_date"] = pd.to_datetime(df["planned_date"], errors="coerce")
    else:
        df["planned_date"] = None

    return df


def get_column_summary(df: pd.DataFrame, column_type: str = "lessons") -> dict:
    """
    Get a summary of columns in the DataFrame.

    Args:
        df: DataFrame to summarize
        column_type: "lessons" or "jobs"

    Returns:
        Dictionary with column summary
    """
    if column_type == "lessons":
        required = LESSONS_REQUIRED_COLUMNS
        optional = LESSONS_OPTIONAL_COLUMNS
    else:
        required = JOBS_REQUIRED_COLUMNS
        optional = JOBS_OPTIONAL_COLUMNS

    present_required = [col for col in required if col in df.columns]
    missing_required = [col for col in required if col not in df.columns]
    present_optional = [col for col in optional if col in df.columns]
    extra_columns = [col for col in df.columns if col not in required + optional]

    return {
        "present_required": present_required,
        "missing_required": missing_required,
        "present_optional": present_optional,
        "extra_columns": extra_columns,
        "total_rows": len(df),
    }
