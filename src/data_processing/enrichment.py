"""LLM-powered metadata enrichment for lessons learned."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field
import logging

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class EnrichmentOutput(BaseModel):
    """Pydantic model for enrichment output validation."""

    specificity_level: str = Field(default="unknown")
    equipment_type: Optional[str] = Field(default=None)
    equipment_family: Optional[str] = Field(default=None)
    applicable_to: List[str] = Field(default_factory=list)
    procedure_tags: List[str] = Field(default_factory=list)
    lesson_scope: str = Field(default="unknown")
    safety_categories: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = Field(default="")


@dataclass
class EnrichmentResult:
    """Result of enrichment for a single lesson."""

    lesson_id: str
    success: bool
    enrichment: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence: float = 0.0
    flag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnrichmentProgress:
    """Track enrichment progress."""

    total: int
    processed: int = 0
    successful: int = 0
    failed: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0

    @property
    def progress_pct(self) -> float:
        return (self.processed / self.total * 100) if self.total > 0 else 0


def create_openai_client(settings) -> AzureOpenAI:
    """Create Azure OpenAI client from settings."""
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai.endpoint,
        api_key=settings.azure_openai.api_key,
        api_version=settings.azure_openai.api_version,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_enrichment_api(
    client: AzureOpenAI,
    system_prompt: str,
    user_prompt: str,
    deployment: str,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI API for enrichment.

    Args:
        client: Azure OpenAI client
        system_prompt: System prompt
        user_prompt: User prompt with lesson data
        deployment: Deployment name
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response

    Returns:
        Parsed JSON response
    """
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


def enrich_single_lesson(
    lesson: Dict[str, Any],
    client: AzureOpenAI,
    system_prompt: str,
    deployment: str,
    settings: Any,
) -> EnrichmentResult:
    """
    Enrich a single lesson with LLM-generated metadata.

    Args:
        lesson: Lesson dictionary
        client: Azure OpenAI client
        system_prompt: System prompt for enrichment
        deployment: Deployment name
        settings: Application settings

    Returns:
        EnrichmentResult with enrichment data
    """
    from config.prompts import format_enrichment_prompt

    lesson_id = lesson.get("lesson_id", "unknown")

    try:
        # Format the user prompt
        user_prompt = format_enrichment_prompt(lesson)

        # Call the API
        response = call_enrichment_api(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            deployment=deployment,
            temperature=settings.enrichment.temperature,
            max_tokens=settings.enrichment.max_tokens,
        )

        # Validate response with Pydantic
        enrichment = EnrichmentOutput(**response)
        enrichment_dict = enrichment.model_dump()

        # Add metadata
        enrichment_dict["enrichment_timestamp"] = datetime.now().isoformat()
        enrichment_dict["enrichment_reviewed"] = False

        # Determine confidence level and flag
        confidence = enrichment_dict.get("confidence_score", 0.0)
        flag = determine_flag(confidence, lesson, enrichment_dict, settings)

        return EnrichmentResult(
            lesson_id=lesson_id,
            success=True,
            enrichment=enrichment_dict,
            confidence=confidence,
            flag=flag,
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for lesson {lesson_id}: {e}")
        return EnrichmentResult(
            lesson_id=lesson_id,
            success=False,
            error=f"Invalid JSON response: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error enriching lesson {lesson_id}: {e}")
        return EnrichmentResult(
            lesson_id=lesson_id,
            success=False,
            error=str(e),
        )


def determine_flag(
    confidence: float,
    lesson: Dict[str, Any],
    enrichment: Dict[str, Any],
    settings: Any,
) -> Optional[str]:
    """
    Determine if the enrichment needs flagging for review.

    Args:
        confidence: Confidence score
        lesson: Original lesson data
        enrichment: Enrichment data
        settings: Application settings

    Returns:
        Flag string or None
    """
    # Safety-critical always requires review
    if lesson.get("severity") == "critical":
        return "SAFETY_CRITICAL"

    # Low confidence requires review
    if confidence < settings.enrichment.medium_confidence_threshold:
        return "LOW_CONFIDENCE"

    # Check for insufficient information
    description_len = len(lesson.get("description", ""))
    if description_len < 50:
        return "INSUFFICIENT_INFORMATION"

    # Check for conflicting signals
    specificity = enrichment.get("specificity_level")
    scope = enrichment.get("lesson_scope")
    if specificity == "equipment_id" and scope == "universal":
        return "CONFLICTING_SIGNALS"

    return None


def enrich_lessons(
    lessons: List[Dict[str, Any]],
    settings: Any,
    progress_callback: Optional[Callable[[EnrichmentProgress], None]] = None,
    batch_size: int = 10,
) -> List[EnrichmentResult]:
    """
    Enrich multiple lessons with LLM-generated metadata.

    Args:
        lessons: List of lesson dictionaries
        settings: Application settings
        progress_callback: Optional callback for progress updates
        batch_size: Number of lessons per batch

    Returns:
        List of EnrichmentResult objects
    """
    from config.prompts import ENRICHMENT_SYSTEM_PROMPT

    results = []
    progress = EnrichmentProgress(total=len(lessons))

    # Create OpenAI client
    client = create_openai_client(settings)

    for i, lesson in enumerate(lessons):
        # Enrich single lesson
        result = enrich_single_lesson(
            lesson=lesson,
            client=client,
            system_prompt=ENRICHMENT_SYSTEM_PROMPT,
            deployment=settings.azure_openai.enrichment_deployment,
            settings=settings,
        )

        results.append(result)

        # Update progress
        progress.processed += 1
        if result.success:
            progress.successful += 1
            conf = result.confidence
            if conf >= settings.enrichment.high_confidence_threshold:
                progress.high_confidence += 1
            elif conf >= settings.enrichment.medium_confidence_threshold:
                progress.medium_confidence += 1
            else:
                progress.low_confidence += 1
        else:
            progress.failed += 1

        # Call progress callback
        if progress_callback:
            progress_callback(progress)

    logger.info(
        f"Enrichment complete: {progress.successful}/{progress.total} successful, "
        f"{progress.high_confidence} high confidence, "
        f"{progress.medium_confidence} medium confidence, "
        f"{progress.low_confidence} low confidence"
    )

    return results


def apply_enrichment_to_dataframe(df, results: List[EnrichmentResult]):
    """
    Apply enrichment results to a DataFrame.

    Args:
        df: pandas DataFrame with lessons
        results: List of EnrichmentResult objects

    Returns:
        DataFrame with enrichment columns added
    """
    import pandas as pd

    df = df.copy()

    # Create enrichment columns
    enrichment_columns = [
        "specificity_level", "equipment_type", "equipment_family",
        "applicable_to", "procedure_tags", "lesson_scope",
        "safety_categories", "enrichment_confidence",
        "enrichment_timestamp", "enrichment_reviewed", "enrichment_flag"
    ]

    for col in enrichment_columns:
        if col not in df.columns:
            df[col] = None

    # Apply enrichment results
    results_by_id = {r.lesson_id: r for r in results}

    for idx, row in df.iterrows():
        lesson_id = row["lesson_id"]
        result = results_by_id.get(lesson_id)

        if result and result.success and result.enrichment:
            enrichment = result.enrichment

            df.at[idx, "specificity_level"] = enrichment.get("specificity_level")
            df.at[idx, "equipment_type"] = enrichment.get("equipment_type")
            df.at[idx, "equipment_family"] = enrichment.get("equipment_family")

            # Convert lists to comma-separated strings
            df.at[idx, "applicable_to"] = ",".join(enrichment.get("applicable_to", []))
            df.at[idx, "procedure_tags"] = ",".join(enrichment.get("procedure_tags", []))
            df.at[idx, "safety_categories"] = ",".join(enrichment.get("safety_categories", []))

            df.at[idx, "lesson_scope"] = enrichment.get("lesson_scope")
            df.at[idx, "enrichment_confidence"] = enrichment.get("confidence_score", 0.0)
            df.at[idx, "enrichment_timestamp"] = enrichment.get("enrichment_timestamp")
            df.at[idx, "enrichment_reviewed"] = enrichment.get("enrichment_reviewed", False)
            df.at[idx, "enrichment_flag"] = result.flag

    return df


def get_enrichment_summary(df) -> Dict[str, Any]:
    """
    Get summary statistics for enrichment.

    Args:
        df: DataFrame with enrichment data

    Returns:
        Dictionary with summary statistics
    """
    total = len(df)

    if "enrichment_confidence" not in df.columns:
        return {
            "total": total,
            "enriched": 0,
            "not_enriched": total,
        }

    enriched = df["enrichment_confidence"].notna().sum()

    # Confidence breakdown
    high = (df["enrichment_confidence"] >= 0.85).sum()
    medium = ((df["enrichment_confidence"] >= 0.70) & (df["enrichment_confidence"] < 0.85)).sum()
    low = (df["enrichment_confidence"] < 0.70).sum()

    # Review status
    reviewed = df["enrichment_reviewed"].sum() if "enrichment_reviewed" in df.columns else 0

    # Flag breakdown
    flags = {}
    if "enrichment_flag" in df.columns:
        flag_counts = df["enrichment_flag"].value_counts()
        flags = flag_counts.to_dict()

    return {
        "total": total,
        "enriched": int(enriched),
        "not_enriched": int(total - enriched),
        "high_confidence": int(high),
        "medium_confidence": int(medium),
        "low_confidence": int(low),
        "reviewed": int(reviewed),
        "pending_review": int(enriched - reviewed),
        "avg_confidence": float(df["enrichment_confidence"].mean()) if enriched > 0 else 0.0,
        "flags": flags,
    }
