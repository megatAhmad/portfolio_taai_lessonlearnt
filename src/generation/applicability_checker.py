"""AI-powered applicability checking for lessons learned against job descriptions."""

import json
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.llm_client import create_chat_client, get_model_name

logger = logging.getLogger(__name__)


# Decision types
ApplicabilityDecision = Literal["yes", "no", "cannot_be_determined"]


class ApplicabilityOutput(BaseModel):
    """Pydantic model for applicability analysis output validation."""

    decision: str = Field(
        default="cannot_be_determined",
        description="Applicability decision: yes, no, or cannot_be_determined"
    )
    justification: str = Field(
        default="",
        description="Detailed justification for the decision"
    )
    mitigation_already_applied: bool = Field(
        default=False,
        description="Whether the lesson's mitigation is already present in job steps"
    )
    risk_not_present: bool = Field(
        default=False,
        description="Whether the risk described in the lesson does not exist in the job context"
    )
    key_factors: List[str] = Field(
        default_factory=list,
        description="Key factors that influenced the decision"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the decision (0.0-1.0)"
    )


@dataclass
class ApplicabilityResult:
    """Result of applicability check for a lesson-job pair."""

    lesson_id: str
    job_id: str
    decision: ApplicabilityDecision
    justification: str
    mitigation_already_applied: bool
    risk_not_present: bool
    key_factors: List[str]
    confidence: float
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def decision_display(self) -> str:
        """Human-readable decision display."""
        display_map = {
            "yes": "Applicable",
            "no": "Not Applicable",
            "cannot_be_determined": "Cannot Be Determined"
        }
        return display_map.get(self.decision, "Unknown")

    @property
    def decision_color(self) -> str:
        """Color code for UI display."""
        color_map = {
            "yes": "green",
            "no": "red",
            "cannot_be_determined": "orange"
        }
        return color_map.get(self.decision, "gray")

    @property
    def decision_emoji(self) -> str:
        """Emoji for UI display."""
        emoji_map = {
            "yes": "✅",
            "no": "❌",
            "cannot_be_determined": "⚠️"
        }
        return emoji_map.get(self.decision, "❓")


class ApplicabilityChecker:
    """Checks whether lessons learned are applicable to specific maintenance jobs."""

    def __init__(self, settings):
        """
        Initialize the applicability checker.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = create_chat_client(settings)
        self.model = get_model_name(settings, "chat")
        self.temperature = 0.2  # Lower temperature for more consistent decisions
        self.max_tokens = settings.generation.max_tokens

        logger.info(f"ApplicabilityChecker initialized with provider: {settings.llm_provider}, model: {self.model}")

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60))
    def _call_applicability_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Call LLM API for applicability analysis.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt with lesson and job data

        Returns:
            Parsed JSON response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)

    def check_applicability(
        self,
        lesson: Dict[str, Any],
        job: Dict[str, Any],
        job_steps: Optional[List[str]] = None,
    ) -> ApplicabilityResult:
        """
        Check if a lesson learned is applicable to a specific job.

        Args:
            lesson: Lesson dictionary
            job: Job dictionary
            job_steps: Optional list of job procedure steps

        Returns:
            ApplicabilityResult with decision and justification
        """
        from config.prompts import (
            APPLICABILITY_SYSTEM_PROMPT,
            format_applicability_prompt,
        )

        lesson_id = lesson.get("lesson_id", "unknown")
        job_id = job.get("job_id", "unknown")

        try:
            # Format the prompt
            user_prompt = format_applicability_prompt(lesson, job, job_steps)

            # Call the API
            response = self._call_applicability_api(
                system_prompt=APPLICABILITY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            # Validate response
            validated = ApplicabilityOutput(**response)

            # Normalize decision to expected values
            decision = validated.decision.lower().replace(" ", "_")
            if decision not in ["yes", "no", "cannot_be_determined"]:
                decision = "cannot_be_determined"

            return ApplicabilityResult(
                lesson_id=lesson_id,
                job_id=job_id,
                decision=decision,
                justification=validated.justification,
                mitigation_already_applied=validated.mitigation_already_applied,
                risk_not_present=validated.risk_not_present,
                key_factors=validated.key_factors,
                confidence=validated.confidence,
                success=True,
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for lesson {lesson_id}, job {job_id}: {e}")
            return ApplicabilityResult(
                lesson_id=lesson_id,
                job_id=job_id,
                decision="cannot_be_determined",
                justification="Error parsing AI response",
                mitigation_already_applied=False,
                risk_not_present=False,
                key_factors=[],
                confidence=0.0,
                success=False,
                error=f"Invalid JSON response: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error checking applicability for lesson {lesson_id}, job {job_id}: {e}")
            return ApplicabilityResult(
                lesson_id=lesson_id,
                job_id=job_id,
                decision="cannot_be_determined",
                justification=f"Error during analysis: {str(e)}",
                mitigation_already_applied=False,
                risk_not_present=False,
                key_factors=[],
                confidence=0.0,
                success=False,
                error=str(e),
            )

    def check_batch(
        self,
        lessons: List[Dict[str, Any]],
        job: Dict[str, Any],
        job_steps: Optional[List[str]] = None,
    ) -> List[ApplicabilityResult]:
        """
        Check applicability for multiple lessons against a single job.

        Args:
            lessons: List of lesson dictionaries
            job: Job dictionary
            job_steps: Optional list of job procedure steps

        Returns:
            List of ApplicabilityResult
        """
        results = []

        for lesson in lessons:
            result = self.check_applicability(lesson, job, job_steps)
            results.append(result)

        return results


def format_applicability_for_display(result: ApplicabilityResult) -> Dict[str, Any]:
    """
    Format an applicability result for UI display.

    Args:
        result: ApplicabilityResult

    Returns:
        Formatted dictionary for display
    """
    return {
        "lesson_id": result.lesson_id,
        "job_id": result.job_id,
        "decision": result.decision,
        "decision_display": result.decision_display,
        "decision_color": result.decision_color,
        "decision_emoji": result.decision_emoji,
        "justification": result.justification,
        "mitigation_already_applied": result.mitigation_already_applied,
        "risk_not_present": result.risk_not_present,
        "key_factors": result.key_factors,
        "confidence": result.confidence,
        "success": result.success,
        "error": result.error,
    }


def create_applicability_checker(settings) -> ApplicabilityChecker:
    """
    Create an applicability checker instance.

    Args:
        settings: Application settings

    Returns:
        ApplicabilityChecker instance
    """
    return ApplicabilityChecker(settings)
