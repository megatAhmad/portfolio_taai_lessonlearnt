"""GPT-4o-mini powered relevance analysis between lessons and jobs."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
import logging

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class RelevanceOutput(BaseModel):
    """Pydantic model for relevance analysis output validation."""

    relevance_score: int = Field(default=0, ge=0, le=100)
    technical_links: List[str] = Field(default_factory=list)
    safety_considerations: str = Field(default="")
    recommended_actions: List[str] = Field(default_factory=list)
    match_reasoning: str = Field(default="")


@dataclass
class RelevanceAnalysis:
    """Result of relevance analysis for a lesson-job pair."""

    lesson_id: str
    job_id: str
    relevance_score: int
    technical_links: List[str]
    safety_considerations: str
    recommended_actions: List[str]
    match_reasoning: str
    match_tier: str = "semantic"
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RelevanceAnalyzer:
    """Analyzes relevance between lessons learned and maintenance jobs."""

    def __init__(self, settings):
        """
        Initialize the relevance analyzer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai.endpoint,
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
        )
        self.deployment = settings.azure_openai.chat_deployment
        self.temperature = settings.generation.temperature
        self.max_tokens = settings.generation.max_tokens

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60))
    def _call_analysis_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Call Azure OpenAI API for relevance analysis.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt with lesson and job data

        Returns:
            Parsed JSON response
        """
        response = self.client.chat.completions.create(
            model=self.deployment,
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

    def analyze_relevance(
        self,
        lesson: Dict[str, Any],
        job: Dict[str, Any],
        match_info: Optional[Dict[str, Any]] = None,
    ) -> RelevanceAnalysis:
        """
        Analyze relevance between a lesson and a job.

        Args:
            lesson: Lesson dictionary
            job: Job dictionary
            match_info: Optional match information (tier, scores)

        Returns:
            RelevanceAnalysis result
        """
        from config.prompts import RELEVANCE_SYSTEM_PROMPT, format_relevance_prompt

        lesson_id = lesson.get("lesson_id", "unknown")
        job_id = job.get("job_id", "unknown")

        match_info = match_info or {}

        try:
            # Format the prompt
            user_prompt = format_relevance_prompt(lesson, job, match_info)

            # Call the API
            response = self._call_analysis_api(
                system_prompt=RELEVANCE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            # Validate response
            validated = RelevanceOutput(**response)

            return RelevanceAnalysis(
                lesson_id=lesson_id,
                job_id=job_id,
                relevance_score=validated.relevance_score,
                technical_links=validated.technical_links,
                safety_considerations=validated.safety_considerations,
                recommended_actions=validated.recommended_actions,
                match_reasoning=validated.match_reasoning,
                match_tier=match_info.get("match_type", "semantic"),
                retrieval_score=match_info.get("retrieval_score", 0.0),
                rerank_score=match_info.get("rerank_score", 0.0),
                success=True,
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for lesson {lesson_id}, job {job_id}: {e}")
            return RelevanceAnalysis(
                lesson_id=lesson_id,
                job_id=job_id,
                relevance_score=0,
                technical_links=[],
                safety_considerations="",
                recommended_actions=[],
                match_reasoning="",
                match_tier=match_info.get("match_type", "semantic"),
                retrieval_score=match_info.get("retrieval_score", 0.0),
                rerank_score=match_info.get("rerank_score", 0.0),
                success=False,
                error=f"Invalid JSON response: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error analyzing relevance for lesson {lesson_id}, job {job_id}: {e}")
            return RelevanceAnalysis(
                lesson_id=lesson_id,
                job_id=job_id,
                relevance_score=0,
                technical_links=[],
                safety_considerations="",
                recommended_actions=[],
                match_reasoning="",
                match_tier=match_info.get("match_type", "semantic"),
                retrieval_score=match_info.get("retrieval_score", 0.0),
                rerank_score=match_info.get("rerank_score", 0.0),
                success=False,
                error=str(e),
            )

    def analyze_batch(
        self,
        lessons: List[Dict[str, Any]],
        job: Dict[str, Any],
        match_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RelevanceAnalysis]:
        """
        Analyze relevance for multiple lessons against a single job.

        Args:
            lessons: List of lesson dictionaries
            job: Job dictionary
            match_infos: Optional list of match information

        Returns:
            List of RelevanceAnalysis results
        """
        results = []

        match_infos = match_infos or [{}] * len(lessons)

        for lesson, match_info in zip(lessons, match_infos):
            analysis = self.analyze_relevance(lesson, job, match_info)
            results.append(analysis)

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results


def format_analysis_for_display(analysis: RelevanceAnalysis) -> Dict[str, Any]:
    """
    Format a relevance analysis for UI display.

    Args:
        analysis: RelevanceAnalysis result

    Returns:
        Formatted dictionary for display
    """
    # Determine score category
    score = analysis.relevance_score
    if score >= 80:
        score_category = "high"
        score_color = "green"
    elif score >= 50:
        score_category = "medium"
        score_color = "orange"
    else:
        score_category = "low"
        score_color = "red"

    # Format tier for display
    tier_display = {
        "equipment_specific": "Equipment-Specific Match",
        "equipment_type": "Equipment-Type Match",
        "generic": "Generic/Universal Match",
        "semantic": "Semantic Match",
    }.get(analysis.match_tier, "Unknown Match")

    return {
        "lesson_id": analysis.lesson_id,
        "job_id": analysis.job_id,
        "relevance_score": score,
        "score_category": score_category,
        "score_color": score_color,
        "match_tier": analysis.match_tier,
        "match_tier_display": tier_display,
        "technical_links": analysis.technical_links,
        "safety_considerations": analysis.safety_considerations,
        "recommended_actions": analysis.recommended_actions,
        "match_reasoning": analysis.match_reasoning,
        "retrieval_score": analysis.retrieval_score,
        "rerank_score": analysis.rerank_score,
    }


def create_relevance_analyzer(settings) -> RelevanceAnalyzer:
    """
    Create a relevance analyzer instance.

    Args:
        settings: Application settings

    Returns:
        RelevanceAnalyzer instance
    """
    return RelevanceAnalyzer(settings)
