"""LLM prompt templates for enrichment and relevance analysis."""

# ============================================================================
# ENRICHMENT PROMPTS
# ============================================================================

ENRICHMENT_SYSTEM_PROMPT = """You are an expert maintenance data analyst specializing in Oil & Gas operations.
Analyze the provided maintenance lesson learned and extract structured metadata.

Return ONLY a valid JSON object with these fields:
{
  "specificity_level": "equipment_id" | "equipment_type" | "generic",
  "equipment_type": "string or null",
  "equipment_family": "rotating_equipment" | "static_equipment" | "instrumentation" | "electrical" | "piping" | null,
  "applicable_to": ["list", "of", "applicability", "scopes"],
  "procedure_tags": ["list", "of", "procedures"],
  "lesson_scope": "specific" | "general" | "universal",
  "safety_categories": ["list", "of", "safety", "categories"],
  "confidence_score": 0.0-1.0,
  "notes": "Any important observations"
}

Guidelines:
- If lesson mentions specific equipment (e.g., "P-101") but corrective action says "ALL" or "mandatory for all", mark as generic/universal
- Safety-critical lessons should have high confidence only if clearly documented
- Use standardized taxonomy:
  * Applicability: "all_pumps", "all_seals", "all_rotating_equipment", "all_heat_exchangers", "all_valves", "all_equipment"
  * Procedures: "installation", "commissioning", "startup", "shutdown", "inspection", "maintenance", "repair", "replacement", "lockout_tagout", "permit_to_work", "isolation", "calibration", "testing", "lubrication", "training", "verification"
  * Safety: "lockout_tagout", "permit_to_work", "confined_space", "hot_work", "pressure_release", "toxic_exposure", "flammable", "electrical_hazard", "fall_protection"
- If unclear or insufficient information, set low confidence (<0.70) and explain in notes
- For vague lessons with insufficient detail, use specificity_level: "unknown" and confidence_score < 0.30"""

ENRICHMENT_USER_PROMPT_TEMPLATE = """Analyze this maintenance lesson:

Lesson ID: {lesson_id}
Title: {title}
Description: {description}
Root Cause: {root_cause}
Corrective Action: {corrective_action}
Equipment Tag: {equipment_tag}
Category: {category}
Severity: {severity}

Extract structured metadata and return as JSON."""


# ============================================================================
# RELEVANCE ANALYSIS PROMPTS
# ============================================================================

RELEVANCE_SYSTEM_PROMPT = """You are an expert maintenance engineer analyzing the relevance between historical lessons learned and future maintenance jobs.

Analyze the provided lesson learned and job description. Consider the match type and lesson scope to provide accurate analysis.

Return a JSON object with:
{
  "relevance_score": 0-100,
  "technical_links": ["list of specific technical connections"],
  "safety_considerations": "text describing safety implications",
  "recommended_actions": ["list of specific action items"],
  "match_reasoning": "explanation of why this lesson matched and how it applies"
}

Guidelines:
- Be concise and focus on actionable insights
- Consider equipment type, failure modes, procedures, and safety implications
- For equipment-specific matches, highlight direct applicability
- For equipment-type matches, explain transferability to similar equipment
- For generic/universal matches, emphasize broad applicability
- Always prioritize safety-critical lessons
- If the lesson scope is "universal", emphasize that the lesson applies regardless of equipment"""

RELEVANCE_USER_PROMPT_TEMPLATE = """Analyze the relevance of this lesson to the job.

CONTEXT:
- Match Type: {match_type}
- Lesson Scope: {lesson_scope}
- Lesson Applicability: {applicable_to}
- Lesson Procedures: {procedure_tags}

LESSON LEARNED:
Lesson ID: {lesson_id}
Title: {title}
Description: {description}
Root Cause: {root_cause}
Corrective Action: {corrective_action}
Category: {category}
Severity: {severity}
Equipment: {equipment_tag}

JOB DESCRIPTION:
Job ID: {job_id}
Title: {job_title}
Description: {job_description}
Equipment: {job_equipment_tag}
Job Type: {job_type}

Provide relevance analysis as JSON."""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_enrichment_prompt(lesson: dict) -> str:
    """Format the enrichment prompt with lesson data."""
    return ENRICHMENT_USER_PROMPT_TEMPLATE.format(
        lesson_id=lesson.get("lesson_id", "N/A"),
        title=lesson.get("title", "N/A"),
        description=lesson.get("description", "N/A"),
        root_cause=lesson.get("root_cause", "N/A"),
        corrective_action=lesson.get("corrective_action", "N/A"),
        equipment_tag=lesson.get("equipment_tag", "N/A"),
        category=lesson.get("category", "N/A"),
        severity=lesson.get("severity", "N/A"),
    )


def format_relevance_prompt(lesson: dict, job: dict, match_info: dict) -> str:
    """Format the relevance analysis prompt with lesson, job, and match data."""
    return RELEVANCE_USER_PROMPT_TEMPLATE.format(
        # Match context
        match_type=match_info.get("match_type", "semantic"),
        lesson_scope=lesson.get("lesson_scope", "unknown"),
        applicable_to=", ".join(lesson.get("applicable_to", [])) if isinstance(lesson.get("applicable_to"), list) else lesson.get("applicable_to", "N/A"),
        procedure_tags=", ".join(lesson.get("procedure_tags", [])) if isinstance(lesson.get("procedure_tags"), list) else lesson.get("procedure_tags", "N/A"),
        # Lesson data
        lesson_id=lesson.get("lesson_id", "N/A"),
        title=lesson.get("title", "N/A"),
        description=lesson.get("description", "N/A"),
        root_cause=lesson.get("root_cause", "N/A"),
        corrective_action=lesson.get("corrective_action", "N/A"),
        category=lesson.get("category", "N/A"),
        severity=lesson.get("severity", "N/A"),
        equipment_tag=lesson.get("equipment_tag", "N/A"),
        # Job data
        job_id=job.get("job_id", "N/A"),
        job_title=job.get("job_title", "N/A"),
        job_description=job.get("job_description", "N/A"),
        job_equipment_tag=job.get("equipment_tag", "N/A"),
        job_type=job.get("job_type", "N/A"),
    )
