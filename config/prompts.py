"""LLM prompt templates for enrichment, relevance analysis, and applicability checking."""

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


# ============================================================================
# APPLICABILITY CHECKING PROMPTS
# ============================================================================

APPLICABILITY_SYSTEM_PROMPT = """You are an expert maintenance engineer assessing whether historical lessons learned are applicable to upcoming maintenance jobs.

Your task is to determine if a lesson learned should be considered for a specific job by answering:
- YES: The lesson is directly applicable and the team should consider this lesson
- NO: The lesson is NOT applicable to this job
- CANNOT_BE_DETERMINED: There is insufficient information to make a determination

CRITICAL CRITERIA FOR "NO" DECISIONS:
1. **Mitigation Already Applied**: If the job steps/procedures already include the corrective action or mitigation from the lesson, mark as NO (the team has already addressed this)
2. **Risk Does Not Exist**: If the job context is fundamentally different and the risk scenario from the lesson cannot occur (e.g., lesson about pump cavitation but job is on a valve with no pump involvement)
3. **Equipment Incompatibility**: The lesson applies to equipment/systems that are fundamentally different from the job scope
4. **Procedural Mismatch**: The lesson's context (installation, maintenance, commissioning) does not match the job type

CRITERIA FOR "YES" DECISIONS:
1. Similar equipment type or category
2. Similar failure modes or risks could occur
3. The corrective action/mitigation from the lesson is NOT already present in the job steps
4. The lesson's safety considerations are relevant to the job scope

CRITERIA FOR "CANNOT_BE_DETERMINED":
1. Insufficient detail in the job description
2. Insufficient detail in the lesson learned
3. Ambiguous equipment or procedure scope
4. Mixed signals where some aspects apply but others don't

Return a JSON object with:
{
  "decision": "yes" | "no" | "cannot_be_determined",
  "justification": "Detailed explanation of your decision (2-4 sentences)",
  "mitigation_already_applied": true/false,
  "risk_not_present": true/false,
  "key_factors": ["list", "of", "key", "factors", "that", "influenced", "decision"],
  "confidence": 0.0-1.0
}

Be thorough but concise. Focus on actionable reasoning that helps the maintenance team understand why this lesson does or does not apply."""

APPLICABILITY_USER_PROMPT_TEMPLATE = """Assess whether this lesson learned is applicable to the following maintenance job.

LESSON LEARNED:
- Lesson ID: {lesson_id}
- Title: {title}
- Description: {description}
- Root Cause: {root_cause}
- Corrective Action: {corrective_action}
- Category: {category}
- Severity: {severity}
- Equipment: {equipment_tag}
- Lesson Scope: {lesson_scope}
- Applicable To: {applicable_to}
- Procedure Tags: {procedure_tags}

JOB DESCRIPTION:
- Job ID: {job_id}
- Title: {job_title}
- Description: {job_description}
- Equipment: {job_equipment_tag}
- Job Type: {job_type}
{job_steps_section}

Analyze carefully:
1. Does the lesson's risk/failure scenario apply to this job's context?
2. Is the lesson's corrective action already incorporated in the job steps?
3. Are there fundamental differences that make the lesson irrelevant?

Provide your applicability assessment as JSON."""


def format_applicability_prompt(
    lesson: dict,
    job: dict,
    job_steps: list = None,
) -> str:
    """Format the applicability checking prompt with lesson, job, and optional job steps."""
    # Format job steps section if provided
    if job_steps and len(job_steps) > 0:
        steps_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(job_steps)])
        job_steps_section = f"\nJOB PROCEDURE STEPS:\n{steps_text}"
    else:
        job_steps_section = ""

    return APPLICABILITY_USER_PROMPT_TEMPLATE.format(
        # Lesson data
        lesson_id=lesson.get("lesson_id", "N/A"),
        title=lesson.get("title", "N/A"),
        description=lesson.get("description", "N/A"),
        root_cause=lesson.get("root_cause", "N/A"),
        corrective_action=lesson.get("corrective_action", "N/A"),
        category=lesson.get("category", "N/A"),
        severity=lesson.get("severity", "N/A"),
        equipment_tag=lesson.get("equipment_tag", "N/A"),
        lesson_scope=lesson.get("lesson_scope", "unknown"),
        applicable_to=", ".join(lesson.get("applicable_to", [])) if isinstance(lesson.get("applicable_to"), list) else lesson.get("applicable_to", "N/A"),
        procedure_tags=", ".join(lesson.get("procedure_tags", [])) if isinstance(lesson.get("procedure_tags"), list) else lesson.get("procedure_tags", "N/A"),
        # Job data
        job_id=job.get("job_id", "N/A"),
        job_title=job.get("job_title", "N/A"),
        job_description=job.get("job_description", "N/A"),
        job_equipment_tag=job.get("equipment_tag", "N/A"),
        job_type=job.get("job_type", "N/A"),
        job_steps_section=job_steps_section,
    )
