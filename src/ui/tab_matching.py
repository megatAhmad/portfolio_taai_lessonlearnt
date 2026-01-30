"""Match & Analyze tab for Streamlit application."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import logging

from src.retrieval import create_hybrid_search, create_reranker, RetrievalResult
from src.generation import (
    create_relevance_analyzer,
    format_analysis_for_display,
    create_applicability_checker,
    format_applicability_for_display,
)

from .utils import (
    dataframe_to_jobs_list,
    dataframe_to_lessons_list,
    create_export_excel,
    get_equipment_type_from_tag,
)
from .components import (
    render_job_card,
    render_match_result,
    render_match_result_with_applicability,
    render_error_message,
    render_success_message,
    render_info_message,
    render_warning_message,
)

logger = logging.getLogger(__name__)


def render_matching_tab(settings) -> None:
    """
    Render the Match & Analyze tab.

    Args:
        settings: Application settings
    """
    st.header("Match Jobs to Lessons")

    # Check prerequisites
    if not st.session_state.get("jobs_uploaded") or st.session_state.get("jobs_df") is None:
        render_info_message("Please upload job descriptions in the 'Upload & Enrich' tab first.")
        return

    if not st.session_state.get("vector_store_initialized"):
        render_warning_message(
            "Vector store not initialized. "
            "Please build the index in the 'Upload & Enrich' tab first."
        )
        return

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        render_job_selection()

    with col2:
        if st.session_state.get("selected_job"):
            render_matching_results(settings)
        else:
            render_info_message("Select a job from the left panel to see matched lessons.")


def render_job_selection() -> None:
    """Render the job selection panel."""
    st.subheader("Select Job")

    jobs_df = st.session_state.jobs_df
    jobs = dataframe_to_jobs_list(jobs_df)

    # Search/filter
    search_query = st.text_input("Search jobs", placeholder="Enter job ID or title...")

    if search_query:
        filtered_jobs = [
            j for j in jobs
            if search_query.lower() in (j.get("job_id", "").lower() + j.get("job_title", "").lower())
        ]
    else:
        filtered_jobs = jobs

    st.caption(f"Showing {len(filtered_jobs)} of {len(jobs)} jobs")

    # Job list
    for job in filtered_jobs:
        job_id = job.get("job_id", "Unknown")
        is_selected = (
            st.session_state.get("selected_job")
            and st.session_state.selected_job.get("job_id") == job_id
        )

        with st.container():
            cols = st.columns([3, 1])

            with cols[0]:
                title = job.get("job_title", "No title")
                st.markdown(f"**{job_id}**: {title[:50]}...")

            with cols[1]:
                button_type = "primary" if is_selected else "secondary"
                if st.button("Select", key=f"select_job_{job_id}", type=button_type):
                    st.session_state.selected_job = job
                    st.session_state.matching_results = None  # Reset results
                    st.rerun()

            st.divider()


def render_matching_results(settings) -> None:
    """
    Render the matching results panel.

    Args:
        settings: Application settings
    """
    job = st.session_state.selected_job

    st.subheader("Job Details")

    # Job details
    with st.container():
        st.markdown(f"**Job ID:** {job.get('job_id', 'N/A')}")
        st.markdown(f"**Title:** {job.get('job_title', 'N/A')}")
        st.markdown(f"**Description:** {job.get('job_description', 'N/A')}")

        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.markdown(f"**Equipment:** {job.get('equipment_tag', 'N/A')}")
        with detail_cols[1]:
            st.markdown(f"**Type:** {job.get('job_type', 'N/A')}")
        with detail_cols[2]:
            st.markdown(f"**Planned:** {job.get('planned_date', 'N/A')}")

    st.divider()

    # Matching controls
    st.subheader("Find Matching Lessons")

    control_cols = st.columns(4)

    with control_cols[0]:
        n_results = st.selectbox(
            "Number of results",
            options=[3, 5, 10, 15],
            index=1,
        )

    with control_cols[1]:
        use_reranker = st.checkbox("Use reranker", value=True)

    with control_cols[2]:
        generate_analysis = st.checkbox("Generate AI analysis", value=True)

    with control_cols[3]:
        check_applicability = st.checkbox(
            "Check applicability",
            value=True,
            help="AI checks if each lesson is applicable to this job (Yes/No/Cannot Determine)"
        )

    # Run matching
    if st.button("Find Matches", type="primary"):
        run_matching(job, n_results, use_reranker, generate_analysis, check_applicability, settings)

    # Display results
    if st.session_state.get("matching_results"):
        st.divider()
        st.subheader("Matched Lessons")
        display_matching_results(job, settings)


def run_matching(
    job: Dict[str, Any],
    n_results: int,
    use_reranker: bool,
    generate_analysis: bool,
    check_applicability: bool,
    settings,
) -> None:
    """
    Run the matching process for a job.

    Args:
        job: Job dictionary
        n_results: Number of results to return
        use_reranker: Whether to use cross-encoder reranking
        generate_analysis: Whether to generate AI analysis
        check_applicability: Whether to check applicability for each lesson
        settings: Application settings
    """
    from src.data_processing.preprocessor import combine_job_text

    progress = st.progress(0, text="Starting matching...")

    try:
        # Get query text
        query = combine_job_text(job, include_metadata=True)

        # Get equipment info for tier matching
        equipment_tag = job.get("equipment_tag")
        equipment_type = get_equipment_type_from_tag(equipment_tag)

        # Step 1: Hybrid search
        progress.progress(0.15, text="Running hybrid search...")

        vector_store = st.session_state.vector_store
        bm25_index = st.session_state.bm25_index

        hybrid_search = create_hybrid_search(vector_store, bm25_index, settings)

        results, tier_results = hybrid_search.multi_tier_search(
            query=query,
            job_equipment_tag=equipment_tag,
            job_equipment_type=equipment_type,
            results_per_tier=n_results * 2,
            total_results=n_results * 3,
        )

        # Step 2: Reranking
        if use_reranker and results:
            progress.progress(0.3, text="Reranking results...")

            reranker = create_reranker()
            results = reranker.rerank_with_tier_preservation(
                query=query,
                results=results,
                top_k=n_results,
                min_per_tier=1,
            )

        # Step 3: Get full lesson data
        progress.progress(0.45, text="Fetching lesson details...")

        lessons_df = st.session_state.lessons_df
        lesson_lookup = {
            row["lesson_id"]: row.to_dict()
            for _, row in lessons_df.iterrows()
        }

        # Step 4: Generate analysis
        if generate_analysis:
            progress.progress(0.55, text="Generating AI analysis...")

            analyzer = create_relevance_analyzer(settings)
            analyses = []

            for result in results[:n_results]:
                lesson_id = result.metadata.get("lesson_id", "")
                lesson = lesson_lookup.get(lesson_id, {})

                # Clean NaN values
                for key, value in lesson.items():
                    if pd.isna(value):
                        lesson[key] = None

                match_info = {
                    "match_type": result.match_tier.value,
                    "retrieval_score": result.boosted_score,
                    "rerank_score": result.metadata.get("rerank_score", 0),
                }

                analysis = analyzer.analyze_relevance(lesson, job, match_info)
                formatted = format_analysis_for_display(analysis)

                # Merge lesson data
                formatted.update({
                    "title": lesson.get("title", ""),
                    "description": lesson.get("description", ""),
                    "root_cause": lesson.get("root_cause", ""),
                    "corrective_action": lesson.get("corrective_action", ""),
                    "category": lesson.get("category", ""),
                    "severity": lesson.get("severity", ""),
                    "equipment_tag": lesson.get("equipment_tag", ""),
                })

                analyses.append(formatted)

            st.session_state.matching_results = analyses
        else:
            # Without AI analysis, just use retrieval scores
            simple_results = []
            for i, result in enumerate(results[:n_results]):
                lesson_id = result.metadata.get("lesson_id", "")
                lesson = lesson_lookup.get(lesson_id, {})

                # Clean NaN values
                for key, value in lesson.items():
                    if pd.isna(value):
                        lesson[key] = None

                simple_results.append({
                    "lesson_id": lesson_id,
                    "job_id": job.get("job_id", ""),
                    "relevance_score": int(result.boosted_score * 100),
                    "match_tier": result.match_tier.value,
                    "match_tier_display": {
                        "equipment_specific": "Equipment-Specific Match",
                        "equipment_type": "Equipment-Type Match",
                        "generic": "Generic/Universal Match",
                        "semantic": "Semantic Match",
                    }.get(result.match_tier.value, "Unknown"),
                    "technical_links": [],
                    "safety_considerations": "",
                    "recommended_actions": [],
                    "match_reasoning": f"Matched via {result.match_tier.value} retrieval",
                    "title": lesson.get("title", ""),
                    "description": lesson.get("description", ""),
                    "root_cause": lesson.get("root_cause", ""),
                    "corrective_action": lesson.get("corrective_action", ""),
                    "category": lesson.get("category", ""),
                    "severity": lesson.get("severity", ""),
                    "equipment_tag": lesson.get("equipment_tag", ""),
                })

            st.session_state.matching_results = simple_results

        # Step 5: Applicability checking (if enabled)
        if check_applicability and st.session_state.matching_results:
            progress.progress(0.75, text="Checking applicability...")

            applicability_checker = create_applicability_checker(settings)
            applicability_results = {}

            # Extract job steps if available
            job_steps = job.get("job_steps", [])
            if isinstance(job_steps, str):
                job_steps = [s.strip() for s in job_steps.split("\n") if s.strip()]

            for i, match_result in enumerate(st.session_state.matching_results):
                lesson_id = match_result.get("lesson_id", "")
                lesson = lesson_lookup.get(lesson_id, {})

                # Clean NaN values
                for key, value in lesson.items():
                    if pd.isna(value):
                        lesson[key] = None

                # Run applicability check
                applicability = applicability_checker.check_applicability(
                    lesson=lesson,
                    job=job,
                    job_steps=job_steps,
                )
                formatted_applicability = format_applicability_for_display(applicability)
                applicability_results[lesson_id] = formatted_applicability

                # Update progress
                progress.progress(0.75 + (0.2 * (i + 1) / len(st.session_state.matching_results)),
                                  text=f"Checking applicability ({i+1}/{len(st.session_state.matching_results)})...")

            st.session_state.applicability_results = applicability_results
        else:
            st.session_state.applicability_results = None

        progress.progress(1.0, text="Matching complete!")
        render_success_message(f"Found {len(st.session_state.matching_results)} matching lessons!")

    except Exception as e:
        logger.exception("Matching failed")
        render_error_message(f"Matching failed: {str(e)}")


def display_matching_results(job: Dict[str, Any], settings) -> None:
    """
    Display the matching results.

    Args:
        job: Job dictionary
        settings: Application settings
    """
    results = st.session_state.matching_results
    applicability_results = st.session_state.get("applicability_results", {})

    if not results:
        render_info_message("No matching lessons found.")
        return

    # Export button
    export_cols = st.columns([3, 1])

    with export_cols[1]:
        # Include applicability in export if available
        export_data = create_export_excel_with_applicability(results, job, applicability_results)
        st.download_button(
            label="Export to Excel",
            data=export_data,
            file_name=f"matches_{job.get('job_id', 'job')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Results summary
    st.markdown(f"**Found {len(results)} matching lessons**")

    # Tier breakdown
    tier_counts = {}
    for r in results:
        tier = r.get("match_tier", "semantic")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    tier_cols = st.columns(len(tier_counts))
    for i, (tier, count) in enumerate(tier_counts.items()):
        with tier_cols[i]:
            tier_display = {
                "equipment_specific": "Equipment-Specific",
                "equipment_type": "Equipment-Type",
                "generic": "Generic/Universal",
                "semantic": "Semantic",
            }.get(tier, tier)
            st.metric(tier_display, count)

    # Applicability breakdown (if available)
    if applicability_results:
        st.divider()
        st.markdown("**Applicability Summary**")

        decision_counts = {"yes": 0, "no": 0, "cannot_be_determined": 0}
        for app_result in applicability_results.values():
            decision = app_result.get("decision", "cannot_be_determined")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        app_cols = st.columns(3)
        with app_cols[0]:
            st.metric("✅ Applicable", decision_counts["yes"])
        with app_cols[1]:
            st.metric("❌ Not Applicable", decision_counts["no"])
        with app_cols[2]:
            st.metric("⚠️ Cannot Determine", decision_counts["cannot_be_determined"])

    st.divider()

    # Filter controls for applicability
    if applicability_results:
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            applicability_filter = st.selectbox(
                "Filter by Applicability",
                options=["All", "Applicable", "Not Applicable", "Cannot Determine"],
                index=0,
            )
    else:
        applicability_filter = "All"

    # Render each result
    lessons_df = st.session_state.lessons_df
    lesson_lookup = {
        row["lesson_id"]: row.to_dict()
        for _, row in lessons_df.iterrows()
    }

    for i, result in enumerate(results, 1):
        lesson_id = result.get("lesson_id", "")
        lesson = lesson_lookup.get(lesson_id, {})

        # Clean NaN values
        for key, value in lesson.items():
            if pd.isna(value):
                lesson[key] = None

        # Get applicability result if available
        applicability = applicability_results.get(lesson_id) if applicability_results else None

        # Apply filter
        if applicability_filter != "All" and applicability:
            decision = applicability.get("decision", "")
            filter_map = {
                "Applicable": "yes",
                "Not Applicable": "no",
                "Cannot Determine": "cannot_be_determined",
            }
            if decision != filter_map.get(applicability_filter, ""):
                continue

        # Render with or without applicability
        if applicability:
            render_match_result_with_applicability(
                result, lesson, applicability, rank=i, show_details=True
            )
        else:
            render_match_result(result, lesson, rank=i, show_details=True)


def create_export_excel_with_applicability(
    results: List[Dict[str, Any]],
    job: Dict[str, Any],
    applicability_results: Dict[str, Dict[str, Any]] = None,
) -> bytes:
    """
    Create Excel export with applicability results.

    Args:
        results: List of matching results
        job: Job dictionary
        applicability_results: Dictionary of applicability results by lesson_id

    Returns:
        Excel file as bytes
    """
    import io

    # Build export data
    export_rows = []
    for result in results:
        lesson_id = result.get("lesson_id", "")

        row = {
            "Job ID": job.get("job_id", ""),
            "Job Title": job.get("job_title", ""),
            "Lesson ID": lesson_id,
            "Lesson Title": result.get("title", ""),
            "Relevance Score": result.get("relevance_score", 0),
            "Match Tier": result.get("match_tier_display", result.get("match_tier", "")),
            "Match Reasoning": result.get("match_reasoning", ""),
            "Technical Links": "; ".join(result.get("technical_links", [])),
            "Safety Considerations": result.get("safety_considerations", ""),
            "Recommended Actions": "; ".join(result.get("recommended_actions", [])),
            "Category": result.get("category", ""),
            "Severity": result.get("severity", ""),
            "Equipment": result.get("equipment_tag", ""),
        }

        # Add applicability columns if available
        if applicability_results and lesson_id in applicability_results:
            app = applicability_results[lesson_id]
            row["Applicability Decision"] = app.get("decision_display", "")
            row["Applicability Justification"] = app.get("justification", "")
            row["Mitigation Already Applied"] = "Yes" if app.get("mitigation_already_applied") else "No"
            row["Risk Not Present"] = "Yes" if app.get("risk_not_present") else "No"
            row["Key Factors"] = "; ".join(app.get("key_factors", []))
            row["Applicability Confidence"] = f"{app.get('confidence', 0):.0%}"

        export_rows.append(row)

    # Create DataFrame and export
    df = pd.DataFrame(export_rows)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Matching Results", index=False)
    output.seek(0)

    return output.getvalue()


def render_batch_matching(settings) -> None:
    """
    Render batch matching for all jobs.

    Args:
        settings: Application settings
    """
    st.subheader("Batch Matching")

    st.markdown("""
    Run matching for all uploaded jobs at once.
    Results can be exported as a comprehensive Excel report.
    """)

    jobs_df = st.session_state.jobs_df
    st.info(f"Will process {len(jobs_df)} jobs")

    n_results = st.selectbox(
        "Results per job",
        options=[3, 5, 10],
        index=1,
        key="batch_n_results",
    )

    if st.button("Run Batch Matching", type="primary", key="batch_match"):
        run_batch_matching(jobs_df, n_results, settings)


def run_batch_matching(
    jobs_df: pd.DataFrame,
    n_results: int,
    settings,
) -> None:
    """
    Run batch matching for all jobs.

    Args:
        jobs_df: DataFrame with jobs
        n_results: Number of results per job
        settings: Application settings
    """
    from src.data_processing.preprocessor import combine_job_text

    jobs = dataframe_to_jobs_list(jobs_df)
    all_results = {}

    progress = st.progress(0, text="Starting batch matching...")

    try:
        vector_store = st.session_state.vector_store
        bm25_index = st.session_state.bm25_index
        hybrid_search = create_hybrid_search(vector_store, bm25_index, settings)
        reranker = create_reranker()
        analyzer = create_relevance_analyzer(settings)

        lessons_df = st.session_state.lessons_df
        lesson_lookup = {
            row["lesson_id"]: row.to_dict()
            for _, row in lessons_df.iterrows()
        }

        for i, job in enumerate(jobs):
            pct = (i + 1) / len(jobs)
            progress.progress(pct, text=f"Processing job {i+1}/{len(jobs)}: {job.get('job_id', '')}")

            query = combine_job_text(job, include_metadata=True)
            equipment_tag = job.get("equipment_tag")
            equipment_type = get_equipment_type_from_tag(equipment_tag)

            # Hybrid search
            results, _ = hybrid_search.multi_tier_search(
                query=query,
                job_equipment_tag=equipment_tag,
                job_equipment_type=equipment_type,
                results_per_tier=n_results * 2,
                total_results=n_results * 3,
            )

            # Rerank
            results = reranker.rerank_with_tier_preservation(
                query=query,
                results=results,
                top_k=n_results,
            )

            # Analyze
            job_matches = []
            for result in results:
                lesson_id = result.metadata.get("lesson_id", "")
                lesson = lesson_lookup.get(lesson_id, {})

                for key, value in lesson.items():
                    if pd.isna(value):
                        lesson[key] = None

                match_info = {
                    "match_type": result.match_tier.value,
                    "retrieval_score": result.boosted_score,
                    "rerank_score": result.metadata.get("rerank_score", 0),
                }

                analysis = analyzer.analyze_relevance(lesson, job, match_info)
                formatted = format_analysis_for_display(analysis)
                formatted.update({
                    "title": lesson.get("title", ""),
                    "category": lesson.get("category", ""),
                    "severity": lesson.get("severity", ""),
                    "equipment_tag": lesson.get("equipment_tag", ""),
                })

                job_matches.append(formatted)

            all_results[job.get("job_id", f"job_{i}")] = {
                "job": job,
                "matches": job_matches,
            }

        st.session_state.batch_results = all_results
        progress.progress(1.0, text="Batch matching complete!")

        render_success_message(f"Processed {len(jobs)} jobs!")

        # Export all results
        # TODO: Create comprehensive batch export

    except Exception as e:
        logger.exception("Batch matching failed")
        render_error_message(f"Batch matching failed: {str(e)}")
