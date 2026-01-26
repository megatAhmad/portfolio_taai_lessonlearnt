"""Match & Analyze tab for Streamlit application."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import logging

from src.retrieval import create_hybrid_search, create_reranker, RetrievalResult
from src.generation import create_relevance_analyzer, format_analysis_for_display

from .utils import (
    dataframe_to_jobs_list,
    dataframe_to_lessons_list,
    create_export_excel,
    get_equipment_type_from_tag,
)
from .components import (
    render_job_card,
    render_match_result,
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

    control_cols = st.columns(3)

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

    # Run matching
    if st.button("Find Matches", type="primary"):
        run_matching(job, n_results, use_reranker, generate_analysis, settings)

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
    settings,
) -> None:
    """
    Run the matching process for a job.

    Args:
        job: Job dictionary
        n_results: Number of results to return
        use_reranker: Whether to use cross-encoder reranking
        generate_analysis: Whether to generate AI analysis
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
        progress.progress(0.2, text="Running hybrid search...")

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
            progress.progress(0.4, text="Reranking results...")

            reranker = create_reranker()
            results = reranker.rerank_with_tier_preservation(
                query=query,
                results=results,
                top_k=n_results,
                min_per_tier=1,
            )

        # Step 3: Get full lesson data
        progress.progress(0.6, text="Fetching lesson details...")

        lessons_df = st.session_state.lessons_df
        lesson_lookup = {
            row["lesson_id"]: row.to_dict()
            for _, row in lessons_df.iterrows()
        }

        # Step 4: Generate analysis
        if generate_analysis:
            progress.progress(0.7, text="Generating AI analysis...")

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

    if not results:
        render_info_message("No matching lessons found.")
        return

    # Export button
    export_cols = st.columns([3, 1])

    with export_cols[1]:
        excel_data = create_export_excel(results, job)
        st.download_button(
            label="Export to Excel",
            data=excel_data,
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

    st.divider()

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

        render_match_result(result, lesson, rank=i, show_details=True)


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
