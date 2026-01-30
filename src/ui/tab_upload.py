"""Upload & Enrich tab for Streamlit application."""

from typing import Optional
import streamlit as st
import pandas as pd
import logging

from src.data_processing.excel_loader import load_lessons_excel, load_jobs_excel, get_column_summary
from src.data_processing.enrichment import (
    enrich_lessons,
    apply_enrichment_to_dataframe,
    get_enrichment_summary,
    EnrichmentProgress,
)
from src.data_processing.chunker import chunk_lessons
from src.retrieval import create_vector_store, create_bm25_index

from .utils import (
    dataframe_to_lessons_list,
    dataframe_to_jobs_list,
)
from .components import (
    render_enrichment_stats,
    render_progress_bar,
    render_error_message,
    render_success_message,
    render_info_message,
    render_warning_message,
)

logger = logging.getLogger(__name__)


def render_upload_tab(settings) -> None:
    """
    Render the Upload & Enrich tab.

    Args:
        settings: Application settings
    """
    st.header("Upload & Enrich Data")

    # Create two columns for lessons and jobs upload
    col1, col2 = st.columns(2)

    # Lessons Upload Section
    with col1:
        st.subheader("Lessons Learned")
        render_lessons_upload(settings)

    # Jobs Upload Section
    with col2:
        st.subheader("Job Descriptions")
        render_jobs_upload()

    st.divider()

    # Enrichment Section
    if st.session_state.get("lessons_uploaded") and st.session_state.get("lessons_df") is not None:
        render_enrichment_section(settings)

    st.divider()

    # Indexing Section
    if st.session_state.get("lessons_enriched") or st.session_state.get("lessons_uploaded"):
        render_indexing_section(settings)


def render_lessons_upload(settings) -> None:
    """
    Render the lessons upload section.

    Args:
        settings: Application settings
    """
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Lessons Learned Excel",
        type=["xlsx", "xls"],
        key="lessons_file",
        help="Upload an Excel file containing lessons learned records",
    )

    if uploaded_file is not None:
        if st.button("Process Lessons", key="process_lessons"):
            with st.spinner("Loading and validating lessons..."):
                df, errors = load_lessons_excel(uploaded_file)

                if errors:
                    for error in errors:
                        if "Missing required columns" in error or "Duplicate lesson IDs" in error:
                            render_error_message(error)
                        else:
                            render_warning_message(error)

                if not df.empty:
                    st.session_state.lessons_df = df
                    st.session_state.lessons_uploaded = True
                    st.session_state.lessons_enriched = False  # Reset enrichment status

                    render_success_message(f"Loaded {len(df)} lessons successfully!")

                    # Show column summary
                    summary = get_column_summary(df, "lessons")
                    with st.expander("Column Summary"):
                        st.json(summary)
                else:
                    render_error_message("No valid lessons could be loaded from the file.")

    # Show current data status
    if st.session_state.get("lessons_uploaded") and st.session_state.get("lessons_df") is not None:
        df = st.session_state.lessons_df
        st.info(f"Current data: {len(df)} lessons loaded")

        with st.expander("Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)

        # Download current data
        if st.button("Download Lessons", key="download_lessons"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="lessons_learned.csv",
                mime="text/csv",
            )


def render_jobs_upload() -> None:
    """Render the jobs upload section."""
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Job Descriptions Excel",
        type=["xlsx", "xls"],
        key="jobs_file",
        help="Upload an Excel file containing job descriptions",
    )

    if uploaded_file is not None:
        if st.button("Process Jobs", key="process_jobs"):
            with st.spinner("Loading and validating jobs..."):
                df, errors = load_jobs_excel(uploaded_file)

                if errors:
                    for error in errors:
                        if "Missing required columns" in error or "Duplicate job IDs" in error:
                            render_error_message(error)
                        else:
                            render_warning_message(error)

                if not df.empty:
                    st.session_state.jobs_df = df
                    st.session_state.jobs_uploaded = True

                    render_success_message(f"Loaded {len(df)} jobs successfully!")

                    # Show column summary
                    summary = get_column_summary(df, "jobs")
                    with st.expander("Column Summary"):
                        st.json(summary)
                else:
                    render_error_message("No valid jobs could be loaded from the file.")

    # Show current data status
    if st.session_state.get("jobs_uploaded") and st.session_state.get("jobs_df") is not None:
        df = st.session_state.jobs_df
        st.info(f"Current data: {len(df)} jobs loaded")

        with st.expander("Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)


def render_enrichment_section(settings) -> None:
    """
    Render the enrichment section.

    Args:
        settings: Application settings
    """
    st.subheader("LLM Metadata Enrichment")

    df = st.session_state.lessons_df

    # Check if already enriched
    if st.session_state.get("lessons_enriched"):
        render_success_message("Lessons have been enriched!")

        # Show enrichment statistics
        stats = get_enrichment_summary(df)
        render_enrichment_stats(stats)

        # Option to re-enrich
        if st.button("Re-enrich All Lessons", key="reenrich"):
            st.session_state.lessons_enriched = False
            st.rerun()

        return

    # Enrichment controls
    st.markdown("""
    This step uses Azure OpenAI to automatically generate metadata for each lesson:
    - Specificity level and scope
    - Equipment type and family classification
    - Applicable procedures and safety categories
    - Confidence scoring
    """)

    col1, col2 = st.columns(2)

    with col1:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of lessons to process at once",
        )

    with col2:
        st.markdown(f"**Lessons to enrich:** {len(df)}")
        st.markdown(f"**Estimated API calls:** {len(df)}")

    # Warning about API costs
    render_warning_message(
        "Enrichment uses Azure OpenAI API calls. "
        f"Estimated cost: ~${len(df) * 0.001:.2f} (at ~$0.001/call)"
    )

    # Enrich button
    if st.button("Start Enrichment", key="start_enrichment", type="primary"):
        run_enrichment(df, settings, batch_size)


def run_enrichment(df: pd.DataFrame, settings, batch_size: int) -> None:
    """
    Run the enrichment process.

    Args:
        df: DataFrame with lessons
        settings: Application settings
        batch_size: Batch size for processing
    """
    lessons = dataframe_to_lessons_list(df)

    # Progress tracking
    progress_bar = st.progress(0, text="Starting enrichment...")
    status_text = st.empty()

    def progress_callback(progress: EnrichmentProgress):
        pct = progress.progress_pct / 100
        progress_bar.progress(
            pct,
            text=f"Processing: {progress.processed}/{progress.total} lessons"
        )
        status_text.markdown(
            f"**Progress:** {progress.successful} successful, "
            f"{progress.failed} failed, "
            f"{progress.high_confidence} high confidence"
        )

    try:
        # Run enrichment
        results = enrich_lessons(
            lessons=lessons,
            settings=settings,
            progress_callback=progress_callback,
            batch_size=batch_size,
        )

        # Apply results to DataFrame
        enriched_df = apply_enrichment_to_dataframe(df, results)

        # Update session state
        st.session_state.lessons_df = enriched_df
        st.session_state.lessons_enriched = True
        st.session_state.enrichment_results = results

        progress_bar.progress(1.0, text="Enrichment complete!")

        # Show summary
        stats = get_enrichment_summary(enriched_df)
        render_enrichment_stats(stats)

        render_success_message(
            f"Enrichment complete! {stats['enriched']} lessons enriched "
            f"with {stats['avg_confidence']:.1%} average confidence."
        )

    except Exception as e:
        logger.exception("Enrichment failed")
        render_error_message(f"Enrichment failed: {str(e)}")


def render_indexing_section(settings) -> None:
    """
    Render the indexing section.

    Args:
        settings: Application settings
    """
    st.subheader("Vector Store Indexing")

    df = st.session_state.lessons_df

    # Check if already indexed
    if st.session_state.get("vector_store_initialized"):
        render_success_message(
            f"Vector store initialized with {st.session_state.documents_indexed} document chunks!"
        )

        # Option to rebuild index
        if st.button("Rebuild Index", key="rebuild_index"):
            st.session_state.vector_store_initialized = False
            st.rerun()

        return

    st.markdown("""
    This step creates embeddings and indexes the lessons for retrieval:
    - Text chunking with metadata preservation
    - Azure OpenAI embeddings generation
    - ChromaDB vector store indexing
    - BM25 sparse index creation
    """)

    include_enrichment = st.checkbox(
        "Include enrichment metadata",
        value=True,
        help="Include LLM-generated metadata in the index",
    )

    # Warning about API costs
    estimated_chunks = len(df) * 1.5  # Rough estimate
    render_info_message(
        f"Estimated embedding cost: ~${estimated_chunks * 0.0001:.3f} "
        f"(~{estimated_chunks:.0f} chunks)"
    )

    if st.button("Build Index", key="build_index", type="primary"):
        run_indexing(df, settings, include_enrichment)


def run_indexing(df: pd.DataFrame, settings, include_enrichment: bool) -> None:
    """
    Run the indexing process.

    Args:
        df: DataFrame with lessons
        settings: Application settings
        include_enrichment: Whether to include enrichment metadata
    """
    lessons = dataframe_to_lessons_list(df)

    progress_bar = st.progress(0, text="Starting indexing...")

    try:
        # Step 1: Chunk lessons
        progress_bar.progress(0.2, text="Chunking lessons...")

        documents = chunk_lessons(
            lessons=lessons,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            include_enrichment=include_enrichment,
        )

        # Step 2: Create vector store
        progress_bar.progress(0.4, text="Creating vector store...")

        vector_store = create_vector_store(settings)

        # Step 3: Add documents to vector store
        progress_bar.progress(0.6, text="Generating embeddings and indexing...")

        vector_store.add_documents(documents)

        # Step 4: Create BM25 index
        progress_bar.progress(0.8, text="Building BM25 index...")

        bm25_index = create_bm25_index(documents)

        # Update session state
        st.session_state.vector_store = vector_store
        st.session_state.bm25_index = bm25_index
        st.session_state.vector_store_initialized = True
        st.session_state.documents_indexed = len(documents)

        progress_bar.progress(1.0, text="Indexing complete!")

        render_success_message(
            f"Index built successfully! {len(documents)} document chunks indexed."
        )

        # Show stats
        stats = vector_store.get_collection_stats()
        with st.expander("Index Statistics"):
            st.json(stats)

    except Exception as e:
        logger.exception("Indexing failed")
        render_error_message(f"Indexing failed: {str(e)}")
