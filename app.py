"""
Maintenance Lessons Learned RAG System

A Streamlit application for analyzing oil and gas turnaround maintenance
lessons learned and matching them to future maintenance jobs.
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import load_settings, Settings
from src.ui import (
    init_session_state,
    reset_session_state,
    render_upload_tab,
    render_review_tab,
    render_matching_tab,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Maintenance Lessons Learned RAG",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header():
    """Render the application header."""
    st.title("🔧 Maintenance Lessons Learned RAG")
    st.markdown(
        """
        *AI-powered analysis of historical maintenance lessons for future job planning*

        Upload your lessons learned and job descriptions to find relevant historical
        insights and safety considerations.
        """
    )


def render_sidebar(settings: Settings):
    """
    Render the application sidebar.

    Args:
        settings: Application settings
    """
    with st.sidebar:
        st.header("Settings")

        # Status indicators
        st.subheader("Data Status")

        if st.session_state.get("lessons_uploaded"):
            lessons_count = len(st.session_state.get("lessons_df", []))
            st.success(f"✓ {lessons_count} lessons loaded")
        else:
            st.warning("○ No lessons loaded")

        if st.session_state.get("lessons_enriched"):
            st.success("✓ Lessons enriched")
        else:
            st.info("○ Enrichment pending")

        if st.session_state.get("vector_store_initialized"):
            doc_count = st.session_state.get("documents_indexed", 0)
            st.success(f"✓ Index ready ({doc_count} chunks)")
        else:
            st.info("○ Index not built")

        if st.session_state.get("jobs_uploaded"):
            jobs_count = len(st.session_state.get("jobs_df", []))
            st.success(f"✓ {jobs_count} jobs loaded")
        else:
            st.warning("○ No jobs loaded")

        st.divider()

        # Configuration info
        st.subheader("Configuration")

        with st.expander("Azure OpenAI"):
            st.caption(f"Endpoint: {settings.azure_openai.endpoint[:30]}...")
            st.caption(f"Embedding: {settings.azure_openai.embedding_deployment}")
            st.caption(f"Chat: {settings.azure_openai.chat_deployment}")

        with st.expander("Retrieval Settings"):
            st.caption(f"Chunk size: {settings.chunking.chunk_size}")
            st.caption(f"Chunk overlap: {settings.chunking.chunk_overlap}")
            st.caption(f"RRF k: {settings.retrieval.rrf_k}")
            st.caption(f"Top k: {settings.retrieval.top_k}")

        with st.expander("Tier Boosts"):
            st.caption(f"Equipment-specific: {settings.retrieval.equipment_specific_boost}x")
            st.caption(f"Equipment-type: {settings.retrieval.equipment_type_boost}x")
            st.caption(f"Generic: {settings.retrieval.generic_boost}x")
            st.caption(f"Universal: {settings.retrieval.universal_boost}x")

        st.divider()

        # Actions
        st.subheader("Actions")

        if st.button("Reset All Data", type="secondary"):
            reset_session_state()
            st.rerun()

        # Debug mode toggle
        if settings.debug:
            st.divider()
            st.subheader("Debug Info")

            with st.expander("Session State"):
                st.json({
                    k: str(v)[:100] if not isinstance(v, (int, float, bool, type(None)))
                    else v
                    for k, v in st.session_state.items()
                    if not k.startswith("_")
                })


def main():
    """Main application entry point."""
    # Configure page
    configure_page()

    # Initialize session state
    init_session_state()

    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        st.error(f"Failed to load settings: {str(e)}")
        st.info("Please ensure your .env file is configured correctly.")
        st.code("""
# Required environment variables in .env:
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
        """)
        return

    # Render header
    render_header()

    # Render sidebar
    render_sidebar(settings)

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload & Enrich",
        "📝 Review & Edit",
        "🔍 Match & Analyze",
    ])

    # Render tab content
    with tab1:
        render_upload_tab(settings)

    with tab2:
        render_review_tab(settings)

    with tab3:
        render_matching_tab(settings)

    # Footer
    st.divider()
    st.caption(
        "Maintenance Lessons Learned RAG System | "
        "Powered by Azure OpenAI & LangChain | "
        "© 2026 Megat"
    )


if __name__ == "__main__":
    main()
