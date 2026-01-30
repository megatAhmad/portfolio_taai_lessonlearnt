"""
Interactive Documentation UI for Maintenance Lessons Learned RAG System

Run with: streamlit run docs_app.py
"""

import streamlit as st
from pathlib import Path


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="RAG System Documentation",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_home():
    """Render the home/overview page."""
    st.title("📚 Maintenance RAG System Documentation")

    st.markdown("""
    Welcome to the interactive documentation for the **Maintenance Lessons Learned RAG System**.

    This application helps maintenance teams in the Oil & Gas industry:
    - **Analyze** historical lessons learned from turnaround maintenance
    - **Match** relevant lessons to future maintenance jobs using multi-tier retrieval
    - **Generate** AI-powered relevance analysis and safety recommendations
    - **Assess applicability** - AI determines if lessons truly apply (Yes/No/Cannot Determine)

    **Supports:** Azure OpenAI and OpenRouter as LLM providers

    ---

    ### Quick Navigation

    Use the sidebar to explore different sections of the documentation.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **🏗️ Architecture**

        Learn about the system architecture, including:
        - Multi-tier retrieval strategy
        - LLM enrichment pipeline
        - Hybrid search components
        """)

    with col2:
        st.success("""
        **📂 Code Structure**

        Explore the codebase organization:
        - Configuration modules
        - Data processing pipeline
        - Retrieval and generation
        """)

    with col3:
        st.warning("""
        **🚀 Getting Started**

        Quick start guide:
        - Installation steps
        - Configuration
        - Running the app
        """)


def render_architecture():
    """Render the architecture documentation page."""
    st.title("🏗️ System Architecture")

    st.markdown("""
    ## Overview

    The system follows a RAG (Retrieval-Augmented Generation) architecture with these key components:
    """)

    # Architecture diagram using text
    st.code("""
    ┌─────────────────────────────────────────────────────────────┐
    │                 STREAMLIT UI (3-TAB LAYOUT)                  │
    │  Tab 1: Upload & Enrich │ Tab 2: Review │ Tab 3: Match       │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │              LLM METADATA ENRICHMENT LAYER                   │
    │  GPT-4o-mini: Auto-generate 11 enrichment columns           │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │                  PROCESSING LAYER                            │
    │  Excel Parser ──► Preprocessor ──► Text Chunker             │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │        MULTI-TIER HYBRID RETRIEVAL ENGINE                    │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │ Tier 1: Equipment-Specific (boost 1.5x)             │    │
    │  │ Tier 2: Equipment-Type (boost 1.2x)                 │    │
    │  │ Tier 3: Generic/Universal (boost 1.3x/1.4x)         │    │
    │  │ Tier 4: Semantic (no filter)                        │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                         ↓                                    │
    │  RRF Fusion + Score Boosting ──► Cross-Encoder Rerank       │
    └─────────────────────┬───────────────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────────────┐
    │      LLM GENERATION LAYER (Azure OpenAI / OpenRouter)       │
    │  ┌───────────────────────┐  ┌───────────────────────────┐   │
    │  │ Relevance Analysis    │  │ Applicability Check       │   │
    │  │ • Score 0-100         │  │ • Yes/No/Cannot Determine │   │
    │  │ • Technical links     │  │ • Justification           │   │
    │  │ • Safety notes        │  │ • Mitigation flags        │   │
    │  └───────────────────────┘  └───────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
    """, language=None)

    st.info("📊 **Interactive Diagram**: Open `docs/architecture_diagram.html` in your browser for a detailed visual flow diagram.")

    st.divider()

    # Multi-tier retrieval explanation
    st.header("Multi-Tier Retrieval Strategy")

    st.markdown("""
    The system uses a **multi-tier retrieval strategy** to ensure both specific and generic lessons are surfaced:
    """)

    tier_data = {
        "Tier": ["1. Equipment-Specific", "2. Equipment-Type", "3. Generic/Universal", "4. Semantic"],
        "Filter": ["equipment_id = job.equipment_id", "equipment_type = job.equipment_type", "lesson_scope = 'universal'", "No filter"],
        "Boost": ["1.5x", "1.2x", "1.3x / 1.4x", "1.0x (base)"],
        "Purpose": [
            "Direct equipment history",
            "Similar equipment patterns",
            "Broadly applicable lessons",
            "Catch-all semantic similarity"
        ]
    }

    st.table(tier_data)

    st.divider()

    # RRF explanation
    st.header("Reciprocal Rank Fusion (RRF)")

    st.markdown("""
    Results from dense (embedding) and sparse (BM25) retrieval are combined using **Reciprocal Rank Fusion**:
    """)

    st.latex(r"RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}")

    st.markdown("""
    Where:
    - `d` = document
    - `R` = set of rankings (dense, sparse)
    - `r(d)` = rank of document d in ranking r
    - `k` = 60 (constant to prevent high-ranked documents from dominating)

    **Why RRF?**
    - Combines semantic understanding (dense) with keyword matching (sparse)
    - Robust to different scoring scales
    - No hyperparameter tuning needed beyond k
    """)


def render_code_structure():
    """Render the code structure documentation page."""
    st.title("📂 Code Structure")

    st.markdown("""
    ## Project Organization

    The codebase is organized into logical modules:
    """)

    # Directory tree
    st.code("""
    portfolio_taai_lessonlearnt/
    ├── app.py                    # Main Streamlit application
    ├── docs_app.py               # This documentation app
    ├── requirements.txt          # Python dependencies
    ├── .env.example              # Environment variable template
    │
    ├── config/                   # Configuration
    │   ├── __init__.py
    │   ├── settings.py           # All application settings
    │   ├── prompts.py            # LLM prompt templates
    │   └── llm_client.py         # LLM client factory (Azure/OpenRouter)
    │
    ├── src/
    │   ├── data_processing/      # Data handling
    │   │   ├── excel_loader.py   # Excel file parsing
    │   │   ├── preprocessor.py   # Text preprocessing
    │   │   ├── chunker.py        # Text chunking
    │   │   └── enrichment.py     # LLM metadata enrichment
    │   │
    │   ├── retrieval/            # Search components
    │   │   ├── embeddings.py     # Embeddings (Azure/OpenRouter)
    │   │   ├── vector_store.py   # ChromaDB operations
    │   │   ├── bm25_search.py    # BM25 sparse retrieval
    │   │   ├── hybrid_search.py  # Multi-tier hybrid search
    │   │   └── reranker.py       # Cross-encoder reranking
    │   │
    │   ├── generation/           # AI generation
    │   │   ├── relevance_analyzer.py    # Relevance analysis
    │   │   └── applicability_checker.py # Applicability assessment
    │   │
    │   └── ui/                   # Streamlit UI
    │       ├── components.py     # Reusable UI components
    │       ├── utils.py          # UI utilities
    │       ├── tab_upload.py     # Upload & Enrich tab
    │       ├── tab_review.py     # Review & Edit tab
    │       └── tab_matching.py   # Match & Analyze tab
    │
    ├── data/                     # Sample data
    │   └── create_sample_data.py # Sample data generator
    │
    ├── docs/                     # Documentation
    │   ├── architecture_diagram.html  # Interactive diagram
    │   └── architecture_diagram.svg   # Vector diagram
    │
    └── chroma_db/                # Vector store persistence
    """, language=None)

    st.divider()

    # Module details
    st.header("Module Details")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Processing",
        "🔍 Retrieval",
        "🤖 Generation",
        "🖥️ UI"
    ])

    with tab1:
        render_data_processing_docs()

    with tab2:
        render_retrieval_docs()

    with tab3:
        render_generation_docs()

    with tab4:
        render_ui_docs()


def render_data_processing_docs():
    """Render data processing module documentation."""
    st.subheader("Data Processing Modules")

    st.markdown("### excel_loader.py")
    st.markdown("""
    Handles Excel file loading and validation:
    - `load_lessons_excel()` - Load lessons learned data
    - `load_jobs_excel()` - Load job descriptions
    - Schema validation with pandera
    - Data cleaning and normalization
    """)

    with st.expander("View Required Columns"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Lessons (Required)**")
            st.code("""
lesson_id       # Unique identifier
title           # Brief title
description     # Detailed description
root_cause      # Root cause analysis
corrective_action # Actions taken
category        # mechanical/electrical/safety/process
            """)
        with col2:
            st.markdown("**Jobs (Required)**")
            st.code("""
job_id          # Unique identifier
job_title       # Brief title
job_description # Work description
            """)

    st.markdown("### preprocessor.py")
    st.markdown("""
    Text preprocessing utilities:
    - Abbreviation expansion (P&ID → Piping and Instrumentation Diagram)
    - Technical pattern normalization
    - Equipment tag extraction
    - Text combination for embedding
    """)

    st.markdown("### chunker.py")
    st.markdown("""
    Text chunking with metadata preservation:
    - Uses LangChain RecursiveCharacterTextSplitter
    - Default: 2000 chars/chunk, 400 char overlap
    - Preserves lesson metadata in chunk metadata
    - Creates LangChain Document objects
    """)

    st.markdown("### enrichment.py")
    st.markdown("""
    LLM-powered metadata enrichment:
    - Calls GPT-4o-mini to extract structured metadata
    - Generates 11 enrichment columns automatically
    - Confidence scoring and flagging
    - Batch processing with progress tracking
    """)


def render_retrieval_docs():
    """Render retrieval module documentation."""
    st.subheader("Retrieval Modules")

    st.markdown("### embeddings.py")
    st.markdown("""
    Embedding management (supports Azure OpenAI and OpenRouter):
    - Uses text-embedding-3-small (1536 dimensions)
    - Automatic provider selection based on `LLM_PROVIDER` env var
    - Disk and memory caching for cost optimization
    - Batch processing support
    - Token counting with tiktoken
    """)

    with st.expander("View Embedding Code Example"):
        st.code("""
from src.retrieval.embeddings import EmbeddingManager

# Initialize
manager = EmbeddingManager(settings)

# Single embedding
embedding = manager.embed_text("pump seal failure")

# Batch embeddings (with caching)
embeddings = manager.embed_texts(["text1", "text2", "text3"])
        """, language="python")

    st.markdown("### vector_store.py")
    st.markdown("""
    ChromaDB vector store operations:
    - Persistent storage in `chroma_db/` directory
    - Metadata filtering support
    - Search by equipment, type, category
    - Collection statistics
    """)

    st.markdown("### bm25_search.py")
    st.markdown("""
    BM25 sparse retrieval:
    - Keyword-based search using rank_bm25
    - Complements semantic search
    - Metadata filtering support
    """)

    st.markdown("### hybrid_search.py")
    st.markdown("""
    Multi-tier hybrid search:
    - Combines dense and sparse retrieval
    - Implements 4-tier search strategy
    - RRF score fusion
    - Tier-based score boosting
    """)

    with st.expander("View Score Boosting Logic"):
        st.code("""
def calculate_boosted_score(base_score, lesson, job):
    score = base_score

    # Tier 1: Equipment-specific boost
    if lesson['equipment_id'] == job['equipment_id']:
        score *= 1.5

    # Tier 2: Equipment-type boost
    elif lesson['equipment_type'] == job['equipment_type']:
        score *= 1.2

    # Tier 3: Universal/generic boost
    if lesson['lesson_scope'] == 'universal':
        score *= 1.3

    # Safety-critical boost (always)
    if lesson.get('severity') == 'critical':
        score *= 1.4

    return score
        """, language="python")

    st.markdown("### reranker.py")
    st.markdown("""
    Cross-encoder reranking:
    - Uses ms-marco-MiniLM-L-6-v2 model
    - Scores query-document pairs directly
    - Tier-aware reranking option
    - Score normalization (0-100)
    """)


def render_generation_docs():
    """Render generation module documentation."""
    st.subheader("Generation Module")

    st.markdown("### relevance_analyzer.py")
    st.markdown("""
    GPT-4o-mini powered relevance analysis:
    - Analyzes lesson-job relevance
    - Generates structured JSON output
    - Includes match reasoning
    - Safety considerations extraction
    """)

    with st.expander("View Output Schema"):
        st.code("""
{
    "relevance_score": 0-100,
    "technical_links": ["specific technical connections"],
    "safety_considerations": "safety implications text",
    "recommended_actions": ["action items"],
    "match_reasoning": "why this lesson applies"
}
        """, language="json")

    st.markdown("""
    **Key Features:**
    - Pydantic validation for output
    - Retry logic with tenacity
    - Match tier context in prompts
    - Batch analysis support
    """)

    st.divider()

    st.markdown("### applicability_checker.py")
    st.markdown("""
    **AI-Powered Applicability Assessment:**

    Determines whether each lesson learned is truly applicable to a specific job,
    going beyond similarity matching to assess actual relevance.
    """)

    st.markdown("""
    **Decision Types:**
    - **YES** - The lesson is directly applicable to the job
    - **NO** - The lesson is NOT applicable to this job
    - **CANNOT BE DETERMINED** - Insufficient information to decide
    """)

    st.info("""
    **Key Criteria for "NO" Decisions:**
    1. Mitigation already applied in job steps
    2. Risk does not exist in job context
    3. Equipment incompatibility
    4. Procedural mismatch
    """)

    with st.expander("View Applicability Output Schema"):
        st.code("""
{
    "decision": "yes" | "no" | "cannot_be_determined",
    "justification": "Detailed explanation (2-4 sentences)",
    "mitigation_already_applied": true/false,
    "risk_not_present": true/false,
    "key_factors": ["factor1", "factor2", ...],
    "confidence": 0.0-1.0
}
        """, language="json")

    st.markdown("""
    **Key Features:**
    - Analyzes job steps against lesson corrective actions
    - Detects when mitigations are already in place
    - Identifies when risks don't apply to job context
    - Provides detailed justification for each decision
    - Confidence scoring for transparency
    """)


def render_ui_docs():
    """Render UI module documentation."""
    st.subheader("UI Modules")

    st.markdown("### Three-Tab Layout")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Tab 1: Upload & Enrich**
        - File upload (lessons, jobs)
        - Validation feedback
        - Enrichment trigger
        - Index building
        """)

    with col2:
        st.success("""
        **Tab 2: Review & Edit**
        - Enrichment review
        - Inline editing
        - Flag management
        - Batch approval
        """)

    with col3:
        st.warning("""
        **Tab 3: Match & Analyze**
        - Job selection
        - Match configuration
        - Results display
        - Excel export
        """)

    st.markdown("### components.py")
    st.markdown("""
    Reusable Streamlit components:
    - `render_lesson_card()` - Display lesson with metadata
    - `render_job_card()` - Display job description
    - `render_match_result()` - Show match with analysis
    - `render_progress_bar()` - Progress tracking
    - Message helpers (error, success, info, warning)
    """)


def render_data_model():
    """Render the data model documentation page."""
    st.title("📊 Data Model")

    st.markdown("""
    ## Two-Stage Data Model

    The system uses a two-stage data model:
    1. **Basic Input** (9 columns) - Provided by user
    2. **Enrichment** (11 columns) - Auto-generated by LLM
    """)

    st.divider()

    # Stage 1
    st.header("Stage 1: Basic Input (User Provides)")

    stage1_data = {
        "Column": ["lesson_id", "title", "description", "root_cause", "corrective_action", "category", "equipment_tag", "date", "severity"],
        "Type": ["str", "str", "str", "str", "str", "str", "str (optional)", "datetime (optional)", "str (optional)"],
        "Description": [
            "Unique identifier (e.g., 'LL-001')",
            "Brief title/summary",
            "Detailed problem description",
            "Root cause analysis findings",
            "Actions taken to resolve",
            "mechanical/electrical/safety/process",
            "Equipment ID (e.g., 'P-101', 'HX-205')",
            "When lesson was learned",
            "low/medium/high/critical"
        ]
    }

    st.table(stage1_data)

    st.divider()

    # Stage 2
    st.header("Stage 2: Auto-Generated Enrichment (LLM Creates)")

    stage2_data = {
        "Column": [
            "specificity_level", "equipment_type", "equipment_family",
            "applicable_to", "procedure_tags", "lesson_scope",
            "safety_categories", "enrichment_confidence",
            "enrichment_timestamp", "enrichment_reviewed", "enrichment_flag"
        ],
        "Type": [
            "str", "str", "str",
            "str (comma-sep)", "str (comma-sep)", "str",
            "str (comma-sep)", "float",
            "datetime", "bool", "str"
        ],
        "Description": [
            "equipment_id / equipment_type / generic",
            "e.g., 'centrifugal_pump', 'heat_exchanger'",
            "rotating_equipment / static_equipment / ...",
            "Comma-separated: 'all_pumps,all_seals'",
            "Comma-separated: 'installation,lockout_tagout'",
            "specific / general / universal",
            "Comma-separated: 'lockout_tagout,pressure_release'",
            "0.0-1.0 confidence score",
            "When enrichment was performed",
            "Whether user reviewed/approved",
            "Review flag (SAFETY_CRITICAL, LOW_CONFIDENCE, etc.)"
        ]
    }

    st.table(stage2_data)

    st.divider()

    # Confidence thresholds
    st.header("Confidence Thresholds")

    threshold_data = {
        "Level": ["High", "Medium", "Low"],
        "Score Range": ["≥ 0.85", "0.70 - 0.84", "< 0.70"],
        "Action": [
            "Auto-accept, no review needed",
            "Yellow flag, suggest review",
            "Red flag, require user review"
        ]
    }

    st.table(threshold_data)


def render_getting_started():
    """Render the getting started guide."""
    st.title("🚀 Getting Started")

    st.markdown("## Installation")

    st.markdown("### 1. Clone the Repository")
    st.code("""
git clone https://github.com/yourusername/portfolio_taai_lessonlearnt.git
cd portfolio_taai_lessonlearnt
    """, language="bash")

    st.markdown("### 2. Create Virtual Environment")
    st.code("""
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    """, language="bash")

    st.markdown("### 3. Install Dependencies")
    st.code("""
pip install -r requirements.txt
    """, language="bash")

    st.markdown("### 4. Configure Environment")
    st.code("""
cp .env.example .env
# Edit .env with your LLM provider credentials (Azure OpenAI or OpenRouter)
    """, language="bash")

    st.markdown("""
    The system supports two LLM providers:
    - **Azure OpenAI** - Microsoft's hosted OpenAI models
    - **OpenRouter** - Multi-provider API gateway (access to OpenAI, Anthropic, and more)

    Choose one provider and configure the relevant environment variables.
    """)

    tab1, tab2 = st.tabs(["Azure OpenAI Configuration", "OpenRouter Configuration"])

    with tab1:
        st.code("""
# LLM Provider
LLM_PROVIDER=azure

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_ENRICHMENT_DEPLOYMENT=gpt-4o-mini
        """)
        st.info("Get your Azure OpenAI credentials from the Azure Portal.")

    with tab2:
        st.code("""
# LLM Provider
LLM_PROVIDER=openrouter

# OpenRouter Configuration
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini
OPENROUTER_ENRICHMENT_MODEL=openai/gpt-4o-mini
        """)
        st.info("Get your OpenRouter API key from https://openrouter.ai/keys")

    st.markdown("### 5. Generate Sample Data (Optional)")
    st.code("""
python data/create_sample_data.py
    """, language="bash")

    st.markdown("### 6. Run the Application")
    st.code("""
streamlit run app.py
    """, language="bash")

    st.divider()

    st.markdown("## Usage Workflow")

    st.markdown("""
    1. **Upload Data** (Tab 1)
       - Upload lessons learned Excel file
       - Upload job descriptions Excel file

    2. **Enrich Lessons** (Tab 1)
       - Click "Start Enrichment"
       - Wait for LLM to process all lessons
       - Review confidence scores

    3. **Build Index** (Tab 1)
       - Click "Build Index"
       - Creates embeddings and vector store

    4. **Review Enrichment** (Tab 2)
       - Review flagged lessons
       - Edit incorrect enrichment data
       - Approve high-confidence items

    5. **Match Jobs** (Tab 3)
       - Select a job to analyze
       - Click "Find Matches"
       - Review matched lessons with AI analysis
       - Export results to Excel
    """)


def render_api_reference():
    """Render the API reference documentation."""
    st.title("📖 API Reference")

    st.markdown("## Key Functions and Classes")

    # Settings
    st.header("Configuration")

    st.markdown("### load_settings()")
    st.code("""
from config.settings import load_settings

settings = load_settings()

# Check LLM provider
settings.llm_provider        # "azure" or "openrouter"
settings.is_azure            # True if using Azure OpenAI
settings.is_openrouter       # True if using OpenRouter

# Get model names (works for both providers)
settings.get_chat_model()     # Returns chat model name
settings.get_embedding_model() # Returns embedding model name

# Access provider-specific settings
if settings.is_azure:
    settings.azure_openai.endpoint
elif settings.is_openrouter:
    settings.openrouter.api_key

# Other settings
settings.embeddings.dimensions
settings.retrieval.rrf_k
settings.enrichment.high_confidence_threshold
    """, language="python")

    st.divider()

    # Data Processing
    st.header("Data Processing")

    st.markdown("### load_lessons_excel()")
    st.code("""
from src.data_processing.excel_loader import load_lessons_excel

df, errors = load_lessons_excel(uploaded_file)
# df: pandas DataFrame with lessons
# errors: list of validation errors/warnings
    """, language="python")

    st.markdown("### enrich_lessons()")
    st.code("""
from src.data_processing.enrichment import enrich_lessons

def progress_callback(progress):
    print(f"Progress: {progress.progress_pct}%")

results = enrich_lessons(
    lessons=lessons_list,
    settings=settings,
    progress_callback=progress_callback,
    batch_size=10
)
    """, language="python")

    st.markdown("### chunk_lessons()")
    st.code("""
from src.data_processing.chunker import chunk_lessons

documents = chunk_lessons(
    lessons=lessons_list,
    chunk_size=2000,
    chunk_overlap=400,
    include_enrichment=True
)
# Returns: List[langchain_core.documents.Document]
    """, language="python")

    st.divider()

    # Retrieval
    st.header("Retrieval")

    st.markdown("### create_vector_store()")
    st.code("""
from src.retrieval import create_vector_store

vector_store = create_vector_store(settings)
vector_store.add_documents(documents)

# Search
results = vector_store.search(
    query_text="pump seal failure",
    n_results=50
)
    """, language="python")

    st.markdown("### create_hybrid_search()")
    st.code("""
from src.retrieval import create_hybrid_search, create_bm25_index

bm25_index = create_bm25_index(documents)
hybrid_search = create_hybrid_search(vector_store, bm25_index, settings)

results, tier_results = hybrid_search.multi_tier_search(
    query="pump seal replacement",
    job_equipment_tag="P-101",
    job_equipment_type="centrifugal_pump",
    results_per_tier=10,
    total_results=50
)
    """, language="python")

    st.divider()

    # Generation
    st.header("Generation")

    st.markdown("### create_relevance_analyzer()")
    st.code("""
from src.generation import create_relevance_analyzer

analyzer = create_relevance_analyzer(settings)

analysis = analyzer.analyze_relevance(
    lesson=lesson_dict,
    job=job_dict,
    match_info={"match_type": "equipment_specific"}
)

# Access results
print(analysis.relevance_score)  # 0-100
print(analysis.technical_links)   # List[str]
print(analysis.safety_considerations)
print(analysis.recommended_actions)
print(analysis.match_reasoning)
    """, language="python")

    st.markdown("### create_applicability_checker()")
    st.code("""
from src.generation import create_applicability_checker

checker = create_applicability_checker(settings)

# Check applicability with optional job steps
result = checker.check_applicability(
    lesson=lesson_dict,
    job=job_dict,
    job_steps=["Step 1: Isolate pump", "Step 2: Drain system", ...]
)

# Access results
print(result.decision)               # "yes", "no", "cannot_be_determined"
print(result.decision_display)       # "Applicable", "Not Applicable", ...
print(result.justification)          # Detailed explanation
print(result.mitigation_already_applied)  # True if already mitigated
print(result.risk_not_present)       # True if risk doesn't apply
print(result.key_factors)            # List of decision factors
print(result.confidence)             # 0.0-1.0

# Batch checking
results = checker.check_batch(
    lessons=[lesson1, lesson2, lesson3],
    job=job_dict,
    job_steps=job_steps_list
)
    """, language="python")


def render_sidebar():
    """Render the documentation sidebar."""
    st.sidebar.title("Navigation")

    pages = {
        "🏠 Home": "home",
        "🏗️ Architecture": "architecture",
        "📂 Code Structure": "code_structure",
        "📊 Data Model": "data_model",
        "🚀 Getting Started": "getting_started",
        "📖 API Reference": "api_reference",
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    st.sidebar.divider()

    st.sidebar.markdown("### Quick Links")
    st.sidebar.markdown("""
    - [Main App](http://localhost:8501)
    - [GitHub Repository](#)
    - [Azure OpenAI Docs](https://learn.microsoft.com/azure/ai-services/openai/)
    - [LangChain Docs](https://python.langchain.com/)
    - [ChromaDB Docs](https://docs.trychroma.com/)
    """)

    st.sidebar.divider()

    st.sidebar.info("""
    **Run Main App:**
    ```
    streamlit run app.py
    ```

    **Run Docs:**
    ```
    streamlit run docs_app.py
    ```
    """)

    return pages[selection]


def main():
    """Main documentation app entry point."""
    configure_page()

    page = render_sidebar()

    if page == "home":
        render_home()
    elif page == "architecture":
        render_architecture()
    elif page == "code_structure":
        render_code_structure()
    elif page == "data_model":
        render_data_model()
    elif page == "getting_started":
        render_getting_started()
    elif page == "api_reference":
        render_api_reference()

    # Footer
    st.divider()
    st.caption(
        "Maintenance Lessons Learned RAG System Documentation | "
        "Version 2.1 | "
        "© 2026 Megat"
    )


if __name__ == "__main__":
    main()
