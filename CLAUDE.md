# CLAUDE.md - AI Assistant Guide for Maintenance RAG Project

## Project Overview

This is a **Proof of Concept (PoC) RAG (Retrieval-Augmented Generation) System** for analyzing oil and gas turnaround maintenance lessons learned records. The system identifies relevant historical lessons for future maintenance jobs to improve safety and efficiency.

| Attribute | Value |
|-----------|-------|
| **Domain** | Oil & Gas Industry (Turnaround Maintenance) |
| **Status** | Specification/Planning Phase |
| **Target User** | Claude Opus (Agentic Coding) |
| **Interface** | Streamlit web application (3-tab layout) |
| **Deployment** | Local development environment |
| **Budget** | <$10/month Azure OpenAI costs |

### Key Objectives
1. Accept Excel uploads containing lessons learned records and job descriptions
2. **Auto-generate metadata enrichment using LLM** (equipment type, applicability, procedures)
3. Use **multi-tier hybrid retrieval** (equipment-specific → type → generic → semantic)
4. Generate AI-powered relevance analysis explaining why lessons apply
5. Present results in an intuitive 3-tab UI with filtering and export capabilities

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py

# Run tests
pytest tests/

# Run with debug mode
DEBUG=true streamlit run app.py
```

## Technology Stack

| Component | Technology | Version/Spec |
|-----------|-----------|--------------|
| Language | Python | ≥3.10 |
| Web Framework | Streamlit | ≥1.30.0 |
| RAG Framework | LangChain | Latest stable |
| Vector Database | ChromaDB | Latest (with persistence) |
| Embeddings | Azure OpenAI text-embedding-3-small | 1536 dimensions |
| LLM | Azure OpenAI GPT-4o-mini | Temperature: 0.3 (analysis), 0.1 (enrichment) |
| Sparse Search | rank_bm25 | Latest |
| Reranker | sentence-transformers cross-encoder | ms-marco-MiniLM-L-6-v2 |
| Data Processing | pandas + openpyxl + pandera | Latest stable |
| JSON Validation | pydantic | ≥2.5.0 |
| Retry Logic | tenacity | ≥8.2.3 |
| Token Counting | tiktoken | ≥0.5.2 |

## Project Structure

```
maintenance_rag_poc/
├── .env                          # Azure OpenAI credentials (GITIGNORED)
├── .env.example                  # Template for environment variables
├── .gitignore                    # Standard Python gitignore
├── requirements.txt              # Python dependencies
├── README.md                     # Setup and usage instructions
├── CLAUDE.md                     # This file - AI assistant guide
├── maintenance_rag_project_requirements.md  # Full project specification
├── app.py                        # Main Streamlit application (3-tab layout)
├── config/
│   ├── __init__.py
│   ├── settings.py               # Configuration management (env vars, params)
│   └── prompts.py                # LLM prompt templates (relevance + enrichment)
├── data/
│   ├── sample_lessons.xlsx       # Sample lessons learned (basic columns only)
│   ├── sample_lessons_enriched.xlsx  # Sample with enrichment (for demo)
│   └── sample_jobs.xlsx          # Sample job descriptions (10-20 records)
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── excel_loader.py       # Excel file reading and validation
│   │   ├── preprocessor.py       # Text cleaning, abbreviation expansion
│   │   ├── chunker.py            # Text chunking (~500 tokens, 100 overlap)
│   │   └── enrichment.py         # NEW: LLM metadata enrichment pipeline
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Azure OpenAI embeddings with caching
│   │   ├── vector_store.py       # ChromaDB operations with metadata
│   │   ├── bm25_search.py        # BM25 sparse retrieval
│   │   ├── hybrid_search.py      # Multi-tier RRF fusion with boosting
│   │   └── reranker.py           # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   └── relevance_analyzer.py # GPT-4o-mini structured analysis
│   └── ui/
│       ├── __init__.py
│       ├── components.py         # Reusable Streamlit components
│       ├── tab_upload.py         # NEW: Tab 1 - Upload & Enrichment
│       ├── tab_review.py         # NEW: Tab 2 - Enrichment Review
│       ├── tab_matching.py       # NEW: Tab 3 - Job Matching
│       └── utils.py              # UI helper functions
├── chroma_db/                    # ChromaDB persistence (GITIGNORED)
└── tests/
    ├── __init__.py
    └── test_*.py                 # Unit tests
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `maintenance_rag_project_requirements.md` | **Complete project specification** - Read this first for full context |
| `app.py` | Main entry point, 3-tab layout, session state management |
| `config/settings.py` | All configuration - no hardcoded values elsewhere |
| `config/prompts.py` | LLM prompts for relevance analysis AND enrichment |
| `src/data_processing/enrichment.py` | **NEW**: LLM metadata enrichment pipeline |
| `src/retrieval/hybrid_search.py` | **Multi-tier** RRF fusion with score boosting |
| `src/generation/relevance_analyzer.py` | GPT-4o-mini with match type context |
| `src/ui/tab_upload.py` | **NEW**: Upload & enrichment trigger UI |
| `src/ui/tab_review.py` | **NEW**: Enrichment review dashboard UI |
| `src/ui/tab_matching.py` | **NEW**: Job matching & results UI |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 STREAMLIT UI (3-TAB LAYOUT)                  │
│  Tab 1: Upload & Enrich | Tab 2: Review | Tab 3: Match      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              LLM METADATA ENRICHMENT LAYER                   │
│  GPT-4o-mini: Auto-generate 11 enrichment columns           │
│  (specificity_level, equipment_type, applicable_to, etc.)  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  PROCESSING LAYER                            │
│  Excel Parser ──► Preprocessor ──► Text Chunker             │
│                    (with enrichment metadata)                │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │        MULTI-TIER HYBRID RETRIEVAL ENGINE              │ │
│  │  Tier 1: Equipment-Specific (boost 1.5x)               │ │
│  │  Tier 2: Equipment-Type (boost 1.2x)                   │ │
│  │  Tier 3: Generic/Universal (boost 1.3x/1.4x)           │ │
│  │  Tier 4: Semantic (no filter)                          │ │
│  │              ↓                                          │ │
│  │  RRF Fusion + Score Boosting ──► Cross-Encoder Rerank  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │      AZURE OPENAI GENERATION LAYER                      │ │
│  │  GPT-4o-mini: Relevance Analysis + Match Reasoning     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  PERSISTENCE LAYER                           │
│  ChromaDB (with metadata) | Session State | Cache           │
└─────────────────────────────────────────────────────────────┘
```

## Two-Stage Data Model

### Stage 1: Basic Input (User Provides - 9 columns)
| Column | Type | Description |
|--------|------|-------------|
| `lesson_id` | str | Unique identifier (e.g., "LL-001") |
| `title` | str | Brief title/summary |
| `description` | str | Detailed problem description (≥10 chars) |
| `root_cause` | str | Root cause analysis findings |
| `corrective_action` | str | Actions taken to resolve |
| `category` | str | "mechanical", "electrical", "safety", "process" |
| `equipment_tag` | str (optional) | Equipment ID (e.g., "P-101", "HX-205") |
| `date` | datetime (optional) | When lesson was learned |
| `severity` | str (optional) | "low", "medium", "high", "critical" |

### Stage 2: Auto-Generated Enrichment (LLM Creates - 11 columns)
| Column | Type | Description |
|--------|------|-------------|
| `specificity_level` | str | "equipment_id", "equipment_type", or "generic" |
| `equipment_type` | str | e.g., "centrifugal_pump", "heat_exchanger" |
| `equipment_family` | str | "rotating_equipment", "static_equipment", etc. |
| `applicable_to` | str | Comma-separated: "all_pumps,all_seals" |
| `procedure_tags` | str | Comma-separated: "installation,lockout_tagout" |
| `lesson_scope` | str | "specific", "general", or "universal" |
| `safety_categories` | str | Comma-separated: "lockout_tagout,pressure_release" |
| `enrichment_confidence` | float | 0.0-1.0 confidence score |
| `enrichment_timestamp` | datetime | When enrichment was performed |
| `enrichment_reviewed` | bool | Whether user reviewed/approved |

## LLM Metadata Enrichment Pipeline

### Enrichment Flow
```
Raw Excel Upload (9 basic columns)
    ↓
LLM Analysis (GPT-4o-mini, batch processing)
    ↓
Auto-Generated Metadata (11 additional columns)
    ↓
User Review & Approval Interface (Tab 2)
    ↓
Enriched Lessons → Multi-Tier RAG System
```

### Confidence Thresholds
| Level | Score Range | Action |
|-------|-------------|--------|
| High | ≥0.85 | Auto-accept, no review needed |
| Medium | 0.70-0.84 | Yellow flag, suggest review |
| Low | <0.70 | Red flag, require user review |

### Enrichment Prompt (Temperature: 0.1)
```
You are an expert maintenance data analyst specializing in Oil & Gas operations.
Analyze the provided maintenance lesson learned and extract structured metadata.

Return ONLY a valid JSON object with these fields:
{
  "specificity_level": "equipment_id" | "equipment_type" | "generic",
  "equipment_type": "string or null",
  "equipment_family": "rotating_equipment" | "static_equipment" | null,
  "applicable_to": ["list", "of", "scopes"],
  "procedure_tags": ["list", "of", "procedures"],
  "lesson_scope": "specific" | "general" | "universal",
  "safety_categories": ["list", "of", "safety", "categories"],
  "confidence_score": 0.0-1.0
}
```

### Cost Estimates
- ~700 tokens per lesson (500 input + 200 output)
- ~$0.0005 per lesson with GPT-4o-mini
- 1000 lessons: ~$0.50 one-time enrichment cost

## Multi-Tier Retrieval Strategy

### Tier Flow
```
Job Query Input
    ↓
Extract Job Metadata (equipment_id, type, procedures)
    ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 1: Equipment-Specific Matches                     │
│ Filter: equipment_id = job.equipment_id                │
│ Retrieve: Top-10 | Boost: ×1.5                         │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 2: Equipment-Type Matches                         │
│ Filter: equipment_type = job.equipment_type            │
│ Retrieve: Top-20 | Boost: ×1.2                         │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 3: Generic/Universal Lessons                      │
│ Filter: lesson_scope = "universal"                     │
│ Retrieve: Top-20 | Boost: ×1.3 (universal), ×1.4 (critical) │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 4: Semantic Similarity (No Filter)                │
│ Pure hybrid search on full text                        │
│ Retrieve: Top-30 | No boost                            │
└────────────────────────┬────────────────────────────────┘
                         ↓
         Combine All Tiers → Deduplicate → RRF → Rerank → Top-N
```

### Score Boosting Logic
```python
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

    # Procedure overlap boost (+10% per match)
    procedure_overlap = len(set(job_procedures) & set(lesson_procedures))
    if procedure_overlap > 0:
        score *= (1 + 0.1 * procedure_overlap)

    return score
```

## Three-Tab UI Layout

### Tab 1: Data Upload & Enrichment
- File uploader for lessons Excel (9 basic columns)
- Validation feedback and preview
- "Start Enrichment" button with batch size selector
- Progress indicator: "Processing 45/100 lessons (45%)"
- Enrichment summary with confidence breakdown
- Navigate to Tab 2 for review

### Tab 2: Enrichment Review Dashboard
- Side-by-side comparison: original lesson vs. enriched metadata
- Color-coded confidence indicators (🟢🟡🔴)
- Inline editing of enrichment tags
- Batch approval for high-confidence items
- Review progress tracker
- Filter by confidence level, category, flags

### Tab 3: Job Analysis & Matching
- Upload job descriptions Excel
- Select job to analyze
- Search configuration with enrichment-based filters
- Multi-tier retrieval with match type badges:
  - 🎯 Equipment-Specific (blue)
  - 🔧 Equipment-Type (green)
  - ⚙️ Generic Process (orange)
  - 💡 Semantic (purple)
- AI relevance analysis with match reasoning
- Export results to Excel

## Environment Configuration

Create `.env` file (never commit this):
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_ENRICHMENT_DEPLOYMENT=gpt-4o-mini
DEBUG=false
```

## Development Guidelines

### Code Conventions
- **Type hints** required for all function signatures
- **PEP 8** style compliance
- **Docstrings** for non-trivial functions (especially enrichment logic)
- Functions should be **< 50 lines** where possible
- **No hardcoded values** - use `config/settings.py`
- Use `tenacity` for API retry logic (exponential backoff, max 6 attempts)
- Use `pydantic` for structured enrichment output validation

### Error Handling
- Implement error handling at API boundaries (especially enrichment batches)
- Use user-friendly messages in UI (no technical jargon)
- Log errors without exposing API keys or secrets
- Graceful handling of missing/null data fields
- Allow pause/resume for long-running enrichment batches

### Performance Targets
| Operation | Target |
|-----------|--------|
| File upload & validation | <5 seconds (1000 rows) |
| Enrichment processing | 2-5 seconds per lesson |
| Batch enrichment (100 lessons) | 3-8 minutes |
| Embedding generation | <30 seconds (100 lessons) |
| Single job query (multi-tier) | <15 seconds |
| Full analysis (5 lessons) | <20 seconds |

### Cost Optimization
- Cache embeddings using content hashes
- Cache enrichment results (don't re-enrich same lessons)
- Batch API calls (10 lessons per enrichment batch)
- Use tiktoken for token counting before API calls
- Target: **<$10/month** Azure OpenAI costs total
- Enrichment: ~$0.50 per 1000 lessons (one-time)
- Ongoing RAG: <$5/month for typical PoC usage

## Implementation Phases

### Phase 1: Data Upload & Enrichment Pipeline (HIGH)
- Excel loading and validation (basic 9 columns)
- LLM metadata enrichment pipeline
- Batch processing with progress tracking
- Confidence scoring and flagging
- Tab 1 UI: Upload and trigger enrichment

### Phase 2: Enrichment Review UI (HIGH)
- Tab 2: Enrichment review dashboard
- Side-by-side comparison layout
- Inline editing of enrichment tags
- Batch approval operations
- User correction tracking

### Phase 3: Core RAG with Multi-Tier Retrieval (HIGH)
- Text chunking with enrichment metadata
- ChromaDB with metadata filtering
- Multi-tier hybrid retrieval (4 tiers)
- Score boosting logic
- Cross-encoder reranking

### Phase 4: Job Matching & Analysis (MEDIUM)
- Tab 3: Job upload and matching
- GPT-4o-mini relevance analysis with match reasoning
- Results display with match type badges
- Excel export with enrichment metadata

### Phase 5: UI Polish & Optimization (MEDIUM)
- Improved layouts and visual design
- Keyboard shortcuts for review tab
- Session state optimization
- Embedding caching

### Phase 6: Evaluation & Documentation (LOW)
- Enrichment accuracy validation
- Multi-tier retrieval effectiveness testing
- Documentation and README

## Quality Metrics (PoC Targets)

| Metric | Target |
|--------|--------|
| Enrichment accuracy | ≥70% accepted without major edits |
| High confidence precision | ≥85% of high-confidence enrichments correct |
| Safety detection recall | ≥90% of safety-critical lessons flagged |
| MRR@5 (Mean Reciprocal Rank) | ≥0.6 |
| Recall@10 | ≥0.7 |
| Generic lesson inclusion | ≥30% of results when applicable |
| User satisfaction | ≥70% rate as "useful" |
| Safety lessons | No critical lessons missed |

## Security Considerations

- **NEVER** commit `.env` file
- **NEVER** log API keys or sensitive data
- Validate and sanitize all user inputs
- Use environment variables for all secrets
- Sanitize file paths when handling uploads
- User corrections should not expose sensitive data

## Troubleshooting

### Common Issues
1. **Azure OpenAI connection errors**: Check `.env` configuration and API version
2. **ChromaDB persistence issues**: Ensure `chroma_db/` directory exists and is writable
3. **Enrichment timeout**: Reduce batch size, implement pause/resume
4. **Low enrichment accuracy**: Iterate on prompt, check sample outputs
5. **Generic lessons not surfacing**: Verify Tier 3 filtering and boost values

### Debug Mode
Set `DEBUG=true` in `.env` to enable verbose logging including:
- Timing information for each processing stage
- Confidence scores and flags for enrichment
- Match type and tier information for retrieval

## Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - combines retrieval with generative AI |
| **Dense Retrieval** | Semantic search using vector embeddings |
| **Sparse Retrieval** | Keyword-based search (BM25) |
| **Hybrid Search** | Combining dense and sparse retrieval |
| **RRF** | Reciprocal Rank Fusion - combines multiple ranked lists |
| **Cross-Encoder** | Neural network that scores query-document pairs |
| **Metadata Enrichment** | LLM-powered extraction of structured tags from lesson text |
| **Specificity Level** | Classification: equipment_id, equipment_type, or generic |
| **Lesson Scope** | Breadth of applicability: specific, general, or universal |
| **Multi-Tier Retrieval** | Search across 4 tiers with different filters and boosts |
| **Score Boosting** | Multiplicative adjustment based on match type/safety |
| **Match Type** | Why lesson matched: equipment_specific, type, generic, semantic |
| **Confidence Score** | LLM's self-assessed enrichment accuracy (0.0-1.0) |

## References

- **Full Requirements**: See `maintenance_rag_project_requirements.md` for complete specifications
- **LangChain Docs**: https://python.langchain.com/docs/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/

---

**Version**: 2.0
**Last Updated**: January 2026
**Project Author**: Megat (Oil & Gas Maintenance AI Systems)

**Major Changes from v1.0**:
- Added LLM-powered metadata enrichment pipeline
- Introduced 3-tab Streamlit UI (Upload → Review → Match)
- Implemented multi-tier hybrid retrieval with score boosting
- Enhanced to handle equipment-specific, equipment-type, and generic lessons
- Expanded data model from 9 to 20 columns (11 auto-generated)
