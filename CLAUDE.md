# CLAUDE.md - AI Assistant Guide for Maintenance RAG Project

## Project Overview

This is a **Proof of Concept (PoC) RAG (Retrieval-Augmented Generation) System** for analyzing oil and gas turnaround maintenance lessons learned records. The system identifies relevant historical lessons for future maintenance jobs to improve safety and efficiency.

| Attribute | Value |
|-----------|-------|
| **Domain** | Oil & Gas Industry (Turnaround Maintenance) |
| **Status** | Specification/Planning Phase |
| **Target User** | Claude Opus (Agentic Coding) |
| **Interface** | Streamlit web application |
| **Deployment** | Local development environment |
| **Budget** | <$10/month Azure OpenAI costs |

### Key Objectives
1. Accept Excel uploads containing lessons learned records and job descriptions
2. Use hybrid retrieval (dense + sparse search) to find relevant lessons for each job
3. Generate AI-powered relevance analysis explaining why lessons apply
4. Present results in an intuitive UI with filtering and export capabilities

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
| LLM | Azure OpenAI GPT-4o-mini | Temperature: 0.3 |
| Sparse Search | rank_bm25 | Latest |
| Reranker | sentence-transformers cross-encoder | ms-marco-MiniLM-L-6-v2 |
| Data Processing | pandas + openpyxl + pandera | Latest stable |
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
├── app.py                        # Main Streamlit application entry point
├── config/
│   ├── __init__.py
│   ├── settings.py               # Configuration management (env vars, params)
│   └── prompts.py                # LLM prompt templates
├── data/
│   ├── sample_lessons.xlsx       # Sample lessons learned (20-50 records)
│   └── sample_jobs.xlsx          # Sample job descriptions (10-20 records)
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── excel_loader.py       # Excel file reading and validation
│   │   ├── preprocessor.py       # Text cleaning, abbreviation expansion
│   │   └── chunker.py            # Text chunking (~500 tokens, 100 overlap)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Azure OpenAI embeddings with caching
│   │   ├── vector_store.py       # ChromaDB operations
│   │   ├── bm25_search.py        # BM25 sparse retrieval
│   │   ├── hybrid_search.py      # RRF fusion algorithm
│   │   └── reranker.py           # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   └── relevance_analyzer.py # GPT-4o-mini structured analysis
│   └── ui/
│       ├── __init__.py
│       ├── components.py         # Reusable Streamlit components
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
| `app.py` | Main entry point, orchestrates UI flow, manages session state |
| `config/settings.py` | All configuration - no hardcoded values elsewhere |
| `config/prompts.py` | LLM prompt templates with variables |
| `src/data_processing/excel_loader.py` | Validates Excel schema using Pandera |
| `src/retrieval/hybrid_search.py` | Core RAG logic - RRF fusion |
| `src/generation/relevance_analyzer.py` | GPT-4o-mini structured prompting |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                        │
│  File Upload | Query Input | Results Display | Filtering    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  PROCESSING LAYER                            │
│  Excel Parser ──► Preprocessor ──► Text Chunker             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │        HYBRID RETRIEVAL ENGINE                          │ │
│  │  Dense Search (Embeddings) + Sparse Search (BM25)      │ │
│  │              ↓                                          │ │
│  │  Reciprocal Rank Fusion (RRF) ──► Cross-Encoder Rerank │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │      AZURE OPENAI GENERATION LAYER                      │ │
│  │  GPT-4o-mini: Relevance Analysis & Explanations        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  PERSISTENCE LAYER                           │
│  ChromaDB (Vector Store) | Session State | Cache            │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval Strategy

| Stage | Method | Output |
|-------|--------|--------|
| 1. Dense Search | Azure OpenAI embeddings → ChromaDB cosine similarity | Top 50 |
| 2. Sparse Search | BM25 keyword matching | Top 50 |
| 3. Fusion | Reciprocal Rank Fusion (RRF) with k=60 | Combined ranking |
| 4. Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) | Top 5 |

**RRF Formula:**
```python
def rrf_score(rank, k=60):
    """Calculate RRF score for combining rankings"""
    return 1 / (k + rank)

# combined_score = rrf_score(dense_rank) + rrf_score(sparse_rank)
```

## Data Formats

### Lessons Learned Excel (Required Columns)
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

### Job Descriptions Excel (Required Columns)
| Column | Type | Description |
|--------|------|-------------|
| `job_id` | str | Unique identifier (e.g., "JOB-001") |
| `job_title` | str | Brief title |
| `job_description` | str | Detailed work description |
| `equipment_tag` | str (optional) | Equipment identifier |
| `job_type` | str (optional) | "inspection", "repair", "replacement" |
| `planned_date` | datetime (optional) | Scheduled execution date |

## Environment Configuration

Create `.env` file (never commit this):
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
DEBUG=false
```

## LLM Configuration

### System Prompt Template
```
You are an expert maintenance engineer analyzing the relevance between
historical lessons learned and future maintenance jobs.

Analyze the following lesson learned and job description. Provide:
1. Relevance Score (0-100): How applicable is this lesson to the job
2. Key Technical Links: Specific technical connections (equipment type, failure mode, procedures)
3. Safety Considerations: Any safety implications from the lesson
4. Recommended Actions: Specific steps to apply this lesson to the job

Be concise and focus on actionable insights.
```

### Relevance Analysis Output Format
```json
{
  "relevance_score": 85,
  "technical_links": [
    "Both involve centrifugal pump mechanical seals",
    "Similar operating pressure (150 psi)",
    "Same failure mode: seal face wear"
  ],
  "safety_considerations": "Ensure proper lockout/tagout procedures. Previous incident involved pressure release injury.",
  "recommended_actions": [
    "Review seal installation procedure from LL-001",
    "Verify correct seal material for fluid service",
    "Include additional technician training"
  ]
}
```

## Development Guidelines

### Code Conventions
- **Type hints** required for all function signatures
- **PEP 8** style compliance
- **Docstrings** for non-trivial functions
- Functions should be **< 50 lines** where possible
- **No hardcoded values** - use `config/settings.py`
- Use `tenacity` for API retry logic (exponential backoff, max 6 attempts)

### Text Chunking Parameters
- Chunk size: ~500 tokens (~2000 characters)
- Overlap: 100 tokens (~400 characters)
- Separator priority: double newlines → single newlines → periods → spaces
- Use LangChain's `RecursiveCharacterTextSplitter`

### Error Handling
- Implement error handling at API boundaries
- Use user-friendly messages in UI (no technical jargon)
- Log errors without exposing API keys or secrets
- Graceful handling of missing/null data fields
- Exponential backoff with jitter for rate limits

### Performance Targets
| Operation | Target |
|-----------|--------|
| File upload & validation | <5 seconds (1000 rows) |
| Embedding generation | <30 seconds (100 lessons) |
| Single job query | <10 seconds |
| Full analysis (5 lessons) | <20 seconds |

### Cost Optimization
- Cache embeddings using content hashes
- Batch API calls (up to 100 texts per embedding call)
- Use tiktoken for token counting before API calls
- Target: **<$10/month** Azure OpenAI costs
- Embedding cost target: <$0.50/month
- LLM cost target: <$5/month

## Implementation Phases

### Phase 1: Core RAG Pipeline (HIGH Priority)
- Excel data loading and validation
- Text preprocessing and chunking
- Azure OpenAI embedding generation
- ChromaDB vector store setup
- Basic dense retrieval (embeddings only)

### Phase 2: Hybrid Search & Reranking (HIGH Priority)
- BM25 sparse retrieval implementation
- RRF fusion logic
- Cross-encoder reranking
- Improved result quality

### Phase 3: LLM Analysis Generation (MEDIUM Priority)
- GPT-4o-mini integration for relevance analysis
- Structured response parsing
- Display AI-generated insights in UI

### Phase 4: UI Polish & Export (MEDIUM Priority)
- Improved Streamlit layout with tabs/pages
- Interactive filtering
- Excel export functionality
- Session state management

### Phase 5: Optimization & Evaluation (LOW Priority)
- Embedding caching
- Performance profiling
- Basic evaluation metrics (optional RAGAS)
- Documentation and README

## UI Specifications

### Results Display
- **Score Badges**: Color-coded (green >80%, orange 50-80%, red <50%)
- **Lesson Cards**: Title, preview (200 chars), metadata, expandable details
- **Export**: Excel file with Summary + Matched Lessons sheets

### Session State
- Persist uploaded data across page interactions
- Cache processed embeddings
- Maintain search results for filtering
- "Reset" button to clear state

## Testing & Validation

### Test Scenarios
1. **Positive Match**: Job clearly related to lesson (e.g., pump seal replacement → pump seal failure)
2. **Negative Match**: Unrelated job should return low scores
3. **Partial Match**: Somewhat related jobs ranked appropriately
4. **Safety-Critical**: Hazardous jobs must surface safety lessons
5. **Equipment-Specific**: Same equipment tag prioritized

### Edge Cases to Handle
- Empty Excel files
- Missing required columns
- Very short descriptions (<10 words)
- Very long descriptions (>5000 words)
- Special characters and encoding issues
- Duplicate lesson IDs

## Security Considerations

- **NEVER** commit `.env` file
- **NEVER** log API keys or sensitive data
- Validate and sanitize all user inputs
- Use environment variables for all secrets
- Sanitize file paths when handling uploads

## Quality Metrics (PoC Targets)

| Metric | Target |
|--------|--------|
| MRR@5 (Mean Reciprocal Rank) | ≥0.6 |
| Recall@10 | ≥0.7 |
| User satisfaction | ≥70% rate as "useful" |
| Safety lessons | No critical lessons missed |

## Out of Scope (PoC)

- Authentication/user management
- Production deployment/scalability
- Real-time data integration
- Multi-language support
- Mobile responsiveness
- Multi-tenancy
- Version control for lessons database

## Troubleshooting

### Common Issues
1. **Azure OpenAI connection errors**: Check `.env` configuration and API version
2. **ChromaDB persistence issues**: Ensure `chroma_db/` directory exists and is writable
3. **Memory issues with large files**: Implement chunked processing
4. **Slow embedding generation**: Use batching (100 texts per call)

### Debug Mode
Set `DEBUG=true` in `.env` to enable verbose logging and timing information.

## Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - combines retrieval with generative AI |
| **Dense Retrieval** | Semantic search using vector embeddings |
| **Sparse Retrieval** | Keyword-based search (BM25) |
| **Hybrid Search** | Combining dense and sparse retrieval |
| **RRF** | Reciprocal Rank Fusion - combines multiple ranked lists |
| **Cross-Encoder** | Neural network that scores query-document pairs |
| **ChromaDB** | Open-source vector database for embeddings |
| **Turnaround Maintenance** | Planned shutdown period for major maintenance |
| **Lessons Learned** | Documented knowledge from past incidents/projects |

## References

- **Full Requirements**: See `maintenance_rag_project_requirements.md` for complete specifications
- **LangChain Docs**: https://python.langchain.com/docs/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/

---

**Version**: 1.1
**Last Updated**: January 2026
**Project Author**: Megat (Oil & Gas Maintenance AI Systems)
