# CLAUDE.md - AI Assistant Guide for Maintenance RAG Project

## Project Overview

This is a **Proof of Concept (PoC) RAG (Retrieval-Augmented Generation) System** for analyzing oil and gas turnaround maintenance lessons learned records. The system identifies relevant historical lessons for future maintenance jobs to improve safety and efficiency.

**Domain:** Oil & Gas Industry (Turnaround Maintenance)
**Status:** Specification/Planning Phase - Implementation not yet started
**Target User:** Claude Opus (Agentic Coding)

## Quick Start Commands

```bash
# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Run the Streamlit application (when app.py exists)
streamlit run app.py

# Run tests (when tests exist)
pytest tests/
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python ≥3.10 | Core development |
| Web Framework | Streamlit ≥1.30.0 | UI layer |
| RAG Framework | LangChain | Orchestrates RAG pipeline |
| Vector Database | ChromaDB | Stores embeddings |
| Embeddings | Azure OpenAI text-embedding-3-small | 1536-dimensional vectors |
| LLM | Azure OpenAI GPT-4o-mini | Relevance analysis generation |
| Sparse Search | rank_bm25 | Keyword-based retrieval |
| Reranker | sentence-transformers cross-encoder | Result reranking |
| Data Processing | pandas + openpyxl + pandera | Excel handling & validation |

## Project Structure (Planned)

```
maintenance_rag_poc/
├── .env                          # Azure OpenAI credentials (GITIGNORED)
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
Streamlit UI Layer
    ↓
Processing Layer (Excel parsing → Preprocessing → Chunking)
    ↓
Hybrid Retrieval Engine (Dense + Sparse → RRF Fusion → Reranking)
    ↓
Azure OpenAI Generation (GPT-4o-mini relevance analysis)
    ↓
Results Display & Export
```

### Retrieval Strategy
1. **Dense Search**: Azure OpenAI embeddings → ChromaDB cosine similarity → Top 50
2. **Sparse Search**: BM25 keyword matching → Top 50
3. **Fusion**: Reciprocal Rank Fusion (RRF) with k=60
4. **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2) → Top 5

## Data Formats

### Lessons Learned Excel (Required Columns)
- `lesson_id` (str): Unique identifier
- `title` (str): Brief title/summary
- `description` (str): Detailed problem description
- `root_cause` (str): Root cause analysis
- `corrective_action` (str): Actions taken
- `category` (str): "mechanical", "electrical", "safety", "process"
- Optional: `equipment_tag`, `date`, `severity`

### Job Descriptions Excel (Required Columns)
- `job_id` (str): Unique identifier
- `job_title` (str): Brief title
- `job_description` (str): Detailed work description
- Optional: `equipment_tag`, `job_type`, `planned_date`

## Environment Configuration

Create `.env` file (never commit this):
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

## Development Guidelines

### Code Conventions
- **Type hints** required for all function signatures
- **PEP 8** style compliance
- **Docstrings** for non-trivial functions
- Functions should be **< 50 lines** where possible
- **No hardcoded values** - use `config/settings.py`
- Use `tenacity` for API retry logic (exponential backoff)

### Error Handling
- Implement error handling at API boundaries
- Use user-friendly messages in UI (no technical jargon)
- Log errors without exposing API keys or secrets
- Graceful handling of missing/null data fields

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

## Implementation Phases

1. **Phase 1 (HIGH)**: Core RAG Pipeline
   - Excel loading, validation, chunking
   - Azure embeddings, ChromaDB setup
   - Basic dense retrieval

2. **Phase 2 (HIGH)**: Hybrid Search & Reranking
   - BM25 sparse retrieval
   - RRF fusion, cross-encoder reranking

3. **Phase 3 (MEDIUM)**: LLM Analysis Generation
   - GPT-4o-mini integration
   - Structured response parsing

4. **Phase 4 (MEDIUM)**: UI Polish & Export
   - Streamlit layout improvements
   - Interactive filtering, Excel export

5. **Phase 5 (LOW)**: Optimization & Evaluation
   - Embedding caching, performance profiling
   - Optional RAGAS metrics

## Common Tasks

### Adding a New Module
1. Create file in appropriate `src/` subdirectory
2. Add `__init__.py` exports
3. Use type hints and docstrings
4. Import configuration from `config/settings.py`

### Modifying Prompts
Edit `config/prompts.py` - all LLM prompts are centralized there.

### Adding Data Validation Rules
Update schema definitions in `src/data_processing/excel_loader.py` using Pandera.

### Testing Retrieval Quality
Use sample data in `data/` directory with known ground truth matches.

## Security Considerations

- **NEVER** commit `.env` file
- **NEVER** log API keys or sensitive data
- Validate and sanitize all user inputs
- Use environment variables for all secrets
- Sanitize file paths when handling uploads

## Quality Metrics (PoC Targets)

- MRR@5 (Mean Reciprocal Rank): ≥0.6
- Recall@10: ≥0.7
- User satisfaction: ≥70% rate results as "useful"
- No critical safety lessons missed in test cases

## Out of Scope (PoC)

- Authentication/user management
- Production deployment/scalability
- Real-time data integration
- Multi-language support
- Mobile responsiveness
- Multi-tenancy

## Troubleshooting

### Common Issues

1. **Azure OpenAI connection errors**: Check `.env` configuration and API version
2. **ChromaDB persistence issues**: Ensure `chroma_db/` directory exists and is writable
3. **Memory issues with large files**: Implement chunked processing
4. **Slow embedding generation**: Use batching (100 texts per call)

### Debug Mode
Set `DEBUG=true` in `.env` to enable verbose logging and timing information.

## References

- **Full Requirements**: See `maintenance_rag_project_requirements.md` for complete specifications
- **LangChain Docs**: https://python.langchain.com/docs/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/

---

**Version**: 1.0
**Last Updated**: January 2026
**Project Author**: Megat (Oil & Gas Maintenance AI Systems)
