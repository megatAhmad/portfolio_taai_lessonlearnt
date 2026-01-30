# Maintenance Lessons Learned RAG System

An AI-powered Retrieval-Augmented Generation (RAG) system for analyzing oil and gas turnaround maintenance lessons learned and matching them to future maintenance jobs.

## Overview

This proof-of-concept application helps maintenance teams:
- Upload and enrich historical lessons learned with AI-generated metadata
- Match upcoming maintenance jobs to relevant historical lessons
- Generate AI-powered relevance analysis explaining why lessons apply
- Export results for planning and safety reviews

## Features

### Three-Tab Interface
1. **Upload & Enrich**: Load Excel files and run LLM metadata enrichment
2. **Review & Edit**: Review and edit enrichment results, manage flags
3. **Match & Analyze**: Find relevant lessons for jobs with AI analysis

### Multi-Tier Retrieval
- **Equipment-Specific**: Exact equipment tag matches (1.5x boost)
- **Equipment-Type**: Same equipment type matches (1.2x boost)
- **Generic/Universal**: Broadly applicable lessons (1.3-1.4x boost)
- **Semantic**: Pure semantic similarity

### AI Capabilities
- LLM metadata enrichment with confidence scoring
- Hybrid search (dense embeddings + sparse BM25)
- Cross-encoder reranking
- Relevance analysis with actionable recommendations
- **Applicability checking** - AI determines if lessons truly apply to jobs (Yes/No/Cannot Determine)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Azure OpenAI or OpenRouter |
| LLM | GPT-4o-mini (via Azure or OpenRouter) |
| Sparse Search | rank_bm25 |
| Reranker | sentence-transformers cross-encoder |

### Supported LLM Providers

| Provider | Description |
|----------|-------------|
| **Azure OpenAI** | Microsoft's hosted OpenAI models |
| **OpenRouter** | Multi-provider API gateway (OpenAI, Anthropic, etc.) |

## Installation

### Prerequisites
- Python 3.10 or higher
- API key for either Azure OpenAI or OpenRouter

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio_taai_lessonlearnt.git
cd portfolio_taai_lessonlearnt
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your LLM provider credentials
```

Choose one of the following configurations:

**Option A: Azure OpenAI**
```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

**Option B: OpenRouter**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini
```

5. Generate sample data (optional):
```bash
python data/create_sample_data.py
```

## Usage

### Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Workflow

1. **Upload Data**
   - Go to "Upload & Enrich" tab
   - Upload lessons learned Excel file
   - Upload job descriptions Excel file (optional)

2. **Enrich Lessons**
   - Click "Start Enrichment" to run LLM metadata generation
   - Review confidence scores and flags

3. **Build Index**
   - Click "Build Index" to create embeddings and vector store

4. **Review Results**
   - Go to "Review & Edit" tab
   - Filter by confidence, flags, or categories
   - Edit enrichment data as needed

5. **Match Jobs**
   - Go to "Match & Analyze" tab
   - Select a job from the list
   - Enable "Check applicability" option (recommended)
   - Click "Find Matches" to retrieve relevant lessons
   - Review AI applicability decisions:
     - ✅ Applicable - Lesson directly applies
     - ❌ Not Applicable - Mitigation exists or risk doesn't apply
     - ⚠️ Cannot Determine - Insufficient information
   - Filter by applicability decision
   - Export results to Excel (includes justifications)

## Data Formats

### Lessons Learned Excel
| Column | Required | Description |
|--------|----------|-------------|
| lesson_id | Yes | Unique identifier (e.g., "LL-001") |
| title | Yes | Brief title/summary |
| description | Yes | Detailed problem description |
| root_cause | Yes | Root cause analysis |
| corrective_action | Yes | Actions taken to resolve |
| category | Yes | "mechanical", "electrical", "safety", "process" |
| equipment_tag | No | Equipment ID (e.g., "P-101") |
| date | No | When lesson was learned |
| severity | No | "low", "medium", "high", "critical" |

### Job Descriptions Excel
| Column | Required | Description |
|--------|----------|-------------|
| job_id | Yes | Unique identifier |
| job_title | Yes | Brief title |
| job_description | Yes | Detailed work description |
| equipment_tag | No | Equipment identifier |
| job_type | No | "inspection", "repair", "replacement" |
| planned_date | No | Scheduled execution date |

## Configuration

Key settings can be adjusted in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| chunk_size | 2000 | Characters per chunk |
| chunk_overlap | 400 | Overlap between chunks |
| rrf_k | 60 | RRF fusion parameter |
| top_k | 5 | Number of final results |
| high_confidence_threshold | 0.85 | High confidence cutoff |
| medium_confidence_threshold | 0.70 | Medium confidence cutoff |

## Project Structure

```
portfolio_taai_lessonlearnt/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── CLAUDE.md                 # AI assistant guide
├── config/
│   ├── settings.py           # Configuration management
│   ├── prompts.py            # LLM prompt templates
│   └── llm_client.py         # LLM client factory (Azure/OpenRouter)
├── src/
│   ├── data_processing/
│   │   ├── excel_loader.py   # Excel file handling
│   │   ├── preprocessor.py   # Text preprocessing
│   │   ├── chunker.py        # Text chunking
│   │   └── enrichment.py     # LLM enrichment pipeline
│   ├── retrieval/
│   │   ├── embeddings.py     # Azure OpenAI embeddings
│   │   ├── vector_store.py   # ChromaDB operations
│   │   ├── bm25_search.py    # BM25 sparse retrieval
│   │   ├── hybrid_search.py  # Multi-tier hybrid search
│   │   └── reranker.py       # Cross-encoder reranking
│   ├── generation/
│   │   ├── relevance_analyzer.py  # GPT-4o-mini analysis
│   │   └── applicability_checker.py # AI applicability assessment
│   └── ui/
│       ├── components.py     # Reusable UI components
│       ├── utils.py          # UI utilities
│       ├── tab_upload.py     # Upload & Enrich tab
│       ├── tab_review.py     # Review & Edit tab
│       └── tab_matching.py   # Match & Analyze tab
├── data/
│   ├── create_sample_data.py # Sample data generator
│   ├── sample_lessons.xlsx   # Sample lessons (generated)
│   └── sample_jobs.xlsx      # Sample jobs (generated)
└── chroma_db/                # Vector store persistence
```

## Cost Considerations

Target: **<$10/month** Azure OpenAI costs

| Operation | Estimated Cost |
|-----------|---------------|
| Enrichment (100 lessons) | ~$0.10 |
| Embeddings (100 lessons) | ~$0.05 |
| Matching (per job) | ~$0.02 |

Tips to minimize costs:
- Use embedding caching (enabled by default)
- Batch enrichment during off-peak hours
- Review high-confidence lessons only for accuracy

## Troubleshooting

### LLM Connection Errors

**Azure OpenAI:**
- Verify `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` in `.env`
- Check API version compatibility
- Ensure models are deployed in Azure OpenAI Studio

**OpenRouter:**
- Verify `OPENROUTER_API_KEY` in `.env`
- Check model names at https://openrouter.ai/models
- Ensure your account has sufficient credits

### ChromaDB Issues
- Delete `chroma_db/` directory to reset
- Check disk space availability

### Memory Issues
- Reduce batch size for large datasets
- Process files incrementally

## License

This project is for demonstration and educational purposes.

## Author

Megat Ahmad - Oil & Gas Maintenance AI Systems

---

*Built with Azure OpenAI/OpenRouter, LangChain, and Streamlit*
