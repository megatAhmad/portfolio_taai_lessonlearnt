# Project Requirements: Maintenance Lessons Learned RAG System
## Oil and Gas Turnaround Maintenance - Proof of Concept

---

## 1. PROJECT OVERVIEW

### 1.1 Purpose
Build a Retrieval-Augmented Generation (RAG) system that analyzes historical turnaround maintenance lessons learned records and identifies which lessons are relevant to apply to future maintenance jobs. The system will help maintenance teams leverage institutional knowledge to improve safety, efficiency, and avoid repeating past mistakes.

### 1.2 Scope
- **Type**: Proof of Concept (PoC) standalone application
- **Interface**: Streamlit web application
- **Data Format**: Excel files (.xlsx, .xls)
- **LLM Backend**: Azure OpenAI services exclusively
- **Deployment**: Local development environment
- **Scale**: No production scaling requirements for PoC

### 1.3 Key Objectives
1. Accept Excel uploads containing lessons learned records and job descriptions
2. Use hybrid retrieval (dense + sparse search) to find relevant lessons for each job
3. Generate AI-powered relevance analysis explaining why lessons apply
4. Present results in an intuitive UI with filtering and export capabilities
5. Maintain cost-effectiveness suitable for PoC (<$10/month Azure OpenAI usage)

---

## 2. TECHNICAL ARCHITECTURE

### 2.1 Technology Stack

| Component | Technology | Version/Specification |
|-----------|-----------|----------------------|
| **Programming Language** | Python | ≥3.10 |
| **RAG Framework** | LangChain | Latest stable |
| **Vector Database** | ChromaDB | Latest (with persistence) |
| **Embeddings** | Azure OpenAI text-embedding-3-small | 1536 dimensions |
| **LLM** | Azure OpenAI GPT-4o-mini | Latest available |
| **Sparse Retrieval** | rank_bm25 | Latest |
| **Reranker** | sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2) | Latest |
| **UI Framework** | Streamlit | ≥1.30.0 |
| **Excel Handling** | pandas + openpyxl | Latest stable |
| **Data Validation** | Pandera | Latest |
| **Environment Management** | python-dotenv | Latest |
| **Evaluation** | RAGAS or DeepEval | Latest (optional for PoC) |

### 2.2 Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                        │
│  File Upload | Query Input | Results Display | Filtering    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  PROCESSING LAYER                            │
│                                                              │
│  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ Excel Parser │───▶│ Preprocessor│───▶│ Text Chunker  │  │
│  └──────────────┘    └─────────────┘    └───────┬───────┘  │
│                                                   │          │
│  ┌──────────────────────────────────────────────▼───────┐  │
│  │        HYBRID RETRIEVAL ENGINE                       │  │
│  │  ┌──────────────┐         ┌──────────────┐          │  │
│  │  │ Dense Search │         │ Sparse Search│          │  │
│  │  │ (Embeddings) │         │   (BM25)     │          │  │
│  │  └──────┬───────┘         └──────┬───────┘          │  │
│  │         └───────────┬─────────────┘                  │  │
│  │                     ▼                                 │  │
│  │          Reciprocal Rank Fusion (RRF)               │  │
│  │                     │                                 │  │
│  │                     ▼                                 │  │
│  │          Cross-Encoder Reranking                     │  │
│  └─────────────────────┬─────────────────────────────────┘  │
│                        │                                    │
│  ┌─────────────────────▼─────────────────────────────────┐  │
│  │      AZURE OPENAI GENERATION LAYER                    │  │
│  │  GPT-4o-mini: Relevance Analysis & Explanations      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  PERSISTENCE LAYER                           │
│  ChromaDB (Vector Store) | Session State | Cache            │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Retrieval Strategy
- **Hybrid Search**: Combine dense embeddings (Azure OpenAI) + sparse keyword search (BM25)
- **Fusion Method**: Reciprocal Rank Fusion (RRF) with k=60
- **Two-Stage Retrieval**: 
  1. Retrieve top-50 candidates via hybrid search
  2. Rerank with cross-encoder to get top-5 most relevant
- **Chunking**: ~500 tokens per chunk, 100-token overlap, preserve logical structure

---

## 3. DATA REQUIREMENTS

### 3.1 Input Data Formats

#### 3.1.1 Lessons Learned Excel File
**Required Columns:**
- `lesson_id` (str): Unique identifier for the lesson
- `title` (str): Brief title/summary of the lesson
- `description` (str): Detailed description of the problem and context
- `root_cause` (str): Root cause analysis findings
- `corrective_action` (str): Actions taken to resolve the issue
- `category` (str): Type of lesson (e.g., "mechanical", "electrical", "safety", "process")
- `equipment_tag` (str, optional): Equipment identifier (e.g., "P-101", "HX-205")
- `date` (datetime, optional): When the lesson was learned
- `severity` (str, optional): "low", "medium", "high", "critical"

**Example Structure:**
```
| lesson_id | title                          | description                    | root_cause              | corrective_action         | category    | equipment_tag | severity |
|-----------|--------------------------------|--------------------------------|-------------------------|---------------------------|-------------|---------------|----------|
| LL-001    | Pump seal failure during TAR   | Centrifugal pump seal failed...| Incorrect installation  | Retrained technicians...  | mechanical  | P-101         | high     |
| LL-002    | Heat exchanger tube leak       | Discovered leak during...      | Corrosion from acid...  | Replaced with resistant...| process     | HX-205        | medium   |
```

#### 3.1.2 Job Descriptions Excel File
**Required Columns:**
- `job_id` (str): Unique identifier for the job
- `job_title` (str): Brief title of the maintenance job
- `job_description` (str): Detailed description of the work to be performed
- `equipment_tag` (str, optional): Equipment identifier
- `job_type` (str, optional): Type of job (e.g., "inspection", "repair", "replacement")
- `planned_date` (datetime, optional): Scheduled execution date

**Example Structure:**
```
| job_id  | job_title                    | job_description                     | equipment_tag | job_type   |
|---------|------------------------------|-------------------------------------|---------------|------------|
| JOB-001 | Replace pump mechanical seal | Remove and replace mechanical seal..| P-102         | repair     |
| JOB-002 | Inspect heat exchanger tubes | Perform eddy current testing...     | HX-210        | inspection |
```

### 3.2 Data Validation Requirements
- Validate required columns exist before processing
- Check for minimum text length in description fields (≥10 characters)
- Verify unique IDs (no duplicates)
- Handle missing/null values gracefully
- Display clear error messages for invalid data
- Support both .xlsx and .xls formats

### 3.3 Text Preprocessing
- Remove excessive whitespace
- Expand common maintenance abbreviations (configurable mapping)
- Standardize technical patterns (e.g., "100 psi", "3600 rpm")
- Preserve technical identifiers (equipment tags, part numbers)
- Handle special characters and encoding issues

---

## 4. AZURE OPENAI INTEGRATION

### 4.1 Environment Configuration
**File**: `.env` (must be in .gitignore)

**Required Variables:**
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

### 4.2 Embedding Configuration
- **Model**: text-embedding-3-small
- **Dimensions**: 1536 (full resolution)
- **Batch Size**: Process up to 100 texts per API call for efficiency
- **Caching**: Implement content-based hashing to cache embeddings and avoid recomputation
- **Cost Target**: <$0.50/month for PoC embedding usage

### 4.3 LLM Configuration
- **Model**: GPT-4o-mini
- **Temperature**: 0.3 (for consistent, factual analysis)
- **Max Tokens**: 500 per response
- **System Prompt Template**:
```
You are an expert maintenance engineer analyzing the relevance between historical lessons learned and future maintenance jobs.

Analyze the following lesson learned and job description. Provide:
1. Relevance Score (0-100): How applicable is this lesson to the job
2. Key Technical Links: Specific technical connections (equipment type, failure mode, procedures)
3. Safety Considerations: Any safety implications from the lesson
4. Recommended Actions: Specific steps to apply this lesson to the job

Be concise and focus on actionable insights.
```

### 4.4 Error Handling & Rate Limiting
- Implement exponential backoff with jitter (tenacity library)
- Retry on RateLimitError and APIError (max 6 attempts)
- Display user-friendly error messages in UI
- Log errors for debugging without exposing API keys

### 4.5 Cost Optimization
- Cache embeddings using content hashes
- Batch API calls when possible
- Use tiktoken for accurate token counting before API calls
- Target: <$5/month total Azure OpenAI costs for PoC

---

## 5. STREAMLIT USER INTERFACE

### 5.1 Application Layout

#### Page 1: Data Upload & Preparation
**Components:**
1. **Header Section**
   - Application title: "Maintenance Lessons Learned RAG System"
   - Subtitle: "Match Historical Lessons to Future Jobs"
   - Brief description of functionality

2. **Lessons Learned Upload**
   - File uploader (accepts .xlsx, .xls)
   - Validation feedback (success/error messages)
   - Preview first 5 rows in expandable table
   - Display total count of lessons loaded

3. **Job Descriptions Upload**
   - File uploader (accepts .xlsx, .xls)
   - Validation feedback
   - Preview first 5 rows
   - Display total count of jobs loaded

4. **Processing Controls**
   - "Process Data" button (only enabled when both files uploaded)
   - Progress indicator showing: parsing → validating → chunking → embedding
   - Status messages for each step

#### Page 2: Query & Analysis
**Components:**
1. **Job Selection**
   - Dropdown or searchable select box to choose a job
   - Display selected job details (title, description, equipment)

2. **Search Configuration**
   - Number of results slider (1-20, default: 5)
   - Minimum relevance score slider (0-100%, default: 50%)
   - Category filter (multiselect)

3. **Search Button**
   - "Find Relevant Lessons" button
   - Processing status indicator

#### Page 3: Results Display
**Components:**
1. **Results Summary**
   - Total lessons found
   - Average relevance score
   - Filtering statistics

2. **Lesson Cards** (for each matched lesson)
   - Card layout with border
   - Header: Lesson title + relevance score badge (color-coded: green >80%, orange 50-80%, red <50%)
   - Preview: First 200 characters of description
   - Metadata: Category, equipment tag, severity
   - Expandable section "View Full Details & AI Analysis" containing:
     - Complete lesson text (description, root cause, corrective action)
     - AI-generated relevance analysis
     - Recommended actions

3. **Export Options**
   - "Export Results to Excel" button
   - Download matched lessons with relevance scores and AI analysis

4. **Interactive Filtering**
   - Search within results (text input)
   - Re-filter by score/category without re-running RAG

### 5.2 Session State Management
- Persist uploaded data across page interactions
- Cache processed embeddings to avoid re-computation
- Maintain search results for filtering operations
- Clear state with "Reset" button

### 5.3 UI/UX Requirements
- Responsive layout (works on desktop browsers)
- Clear visual hierarchy (headers, spacing, containers)
- Loading spinners for long operations
- Informative error messages (no technical jargon)
- Color-coded relevance indicators
- Tooltips for help text
- Accessibility: proper contrast, readable fonts

---

## 6. CORE FUNCTIONALITY SPECIFICATIONS

### 6.1 Text Chunking
**Requirements:**
- Chunk size: ~500 tokens (approximately 2000 characters)
- Overlap: 100 tokens (~400 characters)
- Separator priority: double newlines → single newlines → periods → spaces
- Preserve logical structure: keep root cause analysis with corrective actions
- Add metadata to chunks: source lesson_id, category, equipment_tag

**Implementation Approach:**
- Use LangChain's RecursiveCharacterTextSplitter
- Configure with maintenance-appropriate separators
- Maintain chunk-to-source mapping for result attribution

### 6.2 Hybrid Retrieval Implementation

#### 6.2.1 Dense Retrieval (Embeddings)
- Generate embeddings for all lesson chunks using Azure OpenAI
- Store in ChromaDB with persistent storage
- Cosine similarity for vector search
- Return top-50 candidates

#### 6.2.2 Sparse Retrieval (BM25)
- Use rank_bm25 Python package
- Index lesson text (description + root_cause + corrective_action)
- Return top-50 candidates

#### 6.2.3 Reciprocal Rank Fusion
```python
def rrf_score(rank, k=60):
    """Calculate RRF score for combining rankings"""
    return 1 / (k + rank)

# For each lesson:
# combined_score = rrf_score(dense_rank) + rrf_score(sparse_rank)
```

#### 6.2.4 Reranking
- Load cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Rerank top-50 fusion results
- Return top-5 or top-N (user-configurable)

### 6.3 Relevance Analysis Generation
**For each matched lesson:**
1. Combine lesson text (description + root_cause + corrective_action)
2. Combine job text (job_description)
3. Send to GPT-4o-mini with structured prompt
4. Parse response into structured format:
   - relevance_score (0-100)
   - technical_links (list of specific connections)
   - safety_considerations (text)
   - recommended_actions (list)

**Output Format:**
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

### 6.4 Results Export
**Excel Export Requirements:**
- Filename: `rag_results_[job_id]_[timestamp].xlsx`
- Sheets:
  1. "Summary": Job details, search parameters, results count
  2. "Matched Lessons": All matched lessons with scores and AI analysis
- Columns: lesson_id, title, category, equipment_tag, relevance_score, ai_analysis, description, root_cause, corrective_action

---

## 7. PERFORMANCE & QUALITY REQUIREMENTS

### 7.1 Performance Targets (PoC)
- File upload & validation: <5 seconds for files up to 1000 rows
- Embedding generation: <30 seconds for 100 lessons
- Single job query (hybrid retrieval + reranking): <10 seconds
- Full analysis generation (5 lessons): <20 seconds
- UI responsiveness: No frozen interface during processing

### 7.2 Quality Metrics
**Target Thresholds (evaluated manually for PoC):**
- Mean Reciprocal Rank @ 5 (MRR@5): ≥0.6
- Recall @ 10: ≥0.7
- User satisfaction: ≥70% of domain experts rate results as "useful"
- No critical safety lessons missed in test cases

### 7.3 Data Quality Handling
- Gracefully handle missing/null fields (exclude from search, don't crash)
- Support mixed text encodings (UTF-8 primary, fallback handling)
- Validate date formats explicitly (don't auto-infer)
- Handle merged cells in Excel (forward fill for categorical data)

---

## 8. PROJECT STRUCTURE

### 8.1 Directory Layout
```
maintenance_rag_poc/
├── .env                          # Azure OpenAI credentials (GITIGNORED)
├── .gitignore                    # Ignore .env, __pycache__, chroma_db/
├── requirements.txt              # Python dependencies
├── README.md                     # Setup and usage instructions
├── app.py                        # Main Streamlit application
├── config/
│   ├── __init__.py
│   ├── settings.py               # Configuration management
│   └── prompts.py                # LLM prompt templates
├── data/
│   ├── sample_lessons.xlsx       # Sample lessons learned data
│   └── sample_jobs.xlsx          # Sample job descriptions data
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── excel_loader.py       # Excel file reading and validation
│   │   ├── preprocessor.py       # Text preprocessing
│   │   └── chunker.py            # Text chunking logic
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Azure OpenAI embeddings
│   │   ├── vector_store.py       # ChromaDB operations
│   │   ├── bm25_search.py        # Sparse retrieval
│   │   ├── hybrid_search.py      # RRF fusion
│   │   └── reranker.py           # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   └── relevance_analyzer.py # GPT-4o-mini analysis generation
│   └── ui/
│       ├── __init__.py
│       ├── components.py         # Reusable Streamlit components
│       └── utils.py              # UI helper functions
├── chroma_db/                    # ChromaDB persistence (GITIGNORED)
└── tests/                        # Unit tests (optional for PoC)
    ├── __init__.py
    └── test_*.py
```

### 8.2 Key Files & Responsibilities

#### `app.py`
- Main Streamlit application entry point
- Orchestrate UI flow across pages
- Manage session state
- Call processing functions from src/ modules

#### `src/data_processing/excel_loader.py`
- Read Excel files with pandas + openpyxl
- Validate schema using Pandera
- Return validated DataFrames
- Handle file format errors

#### `src/data_processing/preprocessor.py`
- Text cleaning and normalization
- Abbreviation expansion
- Technical pattern standardization
- Maintain preprocessing glossary

#### `src/data_processing/chunker.py`
- Implement recursive text chunking
- Preserve context with overlap
- Add metadata to chunks
- Return list of Document objects

#### `src/retrieval/embeddings.py`
- Azure OpenAI embedding generation
- Batch processing
- Content-based caching
- Error handling with retries

#### `src/retrieval/vector_store.py`
- ChromaDB initialization and persistence
- Add/update embeddings
- Query vector store
- Metadata filtering

#### `src/retrieval/bm25_search.py`
- BM25 index creation
- Sparse keyword search
- Return ranked results

#### `src/retrieval/hybrid_search.py`
- Combine dense and sparse results
- Implement RRF fusion
- Return fused rankings

#### `src/retrieval/reranker.py`
- Load cross-encoder model
- Rerank candidate results
- Return top-N with scores

#### `src/generation/relevance_analyzer.py`
- Azure OpenAI GPT-4o-mini calls
- Structured prompt formatting
- Parse and structure LLM responses
- Handle generation errors

#### `config/settings.py`
- Load environment variables
- Azure OpenAI configuration
- Model parameters (temperature, max_tokens)
- Chunking parameters
- Retrieval thresholds

#### `config/prompts.py`
- System prompts for relevance analysis
- Prompt templates with variables
- Few-shot examples (if needed)

---

## 9. DEPENDENCIES

### 9.1 Python Packages (requirements.txt)
```
# Core Framework
streamlit>=1.30.0
python-dotenv>=1.0.0

# Azure OpenAI
openai>=1.10.0
tiktoken>=0.5.2

# RAG & LangChain
langchain>=0.1.0
langchain-openai>=0.0.5
chromadb>=0.4.22

# Retrieval
rank-bm25>=0.2.2
sentence-transformers>=2.3.0

# Data Processing
pandas>=2.1.0
openpyxl>=3.1.0
xlrd>=2.0.1
pandera>=0.17.0

# Utilities
tenacity>=8.2.3
numpy>=1.24.0

# Evaluation (Optional)
ragas>=0.1.0  # or deepeval>=0.20.0
```

### 9.2 External Services
- **Azure OpenAI**: Active subscription with deployed models
  - text-embedding-3-small deployment
  - gpt-4o-mini deployment
- **Python**: Version 3.10 or higher
- **pip**: For package management

---

## 10. IMPLEMENTATION PRIORITIES

### 10.1 Phase 1: Core RAG Pipeline (Priority: HIGH)
**Deliverables:**
1. Excel data loading and validation
2. Text preprocessing and chunking
3. Azure OpenAI embedding generation
4. ChromaDB vector store setup
5. Basic dense retrieval (embeddings only)
6. Simple relevance scoring

**Success Criteria:**
- Can load Excel files and validate structure
- Generate embeddings for all lessons
- Retrieve top-5 similar lessons for a job query
- Display results in basic Streamlit UI

### 10.2 Phase 2: Hybrid Search & Reranking (Priority: HIGH)
**Deliverables:**
1. BM25 sparse retrieval implementation
2. RRF fusion logic
3. Cross-encoder reranking
4. Improved result quality

**Success Criteria:**
- Hybrid search returns more relevant results than dense-only
- Reranking improves precision
- Processing time remains acceptable (<10 sec per query)

### 10.3 Phase 3: LLM Analysis Generation (Priority: MEDIUM)
**Deliverables:**
1. GPT-4o-mini integration for relevance analysis
2. Structured response parsing
3. Display AI-generated insights in UI

**Success Criteria:**
- Generate meaningful relevance explanations
- Parse structured outputs correctly
- Handle API errors gracefully

### 10.4 Phase 4: UI Polish & Export (Priority: MEDIUM)
**Deliverables:**
1. Improved Streamlit layout with tabs/pages
2. Interactive filtering
3. Excel export functionality
4. Session state management

**Success Criteria:**
- Intuitive user experience
- Results can be filtered without re-running search
- Export includes all relevant data

### 10.5 Phase 5: Optimization & Evaluation (Priority: LOW for PoC)
**Deliverables:**
1. Embedding caching
2. Performance profiling
3. Basic evaluation metrics (optional)
4. Documentation and README

**Success Criteria:**
- No redundant API calls
- Clear setup instructions
- Optional: RAGAS metrics calculated

---

## 11. TESTING & VALIDATION

### 11.1 Sample Data Requirements
**Provide Sample Excel Files:**
- `sample_lessons.xlsx`: 20-50 example lessons learned
  - Cover multiple categories (mechanical, electrical, safety, process)
  - Include various severity levels
  - Mix of equipment types
  
- `sample_jobs.xlsx`: 10-20 example job descriptions
  - Ensure some jobs should match specific lessons (ground truth)
  - Include edge cases (very generic jobs, very specific jobs)

### 11.2 Manual Validation Tests
**Test Scenarios:**
1. **Positive Match Test**: Job clearly related to a lesson (e.g., pump seal replacement → pump seal failure lesson)
2. **Negative Match Test**: Job unrelated to any lesson (should return low scores)
3. **Partial Match Test**: Job somewhat related (should rank appropriately)
4. **Safety-Critical Test**: Jobs involving safety hazards must surface safety lessons
5. **Equipment-Specific Test**: Jobs on specific equipment tags should prioritize lessons on same equipment

### 11.3 Edge Cases to Handle
- Empty Excel files
- Missing required columns
- Very short descriptions (<10 words)
- Very long descriptions (>5000 words)
- Special characters and encoding issues
- Duplicate lesson IDs
- Jobs with no relevant lessons

---

## 12. SUCCESS CRITERIA

### 12.1 Functional Requirements (Must Have)
✅ Accept Excel uploads for lessons and jobs
✅ Generate embeddings using Azure OpenAI
✅ Perform hybrid retrieval (dense + sparse)
✅ Rerank results with cross-encoder
✅ Generate AI relevance analysis with GPT-4o-mini
✅ Display results in Streamlit UI with scores
✅ Export results to Excel
✅ Handle errors gracefully without crashes
✅ Secure API key management via .env

### 12.2 Quality Requirements (Should Have)
✅ Relevance scores align with domain expert judgment (70%+ accuracy on test cases)
✅ Processing time <30 seconds for typical queries
✅ UI is intuitive and requires minimal training
✅ Cost remains under $10/month for PoC usage
✅ No critical safety lessons missed in validation tests

### 12.3 Documentation Requirements (Should Have)
✅ README with setup instructions
✅ Sample data files included
✅ Code comments for complex logic
✅ Environment variable template (.env.example)

### 12.4 PoC Exit Criteria
**The PoC is considered successful when:**
1. System successfully processes sample data end-to-end
2. Domain experts rate 70%+ of results as "useful" in blind test
3. Hybrid search demonstrably outperforms dense-only search
4. UI enables users to find relevant lessons in <2 minutes
5. No critical bugs or crashes during demo
6. Cost projections confirm sustainability for production scale

---

## 13. CONSTRAINTS & ASSUMPTIONS

### 13.1 Constraints
- **Azure OpenAI Only**: No alternative LLM providers
- **Local Deployment**: No cloud hosting for PoC
- **Excel Files Only**: No database integration
- **English Text**: No multi-language support required
- **Single User**: No concurrent user support needed
- **PoC Budget**: Keep Azure costs under $10/month

### 13.2 Assumptions
- Users have basic Excel skills
- Data quality is reasonably good (minimal garbage entries)
- Domain experts available for validation
- Azure OpenAI services are accessible and reliable
- Lessons learned records follow consistent structure
- Python 3.10+ environment available

### 13.3 Out of Scope (For PoC)
❌ Authentication and user management
❌ Production deployment and scalability
❌ Real-time data integration
❌ Advanced analytics and reporting
❌ Mobile responsiveness
❌ Multi-tenancy
❌ Version control for lessons database
❌ Automated evaluation metrics (RAGAS/DeepEval can be optional)

---

## 14. DELIVERABLES CHECKLIST

When implementation is complete, the following should be delivered:

**Code:**
- [ ] Complete Python codebase following project structure
- [ ] requirements.txt with all dependencies
- [ ] .env.example template (without real credentials)
- [ ] .gitignore (excludes .env, chroma_db/, __pycache__)

**Documentation:**
- [ ] README.md with:
  - Project description
  - Setup instructions (environment, dependencies, Azure OpenAI)
  - How to run the application
  - Sample data location
  - Expected usage workflow
- [ ] Code comments for complex functions
- [ ] Docstrings for key modules

**Sample Data:**
- [ ] sample_lessons.xlsx (20-50 records)
- [ ] sample_jobs.xlsx (10-20 records)

**Application:**
- [ ] Working Streamlit app (app.py)
- [ ] All modules in src/ functional
- [ ] ChromaDB persistence configured

**Validation:**
- [ ] Successfully runs on sample data
- [ ] Produces reasonable results (manual check)
- [ ] No crashes or unhandled errors
- [ ] Export functionality works

---

## 15. NOTES FOR IMPLEMENTATION

### 15.1 Development Best Practices
- Use type hints for function signatures
- Implement error handling at API boundaries
- Log important events (file uploads, API calls, errors)
- Keep functions focused and modular
- Avoid hardcoding values (use config/settings.py)

### 15.2 Code Quality Guidelines
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions under 50 lines where possible
- Add docstrings for non-trivial functions
- Comment complex logic

### 15.3 Security Considerations
- Never commit .env file
- Don't log API keys or sensitive data
- Validate all user inputs
- Sanitize file paths
- Use environment variables for all secrets

### 15.4 Performance Tips
- Cache embeddings aggressively (content hash-based)
- Batch API calls when possible
- Use st.cache_data for expensive computations in Streamlit
- Profile slow operations and optimize if needed
- Consider lazy loading for large datasets

### 15.5 Debugging Support
- Add verbose logging mode (controlled by env var)
- Include timing information for each processing stage
- Display intermediate results in UI (expandable sections)
- Provide clear error messages with context
- Add debug mode that shows full API responses

---

## 16. REFERENCE ARCHITECTURE DIAGRAM

```
User Interface (Streamlit)
    │
    ├─► Upload Lessons Excel ──► Validate Schema ──► Preprocess Text ──┐
    │                                                                    │
    ├─► Upload Jobs Excel ─────► Validate Schema ──► Preprocess Text   │
    │                                                                    │
    │                                                                    ▼
    │                                                            Chunk Text (500 tokens)
    │                                                                    │
    │                                                                    ▼
    │                                                        Generate Embeddings
    │                                                         (Azure OpenAI)
    │                                                                    │
    │                                                                    ▼
    │                                                        Store in ChromaDB
    │                                                        + BM25 Index
    │
    ├─► Select Job ──► Query Input
    │                       │
    │                       ▼
    │               ┌───────────────┐
    │               │ Hybrid Search │
    │               ├───────────────┤
    │               │ Dense (50)    │──► Cosine Similarity
    │               │ Sparse (50)   │──► BM25 Ranking
    │               └───────┬───────┘
    │                       │
    │                       ▼
    │               Reciprocal Rank Fusion (RRF)
    │                       │
    │                       ▼
    │               Cross-Encoder Reranking
    │                       │
    │                       ▼
    │               Top-5 Relevant Lessons
    │                       │
    │                       ▼
    │               GPT-4o-mini Analysis
    │                (Relevance Explanation)
    │                       │
    └───────────────────────┼─────► Display Results
                            │         - Lesson Cards
                            │         - Scores
                            │         - AI Analysis
                            │
                            └─────► Export to Excel
```

---

## 17. GLOSSARY

**RAG (Retrieval-Augmented Generation)**: Architecture that combines information retrieval with generative AI to provide contextually grounded responses.

**Dense Retrieval**: Semantic search using vector embeddings to find conceptually similar documents.

**Sparse Retrieval**: Keyword-based search (like BM25) that matches exact terms.

**Hybrid Search**: Combining dense and sparse retrieval for improved results.

**RRF (Reciprocal Rank Fusion)**: Method to combine multiple ranked lists without tuning parameters.

**Cross-Encoder**: Neural network that scores query-document pairs for reranking.

**ChromaDB**: Open-source vector database optimized for embeddings storage.

**Embedding**: Numerical vector representation of text that captures semantic meaning.

**Turnaround Maintenance**: Planned shutdown period for major maintenance in process industries.

**Lessons Learned**: Documented knowledge from past incidents/projects to prevent recurrence.

---

## END OF REQUIREMENTS DOCUMENT

**Version**: 1.0
**Date**: January 2026
**Author**: Megat (Oil & Gas Maintenance AI Systems)
**Target**: Claude Opus (Agentic Coding)

---

**Next Steps for Implementation:**
1. Set up Python environment (3.10+)
2. Install dependencies from requirements.txt
3. Configure Azure OpenAI credentials in .env
4. Create sample data files
5. Implement Phase 1 (Core RAG Pipeline)
6. Iterate through Phases 2-4
7. Validate with domain experts
8. Refine based on feedback
