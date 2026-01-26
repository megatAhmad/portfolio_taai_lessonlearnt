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

**IMPORTANT: Two-Stage Data Model**

The system supports **basic input** (user-provided columns) and automatically generates **enriched metadata** via LLM processing.

**Stage 1: Basic Input Columns (User Provides)**
These are the ONLY columns required from the user's Excel file:

- `lesson_id` (str): Unique identifier for the lesson
- `title` (str): Brief title/summary of the lesson
- `description` (str): Detailed description of the problem and context
- `root_cause` (str): Root cause analysis findings
- `corrective_action` (str): Actions taken to resolve the issue
- `category` (str): Type of lesson (e.g., "mechanical", "electrical", "safety", "process")
- `equipment_tag` (str, optional): Equipment identifier (e.g., "P-101", "HX-205")
- `date` (datetime, optional): When the lesson was learned
- `severity` (str, optional): "low", "medium", "high", "critical"

**Stage 2: Auto-Generated Enrichment Columns (LLM Creates)**
These columns are automatically generated by the system using GPT-4o-mini analysis:

- `specificity_level` (str): "equipment_id", "equipment_type", or "generic"
- `equipment_type` (str, nullable): e.g., "centrifugal_pump", "heat_exchanger"
- `equipment_family` (str, nullable): e.g., "rotating_equipment", "static_equipment"
- `applicable_to` (str, comma-separated): e.g., "all_pumps,all_seals,all_rotating_equipment"
- `procedure_tags` (str, comma-separated): e.g., "installation,startup,lockout_tagout,commissioning"
- `lesson_scope` (str): "specific", "general", or "universal"
- `safety_categories` (str, comma-separated): e.g., "lockout_tagout,permit_to_work,pressure_release"
- `enrichment_confidence` (float): Overall confidence score (0.0-1.0) for the enrichment
- `enrichment_timestamp` (datetime): When enrichment was performed
- `enrichment_reviewed` (bool): Whether user has reviewed/approved the enrichment

**Example User Input:**
```
| lesson_id | title                          | description                    | root_cause              | corrective_action         | category    | equipment_tag | severity |
|-----------|--------------------------------|--------------------------------|-------------------------|---------------------------|-------------|---------------|----------|
| LL-001    | Pump seal failure during TAR   | Centrifugal pump seal failed...| Incorrect installation  | Retrained technicians...  | mechanical  | P-101         | high     |
| LL-002    | Heat exchanger tube leak       | Discovered leak during...      | Corrosion from acid...  | Replaced with resistant...| process     | HX-205        | medium   |
| LL-003    | Incomplete LOTO injury         | Worker injured when valve...   | No physical verification| Mandatory verification... | safety      | null          | critical |
```

**Example After LLM Enrichment:**
```
| lesson_id | ... | specificity_level | equipment_type   | applicable_to           | procedure_tags              | lesson_scope | enrichment_confidence |
|-----------|-----|-------------------|------------------|-------------------------|-----------------------------|--------------|----------------------|
| LL-001    | ... | equipment_id      | centrifugal_pump | all_seals,all_pumps     | installation,training       | specific     | 0.89                 |
| LL-002    | ... | equipment_type    | heat_exchanger   | all_heat_exchangers     | inspection,corrosion_mgmt   | general      | 0.85                 |
| LL-003    | ... | generic           | null             | all_equipment           | lockout_tagout,isolation    | universal    | 0.95                 |
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

### 3.2 LLM Metadata Enrichment Pipeline

#### 3.2.1 Overview
The system uses Azure OpenAI GPT-4o-mini to automatically generate metadata tags for lessons learned. This enrichment happens **once** when lessons are uploaded, transforming basic user input into semantically rich, searchable metadata.

**Enrichment Flow:**
```
Raw Excel Upload (9 basic columns)
    ↓
LLM Analysis (GPT-4o-mini)
    ↓
Auto-Generated Metadata (11 additional columns)
    ↓
User Review & Approval Interface
    ↓
Enriched Lessons → RAG System
```

#### 3.2.2 Enrichment Process Stages

**Stage 1: Automatic Tag Generation**
- For each lesson, combine: title + description + root_cause + corrective_action
- Send to GPT-4o-mini with specialized metadata extraction prompt
- LLM analyzes text and returns structured JSON with metadata
- Parse JSON and populate enrichment columns
- Calculate confidence scores for each field

**Stage 2: Confidence-Based Flagging**
- High Confidence (≥0.85): Auto-accept, no review needed
- Medium Confidence (0.70-0.84): Yellow flag, suggest review
- Low Confidence (<0.70): Red flag, require user review

**Stage 3: User Review Interface**
- Display side-by-side comparison: original lesson vs. generated metadata
- Allow users to approve, edit, or reject enrichment
- Support batch approval for high-confidence items
- Track which lessons have been reviewed

**Stage 4: Iterative Learning**
- Store user corrections as ground truth examples
- Improve prompts based on common correction patterns
- Optional: Use user-corrected examples in few-shot learning

#### 3.2.3 Enrichment Prompt Strategy

The LLM enrichment prompt should instruct GPT-4o-mini to analyze lessons and extract:

1. **Specificity Level**: Determine if lesson applies to:
   - "equipment_id": Specific to mentioned equipment (e.g., "P-101")
   - "equipment_type": Applicable to all equipment of a type (e.g., all centrifugal pumps)
   - "generic": Applicable across multiple equipment types or purely procedural

2. **Equipment Classification**:
   - Extract equipment_type if mentioned (pump, valve, heat exchanger, etc.)
   - Infer equipment_family (rotating, static, instrumentation, etc.)
   - Handle cases where specific equipment is mentioned but lesson is actually generic

3. **Applicability Scope**:
   - Generate list of equipment/process types this lesson applies to
   - Use standardized taxonomy: "all_pumps", "all_seals", "all_rotating_equipment", "all_equipment"
   - Detect universal keywords: "all", "any", "every", "mandatory for"

4. **Procedure Identification**:
   - Extract maintenance procedures mentioned: installation, commissioning, startup, shutdown, inspection, isolation, etc.
   - Identify safety procedures: lockout_tagout, permit_to_work, confined_space, hot_work, etc.
   - Detect quality/training procedures: verification, torque_spec, training_requirement

5. **Lesson Scope**:
   - "specific": Lesson tied to particular equipment or unique circumstance
   - "general": Lesson applicable to a category of equipment or common procedures
   - "universal": Lesson applicable to all maintenance work (especially safety)

6. **Safety Category Detection**:
   - Identify safety-critical keywords: injury, hazard, LOTO, pressure release, toxic, flammable
   - Extract safety procedure categories
   - Flag lessons with safety implications

#### 3.2.4 Confidence Scoring Methodology

Each enriched field receives a confidence score (0.0-1.0) based on:

- **Text Clarity**: Is the lesson text clear and detailed?
- **Keyword Presence**: Are specific technical terms present?
- **Consistency**: Do multiple analysis signals agree?
- **Pattern Matching**: Does lesson match known patterns?

**Overall Enrichment Confidence**: Weighted average of individual field confidences

**Thresholds:**
- ≥0.85: High confidence (auto-accept)
- 0.70-0.84: Medium confidence (suggest review)
- <0.70: Low confidence (require review)

**Flagging Rules:**
- Safety-critical lessons (severity="critical"): Always require review regardless of confidence
- Lessons with conflicting signals (e.g., mentions specific equipment but has universal corrective action): Flag for review
- Very short descriptions (<50 words): Flag as "insufficient information"

#### 3.2.5 Batch Processing Strategy

For large datasets (100+ lessons):

**Approach 1: Sample-First Validation**
1. Process 10 random sample lessons first
2. User reviews all 10 samples
3. Calculate accuracy rate
4. If accuracy <70%: Improve prompt, retry sample
5. If accuracy ≥70%: Proceed with batch processing

**Approach 2: Progressive Batch Processing**
1. Process in batches of 50 lessons
2. Display progress: "Processing batch 3/20 (150/1000 lessons)"
3. Allow pause/resume functionality
4. Flag low-confidence items for deferred review
5. User reviews flagged items after batch completion

**Time Estimates:**
- LLM processing: 2-5 seconds per lesson
- 100 lessons: 3-8 minutes
- 1000 lessons: 30-80 minutes

**Cost Estimates:**
- Input tokens per lesson: ~500 (title + description + root_cause + corrective_action)
- Output tokens per lesson: ~200 (structured JSON metadata)
- Total per lesson: ~700 tokens
- Cost with GPT-4o-mini: ~$0.0005 per lesson
- 1000 lessons: ~$0.50 one-time cost

#### 3.2.6 Edge Case Handling

**Case 1: Lesson mentions specific equipment but is actually generic**
Example: "Pump P-101 seal failed. Root cause: Technician didn't use torque wrench. Corrective action: **Mandatory torque wrench use for ALL seal installations**"

LLM should detect:
- Keywords "ALL", "Mandatory for all" indicate universal applicability
- specificity_level: "generic" (not "equipment_id")
- applicable_to: ["all_mechanical_seals", "all_pumps", "all_bolted_equipment"]
- lesson_scope: "universal"
- Notes: "Learned from P-101 but applies universally"

**Case 2: Vague or insufficient lesson text**
Example: "Equipment failed. Better maintenance needed."

LLM should return:
```json
{
  "specificity_level": "unknown",
  "applicable_to": [],
  "procedure_tags": [],
  "lesson_scope": "unclear",
  "enrichment_confidence": 0.15,
  "flag": "INSUFFICIENT_INFORMATION",
  "recommendation": "Requires human review and additional detail"
}
```

**Case 3: Multiple equipment types mentioned**
Example: "During turnaround, found that both pumps and compressors had bearing failures due to contaminated lube oil."

LLM should extract:
- equipment_type: "multiple" or null
- applicable_to: ["all_pumps", "all_compressors", "all_rotating_equipment"]
- procedure_tags: ["lubrication", "contamination_control", "bearing_maintenance"]

**Case 4: Safety-critical with low clarity**
Example: "Someone got hurt. Need better safety."

LLM should:
- Flag as "SAFETY_CRITICAL_INSUFFICIENT_DETAIL"
- Set confidence very low
- Require mandatory human review
- Suggest requesting more detail from submitter

#### 3.2.7 User Correction Learning

**Feedback Loop Mechanism:**
1. User corrects LLM-generated tag (e.g., changes "all_pumps" to "all_centrifugal_pumps")
2. System stores: {original_lesson_text, original_tag, corrected_tag, timestamp}
3. After 10+ corrections of similar pattern, system:
   - Updates internal prompt examples
   - Adds correction pattern to validation rules
   - Displays improvement message: "System learned from your edits - accuracy improved by 12%"

**Common Correction Patterns to Learn:**
- Organization-specific terminology (e.g., "PM" means "positive material" not "preventive maintenance" in your org)
- Equipment taxonomy preferences (e.g., prefer "rotating_equipment" over "dynamic_equipment")
- Safety categorization standards (your org's specific LOTO categories)

### 3.3 Data Validation Requirements

**Basic Input Validation (Before Enrichment):**
- Validate required columns exist before processing
- Check for minimum text length in description fields (≥10 characters)
- Verify unique IDs (no duplicates)
- Handle missing/null values gracefully
- Display clear error messages for invalid data
- Support both .xlsx and .xls formats

**Enrichment Validation (After LLM Processing):**
- Verify all enrichment columns are populated
- Check confidence scores are within valid range (0.0-1.0)
- Validate JSON parsing for structured metadata
- Flag lessons with missing enrichment data
- Ensure enrichment_timestamp is recorded

### 3.4 Text Preprocessing
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
AZURE_OPENAI_ENRICHMENT_DEPLOYMENT=gpt-4o-mini  # Can be same as chat deployment
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

### 4.4 LLM Configuration for Metadata Enrichment

**Model**: GPT-4o-mini (same as relevance analysis, can reuse deployment)

**Enrichment System Prompt:**
```
You are an expert maintenance data analyst specializing in Oil & Gas operations. 
Analyze the provided maintenance lesson learned and extract structured metadata.

Return ONLY a valid JSON object with these fields:
{
  "specificity_level": "equipment_id" | "equipment_type" | "generic",
  "equipment_type": "string or null",
  "equipment_family": "rotating_equipment" | "static_equipment" | "instrumentation" | "electrical" | null,
  "applicable_to": ["list", "of", "applicability", "scopes"],
  "procedure_tags": ["list", "of", "procedures"],
  "lesson_scope": "specific" | "general" | "universal",
  "safety_categories": ["list", "of", "safety", "categories"],
  "confidence_score": 0.0-1.0,
  "notes": "Any important observations"
}

Guidelines:
- If lesson mentions specific equipment but corrective action says "ALL" or "mandatory for all", mark as generic/universal
- Safety-critical lessons should have high confidence only if clearly documented
- Use standardized taxonomy: "all_pumps", "all_seals", "lockout_tagout", etc.
- If unclear, set low confidence and explain in notes
```

**Enrichment User Prompt Template:**
```
Analyze this maintenance lesson:

Lesson ID: {lesson_id}
Title: {title}
Description: {description}
Root Cause: {root_cause}
Corrective Action: {corrective_action}
Equipment Tag: {equipment_tag}
Category: {category}
Severity: {severity}
```

**API Call Configuration:**
- Temperature: 0.1 (low for consistent structured output)
- Max Tokens: 300 (sufficient for JSON metadata)
- Response Format: JSON mode (use `response_format={"type": "json_object"}`)
- Retry Logic: 3 attempts with exponential backoff
- Batch Size: Process 10 lessons per batch, with progress tracking

### 4.5 Error Handling & Rate Limiting
- Implement exponential backoff with jitter (tenacity library)
- Retry on RateLimitError and APIError (max 6 attempts)
- Display user-friendly error messages in UI
- Log errors for debugging without exposing API keys

### 4.6 Cost Optimization
- Cache embeddings using content hashes
- Batch API calls when possible (especially for enrichment)
- Use tiktoken for accurate token counting before API calls
- **Enrichment Cost Estimate**: ~$0.50 per 1000 lessons (one-time)
- **Ongoing RAG Cost**: <$5/month for typical PoC usage
- **Total PoC Budget**: <$10/month including enrichment

---

## 5. STREAMLIT USER INTERFACE

### 5.1 Application Layout Structure

The application uses a **three-tab layout** to separate concerns and provide clear workflow:

**Tab 1: Data Upload & Enrichment**
- Upload lessons learned Excel
- Trigger automatic metadata enrichment
- Review and approve enrichment results

**Tab 2: Enrichment Review Dashboard**
- View all enriched lessons with confidence scores
- Side-by-side comparison: original vs. enriched metadata
- Edit and approve enrichment tags
- Batch operations for high-confidence items

**Tab 3: Job Analysis & Matching**
- Upload job descriptions Excel
- Select job to analyze
- View matched lessons with relevance scores
- Export results

### 5.2 Tab 1: Data Upload & Enrichment

#### Components:

1. **Header Section**
   - Application title: "Maintenance Lessons Learned RAG System"
   - Subtitle: "AI-Powered Knowledge Matching for Turnaround Maintenance"
   - Brief description of three-step workflow

2. **Lessons Learned Upload Section**
   - File uploader (accepts .xlsx, .xls)
   - Validation feedback (success/error messages)
   - Preview first 5 rows in expandable table
   - Display total count of lessons loaded
   - Show required vs. provided columns comparison

3. **Automatic Enrichment Section**
   - "Start Enrichment" button (enabled after successful upload)
   - Enrichment configuration options:
     * Batch size selector (10, 25, 50 lessons per batch)
     * Option: "Process sample (10 lessons) first for validation"
   - Progress indicator:
     * Overall: "Processing 45/100 lessons (45%)"
     * Current batch: "Batch 2/4 - Analyzing lesson LL-023..."
     * Status messages: "Extracting metadata... Calculating confidence... Flagging for review..."
   - Pause/Resume capability for large datasets
   - Time estimate: "Estimated time remaining: 3 minutes"

4. **Enrichment Summary**
   - Display after enrichment completes:
     * Total lessons enriched: 100
     * High confidence (auto-approved): 75 (green)
     * Medium confidence (review suggested): 20 (yellow)
     * Low confidence (review required): 5 (red)
     * Average confidence score: 0.87
   - "Proceed to Review Dashboard" button → navigates to Tab 2

5. **Status Indicators**
   - Visual workflow progress:
     ```
     ✅ Upload Complete → 🔄 Enrichment Running → ⏸️ Paused / ✅ Enrichment Complete → 📋 Ready for Review
     ```

### 5.3 Tab 2: Enrichment Review Dashboard

This is the **NEW CRITICAL TAB** that allows users to review and approve LLM-generated metadata.

#### Components:

1. **Dashboard Header**
   - Title: "Enrichment Review Dashboard"
   - Summary statistics:
     * Total lessons: 100
     * Reviewed: 45 / 100
     * Pending review: 55
     * Average confidence: 0.87

2. **Filter & Sort Controls**
   - Filter by confidence level:
     * All lessons
     * High confidence (≥0.85)
     * Medium confidence (0.70-0.84)
     * Low confidence (<0.70)
     * Not yet reviewed
   - Filter by lesson category: mechanical, electrical, safety, process
   - Filter by enrichment flags: "Safety Critical", "Insufficient Info", "Conflicting Signals"
   - Sort by: confidence score, lesson_id, category, date

3. **Batch Action Controls**
   - "Approve All High Confidence (75 lessons)" button
   - "Approve Filtered Selection" button
   - "Flag Selected for Manual Review" button
   - Selection counter: "5 lessons selected"

4. **Enrichment Review Cards** (for each lesson)
   
   **Card Layout (Side-by-Side):**
   
   ```
   ┌─────────────────────────────────────────────────────────────────┐
   │ Lesson LL-045 | Category: Safety | Confidence: 0.92 🟢          │
   ├──────────────────────────┬──────────────────────────────────────┤
   │ ORIGINAL LESSON          │ ENRICHED METADATA                    │
   ├──────────────────────────┼──────────────────────────────────────┤
   │ Title:                   │ Specificity Level: generic ✓         │
   │ Incomplete LOTO injury   │ [Dropdown to edit]                   │
   │                          │                                      │
   │ Description:             │ Equipment Type: null ✓               │
   │ Worker injured when...   │ [Dropdown: pump, valve, null, etc.]  │
   │ [full text shown]        │                                      │
   │                          │ Equipment Family: null ✓             │
   │ Root Cause:              │ [Dropdown: rotating, static, etc.]   │
   │ No physical verification │                                      │
   │ of isolation             │ Applicable To: ✏️                    │
   │                          │ • all_equipment ✓                    │
   │ Corrective Action:       │ • all_isolation_work ✓               │
   │ Mandatory verification   │ [+ Add tag] [× Remove]               │
   │ checklist implemented    │                                      │
   │                          │ Procedure Tags: ✏️                   │
   │ Equipment: null          │ • lockout_tagout ✓                   │
   │ Severity: critical       │ • isolation_verification ✓           │
   │                          │ • permit_to_work ✓                   │
   │                          │ [+ Add tag] [× Remove]               │
   │                          │                                      │
   │                          │ Lesson Scope: universal ✓            │
   │                          │ [Dropdown: specific,general,universal]│
   │                          │                                      │
   │                          │ Safety Categories: ✏️                │
   │                          │ • lockout_tagout ✓                   │
   │                          │ • pressure_release ✓                 │
   │                          │ [+ Add tag] [× Remove]               │
   │                          │                                      │
   │                          │ Confidence: 0.92                     │
   │                          │ Notes: "Clear universal safety       │
   │                          │ lesson with detailed corrective      │
   │                          │ action applicable to all work."      │
   └──────────────────────────┴──────────────────────────────────────┘
   │ [☑️ Approve] [✏️ Edit] [❌ Reject & Manual Entry] [⏭️ Next]      │
   └─────────────────────────────────────────────────────────────────┘
   ```

   **Interactive Features:**
   - Click any enriched field to edit inline
   - Add/remove tags with + and × buttons
   - Color-coded confidence indicators:
     * 🟢 Green (≥0.85): High confidence
     * 🟡 Yellow (0.70-0.84): Medium confidence
     * 🔴 Red (<0.70): Low confidence
   - Keyboard shortcuts: Enter to approve, E to edit, N for next

5. **Review Progress Tracker**
   - Progress bar: "45/100 lessons reviewed (45%)"
   - Category breakdown: "Mechanical: 20/30 ✓ | Electrical: 15/25 ✓ | Safety: 10/15 ✓"
   - Estimated time to complete: "~15 minutes remaining at current pace"

6. **Export Enriched Data**
   - "Export Enriched Lessons to Excel" button
   - Downloads Excel with all original + enriched columns
   - Includes review status and timestamps

7. **Navigation to Next Step**
   - "All High/Medium Confidence Approved → Proceed to Job Matching" button
   - Appears when sufficient lessons are approved (e.g., >70% reviewed or all high-confidence approved)

### 5.4 Tab 3: Job Analysis & Matching

This tab becomes accessible after enrichment is complete (can proceed even if not all lessons reviewed).

#### Components:

1. **Job Upload Section**
   - File uploader for job descriptions Excel
   - Validation and preview (similar to lessons upload)
   - "Load Jobs" button

2. **Job Selection**
2. **Job Selection**
   - Dropdown or searchable select box to choose a job
   - Display selected job details (title, description, equipment)

3. **Search Configuration**
   - Number of results slider (1-20, default: 5)
   - Minimum relevance score slider (0-100%, default: 50%)
   - Category filter (multiselect)
   - **NEW: Enrichment-based filters:**
     * Filter by lesson_scope: specific, general, universal
     * Filter by specificity_level: equipment_id, equipment_type, generic
     * Include only reviewed lessons: Yes/No toggle

4. **Search Button**
   - "Find Relevant Lessons" button
   - Processing status with multi-stage indicator:
     * "Extracting job metadata..."
     * "Multi-tier retrieval (equipment-specific → type → generic → semantic)..."
     * "Reranking with cross-encoder..."
     * "Generating AI analysis..."

### 5.5 Tab 3: Results Display

**Components:**

1. **Results Summary**
   - Total lessons found
   - Average relevance score
   - Breakdown by match type:
     * Equipment-specific matches: 2
     * Equipment-type matches: 1
     * Generic process matches: 2
   - Filtering statistics

2. **Lesson Cards** (for each matched lesson)
   
   **Enhanced Card Layout with Match Type:**
   
   ```
   ┌─────────────────────────────────────────────────────────────────┐
   │ 🎯 Equipment-Specific Match | Score: 94% 🟢                      │
   ├─────────────────────────────────────────────────────────────────┤
   │ Lesson LL-045: Pump Seal Installation Error                     │
   │ Category: Mechanical | Scope: Universal | Reviewed: ✅           │
   ├─────────────────────────────────────────────────────────────────┤
   │ Preview: Technician installed seal backwards due to poor        │
   │ lighting and lack of training on John Crane Type 28 seals...    │
   │                                                                  │
   │ Equipment: P-101 (Centrifugal Pump)                             │
   │ Applicable To: all_seals, all_pumps, all_rotating_equipment     │
   │ Procedures: installation, training, quality_control             │
   │                                                                  │
   │ [📖 View Full Details & AI Analysis ▼]                          │
   │                                                                  │
   │ ┌─ Expanded Section ──────────────────────────────────────────┐ │
   │ │ FULL LESSON CONTENT:                                        │ │
   │ │ Description: [complete text]                                │ │
   │ │ Root Cause: [complete text]                                 │ │
   │ │ Corrective Action: [complete text]                          │ │
   │ │                                                             │ │
   │ │ AI RELEVANCE ANALYSIS:                                      │ │
   │ │ Relevance Score: 94%                                        │ │
   │ │                                                             │ │
   │ │ Key Technical Links:                                        │ │
   │ │ • Both involve centrifugal pump mechanical seals            │ │
   │ │ • Same seal type: John Crane Type 28                        │ │
   │ │ • Installation procedure is directly applicable             │ │
   │ │                                                             │ │
   │ │ Safety Considerations:                                      │ │
   │ │ Ensure adequate lighting at worksite. Verify technician     │ │
   │ │ training completion on specific seal type before work.      │ │
   │ │                                                             │ │
   │ │ Recommended Actions:                                        │ │
   │ │ 1. Review photo installation guide from LL-045              │ │
   │ │ 2. Confirm technician has John Crane Type 28 training       │ │
   │ │ 3. Add task lighting to job package                         │ │
   │ │ 4. Include seal orientation verification in QC checklist    │ │
   │ │                                                             │ │
   │ │ Match Reasoning:                                            │ │
   │ │ This lesson is equipment-specific to P-101 but applies      │ │
   │ │ universally to all mechanical seal installations. The       │ │
   │ │ corrective actions (training, lighting, photo guide) are    │ │
   │ │ directly transferable to your P-205 seal replacement job.   │ │
   │ └─────────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────────┘
   ```

   **Match Type Badges:**
   - 🎯 Equipment-Specific Match (blue badge)
   - 🔧 Equipment-Type Match (green badge)
   - ⚙️ Generic Process Match (orange badge)
   - 💡 Semantic Match (purple badge)

   **Lesson Scope Indicators:**
   - 🌍 Universal
   - 📋 General
   - 🔍 Specific

3. **Interactive Filtering**
   - Search within results (text input)
   - Re-filter by score/category/match-type without re-running RAG
   - "Show only reviewed lessons" toggle
   - "Show only safety-critical" toggle

4. **Export Options**
   - "Export Results to Excel" button
   - Downloads: job details, matched lessons, relevance scores, AI analysis, enrichment metadata
   - Filename: `rag_results_[job_id]_[timestamp].xlsx`

### 5.6 Session State Management
### 5.6 Session State Management

**Critical State Variables:**
- `lessons_uploaded`: DataFrame of raw uploaded lessons
- `lessons_enriched`: DataFrame with enrichment metadata added
- `enrichment_status`: Dict tracking {lesson_id: {status, confidence, reviewed, timestamp}}
- `enrichment_in_progress`: Boolean flag
- `enrichment_progress`: Current progress (e.g., "45/100")
- `jobs_uploaded`: DataFrame of uploaded jobs
- `current_job_selection`: Selected job for analysis
- `search_results`: List of matched lessons with scores
- `user_corrections`: Dict of {lesson_id: {field: {original, corrected}}} for learning

**State Persistence:**
- Enrichment results cached to avoid re-processing on tab switches
- User edits to enrichment immediately update session state
- Search results maintained for filtering operations
- "Reset Enrichment" button clears enrichment cache
- "Clear All Data" button resets entire application state

**Tab-Specific State:**
- Tab 1: Upload status, validation results, enrichment trigger
- Tab 2: Review filters, current lesson being reviewed, approval tracking
- Tab 3: Job selection, search configuration, results

### 5.7 UI/UX Requirements
- Responsive layout (works on desktop browsers)
- Clear visual hierarchy (headers, spacing, containers)
- Loading spinners for long operations (especially enrichment batches)
- Informative error messages (no technical jargon)
- Color-coded confidence indicators (green/yellow/red)
- Color-coded match type badges (blue/green/orange/purple)
- Tooltips for help text on enrichment fields
- Accessibility: proper contrast, readable fonts
- **NEW: Progress tracking with pause/resume for enrichment**
- **NEW: Keyboard shortcuts for enrichment review (Enter=approve, E=edit, N=next)**

---

## 6. CORE FUNCTIONALITY SPECIFICATIONS

### 6.1 Text Chunking
**Requirements:**
- Chunk size: ~500 tokens (approximately 2000 characters)
- Overlap: 100 tokens (~400 characters)
- Separator priority: double newlines → single newlines → periods → spaces
- Preserve logical structure: keep root cause analysis with corrective actions
- **Add metadata to chunks**: 
  * source lesson_id
  * category
  * equipment_tag
  * **NEW: specificity_level** (from enrichment)
  * **NEW: lesson_scope** (from enrichment)
  * **NEW: applicable_to** (from enrichment)
  * **NEW: procedure_tags** (from enrichment)

**Implementation Approach:**
- Use LangChain's RecursiveCharacterTextSplitter
- Configure with maintenance-appropriate separators
- **Include enrichment metadata** in chunk metadata for filtering
- Maintain chunk-to-source mapping for result attribution

### 6.2 Hybrid Retrieval Implementation with Multi-Tier Strategy

**CRITICAL UPDATE**: The retrieval system must leverage enrichment metadata for intelligent multi-tier search.

#### 6.2.1 Multi-Tier Retrieval Flow

```
Job Query Input
    ↓
Extract Job Metadata (equipment_id, equipment_type, procedures)
    ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 1: Equipment-Specific Matches                     │
│ Filter: equipment_id = job.equipment_id                 │
│ Retrieve: Top-10 via dense + sparse hybrid             │
│ Boost Score: ×1.5                                       │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 2: Equipment-Type Matches                         │
│ Filter: equipment_type = job.equipment_type             │
│         OR applicable_to contains job.equipment_type    │
│ Retrieve: Top-20 via dense + sparse hybrid             │
│ Boost Score: ×1.2                                       │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 3: Generic/Universal Lessons                      │
│ Filter: lesson_scope = "universal"                     │
│         OR specificity_level = "generic"                │
│ Retrieve: Top-20 via dense + sparse hybrid             │
│ Boost Score: ×1.3 (for universal)                      │
│ Boost Score: ×1.4 (if severity = "critical")           │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ TIER 4: Semantic Similarity (No Metadata Filter)       │
│ Pure hybrid search on full text                        │
│ Retrieve: Top-30                                        │
│ No boost (base score)                                   │
└────────────────────────┬────────────────────────────────┘
                         ↓
                 Combine All Tiers
                         ↓
                   Deduplicate
                         ↓
           Reciprocal Rank Fusion (RRF)
                         ↓
           Cross-Encoder Reranking
                         ↓
                  Top-N Results
```

#### 6.2.2 Dense Retrieval (Embeddings)
#### 6.2.2 Dense Retrieval (Embeddings)
- Generate embeddings for all lesson chunks using Azure OpenAI
- Store in ChromaDB with persistent storage
- **Include enrichment metadata in ChromaDB metadata fields** for filtering
- Cosine similarity for vector search
- Support metadata filtering (by specificity_level, lesson_scope, etc.)
- Return top-10 to top-50 candidates depending on tier

#### 6.2.3 Sparse Retrieval (BM25)
- Use rank_bm25 Python package
- Index lesson text (description + root_cause + corrective_action)
- Return top-10 to top-50 candidates depending on tier

#### 6.2.4 Reciprocal Rank Fusion with Score Boosting

**RRF Formula:**
```python
def rrf_score(rank, k=60):
    """Calculate RRF score for combining rankings"""
    return 1 / (k + rank)
```

**Score Boosting Logic:**
```python
def calculate_boosted_score(base_score, lesson, job):
    score = base_score
    
    # Tier 1: Equipment-specific boost
    if lesson['equipment_id'] == job['equipment_id']:
        score *= 1.5
    
    # Tier 2: Equipment-type boost
    elif (lesson['equipment_type'] == job['equipment_type'] or
          job['equipment_type'] in lesson.get('applicable_to', [])):
        score *= 1.2
    
    # Tier 3: Universal/generic boost
    if lesson['lesson_scope'] == 'universal':
        score *= 1.3
    
    # Safety-critical boost (always)
    if lesson.get('severity') == 'critical':
        score *= 1.4
    
    # Procedure overlap boost
    job_procedures = extract_procedures(job['description'])
    lesson_procedures = lesson.get('procedure_tags', [])
    procedure_overlap = len(set(job_procedures) & set(lesson_procedures))
    if procedure_overlap > 0:
        score *= (1 + 0.1 * procedure_overlap)  # +10% per overlapping procedure
    
    return score
```

#### 6.2.5 Reranking
- Load cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Rerank top-50 fusion results
- Return top-5 or top-N (user-configurable)

### 6.3 Relevance Analysis Generation with Match Type Context

**For each matched lesson:**
1. Combine lesson text (description + root_cause + corrective_action)
2. Combine job text (job_description)
3. **Include enrichment metadata context** (specificity_level, applicable_to, procedure_tags, lesson_scope)
4. **Include match type** (equipment_specific, equipment_type, generic_process, semantic)
5. Send to GPT-4o-mini with structured prompt
6. Parse response into structured format

**Enhanced Prompt Template:**
```
System: You are an expert maintenance engineer analyzing relevance between a lesson learned and a future job.

Context:
- Match Type: {match_type} (e.g., "equipment_specific", "generic_process")
- Lesson Scope: {lesson_scope} (e.g., "universal", "specific")
- Lesson Applicability: {applicable_to}
- Lesson Procedures: {procedure_tags}

Lesson Learned:
{lesson_text}

Job Description:
{job_text}

Analyze and provide:
1. Relevance Score (0-100): How applicable is this lesson to the job
2. Key Technical Links: Specific technical connections (equipment type, failure mode, procedures)
3. Safety Considerations: Any safety implications from the lesson
4. Recommended Actions: Specific steps to apply this lesson to the job
5. Match Reasoning: Why this lesson matched (considering match type and scope)

Format as JSON.
```

**Output Format:**
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
  ],
  "match_reasoning": "This lesson was matched as an equipment-type lesson (applicable to all centrifugal pumps). While it originated from pump P-101, the corrective actions are marked as universal for all seal installations. The installation procedure mistakes and training requirements are directly transferable to your P-205 job."
}
```

### 6.4 Results Export with Enrichment Metadata
### 6.4 Results Export with Enrichment Metadata
**Excel Export Requirements:**
- Filename: `rag_results_[job_id]_[timestamp].xlsx`
- Sheets:
  1. "Summary": Job details, search parameters, results count, match type breakdown
  2. "Matched Lessons": All matched lessons with scores, AI analysis, **and enrichment metadata**
  
**Sheet 2 Columns (Matched Lessons):**
- lesson_id
- title
- category
- equipment_tag
- relevance_score
- match_type (equipment_specific, equipment_type, generic_process, semantic)
- **specificity_level** (from enrichment)
- **lesson_scope** (from enrichment)
- **applicable_to** (from enrichment)
- **procedure_tags** (from enrichment)
- **enrichment_confidence**
- **enrichment_reviewed** (Yes/No)
- ai_analysis (full JSON or formatted text)
- description
- root_cause
- corrective_action

---

## 7. PERFORMANCE & QUALITY REQUIREMENTS

### 7.1 Performance Targets (PoC)
- File upload & validation: <5 seconds for files up to 1000 rows
- **Enrichment processing: 2-5 seconds per lesson**
- **Batch enrichment (100 lessons): 3-8 minutes**
- Embedding generation: <30 seconds for 100 lessons
- Single job query (multi-tier hybrid retrieval + reranking): <15 seconds
- Full analysis generation (5 lessons): <20 seconds
- UI responsiveness: No frozen interface during processing (use progress indicators)

### 7.2 Quality Metrics

**Enrichment Quality (New):**
- **Enrichment Accuracy**: ≥70% of auto-generated tags accepted by users without edits
- **High Confidence Precision**: ≥85% of high-confidence (≥0.85) enrichments are correct
- **Safety Detection Recall**: ≥90% of safety-critical lessons correctly flagged

**Retrieval Quality:**
- Mean Reciprocal Rank @ 5 (MRR@5): ≥0.6
- Recall @ 10: ≥0.7
- **Multi-Tier Coverage**: Each tier (equipment-specific, type, generic, semantic) contributes to final results
- **Generic Lesson Inclusion**: ≥30% of results include generic/universal lessons when applicable

**Generation Quality:**
- Faithfulness: ≥0.7 (LLM output accurately reflects retrieved context)
- User satisfaction: ≥70% of domain experts rate results as "useful"
- No critical safety lessons missed in test cases

**Overall System Quality:**
- User satisfaction: ≥70% of domain experts rate results as "useful"
- No critical safety lessons missed in test set
- Users prefer RAG output over manual search ≥60% of the time

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
├── app.py                        # Main Streamlit application (3-tab layout)
├── config/
│   ├── __init__.py
│   ├── settings.py               # Configuration management
│   └── prompts.py                # LLM prompt templates (relevance + enrichment)
├── data/
│   ├── sample_lessons.xlsx       # Sample lessons learned data (basic columns only)
│   ├── sample_lessons_enriched.xlsx  # Sample with enrichment (for demo)
│   └── sample_jobs.xlsx          # Sample job descriptions data
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── excel_loader.py       # Excel file reading and validation
│   │   ├── preprocessor.py       # Text preprocessing
│   │   ├── chunker.py            # Text chunking logic
│   │   └── enrichment.py         # NEW: LLM metadata enrichment pipeline
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Azure OpenAI embeddings
│   │   ├── vector_store.py       # ChromaDB operations with metadata
│   │   ├── bm25_search.py        # Sparse retrieval
│   │   ├── hybrid_search.py      # Multi-tier RRF fusion with boosting
│   │   └── reranker.py           # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   └── relevance_analyzer.py # GPT-4o-mini analysis generation
│   └── ui/
│       ├── __init__.py
│       ├── components.py         # Reusable Streamlit components
│       ├── tab_upload.py         # NEW: Tab 1 - Upload & Enrichment
│       ├── tab_review.py         # NEW: Tab 2 - Enrichment Review
│       ├── tab_matching.py       # NEW: Tab 3 - Job Matching
│       └── utils.py              # UI helper functions
├── chroma_db/                    # ChromaDB persistence (GITIGNORED)
└── tests/                        # Unit tests (optional for PoC)
    ├── __init__.py
    └── test_*.py
```

### 8.2 Key Files & Responsibilities

#### `app.py`
- Main Streamlit application entry point with 3-tab layout
- Tab navigation and routing
- Global session state initialization
- Call tab-specific modules (tab_upload, tab_review, tab_matching)

#### `src/data_processing/enrichment.py` (NEW)
- Core enrichment pipeline logic
- Batch processing with progress tracking
- Call Azure OpenAI for metadata extraction
- Confidence score calculation
- Flagging logic (high/medium/low confidence)
- User correction tracking and learning

#### `src/data_processing/enrichment.py` (NEW)
- Core enrichment pipeline logic
- Batch processing with progress tracking
- Call Azure OpenAI for metadata extraction
- Confidence score calculation
- Flagging logic (high/medium/low confidence)
- User correction tracking and learning

#### `src/ui/tab_upload.py` (NEW)
- Tab 1 UI: File upload and enrichment triggering
- File validation and preview
- Enrichment progress display
- Navigation to review tab

#### `src/ui/tab_review.py` (NEW)
- Tab 2 UI: Enrichment review dashboard
- Side-by-side lesson vs. metadata comparison
- Inline editing of enrichment tags
- Batch approval operations
- Review progress tracking

#### `src/ui/tab_matching.py` (NEW)
- Tab 3 UI: Job selection and matching
- Search configuration with enrichment filters
- Results display with match type indicators
- Export functionality

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
- **Multi-tier retrieval strategy** (equipment-specific → type → generic → semantic)
- Combine dense and sparse results per tier
- Implement RRF fusion **with score boosting**
- Apply tier-specific boosts (1.5x, 1.2x, 1.3x, 1.4x)
- Procedure overlap boosting
- Return fused rankings with match_type labels

#### `src/retrieval/reranker.py`
- Load cross-encoder model
- Rerank candidate results
- Return top-N with scores

#### `src/generation/relevance_analyzer.py`
- Azure OpenAI GPT-4o-mini calls
- Structured prompt formatting **with enrichment context**
- Include match_type and lesson_scope in prompts
- Parse and structure LLM responses
- Generate match_reasoning explanations
- Handle generation errors

#### `config/settings.py`
- Load environment variables
- Azure OpenAI configuration
- Model parameters (temperature, max_tokens)
- Chunking parameters
- Retrieval thresholds

#### `config/prompts.py`
- System prompts for relevance analysis
- **System prompts for metadata enrichment**
- Prompt templates with variables
- Few-shot examples (if needed)
- Standardized taxonomy definitions (all_pumps, all_seals, etc.)

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

# JSON Processing (for enrichment)
pydantic>=2.5.0  # For structured enrichment output validation

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

### 10.1 Phase 1: Data Upload & Enrichment Pipeline (Priority: HIGH)
**Deliverables:**
1. Excel data loading and validation (basic columns only)
2. LLM metadata enrichment pipeline
   - GPT-4o-mini integration for tag extraction
   - Batch processing with progress tracking
   - Confidence scoring and flagging
3. Enriched data storage (add enrichment columns to DataFrame)
4. Basic Streamlit Tab 1: Upload and trigger enrichment

**Success Criteria:**
- Can load Excel with 9 basic columns
- Auto-generate 11 enrichment columns via LLM
- Display enrichment progress (batch X/Y)
- Store enriched data in session state
- Processing time: <5 seconds per lesson

### 10.2 Phase 2: Enrichment Review UI (Priority: HIGH)
**Deliverables:**
1. Streamlit Tab 2: Enrichment review dashboard
2. Side-by-side comparison layout
3. Inline editing of enrichment tags
4. Confidence-based filtering and flagging
5. Batch approval operations
6. User correction tracking

**Success Criteria:**
- Display all enriched lessons with confidence scores
- Allow editing of any enrichment field
- Support batch approval of high-confidence items
- Track review progress (X/Y reviewed)
- Save user edits back to DataFrame

### 10.3 Phase 3: Core RAG Pipeline with Multi-Tier Retrieval (Priority: HIGH)
**Deliverables:**
1. Text preprocessing and chunking **with enrichment metadata**
2. Azure OpenAI embedding generation
3. ChromaDB vector store setup **with metadata filtering**
4. **Multi-tier hybrid retrieval** (4 tiers: equipment-specific → type → generic → semantic)
5. **Score boosting logic** (tier-based, safety-critical, procedure overlap)
6. Cross-encoder reranking

**Success Criteria:**
- Chunk text with enrichment metadata attached
- Generate embeddings for all lessons
- Retrieve lessons across all 4 tiers
- Apply correct score boosts per tier
- Generic/universal lessons surface properly
- Reranking improves precision

### 10.4 Phase 4: Job Matching & Analysis (Priority: MEDIUM)
**Deliverables:**
1. Streamlit Tab 3: Job upload and matching
2. Job selection and search configuration
3. GPT-4o-mini relevance analysis **with match type context**
4. Results display with match type badges
5. Excel export **with enrichment metadata**

**Success Criteria:**
- Upload and select jobs
- Execute multi-tier retrieval
- Generate AI relevance analysis with match reasoning
- Display results with match type indicators
- Export includes all enrichment columns

### 10.5 Phase 5: UI Polish & Optimization (Priority: MEDIUM)
**Deliverables:**
1. Improved layouts and visual design
2. Interactive filtering across tabs
3. Keyboard shortcuts for review tab
4. Session state optimization
5. Embedding caching

**Success Criteria:**
- Intuitive user experience across all 3 tabs
- Results can be filtered without re-running search
- No redundant API calls
- Clean tab navigation

### 10.6 Phase 6: Evaluation & Documentation (Priority: LOW for PoC)
**Deliverables:**
1. Enrichment accuracy validation
2. Multi-tier retrieval effectiveness testing
3. Performance profiling
4. Documentation and README
5. Optional: RAGAS metrics

**Success Criteria:**
- Enrichment accuracy ≥70%
- Generic lessons properly surfaced
- Clear setup instructions
- Optional: Automated metrics calculated

---

## 11. TESTING & VALIDATION

### 11.1 Sample Data Requirements

**Provide Sample Excel Files:**

**1. `sample_lessons.xlsx` (Basic columns only - 20-50 lessons)**
- Include ONLY the 9 required input columns (no enrichment columns)
- Cover multiple categories (mechanical, electrical, safety, process)
- Include various severity levels
- Mix of equipment types
- **Include examples of all 3 lesson types:**
  * Equipment-specific: "Pump P-101 had seal failure..."
  * Equipment-type generic: "All centrifugal pumps require seal flush verification..."
  * Process generic: "Incomplete LOTO caused injury on any equipment..."
  
**2. `sample_lessons_enriched.xlsx` (For demo - shows expected output)**
- Same lessons as above but WITH enrichment columns populated
- Demonstrates what enrichment output should look like
- Useful for validation and testing

**3. `sample_jobs.xlsx` (10-20 job descriptions)**
- Ensure some jobs should match specific lessons (ground truth)
- Include jobs that should match:
  * Equipment-specific lessons (same equipment ID)
  * Equipment-type lessons (same pump type)
  * Generic process lessons (isolation, LOTO procedures)
- Include edge cases (very generic jobs, very specific jobs)

### 11.2 Manual Validation Tests

**Enrichment Quality Tests:**
1. **High Confidence Accuracy**: Review 20 high-confidence enrichments → should be ≥85% correct
2. **Safety Detection**: Verify all safety-critical lessons are flagged with safety_categories
3. **Generic vs. Specific**: Check that universal lessons are marked lesson_scope="universal"
4. **Edge Case**: Lesson mentions specific equipment but corrective action says "ALL" → should be generic/universal
5. **Vague Text Handling**: Very short/vague lessons should flag as low confidence

**Retrieval Quality Tests:**
1. **Positive Match Test**: Job clearly related to a lesson (e.g., pump seal replacement → pump seal failure lesson) → Should rank in top 3
2. **Equipment-Specific Match**: Job on P-205 → Should surface lessons on P-205 if they exist (boosted)
3. **Equipment-Type Match**: Job on centrifugal pump → Should surface lessons on "all centrifugal pumps"
4. **Generic Process Match**: Job involving isolation → Should surface "Incomplete LOTO" universal lesson (even if different equipment)
5. **Multi-Tier Coverage**: Results should include mix of match types (specific, type, generic, semantic)
6. **Negative Match Test**: Job unrelated to any lesson → Should return low scores
7. **Safety-Critical Test**: Jobs involving safety hazards must surface safety lessons (boosted scores)

### 11.3 Edge Cases to Handle

**Data Quality:**
- Empty Excel files
- Missing required columns
- Very short descriptions (<10 words)
- Very long descriptions (>5000 words)
- Special characters and encoding issues
- Duplicate lesson IDs
- Jobs with no relevant lessons

**Enrichment Edge Cases:**
- Lesson mentions specific equipment BUT is actually generic
- Vague lesson with insufficient detail for tagging
- Multiple equipment types mentioned in one lesson
- Safety-critical with low text clarity
- Conflicting signals (e.g., specific equipment mentioned, but procedure is universal)

**Retrieval Edge Cases:**
- Job with no equipment_id (generic job)
- Job on equipment type not in any lessons
- Generic job that should match many lessons
- Very specific job that should match only 1-2 lessons

---

## 12. SUCCESS CRITERIA

### 12.1 Functional Requirements (Must Have)
✅ Accept Excel uploads for lessons (basic 9 columns only)
✅ **Auto-generate enrichment metadata using GPT-4o-mini**
✅ **Display enrichment review dashboard with side-by-side comparison**
✅ **Allow user editing and approval of enrichment tags**
✅ **Store enrichment in enriched DataFrame with confidence scores**
✅ Accept Excel uploads for jobs
✅ Generate embeddings using Azure OpenAI
✅ **Perform multi-tier hybrid retrieval (equipment-specific → type → generic → semantic)**
✅ **Apply score boosting per tier (1.5x, 1.2x, 1.3x, 1.4x)**
✅ Rerank results with cross-encoder
✅ Generate AI relevance analysis with GPT-4o-mini **including match reasoning**
✅ Display results in 3-tab Streamlit UI with scores **and match type badges**
✅ Export results to Excel **with enrichment metadata columns**
✅ Handle errors gracefully without crashes
✅ Secure API key management via .env

### 12.2 Quality Requirements (Should Have)
✅ **Enrichment accuracy ≥70% (user accepts tags without major edits)**
✅ **Generic/universal lessons properly surface in results (≥30% of results when applicable)**
✅ Relevance scores align with domain expert judgment (70%+ accuracy on test cases)
✅ Processing time <5 seconds per lesson for enrichment
✅ Query processing time <15 seconds for typical job matching
✅ UI is intuitive and requires minimal training
✅ Cost remains under $10/month for PoC usage (including enrichment)
✅ No critical safety lessons missed in validation tests
✅ **Multi-tier retrieval demonstrably surfaces different match types**

### 12.3 Documentation Requirements (Should Have)
✅ README with setup instructions
✅ Sample data files included (basic + enriched examples)
✅ Code comments for complex logic (especially enrichment and multi-tier retrieval)
✅ Environment variable template (.env.example)
✅ **Enrichment taxonomy documentation** (standardized tag definitions)

### 12.4 PoC Exit Criteria
**The PoC is considered successful when:**
1. System successfully processes sample data end-to-end (upload → enrich → review → match → export)
2. **Enrichment accuracy ≥70% on sample data**
3. **Generic lessons surface in results when relevant (not buried by specific matches)**
4. Domain experts rate 70%+ of job matching results as "useful" in blind test
5. Multi-tier retrieval demonstrably outperforms single-tier approach
6. UI enables users to find relevant lessons in <3 minutes (including enrichment review)
7. No critical bugs or crashes during demo
8. Cost projections confirm sustainability for production scale (~$10/month for PoC)
9. **User can review and approve enrichment for 100 lessons in <30 minutes**

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
❌ Advanced analytics and reporting beyond basic enrichment stats
❌ Mobile responsiveness
❌ Multi-tenancy
❌ Version control for lessons database
❌ **Automated fine-tuning of enrichment model based on corrections** (manual prompt improvement only)
❌ **Multi-language support for enrichment**
❌ Automated evaluation metrics (RAGAS/DeepEval can be optional)

---

## 14. DELIVERABLES CHECKLIST

When implementation is complete, the following should be delivered:

**Code:**
- [ ] Complete Python codebase following project structure
- [ ] **Enrichment pipeline module (src/data_processing/enrichment.py)**
- [ ] **Three-tab Streamlit UI (tab_upload.py, tab_review.py, tab_matching.py)**
- [ ] **Multi-tier hybrid retrieval with score boosting**
- [ ] requirements.txt with all dependencies (including pydantic)
- [ ] .env.example template (without real credentials)
- [ ] .gitignore (excludes .env, chroma_db/, __pycache__)

**Documentation:**
- [ ] README.md with:
  - Project description **including enrichment workflow**
  - Setup instructions (environment, dependencies, Azure OpenAI)
  - **How to use the 3-tab interface** (upload → review → match)
  - Sample data location
  - **Enrichment taxonomy reference** (standardized tag definitions)
  - Expected usage workflow
- [ ] Code comments for complex functions
- [ ] Docstrings for key modules (especially enrichment and multi-tier retrieval)

**Sample Data:**
- [ ] sample_lessons.xlsx (20-50 records, **basic columns only**)
- [ ] **sample_lessons_enriched.xlsx (same records with enrichment metadata - for demo)**
- [ ] sample_jobs.xlsx (10-20 records)

**Application:**
- [ ] Working Streamlit app with 3 tabs (app.py)
- [ ] **Tab 1: Upload and enrichment trigger functional**
- [ ] **Tab 2: Enrichment review dashboard functional**
- [ ] **Tab 3: Job matching and results display functional**
- [ ] All modules in src/ functional
- [ ] ChromaDB persistence configured **with metadata support**

**Validation:**
- [ ] Successfully runs on sample data
- [ ] **Enrichment produces reasonable metadata (manual check)**
- [ ] **Generic lessons surface in multi-tier retrieval results**
- [ ] Produces reasonable job matching results (manual check)
- [ ] No crashes or unhandled errors
- [ ] Export functionality works **with all enrichment columns**

---

## 15. NOTES FOR IMPLEMENTATION

### 15.1 Development Best Practices
- Use type hints for function signatures
- Implement error handling at API boundaries (especially enrichment batches)
- Log important events (file uploads, API calls, errors, **enrichment progress**)
- Keep functions focused and modular
- Avoid hardcoding values (use config/settings.py)
- **For enrichment: Implement progress callbacks for UI updates**

### 15.2 Code Quality Guidelines
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions under 50 lines where possible
- Add docstrings for non-trivial functions (especially enrichment logic)
- Comment complex logic (multi-tier retrieval, score boosting)

### 15.3 Security Considerations
- Never commit .env file
- Don't log API keys or sensitive data
- Validate all user inputs
- Sanitize file paths
- Use environment variables for all secrets
- **Enrichment user corrections should not expose sensitive data**

### 15.4 Performance Tips
- **Cache enrichment results aggressively** (content hash-based, stored in session state)
- Cache embeddings aggressively (content hash-based)
- Batch API calls when possible (especially enrichment - 10 lessons per batch)
- Use st.cache_data for expensive computations in Streamlit
- Profile slow operations and optimize if needed
- Consider lazy loading for large datasets
- **For enrichment: Allow pause/resume for long-running batches**

### 15.5 Debugging Support
- Add verbose logging mode (controlled by env var)
- Include timing information for each processing stage (enrichment, retrieval, generation)
- Display intermediate results in UI (expandable sections)
- **Enrichment: Log confidence scores and flags for review**
- Provide clear error messages with context
- Add debug mode that shows full API responses
- **Show match type and tier information in debug mode**

### 15.6 Enrichment-Specific Implementation Notes

**Prompt Engineering:**
- Start with clear, structured system prompt for enrichment
- Include few-shot examples of good enrichments (optional, test without first)
- Iterate on prompt based on sample results
- Maintain taxonomy consistency (define standardized tags in config)

**Batch Processing:**
- Process in batches of 10 lessons to balance progress visibility and efficiency
- Allow pause/resume by saving batch progress to session state
- Display per-batch progress: "Batch 3/10: Processing lesson LL-023 (23/100 total)"
- Handle API errors gracefully - skip failed lessons, flag for manual review

**Confidence Calibration:**
- Test confidence thresholds on sample data
- Adjust thresholds (0.85, 0.70) if needed based on accuracy
- Higher confidence for simpler fields (specificity_level) vs. complex (applicable_to)

**User Correction Storage:**
- Store corrections in session state: {lesson_id: {field: {original, corrected}}}
- Optionally persist to JSON file for future prompt improvement
- Use corrections to identify systematic LLM errors

### 15.7 Multi-Tier Retrieval Implementation Notes

**Tier Execution:**
- Execute tiers sequentially, not in parallel (simpler for PoC)
- Each tier filters ChromaDB by metadata before hybrid search
- Combine tier results before final RRF fusion
- Track which tier each result came from (for match_type label)

**Score Boosting:**
- Apply boosts AFTER RRF fusion, before reranking
- Boosts are multiplicative (can stack: universal + critical = 1.3 × 1.4 = 1.82x)
- Test boost values on sample data - adjust if needed

**Metadata Filtering:**
- Use ChromaDB's where clause for metadata filtering
- Example: `where={"lesson_scope": "universal"}`
- Ensure enrichment metadata is properly indexed in ChromaDB

---

## 16. REFERENCE ARCHITECTURE DIAGRAM

```
User Interface (Streamlit - 3 Tabs)
    │
TAB 1: UPLOAD & ENRICHMENT
    │
    ├─► Upload Lessons Excel (9 basic columns) ──► Validate Schema ──┐
    │                                                                   │
    │                                                                   ▼
    │                                                    LLM Metadata Enrichment
    │                                                     (GPT-4o-mini, batch)
    │                                                                   │
    │                                                                   ▼
    │                                                    Add 11 enrichment columns:
    │                                                    - specificity_level
    │                                                    - equipment_type/family
    │                                                    - applicable_to
    │                                                    - procedure_tags
    │                                                    - lesson_scope
    │                                                    - safety_categories
    │                                                    - confidence scores
    │                                                                   │
    │                                                                   ▼
    │                                                    Store in Session State
    │                                                    (enriched DataFrame)
    │
TAB 2: ENRICHMENT REVIEW
    │
    ├─► Display Side-by-Side Review ──► User Edits/Approves ──► Update Session State
    │                                                                   │
    │                                                                   ▼
    │                                                    Mark as Reviewed
    │                                                    Track corrections
    │
TAB 3: JOB MATCHING
    │
    ├─► Upload Jobs Excel ─────► Validate Schema ──► Preprocess Text
    │                                                                   │
    │                                                                   ▼
    │                                              Extract Job Metadata
    │                                              (equipment_id, type, procedures)
    │
    ├─► Use Enriched Lessons ──► Preprocess Text ──► Chunk Text (500 tokens)
    │                                                  + Enrichment Metadata
    │                                                                   │
    │                                                                   ▼
    │                                                        Generate Embeddings
    │                                                         (Azure OpenAI)
    │                                                                   │
    │                                                                   ▼
    │                                                        Store in ChromaDB
    │                                                        (with metadata)
    │                                                        + BM25 Index
    │
    ├─► Select Job ──► Query Input
    │                       │
    │                       ▼
    │               ┌───────────────────────┐
    │               │ MULTI-TIER RETRIEVAL  │
    │               ├───────────────────────┤
    │               │ Tier 1: Equipment-ID  │──► Filter + Hybrid (10) ──► Boost 1.5x
    │               │ Tier 2: Equipment-Type│──► Filter + Hybrid (20) ──► Boost 1.2x
    │               │ Tier 3: Generic/Univ. │──► Filter + Hybrid (20) ──► Boost 1.3x/1.4x
    │               │ Tier 4: Semantic      │──► No Filter + Hybrid (30)
    │               └───────┬───────────────┘
    │                       │
    │                       ▼
    │               Combine All Tiers + Deduplicate
    │                       │
    │                       ▼
    │               Reciprocal Rank Fusion (RRF)
    │               + Score Boosting Logic
    │                       │
    │                       ▼
    │               Cross-Encoder Reranking
    │                       │
    │                       ▼
    │               Top-5 Relevant Lessons
    │               (with match_type labels)
    │                       │
    │                       ▼
    │               GPT-4o-mini Analysis
    │                (Relevance + Match Reasoning)
    │                       │
    └───────────────────────┼─────► Display Results
                            │         - Lesson Cards
                            │         - Match Type Badges
                            │         - Scores
                            │         - AI Analysis
                            │
                            └─────► Export to Excel
                                     (with enrichment metadata)
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

**Metadata Enrichment**: LLM-powered process to automatically extract structured tags and metadata from unstructured lesson text.

**Specificity Level**: Classification of how specific a lesson is (equipment_id, equipment_type, or generic).

**Lesson Scope**: Breadth of applicability (specific, general, or universal).

**Applicable To**: List of equipment types or processes that a lesson applies to (e.g., "all_pumps", "all_seals").

**Procedure Tags**: Maintenance procedures mentioned in a lesson (e.g., "installation", "lockout_tagout", "commissioning").

**Multi-Tier Retrieval**: Retrieval strategy that searches across multiple tiers (equipment-specific → type → generic → semantic) with different filters and boosts.

**Score Boosting**: Multiplicative adjustment to relevance scores based on match type, safety criticality, and other factors.

**Match Type**: Classification of why a lesson matched a job (equipment_specific, equipment_type, generic_process, or semantic).

**Confidence Score**: LLM's self-assessed confidence in enrichment accuracy (0.0-1.0 scale).

---

## END OF REQUIREMENTS DOCUMENT

**Version**: 2.0 (Updated with LLM Metadata Enrichment & Multi-Tier Retrieval)
**Date**: January 2026
**Author**: Megat (Oil & Gas Maintenance AI Systems)
**Target**: Claude Opus (Agentic Coding)
**Major Changes from v1.0**:
- Added LLM-powered metadata enrichment pipeline
- Introduced 3-tab Streamlit UI (Upload → Review → Match)
- Implemented multi-tier hybrid retrieval with score boosting
- Enhanced to handle equipment-specific, equipment-type, and generic lessons
- Expanded data model from 9 to 20 columns (11 auto-generated)
- Updated all sections to reflect enrichment workflow

---

**Next Steps for Implementation:**
1. Set up Python environment (3.10+)
2. Install dependencies from requirements.txt
3. Configure Azure OpenAI credentials in .env
4. **Create sample data files (basic columns only for lessons)**
5. Implement Phase 1 (Data Upload & Enrichment Pipeline)
6. Implement Phase 2 (Enrichment Review UI)
7. Implement Phase 3 (Multi-Tier RAG Pipeline)
8. Implement Phase 4 (Job Matching & Analysis)
9. Iterate through Phases 5-6 (Polish & Evaluation)
10. Validate with domain experts
11. Refine based on feedback
