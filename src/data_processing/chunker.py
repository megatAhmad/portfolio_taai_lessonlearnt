"""Text chunking for lessons learned with metadata preservation."""

from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

from .preprocessor import combine_lesson_text, preprocess_text

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 2000  # ~500 tokens
DEFAULT_CHUNK_OVERLAP = 400  # ~100 tokens
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_lessons(
    lessons: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    include_enrichment: bool = True,
) -> List[Document]:
    """
    Chunk lessons learned into documents for embedding.

    Args:
        lessons: List of lesson dictionaries
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        include_enrichment: Whether to include enrichment metadata

    Returns:
        List of LangChain Document objects
    """
    documents = []

    for lesson in lessons:
        lesson_docs = create_chunks_with_metadata(
            lesson,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_enrichment=include_enrichment,
        )
        documents.extend(lesson_docs)

    logger.info(f"Created {len(documents)} chunks from {len(lessons)} lessons")
    return documents


def create_chunks_with_metadata(
    lesson: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    include_enrichment: bool = True,
) -> List[Document]:
    """
    Create chunks from a single lesson with metadata.

    Args:
        lesson: Lesson dictionary
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        include_enrichment: Whether to include enrichment metadata

    Returns:
        List of LangChain Document objects
    """
    # Combine lesson text
    text = combine_lesson_text(lesson, include_metadata=False)

    # Preprocess text
    text = preprocess_text(text, expand_abbr=True, normalize=True)

    if not text.strip():
        logger.warning(f"Empty text for lesson {lesson.get('lesson_id', 'unknown')}")
        return []

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Create documents with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = create_chunk_metadata(lesson, chunk_index=i, total_chunks=len(chunks))

        # Add enrichment metadata if available and requested
        if include_enrichment:
            metadata.update(get_enrichment_metadata(lesson))

        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)

    return documents


def create_chunk_metadata(
    lesson: Dict[str, Any],
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> Dict[str, Any]:
    """
    Create metadata for a chunk from lesson data.

    Args:
        lesson: Lesson dictionary
        chunk_index: Index of this chunk
        total_chunks: Total number of chunks for this lesson

    Returns:
        Metadata dictionary
    """
    metadata = {
        # Core identifiers
        "lesson_id": lesson.get("lesson_id", ""),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,

        # Basic metadata
        "title": lesson.get("title", ""),
        "category": lesson.get("category", ""),
        "equipment_tag": lesson.get("equipment_tag") or "",
        "severity": lesson.get("severity", "medium"),

        # Source tracking
        "source": "lessons_learned",
    }

    # Add date if available
    if lesson.get("date"):
        metadata["date"] = str(lesson["date"])

    return metadata


def get_enrichment_metadata(lesson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract enrichment metadata from a lesson.

    Args:
        lesson: Lesson dictionary with enrichment data

    Returns:
        Enrichment metadata dictionary
    """
    enrichment = {}

    # Enrichment fields
    enrichment_fields = [
        "specificity_level",
        "equipment_type",
        "equipment_family",
        "lesson_scope",
        "enrichment_confidence",
        "enrichment_reviewed",
    ]

    for field in enrichment_fields:
        if field in lesson and lesson[field] is not None:
            enrichment[field] = lesson[field]

    # Handle list fields (convert to comma-separated strings for ChromaDB)
    list_fields = ["applicable_to", "procedure_tags", "safety_categories"]
    for field in list_fields:
        if field in lesson and lesson[field]:
            value = lesson[field]
            if isinstance(value, list):
                enrichment[field] = ",".join(value)
            elif isinstance(value, str):
                enrichment[field] = value
            else:
                enrichment[field] = str(value)

    return enrichment


def create_job_document(job: Dict[str, Any]) -> Document:
    """
    Create a document from a job description for querying.

    Args:
        job: Job dictionary

    Returns:
        LangChain Document object
    """
    from .preprocessor import combine_job_text

    text = combine_job_text(job, include_metadata=False)
    text = preprocess_text(text, expand_abbr=True, normalize=True)

    metadata = {
        "job_id": job.get("job_id", ""),
        "job_title": job.get("job_title", ""),
        "equipment_tag": job.get("equipment_tag") or "",
        "job_type": job.get("job_type") or "",
        "source": "job_description",
    }

    return Document(page_content=text, metadata=metadata)


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Uses a simple heuristic: ~4 characters per token on average.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def get_optimal_chunk_params(avg_lesson_length: int) -> Dict[str, int]:
    """
    Get optimal chunking parameters based on average lesson length.

    Args:
        avg_lesson_length: Average length of lessons in characters

    Returns:
        Dictionary with chunk_size and chunk_overlap
    """
    if avg_lesson_length < 1000:
        # Short lessons: smaller chunks, less overlap
        return {"chunk_size": 1000, "chunk_overlap": 200}
    elif avg_lesson_length < 3000:
        # Medium lessons: default parameters
        return {"chunk_size": 2000, "chunk_overlap": 400}
    else:
        # Long lessons: larger chunks, more overlap
        return {"chunk_size": 3000, "chunk_overlap": 600}
