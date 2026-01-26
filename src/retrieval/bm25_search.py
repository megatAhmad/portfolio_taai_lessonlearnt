"""BM25 sparse retrieval for lessons learned."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """Document representation for BM25 search."""

    id: str
    content: str
    metadata: Dict[str, Any]
    tokens: List[str] = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = tokenize(self.content)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Lowercase and split on non-alphanumeric characters
    text = text.lower()

    # Keep alphanumeric characters and hyphens (for equipment tags)
    tokens = re.findall(r"[a-z0-9]+-?[a-z0-9]*", text)

    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


class BM25Search:
    """BM25 sparse retrieval index."""

    def __init__(self):
        """Initialize the BM25 search index."""
        self.documents: List[BM25Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._index_built = False

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of document dictionaries with 'id', 'content', 'metadata'
        """
        for doc in documents:
            bm25_doc = BM25Document(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
            )
            self.documents.append(bm25_doc)

        self._index_built = False

    def add_langchain_documents(self, documents: List[Any]) -> None:
        """
        Add LangChain Document objects to the index.

        Args:
            documents: List of LangChain Document objects
        """
        for i, doc in enumerate(documents):
            lesson_id = doc.metadata.get("lesson_id", f"unknown_{i}")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            doc_id = f"{lesson_id}_chunk_{chunk_idx}"

            bm25_doc = BM25Document(
                id=doc_id,
                content=doc.page_content,
                metadata=doc.metadata,
            )
            self.documents.append(bm25_doc)

        self._index_built = False

    def build_index(self) -> None:
        """Build the BM25 index from added documents."""
        if not self.documents:
            logger.warning("No documents to index")
            return

        # Extract tokenized corpus
        corpus = [doc.tokens for doc in self.documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(corpus)
        self._index_built = True

        logger.info(f"Built BM25 index with {len(self.documents)} documents")

    def search(
        self,
        query: str,
        n_results: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25.

        Args:
            query: Query text
            n_results: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of search results with scores
        """
        if not self._index_built:
            self.build_index()

        if not self.bm25 or not self.documents:
            return []

        # Tokenize query
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create result list
        results = []
        for i, (doc, score) in enumerate(zip(self.documents, scores)):
            # Apply metadata filter if provided
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": float(score),
                "rank": 0,  # Will be set after sorting
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Set ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1

        # Return top n results
        return results[:n_results]

    def search_by_equipment(
        self,
        query: str,
        equipment_tag: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching specific equipment.

        Args:
            query: Query text
            equipment_tag: Equipment tag to filter by
            n_results: Number of results

        Returns:
            List of search results
        """
        return self.search(
            query,
            n_results=n_results,
            metadata_filter={"equipment_tag": equipment_tag},
        )

    def search_by_equipment_type(
        self,
        query: str,
        equipment_type: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching equipment type.

        Args:
            query: Query text
            equipment_type: Equipment type to filter by
            n_results: Number of results

        Returns:
            List of search results
        """
        return self.search(
            query,
            n_results=n_results,
            metadata_filter={"equipment_type": equipment_type},
        )

    def clear(self) -> None:
        """Clear all documents from the index."""
        self.documents = []
        self.bm25 = None
        self._index_built = False
        logger.info("BM25 index cleared")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            Document count
        """
        return len(self.documents)

    def get_vocabulary_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            Number of unique terms
        """
        if not self._index_built or not self.bm25:
            return 0
        return len(self.bm25.idf)


def create_bm25_index(documents: List[Any]) -> BM25Search:
    """
    Create a BM25 index from documents.

    Args:
        documents: List of documents (dict or LangChain Document)

    Returns:
        BM25Search instance
    """
    bm25 = BM25Search()

    if documents:
        # Check if documents are LangChain Documents
        if hasattr(documents[0], "page_content"):
            bm25.add_langchain_documents(documents)
        else:
            bm25.add_documents(documents)

        bm25.build_index()

    return bm25
