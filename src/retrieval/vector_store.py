"""ChromaDB vector store operations with metadata filtering."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document

from .embeddings import EmbeddingManager, create_embedding_function

logger = logging.getLogger(__name__)

# Default persistence directory
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "lessons_learned"


class VectorStore:
    """ChromaDB vector store for lessons learned."""

    def __init__(self, settings, persist_dir: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            settings: Application settings
            persist_dir: Optional persistence directory
        """
        self.settings = settings
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR

        # Ensure persistence directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(settings)

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Get or create the lessons learned collection.

        Returns:
            ChromaDB collection
        """
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Maintenance lessons learned embeddings"},
        )

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents per batch

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_manager.embed_texts(texts)

        # Prepare data for ChromaDB
        ids = []
        metadatas = []

        for i, doc in enumerate(documents):
            # Create unique ID from lesson_id and chunk_index
            lesson_id = doc.metadata.get("lesson_id", f"unknown_{i}")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            doc_id = f"{lesson_id}_chunk_{chunk_idx}"
            ids.append(doc_id)

            # Filter metadata to ChromaDB-compatible types
            metadata = self._filter_metadata(doc.metadata)
            metadatas.append(metadata)

        # Add to collection in batches
        total_added = 0
        for batch_start in range(0, len(documents), batch_size):
            batch_end = min(batch_start + batch_size, len(documents))

            batch_ids = ids[batch_start:batch_end]
            batch_texts = texts[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]

            # Upsert to handle duplicates
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

            total_added += len(batch_ids)

        logger.info(f"Added {total_added} documents to vector store")
        return total_added

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to ChromaDB-compatible types.

        ChromaDB only supports: str, int, float, bool

        Args:
            metadata: Original metadata

        Returns:
            Filtered metadata
        """
        filtered = {}

        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                filtered[key] = ",".join(str(v) for v in value)
            else:
                # Convert other types to string
                filtered[key] = str(value)

        return filtered

    def search(
        self,
        query_text: str,
        n_results: int = 50,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter

        Returns:
            List of search results with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query_text)

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    # Convert distance to similarity score (cosine distance to similarity)
                    "score": 1 - (results["distances"][0][i] if results["distances"] else 0.0),
                }
                formatted_results.append(result)

        return formatted_results

    def search_by_equipment(
        self,
        query_text: str,
        equipment_tag: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching specific equipment.

        Args:
            query_text: Query text
            equipment_tag: Equipment tag to filter by
            n_results: Number of results

        Returns:
            List of search results
        """
        where_filter = {"equipment_tag": equipment_tag}
        return self.search(query_text, n_results=n_results, where=where_filter)

    def search_by_equipment_type(
        self,
        query_text: str,
        equipment_type: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching equipment type.

        Args:
            query_text: Query text
            equipment_type: Equipment type to filter by
            n_results: Number of results

        Returns:
            List of search results
        """
        where_filter = {"equipment_type": equipment_type}
        return self.search(query_text, n_results=n_results, where=where_filter)

    def search_universal_lessons(
        self,
        query_text: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for universal/generic lessons.

        Args:
            query_text: Query text
            n_results: Number of results

        Returns:
            List of search results
        """
        where_filter = {"lesson_scope": "universal"}
        return self.search(query_text, n_results=n_results, where=where_filter)

    def search_by_category(
        self,
        query_text: str,
        category: str,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in a specific category.

        Args:
            query_text: Query text
            category: Category to filter by
            n_results: Number of results

        Returns:
            List of search results
        """
        where_filter = {"category": category}
        return self.search(query_text, n_results=n_results, where=where_filter)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the collection.

        Returns:
            List of all documents
        """
        results = self.collection.get(include=["documents", "metadatas"])

        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                doc = {
                    "id": doc_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
                documents.append(doc)

        return documents

    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Document count
        """
        return self.collection.count()

    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self._get_or_create_collection()
        logger.info("Vector store cleared")

    def delete_by_lesson_id(self, lesson_id: str) -> None:
        """
        Delete all chunks for a specific lesson.

        Args:
            lesson_id: Lesson ID to delete
        """
        # Get all documents for this lesson
        results = self.collection.get(
            where={"lesson_id": lesson_id},
            include=["metadatas"],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for lesson {lesson_id}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()

        # Get unique lesson IDs
        all_docs = self.collection.get(include=["metadatas"])
        lesson_ids = set()
        categories = set()
        equipment_types = set()

        for metadata in all_docs["metadatas"] or []:
            if metadata.get("lesson_id"):
                lesson_ids.add(metadata["lesson_id"])
            if metadata.get("category"):
                categories.add(metadata["category"])
            if metadata.get("equipment_type"):
                equipment_types.add(metadata["equipment_type"])

        return {
            "total_chunks": count,
            "unique_lessons": len(lesson_ids),
            "categories": list(categories),
            "equipment_types": list(equipment_types),
        }


def create_vector_store(settings, persist_dir: Optional[str] = None) -> VectorStore:
    """
    Create a vector store instance.

    Args:
        settings: Application settings
        persist_dir: Optional persistence directory

    Returns:
        VectorStore instance
    """
    return VectorStore(settings, persist_dir)
