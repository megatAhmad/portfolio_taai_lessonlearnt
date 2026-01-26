"""Azure OpenAI embeddings with caching and batching."""

import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

logger = logging.getLogger(__name__)

# Cache directory for embeddings
CACHE_DIR = Path("chroma_db/embedding_cache")


class EmbeddingManager:
    """Manages Azure OpenAI embeddings with caching."""

    def __init__(self, settings):
        """
        Initialize the embedding manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai.endpoint,
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
        )
        self.deployment = settings.azure_openai.embedding_deployment
        self.dimensions = settings.embeddings.dimensions
        self.batch_size = settings.embeddings.batch_size
        self.cache_enabled = settings.embeddings.cache_embeddings

        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Ensure cache directory exists
        if self.cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, List[float]] = {}

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.tokenizer.encode(text))

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key from text content.

        Args:
            text: Input text

        Returns:
            MD5 hash of text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get cache file path for a cache key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return CACHE_DIR / f"{cache_key}.json"

    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Load embedding from cache.

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(text)

        # Check memory cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    embedding = json.load(f)
                self._cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")

        return None

    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embedding to cache.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        if not self.cache_enabled:
            return

        cache_key = self._get_cache_key(text)
        self._cache[cache_key] = embedding

        # Save to disk
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "w") as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error saving embedding to cache: {e}")

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60))
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call Azure OpenAI embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.deployment,
            input=texts,
        )

        # Sort by index to maintain order
        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding

        return embeddings

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Check cache
        cached = self._load_from_cache(text)
        if cached is not None:
            return cached

        # Generate embedding
        embeddings = self._call_embedding_api([text])
        embedding = embeddings[0]

        # Cache result
        self._save_to_cache(text, embedding)

        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching and caching.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if not texts_to_embed:
            logger.info(f"All {len(texts)} embeddings loaded from cache")
            return results

        logger.info(
            f"Generating embeddings: {len(texts_to_embed)} new, "
            f"{len(texts) - len(texts_to_embed)} from cache"
        )

        # Process in batches
        for batch_start in range(0, len(texts_to_embed), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            batch_indices = indices_to_embed[batch_start:batch_end]

            # Generate embeddings
            batch_embeddings = self._call_embedding_api(batch_texts)

            # Store results and cache
            for text, idx, embedding in zip(batch_texts, batch_indices, batch_embeddings):
                results[idx] = embedding
                self._save_to_cache(text, embedding)

        return results

    def embed_documents(self, documents: List[Any]) -> List[List[float]]:
        """
        Generate embeddings for LangChain documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of embedding vectors
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

        if CACHE_DIR.exists():
            for cache_file in CACHE_DIR.glob("*.json"):
                cache_file.unlink()

        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        memory_entries = len(self._cache)
        disk_entries = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0

        return {
            "memory_entries": memory_entries,
            "disk_entries": disk_entries,
        }


def create_embedding_function(settings):
    """
    Create a LangChain-compatible embedding function.

    Args:
        settings: Application settings

    Returns:
        Embedding function for ChromaDB
    """
    manager = EmbeddingManager(settings)

    class EmbeddingFunction:
        """LangChain-compatible embedding function wrapper."""

        def __init__(self, manager: EmbeddingManager):
            self.manager = manager

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.manager.embed_texts(texts)

        def embed_query(self, text: str) -> List[float]:
            return self.manager.embed_text(text)

    return EmbeddingFunction(manager)
