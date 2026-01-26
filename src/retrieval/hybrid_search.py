"""Multi-tier hybrid search with RRF fusion and score boosting."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .vector_store import VectorStore
from .bm25_search import BM25Search

logger = logging.getLogger(__name__)


class MatchTier(Enum):
    """Match tier levels for multi-tier retrieval."""

    EQUIPMENT_SPECIFIC = "equipment_specific"  # Exact equipment tag match
    EQUIPMENT_TYPE = "equipment_type"  # Same equipment type
    GENERIC = "generic"  # Generic/universal lessons
    SEMANTIC = "semantic"  # Pure semantic similarity


@dataclass
class RetrievalResult:
    """Result from retrieval with tier information."""

    id: str
    content: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    boosted_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0
    match_tier: MatchTier = MatchTier.SEMANTIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rrf_score": self.rrf_score,
            "boosted_score": self.boosted_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "match_tier": self.match_tier.value,
        }


class HybridSearch:
    """Multi-tier hybrid search combining dense and sparse retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_search: BM25Search,
        settings,
    ):
        """
        Initialize hybrid search.

        Args:
            vector_store: Dense retrieval vector store
            bm25_search: Sparse retrieval BM25 index
            settings: Application settings
        """
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.settings = settings

        # RRF parameters
        self.rrf_k = settings.retrieval.rrf_k

        # Tier boost factors
        self.equipment_specific_boost = settings.retrieval.equipment_specific_boost
        self.equipment_type_boost = settings.retrieval.equipment_type_boost
        self.generic_boost = settings.retrieval.generic_boost
        self.universal_boost = settings.retrieval.universal_boost

    def rrf_score(self, rank: int) -> float:
        """
        Calculate Reciprocal Rank Fusion score.

        Args:
            rank: Rank position (1-indexed)

        Returns:
            RRF score
        """
        if rank <= 0:
            return 0.0
        return 1.0 / (self.rrf_k + rank)

    def _combine_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> Dict[str, RetrievalResult]:
        """
        Combine dense and sparse results using RRF.

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search

        Returns:
            Dictionary of combined results by ID
        """
        combined = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = RetrievalResult(
                    id=doc_id,
                    content=result["content"],
                    metadata=result["metadata"],
                )
            combined[doc_id].dense_score = result.get("score", 0.0)
            combined[doc_id].dense_rank = rank

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = RetrievalResult(
                    id=doc_id,
                    content=result["content"],
                    metadata=result["metadata"],
                )
            combined[doc_id].sparse_score = result.get("score", 0.0)
            combined[doc_id].sparse_rank = rank

        # Calculate RRF scores
        for result in combined.values():
            dense_rrf = self.rrf_score(result.dense_rank) if result.dense_rank > 0 else 0.0
            sparse_rrf = self.rrf_score(result.sparse_rank) if result.sparse_rank > 0 else 0.0
            result.rrf_score = dense_rrf + sparse_rrf

        return combined

    def _determine_tier(
        self,
        result: RetrievalResult,
        job_equipment_tag: Optional[str],
        job_equipment_type: Optional[str],
    ) -> MatchTier:
        """
        Determine the match tier for a result.

        Args:
            result: Retrieval result
            job_equipment_tag: Equipment tag from job
            job_equipment_type: Equipment type from job

        Returns:
            Match tier
        """
        metadata = result.metadata

        # Check for equipment-specific match
        if job_equipment_tag and metadata.get("equipment_tag"):
            if metadata["equipment_tag"].upper() == job_equipment_tag.upper():
                return MatchTier.EQUIPMENT_SPECIFIC

        # Check for equipment-type match
        if job_equipment_type and metadata.get("equipment_type"):
            if metadata["equipment_type"].lower() == job_equipment_type.lower():
                return MatchTier.EQUIPMENT_TYPE

        # Check for universal/generic lessons
        lesson_scope = metadata.get("lesson_scope", "").lower()
        if lesson_scope == "universal":
            return MatchTier.GENERIC
        if lesson_scope == "generic":
            return MatchTier.GENERIC

        # Default to semantic match
        return MatchTier.SEMANTIC

    def _apply_tier_boost(self, result: RetrievalResult) -> float:
        """
        Apply tier-based boost to RRF score.

        Args:
            result: Retrieval result

        Returns:
            Boosted score
        """
        boost = 1.0

        if result.match_tier == MatchTier.EQUIPMENT_SPECIFIC:
            boost = self.equipment_specific_boost
        elif result.match_tier == MatchTier.EQUIPMENT_TYPE:
            boost = self.equipment_type_boost
        elif result.match_tier == MatchTier.GENERIC:
            # Use generic or universal boost based on scope
            scope = result.metadata.get("lesson_scope", "").lower()
            if scope == "universal":
                boost = self.universal_boost
            else:
                boost = self.generic_boost

        return result.rrf_score * boost

    def search(
        self,
        query: str,
        n_results: int = 50,
        job_equipment_tag: Optional[str] = None,
        job_equipment_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Perform multi-tier hybrid search.

        Args:
            query: Query text
            n_results: Number of results to return
            job_equipment_tag: Equipment tag from job for tier matching
            job_equipment_type: Equipment type from job for tier matching

        Returns:
            List of retrieval results sorted by boosted score
        """
        # Get dense results
        dense_results = self.vector_store.search(query, n_results=n_results * 2)

        # Get sparse results
        sparse_results = self.bm25_search.search(query, n_results=n_results * 2)

        # Combine with RRF
        combined = self._combine_results(dense_results, sparse_results)

        # Determine tiers and apply boosts
        for result in combined.values():
            result.match_tier = self._determine_tier(
                result, job_equipment_tag, job_equipment_type
            )
            result.boosted_score = self._apply_tier_boost(result)

        # Sort by boosted score
        results = sorted(combined.values(), key=lambda x: x.boosted_score, reverse=True)

        return results[:n_results]

    def multi_tier_search(
        self,
        query: str,
        job_equipment_tag: Optional[str] = None,
        job_equipment_type: Optional[str] = None,
        results_per_tier: int = 10,
        total_results: int = 50,
    ) -> Tuple[List[RetrievalResult], Dict[str, List[RetrievalResult]]]:
        """
        Perform explicit multi-tier search with separate tier results.

        This method explicitly searches each tier and returns both combined
        results and per-tier results.

        Args:
            query: Query text
            job_equipment_tag: Equipment tag from job
            job_equipment_type: Equipment type from job
            results_per_tier: Number of results per tier
            total_results: Total number of combined results

        Returns:
            Tuple of (combined_results, tier_results_dict)
        """
        tier_results = {
            MatchTier.EQUIPMENT_SPECIFIC.value: [],
            MatchTier.EQUIPMENT_TYPE.value: [],
            MatchTier.GENERIC.value: [],
            MatchTier.SEMANTIC.value: [],
        }

        all_results = {}

        # Tier 1: Equipment-specific search
        if job_equipment_tag:
            equipment_dense = self.vector_store.search_by_equipment(
                query, job_equipment_tag, n_results=results_per_tier
            )
            equipment_sparse = self.bm25_search.search_by_equipment(
                query, job_equipment_tag, n_results=results_per_tier
            )
            tier_combined = self._combine_results(equipment_dense, equipment_sparse)

            for result in tier_combined.values():
                result.match_tier = MatchTier.EQUIPMENT_SPECIFIC
                result.boosted_score = result.rrf_score * self.equipment_specific_boost
                tier_results[MatchTier.EQUIPMENT_SPECIFIC.value].append(result)
                all_results[result.id] = result

        # Tier 2: Equipment-type search
        if job_equipment_type:
            type_dense = self.vector_store.search_by_equipment_type(
                query, job_equipment_type, n_results=results_per_tier
            )
            type_sparse = self.bm25_search.search_by_equipment_type(
                query, job_equipment_type, n_results=results_per_tier
            )
            tier_combined = self._combine_results(type_dense, type_sparse)

            for result in tier_combined.values():
                if result.id not in all_results:
                    result.match_tier = MatchTier.EQUIPMENT_TYPE
                    result.boosted_score = result.rrf_score * self.equipment_type_boost
                    tier_results[MatchTier.EQUIPMENT_TYPE.value].append(result)
                    all_results[result.id] = result

        # Tier 3: Universal/generic lessons
        universal_dense = self.vector_store.search_universal_lessons(
            query, n_results=results_per_tier
        )
        # For BM25, search all and filter
        universal_sparse = self.bm25_search.search(
            query,
            n_results=results_per_tier * 2,
            metadata_filter={"lesson_scope": "universal"},
        )
        tier_combined = self._combine_results(universal_dense, universal_sparse)

        for result in tier_combined.values():
            if result.id not in all_results:
                result.match_tier = MatchTier.GENERIC
                result.boosted_score = result.rrf_score * self.universal_boost
                tier_results[MatchTier.GENERIC.value].append(result)
                all_results[result.id] = result

        # Tier 4: Semantic search (catch-all)
        semantic_dense = self.vector_store.search(query, n_results=results_per_tier * 2)
        semantic_sparse = self.bm25_search.search(query, n_results=results_per_tier * 2)
        tier_combined = self._combine_results(semantic_dense, semantic_sparse)

        for result in tier_combined.values():
            if result.id not in all_results:
                result.match_tier = MatchTier.SEMANTIC
                result.boosted_score = result.rrf_score  # No boost for semantic
                tier_results[MatchTier.SEMANTIC.value].append(result)
                all_results[result.id] = result

        # Sort tier results
        for tier in tier_results:
            tier_results[tier].sort(key=lambda x: x.boosted_score, reverse=True)

        # Combine all results and sort
        combined_results = sorted(
            all_results.values(), key=lambda x: x.boosted_score, reverse=True
        )

        return combined_results[:total_results], tier_results


def create_hybrid_search(
    vector_store: VectorStore,
    bm25_search: BM25Search,
    settings,
) -> HybridSearch:
    """
    Create a hybrid search instance.

    Args:
        vector_store: Dense retrieval vector store
        bm25_search: Sparse retrieval BM25 index
        settings: Application settings

    Returns:
        HybridSearch instance
    """
    return HybridSearch(vector_store, bm25_search, settings)
