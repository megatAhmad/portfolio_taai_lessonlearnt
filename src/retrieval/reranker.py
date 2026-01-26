"""Cross-encoder reranking for retrieval results."""

from typing import List, Dict, Any, Optional
import logging

from sentence_transformers import CrossEncoder
import numpy as np

from .hybrid_search import RetrievalResult

logger = logging.getLogger(__name__)

# Default reranker model
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"Loaded reranker model: {model_name}")

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder.

        Args:
            query: Query text
            results: List of retrieval results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Add rerank scores to results
        for i, result in enumerate(results):
            result.metadata["rerank_score"] = float(scores[i])

        # Sort by rerank score
        reranked = sorted(
            results,
            key=lambda x: x.metadata.get("rerank_score", 0),
            reverse=True,
        )

        return reranked[:top_k]

    def rerank_with_tier_preservation(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
        min_per_tier: int = 1,
    ) -> List[RetrievalResult]:
        """
        Rerank results while preserving tier diversity.

        This method ensures that at least min_per_tier results from each
        represented tier are included in the final results.

        Args:
            query: Query text
            results: List of retrieval results
            top_k: Number of results to return
            min_per_tier: Minimum results per tier

        Returns:
            Reranked results with tier diversity
        """
        if not results:
            return []

        # Group results by tier
        tier_groups: Dict[str, List[RetrievalResult]] = {}
        for result in results:
            tier = result.match_tier.value
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(result)

        # Rerank each tier separately
        reranked_tiers: Dict[str, List[RetrievalResult]] = {}
        for tier, tier_results in tier_groups.items():
            if tier_results:
                pairs = [(query, r.content) for r in tier_results]
                scores = self.model.predict(pairs)

                for i, result in enumerate(tier_results):
                    result.metadata["rerank_score"] = float(scores[i])

                reranked_tiers[tier] = sorted(
                    tier_results,
                    key=lambda x: x.metadata.get("rerank_score", 0),
                    reverse=True,
                )

        # Select results ensuring tier diversity
        final_results = []
        tier_counts = {tier: 0 for tier in reranked_tiers}

        # First pass: ensure minimum per tier
        for tier, tier_results in reranked_tiers.items():
            for result in tier_results[:min_per_tier]:
                if len(final_results) < top_k:
                    final_results.append(result)
                    tier_counts[tier] += 1

        # Second pass: fill remaining slots with highest scores
        all_remaining = []
        for tier, tier_results in reranked_tiers.items():
            all_remaining.extend(tier_results[tier_counts[tier]:])

        all_remaining.sort(
            key=lambda x: x.metadata.get("rerank_score", 0),
            reverse=True,
        )

        for result in all_remaining:
            if len(final_results) >= top_k:
                break
            if result not in final_results:
                final_results.append(result)

        # Final sort by rerank score
        final_results.sort(
            key=lambda x: x.metadata.get("rerank_score", 0),
            reverse=True,
        )

        return final_results

    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[RetrievalResult]],
        top_k: int = 5,
    ) -> List[List[RetrievalResult]]:
        """
        Batch rerank multiple queries' results.

        Args:
            queries: List of query texts
            results_list: List of result lists, one per query
            top_k: Number of results per query

        Returns:
            List of reranked result lists
        """
        reranked_list = []

        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            reranked_list.append(reranked)

        return reranked_list


class RerankerWithScoreNormalization(Reranker):
    """Reranker that normalizes scores for better interpretability."""

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Rerank with score normalization.

        Args:
            query: Query text
            results: List of retrieval results
            top_k: Number of results to return

        Returns:
            Reranked results with normalized scores
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        # Get cross-encoder scores
        raw_scores = self.model.predict(pairs)

        # Normalize scores to 0-100 range
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        if score_range > 0:
            normalized_scores = [
                ((s - min_score) / score_range) * 100 for s in raw_scores
            ]
        else:
            normalized_scores = [50.0] * len(raw_scores)

        # Add scores to results
        for i, result in enumerate(results):
            result.metadata["rerank_score_raw"] = float(raw_scores[i])
            result.metadata["rerank_score"] = float(normalized_scores[i])

        # Sort by normalized score
        reranked = sorted(
            results,
            key=lambda x: x.metadata.get("rerank_score", 0),
            reverse=True,
        )

        return reranked[:top_k]


def create_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    normalize_scores: bool = True,
    device: Optional[str] = None,
) -> Reranker:
    """
    Create a reranker instance.

    Args:
        model_name: Name of the cross-encoder model
        normalize_scores: Whether to normalize scores
        device: Device to use

    Returns:
        Reranker instance
    """
    if normalize_scores:
        return RerankerWithScoreNormalization(model_name, device)
    return Reranker(model_name, device)
