"""Relevance filter — second-stage scoring and threshold cut for RAG results."""

from typing import List, Optional, Tuple

from .models import ChunkMetadata, RetrievalResult


class RelevanceFilter:
    """Convert raw L2 distances to relevance scores and apply filtering."""

    def __init__(self, threshold: float = 0.0, top_k_after: Optional[int] = None):
        self.threshold = threshold
        self.top_k_after = top_k_after

    def filter(
        self,
        scored_chunks: List[Tuple[ChunkMetadata, float]],
        query: str = "",
    ) -> List[RetrievalResult]:
        """Score, filter, and rank chunks.

        Args:
            scored_chunks: List of (chunk, L2_distance) from FAISS.
            query: The query string used to retrieve these chunks.

        Returns:
            Filtered and sorted list of RetrievalResult (highest score first).
        """
        results: List[RetrievalResult] = []
        for chunk, distance in scored_chunks:
            score = 1.0 / (1.0 + distance)
            if score >= self.threshold:
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=score,
                        distance=distance,
                        query=query,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)

        if self.top_k_after is not None:
            results = results[: self.top_k_after]

        return results
