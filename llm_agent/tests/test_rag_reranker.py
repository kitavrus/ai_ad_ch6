"""Tests for RelevanceFilter, QueryRewriter, extended RAGRetriever, and FAISSIndex.search_with_scores."""

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_agent.rag.models import ChunkMetadata, RetrievalResult
from llm_agent.rag.reranker import RelevanceFilter
from llm_agent.rag.query_rewrite import QueryRewriter
from llm_agent.rag.retriever import RAGRetriever
from llm_agent.rag.index import FAISSIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(i: int, strategy: str = "structure") -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=f"doc_{i}",
        source=f"docs/file{i}.md",
        title=f"File {i}",
        section="Section A",
        strategy=strategy,
        char_start=0,
        char_end=100,
        text=f"Content of chunk {i}",
    )


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    msg = MagicMock()
    msg.content = "improved search query"
    client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
    return client


@pytest.fixture()
def mock_embedder():
    emb = MagicMock()
    emb.generate.return_value = np.zeros((1, 1536), dtype=np.float32)
    return emb


# ---------------------------------------------------------------------------
# TestRelevanceFilter
# ---------------------------------------------------------------------------


class TestRelevanceFilter:
    def test_score_formula(self):
        rf = RelevanceFilter()
        scored = [(_chunk(0), 0.0), (_chunk(1), 1.0)]
        results = rf.filter(scored, query="q")
        # chunk 0 distance=0 → score=1.0; chunk 1 distance=1 → score=0.5
        assert abs(results[0].score - 1.0) < 1e-6
        assert abs(results[1].score - 0.5) < 1e-6

    def test_threshold_removes_low_scores(self):
        rf = RelevanceFilter(threshold=0.6)
        # distance=4 → score=0.2 (filtered); distance=0 → score=1.0 (kept)
        scored = [(_chunk(0), 0.0), (_chunk(1), 4.0)]
        results = rf.filter(scored, query="q")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "doc_0"

    def test_top_k_after_limits_results(self):
        rf = RelevanceFilter(top_k_after=2)
        scored = [(_chunk(i), float(i)) for i in range(5)]
        results = rf.filter(scored, query="q")
        assert len(results) == 2

    def test_sorted_descending(self):
        rf = RelevanceFilter()
        scored = [(_chunk(0), 3.0), (_chunk(1), 0.0), (_chunk(2), 1.0)]
        results = rf.filter(scored, query="q")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        rf = RelevanceFilter()
        assert rf.filter([], query="q") == []

    def test_query_stored_in_result(self):
        rf = RelevanceFilter()
        results = rf.filter([(_chunk(0), 0.5)], query="test query")
        assert results[0].query == "test query"

    def test_distance_stored_in_result(self):
        rf = RelevanceFilter()
        results = rf.filter([(_chunk(0), 2.5)], query="q")
        assert abs(results[0].distance - 2.5) < 1e-6

    def test_threshold_zero_keeps_all(self):
        rf = RelevanceFilter(threshold=0.0)
        scored = [(_chunk(i), float(i * 10)) for i in range(5)]
        results = rf.filter(scored, query="q")
        assert len(results) == 5


# ---------------------------------------------------------------------------
# TestQueryRewriter
# ---------------------------------------------------------------------------


class TestQueryRewriter:
    def test_rewrite_calls_llm(self, mock_llm_client):
        qr = QueryRewriter(client=mock_llm_client)
        result = qr.rewrite("что такое чанкинг")
        mock_llm_client.chat.completions.create.assert_called_once()
        assert result == "improved search query"

    def test_rewrite_fallback_on_exception(self, mock_llm_client):
        mock_llm_client.chat.completions.create.side_effect = RuntimeError("API error")
        qr = QueryRewriter(client=mock_llm_client)
        result = qr.rewrite("original query")
        assert result == "original query"

    def test_rewrite_multi_returns_list(self, mock_llm_client):
        msg = MagicMock()
        msg.content = "variant one\nvariant two\nvariant three"
        mock_llm_client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
        qr = QueryRewriter(client=mock_llm_client)
        results = qr.rewrite_multi("query", n=3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0] == "variant one"

    def test_rewrite_multi_deduplicates(self, mock_llm_client):
        msg = MagicMock()
        msg.content = "same\nsame\ndifferent"
        mock_llm_client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
        qr = QueryRewriter(client=mock_llm_client)
        results = qr.rewrite_multi("query", n=3)
        assert len(results) == 2
        assert results[0] == "same"
        assert results[1] == "different"

    def test_rewrite_multi_fallback_on_exception(self, mock_llm_client):
        mock_llm_client.chat.completions.create.side_effect = RuntimeError("fail")
        qr = QueryRewriter(client=mock_llm_client)
        results = qr.rewrite_multi("original")
        assert results == ["original"]

    def test_rewrite_empty_response_returns_original(self, mock_llm_client):
        msg = MagicMock()
        msg.content = ""
        mock_llm_client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]
        qr = QueryRewriter(client=mock_llm_client)
        result = qr.rewrite("original query")
        assert result == "original query"


# ---------------------------------------------------------------------------
# TestRAGRetrieverWithScores
# ---------------------------------------------------------------------------


class TestRAGRetrieverWithScores:
    def _make_index_mock(self, chunks_and_dists):
        """Create a FAISSIndex mock that returns (chunk, dist) pairs."""
        idx = MagicMock()
        idx.search_with_scores.return_value = chunks_and_dists
        idx.search.return_value = [c for c, _ in chunks_and_dists]
        return idx

    def test_scores_propagated(self, mock_embedder):
        retriever = RAGRetriever()
        retriever._embedder = mock_embedder
        retriever._indexes["structure"] = self._make_index_mock(
            [(_chunk(0), 0.0), (_chunk(1), 1.0)]
        )
        results = retriever.search_with_scores("query", strategy="structure", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score >= results[1].score

    def test_threshold_removes_far_chunks(self, mock_embedder):
        retriever = RAGRetriever(relevance_threshold=0.6)
        retriever._embedder = mock_embedder
        # distance=4 → score=0.2 (below threshold); distance=0 → score=1.0
        retriever._indexes["structure"] = self._make_index_mock(
            [(_chunk(0), 0.0), (_chunk(1), 4.0)]
        )
        results = retriever.search_with_scores("query", strategy="structure", top_k=2)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "doc_0"

    def test_rewriter_called_when_enabled(self, mock_embedder, mock_llm_client):
        retriever = RAGRetriever(rewrite_query=True, llm_client=mock_llm_client)
        retriever._embedder = mock_embedder
        retriever._indexes["structure"] = self._make_index_mock([(_chunk(0), 0.5)])
        retriever.search_with_scores("original", strategy="structure", top_k=1)
        mock_llm_client.chat.completions.create.assert_called_once()

    def test_multi_query_merge_dedup(self, mock_embedder, mock_llm_client):
        msg = MagicMock()
        msg.content = "variant one\nvariant two"
        mock_llm_client.chat.completions.create.return_value.choices = [MagicMock(message=msg)]

        retriever = RAGRetriever(rewrite_multi=True, llm_client=mock_llm_client)
        retriever._embedder = mock_embedder
        # Both variants return the same chunk (doc_0) — should appear once
        idx = MagicMock()
        idx.search_with_scores.return_value = [(_chunk(0), 0.5)]
        retriever._indexes["structure"] = idx
        results = retriever.search_with_scores("query", strategy="structure", top_k=3)
        chunk_ids = [r.chunk.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks in merged results"

    def test_backward_compat_search_returns_chunks(self, mock_embedder):
        retriever = RAGRetriever()
        retriever._embedder = mock_embedder
        retriever._indexes["structure"] = self._make_index_mock(
            [(_chunk(0), 0.5), (_chunk(1), 1.0)]
        )
        results = retriever.search("query", strategy="structure", top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(c, ChunkMetadata) for c in results)

    def test_top_k_after_limits_output(self, mock_embedder):
        retriever = RAGRetriever(top_k_after=1)
        retriever._embedder = mock_embedder
        retriever._indexes["structure"] = self._make_index_mock(
            [(_chunk(i), float(i)) for i in range(5)]
        )
        results = retriever.search_with_scores("query", strategy="structure", top_k=5)
        assert len(results) == 1

    def test_rewriter_not_created_when_disabled(self):
        retriever = RAGRetriever()
        assert retriever._rewriter is None


# ---------------------------------------------------------------------------
# TestFAISSIndexSearchWithScores
# ---------------------------------------------------------------------------


class TestFAISSIndexSearchWithScores:
    def _build_index(self, n_chunks: int = 5, dim: int = 4) -> FAISSIndex:
        idx = FAISSIndex()
        chunks = [_chunk(i) for i in range(n_chunks)]
        embeddings = np.random.rand(n_chunks, dim).astype(np.float32)
        idx.build(embeddings, chunks)
        return idx, embeddings

    def test_returns_tuples(self):
        idx, _ = self._build_index()
        query = np.random.rand(4).astype(np.float32)
        results = idx.search_with_scores(query, top_k=3)
        assert len(results) == 3
        for chunk, dist in results:
            assert isinstance(chunk, ChunkMetadata)
            assert isinstance(dist, float)

    def test_distances_non_negative(self):
        idx, _ = self._build_index()
        query = np.random.rand(4).astype(np.float32)
        results = idx.search_with_scores(query, top_k=5)
        assert all(dist >= 0.0 for _, dist in results)

    def test_exact_match_near_zero(self):
        idx, embeddings = self._build_index(n_chunks=3, dim=4)
        # Query is exactly the first embedding
        query = embeddings[0].copy()
        results = idx.search_with_scores(query, top_k=1)
        _, dist = results[0]
        assert dist < 1e-4

    def test_raises_when_not_built(self):
        idx = FAISSIndex()
        with pytest.raises(RuntimeError):
            idx.search_with_scores(np.zeros(4, dtype=np.float32), top_k=1)

    def test_top_k_respected(self):
        idx, _ = self._build_index(n_chunks=10)
        query = np.random.rand(4).astype(np.float32)
        results = idx.search_with_scores(query, top_k=3)
        assert len(results) == 3
