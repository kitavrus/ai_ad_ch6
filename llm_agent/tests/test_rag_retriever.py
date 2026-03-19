"""Tests for RAGRetriever."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_agent.rag.models import ChunkMetadata
from llm_agent.rag.retriever import RAGRetriever


def _make_chunk(i: int) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=f"doc_{i}",
        source=f"docs/file{i}.md",
        title=f"File {i}",
        section="Section A",
        strategy="structure",
        char_start=0,
        char_end=100,
        text=f"Content of chunk {i}",
    )


@pytest.fixture()
def mock_index():
    idx = MagicMock()
    chunks = [_make_chunk(0), _make_chunk(1), _make_chunk(2)]
    idx.search.return_value = chunks
    idx.search_with_scores.return_value = [(c, float(i)) for i, c in enumerate(chunks)]
    return idx


@pytest.fixture()
def mock_embedder():
    emb = MagicMock()
    emb.generate.return_value = np.zeros((1, 1536), dtype=np.float32)
    return emb


def test_search_returns_chunks(mock_index, mock_embedder):
    """search() returns top_k chunks from the index."""
    retriever = RAGRetriever(index_dir="rag_index")
    retriever._embedder = mock_embedder
    retriever._indexes["structure"] = mock_index

    results = retriever.search("test query", strategy="structure", top_k=3)

    mock_embedder.generate.assert_called_once_with(["test query"])
    mock_index.search_with_scores.assert_called_once()
    assert len(results) == 3
    assert all(isinstance(c, ChunkMetadata) for c in results)


def test_lazy_loading(mock_embedder, tmp_path):
    """Index is loaded from disk exactly once (lazy singleton per strategy)."""
    retriever = RAGRetriever(index_dir=str(tmp_path))
    retriever._embedder = mock_embedder

    fake_index = MagicMock()
    fake_index.search_with_scores.return_value = [(_make_chunk(0), 0.5)]

    with patch("llm_agent.rag.retriever.FAISSIndex.load", return_value=fake_index) as mock_load:
        retriever.search("q1", strategy="structure", top_k=1)
        retriever.search("q2", strategy="structure", top_k=1)

    # FAISSIndex.load should be called only once despite two searches
    mock_load.assert_called_once()
    assert fake_index.search_with_scores.call_count == 2


def test_unknown_strategy_raises(mock_embedder, tmp_path):
    """Searching with a strategy that has no index file raises FileNotFoundError."""
    retriever = RAGRetriever(index_dir=str(tmp_path))
    retriever._embedder = mock_embedder

    with pytest.raises(Exception):  # FAISSIndex.load raises when file absent
        retriever.search("query", strategy="nonexistent", top_k=3)


def test_search_passes_top_k(mock_index, mock_embedder):
    """search() forwards the top_k argument to FAISSIndex.search_with_scores."""
    retriever = RAGRetriever(index_dir="rag_index")
    retriever._embedder = mock_embedder
    retriever._indexes["fixed"] = mock_index
    mock_index.search_with_scores.return_value = [(_make_chunk(0), 0.5)]

    retriever.search("q", strategy="fixed", top_k=5)

    args, kwargs = mock_index.search_with_scores.call_args
    top_k_value = args[1] if len(args) > 1 else kwargs.get("top_k")
    assert top_k_value == 5
