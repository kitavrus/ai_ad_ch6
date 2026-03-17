"""Tests for rag/index.py — uses random embeddings, no network calls."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag.index import FAISSIndex
from rag.models import ChunkMetadata


def make_chunks(n: int, strategy: str = "fixed") -> list[ChunkMetadata]:
    return [
        ChunkMetadata(
            chunk_id=f"doc_{i}",
            source=f"doc_{i}.md",
            title=f"doc_{i}",
            section="",
            strategy=strategy,
            char_start=0,
            char_end=10,
            text=f"text chunk {i}",
        )
        for i in range(n)
    ]


def make_embeddings(n: int, dim: int = 1536) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


# ── build ──────────────────────────────────────────────────────────────────────

class TestFAISSIndexBuild:
    def test_build_basic(self):
        idx = FAISSIndex()
        idx.build(make_embeddings(5), make_chunks(5))
        assert idx._index is not None
        assert idx._index.ntotal == 5

    def test_build_stores_chunks(self):
        idx = FAISSIndex()
        chunks = make_chunks(3)
        idx.build(make_embeddings(3), chunks)
        assert idx._chunks == chunks

    def test_build_empty_raises(self):
        idx = FAISSIndex()
        with pytest.raises(ValueError, match="zero embeddings"):
            idx.build(np.zeros((0, 1536), dtype=np.float32), [])


# ── search ─────────────────────────────────────────────────────────────────────

class TestFAISSIndexSearch:
    def setup_method(self):
        self.idx = FAISSIndex()
        self.embeddings = make_embeddings(10)
        self.chunks = make_chunks(10)
        self.idx.build(self.embeddings, self.chunks)

    def test_search_returns_chunks(self):
        query = make_embeddings(1)[0]
        results = self.idx.search(query, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, ChunkMetadata) for r in results)

    def test_search_top_k_limit(self):
        query = make_embeddings(1)[0]
        results = self.idx.search(query, top_k=5)
        assert len(results) <= 5

    def test_search_exact_match(self):
        # Querying with an existing embedding should return that chunk as top-1
        query = self.embeddings[0]
        results = self.idx.search(query, top_k=1)
        assert results[0].chunk_id == "doc_0"

    def test_search_on_empty_raises(self):
        idx = FAISSIndex()
        with pytest.raises(RuntimeError):
            idx.search(make_embeddings(1)[0], top_k=1)

    def test_search_top_k_exceeds_total(self):
        query = make_embeddings(1)[0]
        results = self.idx.search(query, top_k=100)
        assert len(results) == 10  # only 10 chunks exist


# ── save / load ────────────────────────────────────────────────────────────────

class TestFAISSIndexSaveLoad:
    def test_round_trip(self, tmp_path):
        idx = FAISSIndex()
        embeddings = make_embeddings(4)
        chunks = make_chunks(4)
        idx.build(embeddings, chunks)

        save_path = str(tmp_path / "test_index")
        idx.save(save_path)

        assert Path(save_path + ".faiss").exists()
        assert Path(save_path + "_metadata.json").exists()

        loaded = FAISSIndex.load(save_path)
        assert loaded._index.ntotal == 4
        assert len(loaded._chunks) == 4
        assert loaded._chunks[0].chunk_id == "doc_0"

    def test_metadata_json_content(self, tmp_path):
        idx = FAISSIndex()
        idx.build(make_embeddings(2), make_chunks(2))
        save_path = str(tmp_path / "idx")
        idx.save(save_path)

        with open(save_path + "_metadata.json") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["chunk_id"] == "doc_0"

    def test_search_after_load(self, tmp_path):
        embeddings = make_embeddings(5)
        chunks = make_chunks(5)
        idx = FAISSIndex()
        idx.build(embeddings, chunks)
        save_path = str(tmp_path / "idx")
        idx.save(save_path)

        loaded = FAISSIndex.load(save_path)
        results = loaded.search(embeddings[2], top_k=1)
        assert results[0].chunk_id == "doc_2"

    def test_creates_parent_dirs(self, tmp_path):
        idx = FAISSIndex()
        idx.build(make_embeddings(2), make_chunks(2))
        deep_path = str(tmp_path / "a" / "b" / "c" / "idx")
        idx.save(deep_path)
        assert Path(deep_path + ".faiss").exists()
