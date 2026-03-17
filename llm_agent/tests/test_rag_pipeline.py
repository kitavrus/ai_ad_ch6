"""Tests for rag/pipeline.py and rag/compare.py — embeddings mocked."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.models import IndexStats
from rag.pipeline import IndexingPipeline, _compute_stats
from rag.chunking import FixedSizeChunker
from rag.models import ChunkMetadata


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def docs_dir(tmp_path):
    """Create a small markdown corpus in a temp directory."""
    (tmp_path / "README.md").write_text("# README\nSome intro text.\n## Section\nMore text here.")
    (tmp_path / "GUIDE.md").write_text("# Guide\nInstallation instructions.\n## Usage\nUse it like this.")
    return str(tmp_path)


@pytest.fixture()
def mock_embedder():
    """EmbeddingGenerator that returns deterministic random embeddings."""
    gen = MagicMock()
    def fake_generate(texts):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), 1536)).astype(np.float32)
    gen.generate.side_effect = fake_generate
    return gen


# ── _compute_stats ─────────────────────────────────────────────────────────────

class TestComputeStats:
    def _make_chunk(self, text: str) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_id="x_0", source="x.md", title="x",
            section="", strategy="fixed",
            char_start=0, char_end=len(text), text=text,
        )

    def test_basic(self):
        chunks = [self._make_chunk("a" * n) for n in [100, 200, 300]]
        stats = _compute_stats("fixed", chunks)
        assert stats.strategy == "fixed"
        assert stats.total_chunks == 3
        assert stats.min_chars == 100
        assert stats.max_chars == 300
        assert stats.avg_chars == 200.0

    def test_empty_chunks(self):
        stats = _compute_stats("fixed", [])
        assert stats.total_chunks == 0
        assert stats.avg_chars == 0.0

    def test_sources_count(self):
        chunks = [
            ChunkMetadata(chunk_id="a_0", source="a.md", title="a", section="",
                          strategy="fixed", char_start=0, char_end=5, text="hello"),
            ChunkMetadata(chunk_id="b_0", source="b.md", title="b", section="",
                          strategy="fixed", char_start=0, char_end=5, text="world"),
        ]
        stats = _compute_stats("fixed", chunks)
        assert stats.sources == 2


# ── IndexingPipeline ───────────────────────────────────────────────────────────

class TestIndexingPipeline:
    def test_run_fixed(self, docs_dir, mock_embedder, tmp_path):
        pipeline = IndexingPipeline(embedding_generator=mock_embedder)
        stats = pipeline.run(docs_dir=docs_dir, strategy="fixed",
                             output_path=str(tmp_path / "fixed"))
        assert stats.strategy == "fixed"
        assert stats.total_chunks > 0
        assert stats.sources == 2
        assert Path(str(tmp_path / "fixed.faiss")).exists()
        assert Path(str(tmp_path / "fixed_metadata.json")).exists()

    def test_run_structure(self, docs_dir, mock_embedder, tmp_path):
        pipeline = IndexingPipeline(embedding_generator=mock_embedder)
        stats = pipeline.run(docs_dir=docs_dir, strategy="structure",
                             output_path=str(tmp_path / "structure"))
        assert stats.strategy == "structure"
        assert stats.total_chunks > 0

    def test_unknown_strategy_raises(self, docs_dir, mock_embedder, tmp_path):
        pipeline = IndexingPipeline(embedding_generator=mock_embedder)
        with pytest.raises(ValueError, match="Unknown strategy"):
            pipeline.run(docs_dir=docs_dir, strategy="magic",
                         output_path=str(tmp_path / "x"))

    def test_no_md_files_raises(self, mock_embedder, tmp_path):
        (tmp_path / "data").mkdir()
        pipeline = IndexingPipeline(embedding_generator=mock_embedder)
        with pytest.raises(FileNotFoundError):
            pipeline.run(docs_dir=str(tmp_path / "data"), strategy="fixed",
                         output_path=str(tmp_path / "out"))

    def test_embedder_called_once(self, docs_dir, mock_embedder, tmp_path):
        pipeline = IndexingPipeline(embedding_generator=mock_embedder)
        pipeline.run(docs_dir=docs_dir, strategy="fixed",
                     output_path=str(tmp_path / "idx"))
        mock_embedder.generate.assert_called_once()

    def test_stats_fields_present(self, docs_dir, mock_embedder, tmp_path):
        pipeline = IndexingPipeline(embedding_generator=mock_embedder)
        stats = pipeline.run(docs_dir=docs_dir, strategy="fixed",
                             output_path=str(tmp_path / "idx"))
        assert isinstance(stats, IndexStats)
        assert stats.min_chars > 0
        assert stats.max_chars >= stats.min_chars
        assert stats.std_chars >= 0


# ── run_comparison ─────────────────────────────────────────────────────────────

class TestRunComparison:
    def test_produces_report(self, docs_dir, mock_embedder, tmp_path):
        from rag.compare import run_comparison
        report = run_comparison(docs_dir=docs_dir, output_dir=str(tmp_path),
                                embedding_generator=mock_embedder)
        assert "fixed" in report
        assert "structure" in report
        assert report["fixed"]["strategy"] == "fixed"
        assert report["structure"]["strategy"] == "structure"

    def test_saves_json(self, docs_dir, mock_embedder, tmp_path):
        from rag.compare import run_comparison
        run_comparison(docs_dir=docs_dir, output_dir=str(tmp_path),
                       embedding_generator=mock_embedder)
        report_path = tmp_path / "compare_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "fixed" in data
        assert "structure" in data

    def test_creates_both_indexes(self, docs_dir, mock_embedder, tmp_path):
        from rag.compare import run_comparison
        run_comparison(docs_dir=docs_dir, output_dir=str(tmp_path),
                       embedding_generator=mock_embedder)
        assert (tmp_path / "fixed.faiss").exists()
        assert (tmp_path / "structure.faiss").exists()
