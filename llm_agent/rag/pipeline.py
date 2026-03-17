"""IndexingPipeline — orchestrates corpus → chunks → embeddings → FAISS index."""

import math
import sys
from pathlib import Path
from typing import List


def _print_bar(label: str, current: int, total: int, detail: str = "") -> None:
    width = 30
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    detail_short = detail[:30].ljust(30) if detail else ""
    print(f"\r{label}: [{bar}] {pct:3d}% ({current}/{total}) {detail_short}", end="", flush=True)

import numpy as np

from .chunking import FixedSizeChunker, StructureChunker
from .embeddings import EmbeddingGenerator
from .index import FAISSIndex
from .models import ChunkMetadata, IndexStats


def _compute_stats(strategy: str, chunks: List[ChunkMetadata]) -> IndexStats:
    lengths = [len(c.text) for c in chunks]
    if not lengths:
        return IndexStats(
            strategy=strategy,
            total_chunks=0,
            avg_chars=0.0,
            min_chars=0,
            max_chars=0,
            std_chars=0.0,
            sources=0,
        )
    avg = sum(lengths) / len(lengths)
    variance = sum((x - avg) ** 2 for x in lengths) / len(lengths)
    sources = len({c.source for c in chunks})
    return IndexStats(
        strategy=strategy,
        total_chunks=len(chunks),
        avg_chars=round(avg, 1),
        min_chars=min(lengths),
        max_chars=max(lengths),
        std_chars=round(math.sqrt(variance), 1),
        sources=sources,
    )


class IndexingPipeline:
    """Runs the full indexing pipeline for a given strategy."""

    def __init__(self, embedding_generator: EmbeddingGenerator | None = None):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def run(self, docs_dir: str, strategy: str, output_path: str) -> IndexStats:
        """
        1. Collect .md files from docs_dir
        2. Read text
        3. Chunk with chosen strategy
        4. Generate embeddings
        5. Build + save FAISS index
        6. Return stats
        """
        md_files = sorted(Path(docs_dir).rglob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown files found in {docs_dir}")

        all_chunks: List[ChunkMetadata] = []
        chunker = self._make_chunker(strategy)

        total_files = len(md_files)
        for i, md_file in enumerate(md_files, 1):
            _print_bar("Chunking", i, total_files, md_file.name)
            text = md_file.read_text(encoding="utf-8")
            title = md_file.stem
            source = str(md_file)
            chunks = chunker.chunk(text, source=source, title=title)
            all_chunks.extend(chunks)
        print()  # newline after progress bar

        texts = [c.text for c in all_chunks]
        embeddings: np.ndarray = self.embedding_generator.generate(texts, progress=True)

        idx = FAISSIndex()
        idx.build(embeddings, all_chunks)
        idx.save(output_path)

        return _compute_stats(strategy, all_chunks)

    def _make_chunker(self, strategy: str):
        if strategy == "fixed":
            return FixedSizeChunker()
        elif strategy == "structure":
            return StructureChunker()
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'fixed' or 'structure'.")
