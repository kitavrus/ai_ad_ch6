"""FAISS-based vector index with JSON metadata sidecar."""

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .models import ChunkMetadata


class FAISSIndex:
    """Exact L2 nearest-neighbor index over chunk embeddings."""

    def __init__(self):
        self._index: faiss.IndexFlatL2 | None = None
        self._chunks: List[ChunkMetadata] = []

    def build(self, embeddings: np.ndarray, chunks: List[ChunkMetadata]) -> None:
        """Build the index from embeddings and corresponding chunk metadata."""
        if len(embeddings) == 0:
            raise ValueError("Cannot build index with zero embeddings.")
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings.astype(np.float32))
        self._chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[ChunkMetadata]:
        """Return top_k most similar chunks to the query embedding."""
        if self._index is None:
            raise RuntimeError("Index is empty. Call build() or load() first.")
        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._chunks))
        _, indices = self._index.search(query, k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    def search_with_scores(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Return top_k chunks with their raw L2 distances."""
        if self._index is None:
            raise RuntimeError("Index is empty. Call build() or load() first.")
        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._chunks))
        distances, indices = self._index.search(query, k)
        return [
            (self._chunks[i], float(distances[0][pos]))
            for pos, i in enumerate(indices[0])
            if i < len(self._chunks)
        ]

    def save(self, path: str) -> None:
        """Save the FAISS index and metadata sidecar to disk."""
        faiss_path = Path(path).with_suffix(".faiss")
        meta_path = Path(path + "_metadata.json")
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self._chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        """Load a previously saved index from disk."""
        faiss_path = Path(path).with_suffix(".faiss")
        meta_path = Path(path + "_metadata.json")
        instance = cls()
        instance._index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        instance._chunks = [ChunkMetadata(**item) for item in raw]
        return instance
