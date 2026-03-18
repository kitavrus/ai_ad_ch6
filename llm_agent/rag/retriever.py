"""RAG retriever — lazy-loads FAISS indexes and searches by query."""

from pathlib import Path
from typing import Dict, List

from llm_agent.rag.embeddings import EmbeddingGenerator
from llm_agent.rag.index import FAISSIndex
from llm_agent.rag.models import ChunkMetadata

_DEFAULT_INDEX_DIR = "rag_index"


class RAGRetriever:
    """Lazy-loading retriever backed by per-strategy FAISS indexes."""

    def __init__(self, index_dir: str = _DEFAULT_INDEX_DIR):
        self.index_dir = index_dir
        self._indexes: Dict[str, FAISSIndex] = {}
        self._embedder = EmbeddingGenerator()

    def _load_index(self, strategy: str) -> FAISSIndex:
        if strategy not in self._indexes:
            path = str(Path(self.index_dir) / strategy)
            self._indexes[strategy] = FAISSIndex.load(path)
        return self._indexes[strategy]

    def search(self, query: str, strategy: str = "structure", top_k: int = 3) -> List[ChunkMetadata]:
        """Search the index for the top_k most relevant chunks.

        Args:
            query: User query text.
            strategy: Index strategy to search ("fixed" | "structure").
            top_k: Number of chunks to return.

        Returns:
            List of ChunkMetadata ordered by relevance.
        """
        index = self._load_index(strategy)
        query_embedding = self._embedder.generate([query])[0]
        return index.search(query_embedding, top_k)
