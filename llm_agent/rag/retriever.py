"""RAG retriever — lazy-loads FAISS indexes and searches by query."""

from pathlib import Path
from typing import Dict, List, Optional

from llm_agent.rag.embeddings import EmbeddingGenerator
from llm_agent.rag.index import FAISSIndex
from llm_agent.rag.models import ChunkMetadata, RetrievalResult
from llm_agent.rag.reranker import RelevanceFilter

_DEFAULT_INDEX_DIR = "rag_index"


class RAGRetriever:
    """Lazy-loading retriever backed by per-strategy FAISS indexes."""

    def __init__(
        self,
        index_dir: str = _DEFAULT_INDEX_DIR,
        relevance_threshold: float = 0.0,
        top_k_before: Optional[int] = None,
        top_k_after: Optional[int] = None,
        rewrite_query: bool = False,
        rewrite_multi: bool = False,
        rewrite_n: int = 3,
        llm_client=None,
    ):
        self.index_dir = index_dir
        self.relevance_threshold = relevance_threshold
        self.top_k_before = top_k_before
        self.top_k_after = top_k_after
        self.rewrite_query = rewrite_query
        self.rewrite_multi = rewrite_multi
        self.rewrite_n = rewrite_n
        self._indexes: Dict[str, FAISSIndex] = {}
        self._embedder = EmbeddingGenerator()
        self._rewriter = None
        if rewrite_query or rewrite_multi:
            from llm_agent.rag.query_rewrite import QueryRewriter
            self._rewriter = QueryRewriter(client=llm_client)

    def _load_index(self, strategy: str) -> FAISSIndex:
        if strategy not in self._indexes:
            path = str(Path(self.index_dir) / strategy)
            self._indexes[strategy] = FAISSIndex.load(path)
        return self._indexes[strategy]

    def search_with_scores(
        self, query: str, strategy: str = "structure", top_k: int = 3
    ) -> List[RetrievalResult]:
        """Search with relevance scores; supports query rewriting and filtering.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        # 1. Determine queries to run
        queries: List[str] = [query]
        if self._rewriter is not None:
            if self.rewrite_multi:
                queries = self._rewriter.rewrite_multi(query, n=self.rewrite_n)
            elif self.rewrite_query:
                queries = [self._rewriter.rewrite(query)]

        # 2. Determine how many to fetch from FAISS
        fetch_k = self.top_k_before if self.top_k_before is not None else top_k

        # 3. Fetch and merge — keep lowest distance per chunk_id
        index = self._load_index(strategy)
        best: Dict[str, tuple] = {}  # chunk_id -> (chunk, distance, query_used)
        for q in queries:
            embedding = self._embedder.generate([q])[0]
            scored = index.search_with_scores(embedding, top_k=fetch_k)
            for chunk, distance in scored:
                cid = chunk.chunk_id
                if cid not in best or distance < best[cid][1]:
                    best[cid] = (chunk, distance, q)

        scored_merged = [(chunk, dist) for chunk, dist, _ in best.values()]
        query_map = {chunk.chunk_id: q for chunk, _, q in best.values()}

        # 4. Apply relevance filter
        after_k = self.top_k_after if self.top_k_after is not None else top_k
        rf = RelevanceFilter(threshold=self.relevance_threshold, top_k_after=after_k)
        results = rf.filter(scored_merged, query=query)

        # Patch query field per-chunk to show which query variant retrieved it
        for r in results:
            r.query = query_map.get(r.chunk.chunk_id, query)

        return results

    def search(self, query: str, strategy: str = "structure", top_k: int = 3) -> List[ChunkMetadata]:
        """Search the index for the top_k most relevant chunks.

        Backward-compatible: returns List[ChunkMetadata].
        """
        return [r.chunk for r in self.search_with_scores(query, strategy, top_k)]
