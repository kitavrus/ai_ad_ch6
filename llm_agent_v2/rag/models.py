"""Pydantic models for the RAG pipeline."""

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    chunk_id: str       # "{source_slug}_{chunk_index}"
    source: str         # file path relative to project root
    title: str          # filename without extension
    section: str        # nearest heading above, or "" for fixed-size
    strategy: str       # "fixed" | "structure"
    char_start: int
    char_end: int
    text: str


class IndexStats(BaseModel):
    strategy: str
    total_chunks: int
    avg_chars: float
    min_chars: int
    max_chars: int
    std_chars: float
    sources: int


class RetrievalResult(BaseModel):
    chunk: ChunkMetadata
    score: float    # 0..1, higher = better; 1/(1+L2_distance)
    distance: float # raw L2 from FAISS
    query: str      # query used (may be rewritten)
