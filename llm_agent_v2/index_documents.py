#!/usr/bin/env python3
"""CLI entry point for the RAG document indexing pipeline.

Usage:
    python index_documents.py                    # index with both strategies
    python index_documents.py --strategy fixed   # only fixed
    python index_documents.py --compare          # show comparison report
    python index_documents.py --query "memory"   # search the index
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the llm_agent_v2 package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = str(Path(__file__).parent)
OUTPUT_DIR = str(Path(__file__).parent / "rag_index")
ROOT_README = str(Path(__file__).parent.parent / "README.md")


def cmd_compare() -> None:
    from rag.compare import run_comparison
    run_comparison(docs_dir=DOCS_DIR, output_dir=OUTPUT_DIR)


def cmd_index(strategy: str) -> None:
    from rag.pipeline import IndexingPipeline
    pipeline = IndexingPipeline()
    output_path = str(Path(OUTPUT_DIR) / strategy)
    extra = [ROOT_README] if Path(ROOT_README).exists() else []
    print(f"Indexing with strategy: {strategy}")
    stats = pipeline.run(docs_dir=DOCS_DIR, strategy=strategy, output_path=output_path, extra_files=extra)
    print(f"Done. {stats.total_chunks} chunks from {stats.sources} files.")
    print(f"  avg={stats.avg_chars:.0f} chars, min={stats.min_chars}, max={stats.max_chars}, std={stats.std_chars:.0f}")
    print(f"Index saved to {output_path}.faiss")


def cmd_query(query: str, strategy: str = "fixed", top_k: int = 5) -> None:
    from rag.embeddings import EmbeddingGenerator
    from rag.index import FAISSIndex

    index_path = str(Path(OUTPUT_DIR) / strategy)
    faiss_path = index_path + ".faiss"
    if not Path(faiss_path).exists():
        print(f"Index not found at {faiss_path}. Run indexing first.")
        sys.exit(1)

    print(f"Loading index from {faiss_path} ...")
    idx = FAISSIndex.load(index_path)

    gen = EmbeddingGenerator()
    q_emb = gen.generate([query])
    results = idx.search(q_emb[0], top_k=top_k)

    print(f"\nTop {len(results)} results for query: {query!r}\n")
    for i, chunk in enumerate(results, 1):
        section = f" [{chunk.section}]" if chunk.section else ""
        print(f"{i}. {chunk.title}{section} ({chunk.source})")
        preview = chunk.text[:200].replace("\n", " ")
        print(f"   {preview}...\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Document Indexing Pipeline")
    parser.add_argument("--strategy", choices=["fixed", "structure"], default=None,
                        help="Chunking strategy (default: both)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both strategies and print comparison report")
    parser.add_argument("--query", type=str, default=None,
                        help="Search the index with a natural language query")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return for --query (default: 5)")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.compare:
        cmd_compare()
    elif args.query:
        strategy = args.strategy or "fixed"
        cmd_query(args.query, strategy=strategy, top_k=args.top_k)
    elif args.strategy:
        cmd_index(args.strategy)
    else:
        # Default: index with both strategies
        cmd_index("fixed")
        cmd_index("structure")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано.")
        sys.exit(1)
