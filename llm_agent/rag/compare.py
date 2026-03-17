"""Comparison report for fixed vs structure chunking strategies."""

import json
from pathlib import Path

from .models import IndexStats
from .pipeline import IndexingPipeline


def run_comparison(docs_dir: str, output_dir: str, embedding_generator=None) -> dict:
    """Run both strategies, print a comparison table, and save compare_report.json."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipeline = IndexingPipeline(embedding_generator=embedding_generator)

    print("Running fixed-size chunking...")
    fixed_stats = pipeline.run(
        docs_dir=docs_dir,
        strategy="fixed",
        output_path=str(Path(output_dir) / "fixed"),
    )

    print("Running structure-based chunking...")
    struct_stats = pipeline.run(
        docs_dir=docs_dir,
        strategy="structure",
        output_path=str(Path(output_dir) / "structure"),
    )

    _print_table(fixed_stats, struct_stats)

    report = {
        "fixed": fixed_stats.model_dump(),
        "structure": struct_stats.model_dump(),
    }
    report_path = Path(output_dir) / "compare_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    return report


def _print_table(fixed: IndexStats, struct: IndexStats) -> None:
    header = f"{'Strategy':<14} {'Chunks':>7} {'Avg chars':>10} {'Min':>6} {'Max':>6} {'Std':>8} {'Sources':>8}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for s in (fixed, struct):
        print(
            f"{s.strategy:<14} {s.total_chunks:>7} {s.avg_chars:>10.1f} "
            f"{s.min_chars:>6} {s.max_chars:>6} {s.std_chars:>8.1f} {s.sources:>8}"
        )
    print(separator + "\n")
