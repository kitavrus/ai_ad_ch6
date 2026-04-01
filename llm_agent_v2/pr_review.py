"""PR Review pipeline: RAG + LLM → structured Markdown review."""

from __future__ import annotations

import re
import asyncio
from pathlib import Path
from typing import Optional

from .config import SessionConfig
from .llm_client import LLMClient

MAX_DIFF_BYTES = 80_000
RAG_INDEX_PATH = Path(__file__).parent / "rag_index" / "fixed.faiss"

SYSTEM_PROMPT = """\
You are an expert code reviewer. Analyze the provided git diff carefully.
Return a structured Markdown review with EXACTLY these three sections (keep the headers):

## 🐛 Potential Bugs
List specific bugs, null-pointer risks, off-by-one errors, unhandled exceptions,
race conditions, or incorrect logic. Cite file names and line numbers from the diff.
If none found, write: No issues found.

## 🏗️ Architectural Issues
List design problems: tight coupling, violated SOLID principles, missing abstractions,
inconsistent patterns compared to existing codebase context.
If none found, write: No issues found.

## 💡 Recommendations
Concrete, actionable improvements: naming, tests, edge cases not covered, performance.
If none found, write: No recommendations.

Be specific and concise. Do NOT repeat the diff back. Focus on problems, not praise.\
"""


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def _extract_rag_queries(diff: str, changed_files: list[str]) -> list[str]:
    """Extract up to 5 meaningful search queries from the diff."""
    queries: list[str] = []

    # File module names (e.g. "foo/bar/baz.py" → "baz")
    for f in changed_files[:3]:
        stem = Path(f).stem
        if stem and stem not in ("__init__", "test", "tests"):
            queries.append(stem)

    # Function/class names defined or modified in the diff
    for m in re.finditer(r"^\+.*(?:def|class)\s+(\w+)", diff, re.MULTILINE):
        name = m.group(1)
        if name not in queries:
            queries.append(name)
        if len(queries) >= 5:
            break

    return queries[:5]


def _build_rag_context(diff: str, changed_files: list[str]) -> str:
    """Query existing RAG index and return formatted context string."""
    if not RAG_INDEX_PATH.exists():
        return ""

    try:
        from .rag.retriever import RAGRetriever

        retriever = RAGRetriever(str(RAG_INDEX_PATH.with_suffix("")))
        queries = _extract_rag_queries(diff, changed_files)
        if not queries:
            return ""

        seen: set[str] = set()
        chunks: list[str] = []
        for q in queries:
            for result in retriever.search(q, top_k=2):
                text = result.chunk.text.strip()
                if text and text not in seen:
                    seen.add(text)
                    source = result.chunk.metadata.source if result.chunk.metadata else "?"
                    chunks.append(f"[{source}]\n{text}")

        if not chunks:
            return ""
        return "## Codebase context (RAG)\n\n" + "\n\n---\n\n".join(chunks[:6])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_review_messages(diff: str, changed_files: list[str]) -> list[dict]:
    """Build the messages list for the LLM call."""
    if len(diff.encode()) > MAX_DIFF_BYTES:
        diff = diff.encode()[:MAX_DIFF_BYTES].decode(errors="ignore")
        diff += "\n\n[...diff truncated at 80KB...]"

    rag_context = _build_rag_context(diff, changed_files)

    parts = ["Here is the git diff to review:\n\n```diff\n", diff, "\n```"]
    if rag_context:
        parts += ["\n\n", rag_context]
    if changed_files:
        parts += ["\n\nChanged files: ", ", ".join(changed_files)]

    return [{"role": "user", "content": "".join(parts)}]


# ---------------------------------------------------------------------------
# TODO / blocking-issue detection
# ---------------------------------------------------------------------------

_BLOCKING_PATTERN = re.compile(
    r"^\+.*\b(TODO|FIXME|HACK|XXX)\b", re.MULTILINE | re.IGNORECASE
)


def detect_blocking_issues(diff: str) -> list[str]:
    """Return list of added lines containing TODO/FIXME/HACK/XXX."""
    return [m.group(0).lstrip("+").strip() for m in _BLOCKING_PATTERN.finditer(diff)]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def format_comment(review_text: str, pr_number: int, blocking: list[str]) -> str:
    """Wrap review text in a GitHub comment with a header."""
    parts: list[str] = []

    if blocking:
        parts.append("## ❌ Blocking Issues — Review Failed\n\n")
        parts.append(
            "The following **TODO / FIXME / HACK** markers were found in the diff "
            "and must be resolved before merging:\n\n"
        )
        for line in blocking:
            parts.append(f"- `{line}`\n")
        parts.append("\n---\n\n")

    parts.append("## 🤖 AI Code Review\n\n")
    parts.append(review_text.strip())
    parts.append("\n\n---\n*Generated by AI Code Reviewer · llm_agent_v2*")
    return "".join(parts)


def run_review_sync(
    diff: str,
    changed_files: list[str],
    config: Optional[SessionConfig] = None,
) -> tuple[str, list[str]]:
    """Run the full review pipeline synchronously.

    Returns (review_text, blocking_issues).
    """
    if not diff.strip():
        return "No changes to review.", []

    blocking = detect_blocking_issues(diff)

    review_config = SessionConfig(
        **(config.model_dump() if config else {}),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=2048,
        temperature=0.2,
    )
    llm = LLMClient(config=review_config)
    messages = build_review_messages(diff, changed_files)
    review_text = llm.chat(messages)
    return review_text, blocking


async def run_review(
    diff: str,
    changed_files: list[str],
    config: Optional[SessionConfig] = None,
) -> tuple[str, list[str]]:
    """Async wrapper around run_review_sync."""
    return await asyncio.to_thread(run_review_sync, diff, changed_files, config)
