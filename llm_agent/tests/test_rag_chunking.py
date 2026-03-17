"""Tests for rag/chunking.py — no network calls needed."""

import pytest

from rag.chunking import FixedSizeChunker, StructureChunker, _source_slug


# ── helpers ────────────────────────────────────────────────────────────────────

SIMPLE_TEXT = "A" * 600  # 600 characters

MARKDOWN_TEXT = """\
# Introduction
Some intro text here.

## Installation
Run pip install.

### Sub-section
Details about installation.

# Usage
How to use the tool.
"""

LONG_SECTION_TEXT = "# BigSection\n" + "\n\n".join(["word " * 50] * 12)  # >2000 chars, blank-line separated paragraphs


# ── _source_slug ───────────────────────────────────────────────────────────────

def test_source_slug_basic():
    assert _source_slug("docs/my-file.md") == "my_file"


def test_source_slug_underscores():
    assert _source_slug("some_file.md") == "some_file"


# ── FixedSizeChunker ───────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_produces_chunks(self):
        chunker = FixedSizeChunker(chunk_size=512, overlap=100)
        chunks = chunker.chunk(SIMPLE_TEXT, source="test.md", title="test")
        assert len(chunks) >= 1

    def test_chunk_ids_unique(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(SIMPLE_TEXT, source="test.md", title="test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_strategy_is_fixed(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("hello world " * 50, source="x.md", title="x")
        assert all(c.strategy == "fixed" for c in chunks)

    def test_section_is_empty(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("hello world " * 50, source="x.md", title="x")
        assert all(c.section == "" for c in chunks)

    def test_chunk_size_respected(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        text = "x" * 350
        chunks = chunker.chunk(text, source="t.md", title="t")
        # last chunk may be smaller
        assert all(len(c.text) <= 100 for c in chunks)

    def test_overlap_creates_extra_chunks(self):
        chunker_no_overlap = FixedSizeChunker(chunk_size=200, overlap=0)
        chunker_overlap = FixedSizeChunker(chunk_size=200, overlap=50)
        text = "y" * 600
        assert len(chunker_overlap.chunk(text, "f.md", "f")) >= len(
            chunker_no_overlap.chunk(text, "f.md", "f")
        )

    def test_char_start_end(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        text = "a" * 250
        chunks = chunker.chunk(text, source="f.md", title="f")
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 100

    def test_title_and_source_preserved(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("hello " * 100, source="path/to/doc.md", title="doc")
        assert all(c.source == "path/to/doc.md" for c in chunks)
        assert all(c.title == "doc" for c in chunks)

    def test_empty_text_returns_no_chunks(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk("   \n  ", source="e.md", title="e") == []


# ── StructureChunker ──────────────────────────────────────────────────────────

class TestStructureChunker:
    def test_produces_chunks(self):
        chunker = StructureChunker()
        chunks = chunker.chunk(MARKDOWN_TEXT, source="test.md", title="test")
        assert len(chunks) >= 1

    def test_strategy_is_structure(self):
        chunker = StructureChunker()
        chunks = chunker.chunk(MARKDOWN_TEXT, source="test.md", title="test")
        assert all(c.strategy == "structure" for c in chunks)

    def test_sections_captured(self):
        chunker = StructureChunker()
        chunks = chunker.chunk(MARKDOWN_TEXT, source="test.md", title="test")
        sections = [c.section for c in chunks]
        assert any("Introduction" in s for s in sections)
        assert any("Installation" in s for s in sections)
        assert any("Usage" in s for s in sections)

    def test_large_section_sub_split(self):
        chunker = StructureChunker(min_chunk_size=200, max_section_size=500)
        chunks = chunker.chunk(LONG_SECTION_TEXT, source="big.md", title="big")
        # Should produce multiple chunks since section > max_section_size
        assert len(chunks) > 1

    def test_unique_chunk_ids(self):
        chunker = StructureChunker()
        chunks = chunker.chunk(MARKDOWN_TEXT, source="test.md", title="test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_heading_text(self):
        chunker = StructureChunker()
        plain = "Just plain text without any headings.\n" * 10
        chunks = chunker.chunk(plain, source="plain.md", title="plain")
        assert len(chunks) >= 1
        assert all(c.section == "" for c in chunks)

    def test_content_complete(self):
        chunker = StructureChunker(min_chunk_size=10)
        chunks = chunker.chunk(MARKDOWN_TEXT, source="test.md", title="test")
        combined = "".join(c.text for c in chunks)
        # All non-whitespace content should appear in some chunk
        for word in ["Introduction", "Installation", "Usage"]:
            assert word in combined

    def test_empty_text_returns_no_chunks(self):
        chunker = StructureChunker()
        chunks = chunker.chunk("", source="e.md", title="e")
        assert chunks == []

    def test_title_and_source_preserved(self):
        chunker = StructureChunker()
        chunks = chunker.chunk(MARKDOWN_TEXT, source="docs/guide.md", title="guide")
        assert all(c.source == "docs/guide.md" for c in chunks)
        assert all(c.title == "guide" for c in chunks)
