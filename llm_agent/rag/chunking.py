"""Chunking strategies for document indexing."""

import re
from pathlib import Path
from typing import List

from .models import ChunkMetadata


def _source_slug(source: str) -> str:
    """Convert a file path to a slug for chunk IDs."""
    return Path(source).stem.replace(" ", "_").replace("-", "_").lower()


class FixedSizeChunker:
    """Splits text by character count with sliding overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str, title: str) -> List[ChunkMetadata]:
        chunks: List[ChunkMetadata] = []
        slug = _source_slug(source)
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(ChunkMetadata(
                    chunk_id=f"{slug}_{idx}",
                    source=source,
                    title=title,
                    section="",
                    strategy="fixed",
                    char_start=start,
                    char_end=end,
                    text=chunk_text,
                ))
                idx += 1
            if end >= len(text):
                break
            start = end - self.overlap
        return chunks


class StructureChunker:
    """Splits markdown by headings, with fallback splitting by blank lines for large sections."""

    HEADING_RE = re.compile(r'^#{1,6}\s', re.MULTILINE)

    def __init__(self, min_chunk_size: int = 200, max_section_size: int = 2000):
        self.min_chunk_size = min_chunk_size
        self.max_section_size = max_section_size

    def chunk(self, text: str, source: str, title: str) -> List[ChunkMetadata]:
        slug = _source_slug(source)
        sections = self._split_by_headings(text)
        chunks: List[ChunkMetadata] = []
        idx = 0
        char_cursor = 0

        for section_heading, section_text in sections:
            section_start = text.find(section_text, char_cursor)
            if section_start == -1:
                section_start = char_cursor
            char_cursor = section_start

            if len(section_text) > self.max_section_size:
                sub_chunks = self._split_by_blank_lines(section_text, self.min_chunk_size)
            else:
                sub_chunks = [section_text]

            sub_cursor = section_start
            for sub in sub_chunks:
                sub_start = text.find(sub, sub_cursor)
                if sub_start == -1:
                    sub_start = sub_cursor
                sub_end = sub_start + len(sub)
                if sub.strip():
                    chunks.append(ChunkMetadata(
                        chunk_id=f"{slug}_{idx}",
                        source=source,
                        title=title,
                        section=section_heading,
                        strategy="structure",
                        char_start=sub_start,
                        char_end=sub_end,
                        text=sub,
                    ))
                    idx += 1
                sub_cursor = sub_end

            char_cursor = section_start + len(section_text)

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple]:
        """Return list of (heading_text, section_content) tuples."""
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            return [("", text)]

        sections = []
        # Content before first heading
        if matches[0].start() > 0:
            pre = text[:matches[0].start()]
            if pre.strip():
                sections.append(("", pre))

        for i, m in enumerate(matches):
            heading_line_end = text.find('\n', m.start())
            if heading_line_end == -1:
                heading_line_end = len(text)
            heading = text[m.start():heading_line_end].strip()
            content_start = heading_line_end + 1
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[m.start():content_end]
            sections.append((heading, section_text))

        return sections

    def _split_by_blank_lines(self, text: str, min_size: int) -> List[str]:
        """Split text by blank lines, merging small pieces."""
        paragraphs = re.split(r'\n\s*\n', text)
        result = []
        current = ""
        for para in paragraphs:
            if not para.strip():
                continue
            if len(current) + len(para) < min_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    result.append(current)
                current = para.strip()
        if current:
            result.append(current)
        return result if result else [text]
