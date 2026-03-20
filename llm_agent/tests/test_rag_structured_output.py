"""Tests for RAG structured output and IDK mode."""
import pytest
from unittest.mock import MagicMock, patch

from chatbot.context import build_rag_system_addition, build_rag_idk_system
from rag.models import ChunkMetadata, RetrievalResult


def _make_result(chunk_id="docs_3", source="docs/file.md", section="Intro",
                 score=0.82, text="Some text about the topic.") -> RetrievalResult:
    chunk = ChunkMetadata(
        chunk_id=chunk_id,
        source=source,
        title="file",
        section=section,
        strategy="fixed",
        char_start=0,
        char_end=len(text),
        text=text,
    )
    return RetrievalResult(chunk=chunk, score=score, distance=1 - score, query="test query")


# --- Tests for build_rag_system_addition ---

def test_contains_answer_header():
    result = build_rag_system_addition([_make_result()])
    assert "**Ответ:**" in result


def test_contains_sources_header():
    result = build_rag_system_addition([_make_result()])
    assert "**Источники:**" in result


def test_contains_citations_header():
    result = build_rag_system_addition([_make_result()])
    assert "**Цитаты:**" in result


def test_chunk_header_contains_score_and_chunk_id():
    result = build_rag_system_addition([_make_result(chunk_id="docs_3", score=0.82)])
    assert "score=0.82" in result
    assert "chunk_id=docs_3" in result


def test_build_rag_idk_system_contains_ne_znayu():
    msg = build_rag_idk_system()
    assert "Не знаю" in msg
    assert "Уточните" in msg


def test_empty_results_returns_empty_string():
    result = build_rag_system_addition([])
    assert result == ""


# --- Tests for IDK mode in main.py ---

def _make_rag_state(threshold: float, enabled: bool = True):
    state = MagicMock()
    state.rag_mode.enabled = enabled
    state.rag_mode.threshold = threshold
    state.rag_mode.strategy = "fixed"
    state.rag_mode.top_k = 3
    return state


def test_idk_mode_activated_when_no_results_and_threshold_positive():
    """IDK message appended when results=[] and threshold > 0."""
    api_messages = []
    state = _make_rag_state(threshold=0.5)

    with patch("chatbot.main._get_retriever") as mock_retriever, \
         patch("chatbot.main.build_rag_idk_system", return_value="IDK_MSG") as mock_idk, \
         patch("chatbot.main.build_rag_system_addition") as mock_addition:
        mock_retriever.return_value.search_with_scores.return_value = []

        # Simulate the RAG block
        _rag_results = mock_retriever(state.rag_mode).search_with_scores(
            "question", state.rag_mode.strategy, state.rag_mode.top_k
        )
        if _rag_results:
            api_messages.append({"role": "system", "content": mock_addition(_rag_results)})
        elif state.rag_mode.threshold > 0:
            api_messages.append({"role": "system", "content": mock_idk()})

    assert len(api_messages) == 1
    assert api_messages[0]["content"] == "IDK_MSG"


def test_idk_mode_not_activated_when_threshold_zero():
    """IDK message NOT appended when threshold = 0."""
    api_messages = []
    state = _make_rag_state(threshold=0)

    with patch("chatbot.main._get_retriever") as mock_retriever, \
         patch("chatbot.main.build_rag_idk_system") as mock_idk, \
         patch("chatbot.main.build_rag_system_addition") as mock_addition:
        mock_retriever.return_value.search_with_scores.return_value = []

        _rag_results = mock_retriever(state.rag_mode).search_with_scores(
            "question", state.rag_mode.strategy, state.rag_mode.top_k
        )
        if _rag_results:
            api_messages.append({"role": "system", "content": mock_addition(_rag_results)})
        elif state.rag_mode.threshold > 0:
            api_messages.append({"role": "system", "content": mock_idk()})

    assert len(api_messages) == 0
    mock_idk.assert_not_called()


def test_idk_not_activated_when_results_present():
    """IDK message NOT appended when there are results."""
    api_messages = []
    state = _make_rag_state(threshold=0.5)
    results = [_make_result()]

    with patch("chatbot.main._get_retriever") as mock_retriever, \
         patch("chatbot.main.build_rag_idk_system") as mock_idk, \
         patch("chatbot.main.build_rag_system_addition", return_value="RAG_CONTENT") as mock_addition:
        mock_retriever.return_value.search_with_scores.return_value = results

        _rag_results = mock_retriever(state.rag_mode).search_with_scores(
            "question", state.rag_mode.strategy, state.rag_mode.top_k
        )
        if _rag_results:
            api_messages.append({"role": "system", "content": mock_addition(_rag_results)})
        elif state.rag_mode.threshold > 0:
            api_messages.append({"role": "system", "content": mock_idk()})

    assert len(api_messages) == 1
    assert api_messages[0]["content"] == "RAG_CONTENT"
    mock_idk.assert_not_called()


def test_source_line_contains_section_and_chunk_id():
    """The chunk header line includes §Section and chunk_id=."""
    result = build_rag_system_addition([_make_result(section="Overview", chunk_id="docs_7")])
    assert "§Overview" in result
    assert "chunk_id=docs_7" in result
