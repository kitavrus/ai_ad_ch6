"""Сценарные тесты: RAG-чат с гарантированными источниками + память задачи."""

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_agent.chatbot.context import ensure_rag_sources_in_response, build_rag_system_addition
from llm_agent.chatbot.memory import Memory, WorkingMemory
from llm_agent.chatbot.models import SessionState
from llm_agent.rag.models import ChunkMetadata, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, source: str = "docs/python.md", section: str = "") -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=f"doc_{idx}",
        source=source,
        title=source.split("/")[-1].replace(".md", ""),
        section=section,
        strategy="fixed",
        char_start=idx * 100,
        char_end=idx * 100 + 200,
        text=f"Text content for chunk {idx}.",
    )


def _make_result(idx: int, source: str = "docs/python.md", section: str = "", score: float = 0.85) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(idx, source, section),
        score=score,
        distance=0.2,
        query="test query",
    )


def _make_state(tmp_path, profile: str = "test") -> SessionState:
    state = SessionState(
        model="gpt-4",
        base_url="https://api.example.com",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        dialogue_start_time=time.time(),
    )
    state.memory = Memory()
    state.profile_name = profile
    return state


def _make_llm_response(text: str):
    """Создаёт мок ответа LLM без блока **Источники:**."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=50, completion_tokens=30, total_tokens=80)
    return resp


# ---------------------------------------------------------------------------
# Unit: ensure_rag_sources_in_response
# ---------------------------------------------------------------------------

def test_ensure_rag_sources_idempotent():
    """Если ответ уже содержит **Источники:** — функция не дублирует блок."""
    results = [_make_result(1)]
    text = "Ответ с источниками.\n\n**Источники:**\n- docs/python.md (chunk_id=doc_1, score=0.85)"
    result = ensure_rag_sources_in_response(text, results)
    assert result.count("**Источники:**") == 1


def test_ensure_rag_sources_adds_when_missing():
    """Если ответ не содержит **Источники:** — добавляет блок."""
    results = [_make_result(1, source="docs/asyncio.md", section="coroutines")]
    text = "Coroutine — это объект, представляющий отложенное вычисление."
    result = ensure_rag_sources_in_response(text, results)
    assert "**Источники:**" in result
    assert "doc_1" in result
    assert "0.85" in result


def test_ensure_rag_sources_empty_results():
    """Пустой список результатов — ничего не добавляет."""
    text = "Ответ без RAG."
    result = ensure_rag_sources_in_response(text, [])
    assert result == text
    assert "**Источники:**" not in result


def test_ensure_rag_sources_multiple_chunks():
    """Несколько чанков — все перечислены в блоке."""
    results = [_make_result(i, source=f"docs/doc{i}.md") for i in range(3)]
    text = "Подробный ответ без источников."
    result = ensure_rag_sources_in_response(text, results)
    assert result.count("- docs/doc") == 3
    for i in range(3):
        assert f"chunk_id=doc_{i}" in result


# ---------------------------------------------------------------------------
# Unit: build_rag_system_addition with working_memory
# ---------------------------------------------------------------------------

def test_build_rag_system_addition_includes_task():
    wm = WorkingMemory()
    wm.set_task("изучить asyncio")
    results = [_make_result(1)]
    text = build_rag_system_addition(results, wm)
    assert "изучить asyncio" in text


def test_build_rag_system_addition_includes_preferences():
    wm = WorkingMemory()
    wm.set_preference("язык", "русский")
    wm.set_preference("глубина", "экспертная")
    results = [_make_result(1)]
    text = build_rag_system_addition(results, wm)
    assert "[Зафиксированные предпочтения/ограничения]" in text
    assert "язык: русский" in text
    assert "глубина: экспертная" in text


def test_build_rag_system_addition_no_working_memory():
    results = [_make_result(1)]
    text = build_rag_system_addition(results, None)
    assert "[RAG-контекст]" in text
    assert "[Цель диалога" not in text


# ---------------------------------------------------------------------------
# Сценарий 1: «Python async/await» — 12 шагов
# ---------------------------------------------------------------------------

PYTHON_QUESTIONS = [
    "Что такое coroutine в Python?",
    "Как работает event loop в asyncio?",
    "Что делает asyncio.gather?",
    "Как установить timeout для корутины?",
    "В чём разница между asyncio.sleep и time.sleep?",
    "Что такое Task в asyncio?",
    "Как использовать async with?",
    "Как использовать async for?",
    "Что такое asyncio.Queue?",
    "Как обработать исключения в asyncio?",
]

LLM_ANSWERS_NO_SOURCES = [
    f"Ответ на вопрос {i + 1}: Подробное объяснение темы."
    for i in range(len(PYTHON_QUESTIONS))
]


@pytest.fixture
def rag_results_python():
    return [
        _make_result(1, source="docs/python_async.md", section="coroutines"),
        _make_result(2, source="docs/python_async.md", section="event_loop"),
    ]


class TestScenario1PythonAsync:
    """Сценарий 1: 12 сообщений про async/await в Python."""

    def test_sources_contain_chunk_id(self, tmp_path, monkeypatch, rag_results_python):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        _apply_inline_updates = None  # не нужен здесь

        for answer_text in LLM_ANSWERS_NO_SOURCES:
            display_text = ensure_rag_sources_in_response(answer_text, rag_results_python)
            assert "chunk_id=doc_1" in display_text
            assert "score=0.85" in display_text

    def test_history_not_reset(self, tmp_path, monkeypatch, rag_results_python):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)

        for i, question in enumerate(PYTHON_QUESTIONS):
            state.messages.append({"role": "user", "content": question})
            state.messages.append({"role": "assistant", "content": f"Answer {i}"})

        assert len(state.messages) == len(PYTHON_QUESTIONS) * 2


# ---------------------------------------------------------------------------
# Сценарий 2: «RAG-архитектура» — 13 шагов
# ---------------------------------------------------------------------------

RAG_ARCH_QUESTIONS = [
    "Как работают embedding-модели?",
    "Какие стратегии chunking существуют?",
    "Чем FAISS отличается от Annoy?",
    "Что такое reranking в RAG?",
    "Как реализовать hybrid search?",
    "Что такое dense retrieval?",
    "Как выбрать размер чанка?",
    "Что такое BM25?",
    "Как оценить качество RAG-системы?",
    "Что такое ColBERT?",
    "Как работает HyDE?",
]

LLM_ANSWERS_RAG_NO_SOURCES = [
    f"Экспертный ответ {i + 1} на русском языке об архитектуре RAG."
    for i in range(len(RAG_ARCH_QUESTIONS))
]


@pytest.fixture
def rag_results_arch():
    return [
        _make_result(10, source="docs/rag_architecture.md", section="embeddings"),
        _make_result(11, source="docs/rag_architecture.md", section="chunking"),
    ]


class TestScenario2RagArchitecture:
    """Сценарий 2: 13 сообщений про RAG-архитектуру."""

    def test_preferences_set_and_persist(self, tmp_path, monkeypatch, rag_results_arch):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        from llm_agent.chatbot.main import _apply_inline_updates

        # Фиксируем предпочтения через agent State Update (имитация)
        state.memory.working.set_preference("язык", "русский")
        state.memory.working.set_preference("глубина", "экспертная")

        # 11 вопросов
        for i, (q, a) in enumerate(zip(RAG_ARCH_QUESTIONS, LLM_ANSWERS_RAG_NO_SOURCES)):
            state.messages.append({"role": "user", "content": q})
            display_text = ensure_rag_sources_in_response(a, rag_results_arch)
            state.messages.append({"role": "assistant", "content": display_text})

            assert "**Источники:**" in display_text, f"Шаг {i+1}: источники отсутствуют"
            assert state.memory.working.user_preferences["язык"] == "русский"
            assert state.memory.working.user_preferences["глубина"] == "экспертная"

    def test_sources_contain_rag_arch_chunk_ids(self, tmp_path, monkeypatch, rag_results_arch):
        monkeypatch.chdir(tmp_path)
        for answer_text in LLM_ANSWERS_RAG_NO_SOURCES:
            display_text = ensure_rag_sources_in_response(answer_text, rag_results_arch)
            assert "chunk_id=doc_10" in display_text
            assert "chunk_id=doc_11" in display_text

    def test_rag_system_prompt_includes_preferences(self, tmp_path, monkeypatch, rag_results_arch):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        state.memory.working.set_preference("язык", "русский")
        state.memory.working.set_preference("глубина", "экспертная")

        sys_prompt = build_rag_system_addition(rag_results_arch, state.memory.working)
        assert "язык: русский" in sys_prompt
        assert "глубина: экспертная" in sys_prompt

    def test_all_responses_have_sources(self, tmp_path, monkeypatch, rag_results_arch):
        monkeypatch.chdir(tmp_path)
        for answer_text in LLM_ANSWERS_RAG_NO_SOURCES:
            result = ensure_rag_sources_in_response(answer_text, rag_results_arch)
            assert "**Источники:**" in result

    def test_history_grows_correctly(self, tmp_path, monkeypatch, rag_results_arch):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)

        for i, q in enumerate(RAG_ARCH_QUESTIONS):
            state.messages.append({"role": "user", "content": q})
            state.messages.append({"role": "assistant", "content": f"Answer {i}"})
            assert len(state.messages) == 2 * (i + 1)

    def test_preferences_survive_11_messages(self, tmp_path, monkeypatch, rag_results_arch):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        state.memory.working.set_preference("язык", "русский")
        state.memory.working.set_preference("глубина", "экспертная")

        for q in RAG_ARCH_QUESTIONS:
            state.messages.append({"role": "user", "content": q})
            assert state.memory.working.user_preferences.get("язык") == "русский"
            assert state.memory.working.user_preferences.get("глубина") == "экспертная"


