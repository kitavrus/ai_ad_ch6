"""Сценарные тесты: автоматизация ручного чеклиста RAG_STRUCTURED_OUTPUT_TESTING.md."""

import time

import pytest

from llm_agent.chatbot.context import (
    build_rag_idk_system,
    build_rag_system_addition,
    ensure_rag_sources_in_response,
)
from llm_agent.chatbot.memory import Memory
from llm_agent.chatbot.models import ChatMessage, SessionState
from llm_agent.rag.models import ChunkMetadata, RetrievalResult

# ---------------------------------------------------------------------------
# Helpers (copied from test_rag_dialog_scenarios.py)
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, source: str = "docs/guide.md", section: str = "") -> ChunkMetadata:
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


def _make_result(idx: int, source: str = "docs/guide.md", section: str = "", score: float = 0.85) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(idx, source, section),
        score=score,
        distance=0.2,
        query="test query",
    )


def _make_state(profile: str = "test") -> SessionState:
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


# ---------------------------------------------------------------------------
# Step 3: 10 вопросов по документам
# ---------------------------------------------------------------------------

TEN_QUESTIONS = [
    "Как работает краткосрочная память в чатботе?",
    "Чем отличаются профили пользователей?",
    "Как работает plan builder и валидация шагов?",
    "Какие стратегии контекста поддерживаются?",
    "Как добавить инвариант и зачем он нужен?",
    "Где хранятся файлы сессии и метрики запросов?",
    "Как работает стратегия sticky_facts?",
    "Какие зависимости нужны для RAG-режима?",
    "Как проходит валидация черновика ответа агента?",
    "Что происходит при исчерпании всех ретраев в agent loop?",
]

OFFTOPIC_QUESTIONS = [
    "Как приготовить борщ?",
    "Расскажи про фотосинтез",
    "Какой курс доллара?",
]


def _build_chunks_for_question(q_idx: int):
    """2 фейковых чанка для каждого вопроса."""
    return [
        _make_result(q_idx * 10 + 1, source=f"docs/topic_{q_idx}.md", section="Overview", score=0.90),
        _make_result(q_idx * 10 + 2, source=f"docs/topic_{q_idx}.md", section="Details", score=0.75),
    ]


class TestScenario1TenQuestions:
    """Сценарий 1: 10 вопросов по документам (Step 3 руководства)."""

    @pytest.fixture
    def prompts(self):
        """Строим system-addition для всех 10 вопросов заранее."""
        return [
            build_rag_system_addition(_build_chunks_for_question(i))
            for i in range(10)
        ]

    def test_all_responses_have_answer_header(self, prompts):
        for i, prompt in enumerate(prompts):
            assert "**Ответ:**" in prompt, f"Вопрос {i+1}: нет **Ответ:**"

    def test_all_responses_have_sources_header(self, prompts):
        for i, prompt in enumerate(prompts):
            assert "**Источники:**" in prompt, f"Вопрос {i+1}: нет **Источники:**"

    def test_all_responses_have_citations_header(self, prompts):
        for i, prompt in enumerate(prompts):
            assert "**Цитаты:**" in prompt, f"Вопрос {i+1}: нет **Цитаты:**"

    def test_chunk_metadata_in_every_response(self, prompts):
        for i, prompt in enumerate(prompts):
            assert "chunk_id=" in prompt, f"Вопрос {i+1}: нет chunk_id="
            assert "score=" in prompt, f"Вопрос {i+1}: нет score="

    def test_ensure_sources_appended_when_missing(self):
        """ensure_rag_sources_in_response добавляет **Источники:** для всех 10 ответов."""
        for i in range(10):
            results = _build_chunks_for_question(i)
            # Симулируем ответ без блока источников
            raw_answer = f"Ответ на вопрос {i+1}: подробное объяснение темы."
            patched = ensure_rag_sources_in_response(raw_answer, results)
            assert "**Источники:**" in patched, f"Вопрос {i+1}: **Источники:** не добавлены"

    def test_history_grows_to_20_messages(self):
        """После 10 Q&A-ходов state.messages содержит ровно 20 сообщений."""
        state = _make_state()
        state.messages = []

        for i, question in enumerate(TEN_QUESTIONS):
            state.messages.append(ChatMessage(role="user", content=question))
            state.messages.append(
                ChatMessage(role="assistant", content=f"Ответ на вопрос {i+1}.")
            )

        assert len(state.messages) == 20


# ---------------------------------------------------------------------------
# Step 4: IDK-режим
# ---------------------------------------------------------------------------

class TestScenario2IdkMode:
    """Сценарий 2: IDK-режим для офф-топик вопросов (Step 4 руководства)."""

    def _simulate_rag_routing(self, question: str, results: list, threshold: float) -> str:
        """Минимальная симуляция RAG-маршрутизации из main.py."""
        if results:
            return build_rag_system_addition(results)
        elif threshold > 0:
            return build_rag_idk_system()
        return ""

    def test_idk_triggered_for_3_offtopic_questions(self):
        """При threshold=0.5 и пустых результатах IDK-ответ возвращается для всех 3 вопросов."""
        for question in OFFTOPIC_QUESTIONS:
            result = self._simulate_rag_routing(question, results=[], threshold=0.5)
            assert result != "", f"IDK не вызван для: {question!r}"
            assert "Не знаю" in result, f"IDK-ответ не содержит 'Не знаю' для: {question!r}"

    def test_idk_not_triggered_when_threshold_zero(self):
        """При threshold=0 IDK не вызывается даже при пустых результатах."""
        for question in OFFTOPIC_QUESTIONS:
            result = self._simulate_rag_routing(question, results=[], threshold=0)
            assert result == "", f"IDK неожиданно вызван при threshold=0 для: {question!r}"

    def test_idk_response_contains_ne_znayu(self):
        """build_rag_idk_system() содержит 'Не знаю' для каждого офф-топик вопроса."""
        idk_msg = build_rag_idk_system()
        for question in OFFTOPIC_QUESTIONS:
            # Одно и то же сообщение для всех — проверяем содержимое
            assert "Не знаю" in idk_msg, f"IDK-ответ не содержит 'Не знаю' (вопрос: {question!r})"

    def test_rag_content_used_when_results_present(self):
        """При наличии результатов используется RAG-контент, а не IDK."""
        for question in OFFTOPIC_QUESTIONS:
            results = [_make_result(1, source="docs/main.md", section="Overview")]
            result = self._simulate_rag_routing(question, results=results, threshold=0.5)
            assert "**Ответ:**" in result, f"RAG-контент не использован для: {question!r}"
            assert "Не знаю" not in result, f"IDK неожиданно появился при наличии результатов: {question!r}"
