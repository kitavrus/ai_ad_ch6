"""Сценарные тесты: RAG + задача (goal/clarifications/constraints) через 12–15 шагов."""

import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from llm_agent.chatbot.context import build_rag_system_addition, ensure_rag_sources_in_response
from llm_agent.chatbot.memory import Memory, WorkingMemory
from llm_agent.chatbot.models import SessionState, TaskPlan
from llm_agent.rag.models import ChunkMetadata, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, source: str = "docs/python.md", section: str = "") -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=f"chunk_{idx}",
        source=source,
        title=source.split("/")[-1].replace(".md", ""),
        section=section,
        strategy="fixed",
        char_start=idx * 100,
        char_end=idx * 100 + 200,
        text=f"Content for chunk {idx}.",
    )


def _make_result(
    idx: int,
    source: str = "docs/python.md",
    section: str = "",
    score: float = 0.88,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(idx, source, section),
        score=score,
        distance=0.15,
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


def _make_plan(goal: str) -> TaskPlan:
    now = datetime.now(timezone.utc).isoformat()
    return TaskPlan(
        task_id=str(uuid.uuid4()),
        name=goal,
        title=goal,
        description=goal,
        steps=[],
        clarifications=[],
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Scenario 1 fixtures
# ---------------------------------------------------------------------------

DECORATOR_QUESTIONS = [
    "Что такое декораторы в Python?",
    "Как написать простой декоратор?",
    "Что такое functools.wraps?",
    "Как передать аргументы декоратору?",
    "Можно ли применять несколько декораторов к одной функции?",
    "Как декораторы работают с методами класса?",
    "Что такое @property?",
    "Как создать декоратор класса?",
    "Чем отличаются декораторы от metaclass?",
    "Как тестировать функции с декораторами?",
    "Что такое @dataclass?",
    "Как декораторы влияют на производительность?",
    "Можно ли снять декоратор после применения?",
    "Как использовать декоратор для кэширования?",
    "Когда НЕ стоит использовать декораторы?",
]

LLM_ANSWERS_DECORATORS = [
    f"Ответ {i + 1}: подробное объяснение декораторов Python 3.12."
    for i in range(len(DECORATOR_QUESTIONS))
]


@pytest.fixture
def rag_results_decorators():
    return [
        _make_result(1, source="docs/decorators.md", section="basics"),
        _make_result(2, source="docs/decorators.md", section="advanced"),
        _make_result(3, source="docs/python312.md", section="functools"),
    ]


# ---------------------------------------------------------------------------
# Scenario 1 — "Python decorators" (15 messages)
# ---------------------------------------------------------------------------

class TestScenario1Decorators:
    """15-шаговый диалог: цель, уточнения, ограничения, источники."""

    def test_goal_persists_across_15_messages(self, tmp_path, monkeypatch, rag_results_decorators):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        goal = "изучить декораторы Python 3.12"
        state.memory.working.set_task(goal)

        for i, (q, a) in enumerate(zip(DECORATOR_QUESTIONS, LLM_ANSWERS_DECORATORS)):
            state.messages.append({"role": "user", "content": q})
            display = ensure_rag_sources_in_response(a, rag_results_decorators)
            state.messages.append({"role": "assistant", "content": display})

            assert state.memory.working.current_task == goal, (
                f"Шаг {i + 1}: цель изменилась"
            )

        assert len(state.messages) == len(DECORATOR_QUESTIONS) * 2

    def test_clarifications_accumulate(self, tmp_path, monkeypatch, rag_results_decorators):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan("изучить декораторы Python 3.12")

        clarifications = [
            {"question": "Какую версию Python использовать?", "answer": "Python 3.12"},
            {"question": "Нужно ли изучать metaclass?", "answer": "нет, избегать"},
            {"question": "Формат примеров?", "answer": "короткие, рабочие"},
            {"question": "Нужна ли теория?", "answer": "да, с примерами"},
            {"question": "Тестирование?", "answer": "pytest, без mocks для декораторов"},
        ]
        for c in clarifications:
            plan.clarifications.append(c)

        assert len(plan.clarifications) == 5
        for orig, stored in zip(clarifications, plan.clarifications):
            assert stored["question"] == orig["question"]
            assert stored["answer"] == orig["answer"]

    def test_constraints_survive_all_turns(self, tmp_path, monkeypatch, rag_results_decorators):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        state.memory.working.set_preference("версия_python", "3.12")
        state.memory.working.set_preference("запрет", "metaclass")
        state.memory.working.set_preference("язык_ответа", "русский")

        for i, (q, a) in enumerate(zip(DECORATOR_QUESTIONS, LLM_ANSWERS_DECORATORS)):
            state.messages.append({"role": "user", "content": q})
            state.messages.append({"role": "assistant", "content": a})

            assert state.memory.working.user_preferences["версия_python"] == "3.12", (
                f"Шаг {i + 1}: constraint версия_python потерян"
            )
            assert state.memory.working.user_preferences["запрет"] == "metaclass", (
                f"Шаг {i + 1}: constraint запрет потерян"
            )
            assert state.memory.working.user_preferences["язык_ответа"] == "русский", (
                f"Шаг {i + 1}: constraint язык_ответа потерян"
            )

    def test_sources_in_every_response(self, tmp_path, monkeypatch, rag_results_decorators):
        monkeypatch.chdir(tmp_path)
        for i, answer_text in enumerate(LLM_ANSWERS_DECORATORS):
            result = ensure_rag_sources_in_response(answer_text, rag_results_decorators)
            assert "**Источники:**" in result, f"Шаг {i + 1}: блок источников отсутствует"
            assert "chunk_1" in result
            assert "chunk_2" in result

    def test_system_prompt_includes_goal(self, tmp_path, monkeypatch, rag_results_decorators):
        monkeypatch.chdir(tmp_path)
        wm = WorkingMemory()
        wm.set_task("изучить декораторы Python 3.12")
        wm.set_preference("версия_python", "3.12")
        wm.set_preference("запрет", "metaclass")

        sys_prompt = build_rag_system_addition(rag_results_decorators, wm)
        assert "изучить декораторы Python 3.12" in sys_prompt
        assert "версия_python" in sys_prompt
        assert "metaclass" in sys_prompt


# ---------------------------------------------------------------------------
# Scenario 2 fixtures
# ---------------------------------------------------------------------------

DB_QUESTIONS = [
    "Какие базы данных подходят для SaaS?",
    "Сравни PostgreSQL и MySQL для SaaS.",
    "Как выбрать БД с бюджетом до $100/мес?",
    "Поддерживает ли PostgreSQL геозапросы?",
    "Как PostGIS помогает с гео-данными?",
    "Какие managed-сервисы дешевле $100/мес?",
    "Как SQL-команда быстро освоит новую БД?",
    "Что такое connection pooling для SaaS?",
    "Как масштабировать PostgreSQL горизонтально?",
    "Какие индексы нужны для геозапросов?",
    "Как настроить бэкапы в Supabase?",
    "Итоговая рекомендация по БД для нашего SaaS.",
]

LLM_ANSWERS_DB = [
    f"Ответ {i + 1}: рекомендации по выбору базы данных для SaaS."
    for i in range(len(DB_QUESTIONS))
]


@pytest.fixture
def rag_results_db():
    return [
        _make_result(20, source="docs/databases.md", section="postgresql"),
        _make_result(21, source="docs/databases.md", section="geo_queries"),
        _make_result(22, source="docs/saas_infra.md", section="managed_services"),
    ]


# ---------------------------------------------------------------------------
# Scenario 2 — "Database choice" (12 messages)
# ---------------------------------------------------------------------------

class TestScenario2DatabaseChoice:
    """12-шаговый диалог: цель выбора БД, ограничения добавляются поступенно."""

    def test_constraints_accumulate_across_12_turns(self, tmp_path, monkeypatch, rag_results_db):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        state.memory.working.set_task("выбрать БД для SaaS-проекта")

        # Ограничения добавляются постепенно в разных шагах
        constraints_schedule = {
            2: ("бюджет", "менее $100/мес"),
            5: ("гео_запросы", "обязательны"),
            7: ("команда", "знает SQL"),
            9: ("managed", "предпочтительно"),
        }

        for i, (q, a) in enumerate(zip(DB_QUESTIONS, LLM_ANSWERS_DB)):
            turn = i + 1
            if turn in constraints_schedule:
                key, val = constraints_schedule[turn]
                state.memory.working.set_preference(key, val)

            state.messages.append({"role": "user", "content": q})
            state.messages.append({"role": "assistant", "content": a})

        # После 12 шагов все 4 ограничения должны быть на месте
        prefs = state.memory.working.user_preferences
        assert prefs.get("бюджет") == "менее $100/мес"
        assert prefs.get("гео_запросы") == "обязательны"
        assert prefs.get("команда") == "знает SQL"
        assert prefs.get("managed") == "предпочтительно"

    def test_goal_not_lost_after_clarifications(self, tmp_path, monkeypatch, rag_results_db):
        monkeypatch.chdir(tmp_path)
        state = _make_state(tmp_path)
        goal = "выбрать БД для SaaS-проекта"
        state.memory.working.set_task(goal)

        for key, val in [
            ("бюджет", "менее $100/мес"),
            ("гео_запросы", "обязательны"),
            ("команда", "знает SQL"),
            ("managed", "предпочтительно"),
        ]:
            state.memory.working.set_preference(key, val)
            # Цель не должна измениться после добавления предпочтений
            assert state.memory.working.current_task == goal

        for i, q in enumerate(DB_QUESTIONS):
            state.messages.append({"role": "user", "content": q})
            state.messages.append({"role": "assistant", "content": LLM_ANSWERS_DB[i]})
            assert state.memory.working.current_task == goal, (
                f"Шаг {i + 1}: цель потеряна после уточнений"
            )

    def test_all_responses_have_sources(self, tmp_path, monkeypatch, rag_results_db):
        monkeypatch.chdir(tmp_path)
        for i, answer_text in enumerate(LLM_ANSWERS_DB):
            result = ensure_rag_sources_in_response(answer_text, rag_results_db)
            assert "**Источники:**" in result, f"Шаг {i + 1}: блок источников отсутствует"

    def test_source_chunk_ids_consistent(self, tmp_path, monkeypatch, rag_results_db):
        monkeypatch.chdir(tmp_path)
        for i, answer_text in enumerate(LLM_ANSWERS_DB):
            result = ensure_rag_sources_in_response(answer_text, rag_results_db)
            assert "chunk_20" in result, f"Шаг {i + 1}: chunk_20 отсутствует"
            assert "chunk_21" in result, f"Шаг {i + 1}: chunk_21 отсутствует"
            assert "chunk_22" in result, f"Шаг {i + 1}: chunk_22 отсутствует"

    def test_system_prompt_reflects_all_constraints(self, tmp_path, monkeypatch, rag_results_db):
        monkeypatch.chdir(tmp_path)
        wm = WorkingMemory()
        wm.set_task("выбрать БД для SaaS-проекта")
        wm.set_preference("бюджет", "менее $100/мес")
        wm.set_preference("гео_запросы", "обязательны")
        wm.set_preference("команда", "знает SQL")
        wm.set_preference("managed", "предпочтительно")

        # Имитируем 12 шагов диалога, затем проверяем системный промпт
        state = _make_state(tmp_path)
        state.memory.working = wm
        for i, q in enumerate(DB_QUESTIONS):
            state.messages.append({"role": "user", "content": q})
            state.messages.append({"role": "assistant", "content": LLM_ANSWERS_DB[i]})

        sys_prompt = build_rag_system_addition(rag_results_db, state.memory.working)
        assert "выбрать БД для SaaS-проекта" in sys_prompt
        assert "бюджет" in sys_prompt
        assert "гео_запросы" in sys_prompt
        assert "команда" in sys_prompt
        assert "managed" in sys_prompt

    def test_clarifications_stored_in_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan("выбрать БД для SaaS-проекта")

        qa_pairs = [
            ("Каков бюджет?", "менее $100/мес"),
            ("Нужны ли геозапросы?", "да, обязательно"),
            ("Какой стек у команды?", "SQL, немного NoSQL"),
            ("Managed или self-hosted?", "предпочтительно managed"),
        ]
        for q, a in qa_pairs:
            plan.clarifications.append({"question": q, "answer": a})

        # После 12 шагов диалога уточнения не должны сбрасываться
        for _ in range(12):
            pass  # симуляция шагов (память хранится в plan, не сессии)

        assert len(plan.clarifications) == 4
        answers = [c["answer"] for c in plan.clarifications]
        assert "менее $100/мес" in answers
        assert "да, обязательно" in answers
        assert "SQL, немного NoSQL" in answers
        assert "предпочтительно managed" in answers
