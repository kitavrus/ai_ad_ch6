"""Скрипт сравнения ответов LLM с RAG и без RAG.

Запуск:
    python compare_rag.py --output rag_comparison_report.md
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))

# ---------------------------------------------------------------------------
# 10 контрольных вопросов
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "question": "Как устроена трёхуровневая система памяти и где физически хранятся данные каждого уровня?",
        "expected": (
            "Три уровня: ShortTermMemory (текущая сессия), WorkingMemory (текущая задача + предпочтения), "
            "LongTermMemory (профиль + решения + знания). "
            "Пути: dialogues/{profile}/memory/short_term/, working/, long_term/. "
            "Класс Memory — единый фасад над тремя уровнями."
        ),
        "sources": "CLAUDE.md §Трехуровневая система памяти, schema-graph.md §Уровень 3",
    },
    {
        "question": "Какие стратегии управления контекстом поддерживаются и чем они отличаются?",
        "expected": (
            "sliding_window — последние N сообщений + суммаризация старых; "
            "sticky_facts — ключевые факты в системном сообщении, /setfact /delfact; "
            "branching — независимые ветки от чекпоинта, /branch /switch."
        ),
        "sources": "CLAUDE.md §Стратегии контекста, schema-graph.md §Уровень 2",
    },
    {
        "question": "Как работает агентский цикл /plan on? Опишите последовательность шагов от ввода до финального ответа.",
        "expected": (
            "Онбординг: awaiting_task → awaiting_invariants → active. "
            "В active: build_agent_system_prompt → LLM-черновик → validate_draft_against_invariants → "
            "retry (до max_retries) → parse_agent_output. "
            "State Update сохраняется в working.preferences."
        ),
        "sources": "EXAMPLES.md §/plan on — пример сессии, CLAUDE.md §Режим Plan",
    },
    {
        "question": "Какие категории инвариантов поддерживаются? Приведи примеры каждой.",
        "expected": (
            "Стек-ограничение (только Python), техническое решение (использовать PostgreSQL), "
            "бизнес-правило (ответ на русском), архитектурное ограничение (без глобальных переменных), "
            "противоречивые инварианты."
        ),
        "sources": "SCENARIOS.md §Категории инвариантов",
    },
    {
        "question": "Что происходит при нарушении инварианта в plan builder — как работает retry-цикл и что при исчерпании попыток?",
        "expected": (
            "При нарушении: clarification question + повторный черновик, до max_retries раз. "
            "При исчерпании: _prompt_invariant_resolution предлагает edit <N>, remove <N>, abort. "
            "Шаги нельзя пропустить."
        ),
        "sources": "SCENARIOS.md §Retry-цикл §Исчерпание попыток, CLAUDE.md §Builder invariant resolution",
    },
    {
        "question": "Как устроена иерархия уровней команд (schema graph)? Сколько уровней и что на каждом?",
        "expected": (
            "0: profile_name (глобальное), 1: SessionState/session_*.json, "
            "2: контекстные стратегии (взаимоисключающие), 3: трёхуровневая память, "
            "4: задачи, 5: план/агент + инварианты, 6: проекты."
        ),
        "sources": "schema-graph.md §Уровень 0–6",
    },
    {
        "question": "Как работает Notification Watcher? Опиши сценарий от создания напоминания до автоматического показа.",
        "expected": (
            "Запустить чатбот → создать напоминание через LLM → не вводить ничего → "
            "через ~10 сек бот сам выводит уведомление. "
            "NotificationServer слушает webhook, _start_notification_watcher — фоновый поток."
        ),
        "sources": "USE_CASE_NOTIFICATION_WATCHER.md §Способ 1",
    },
    {
        "question": "Как проиндексировать документы для RAG? Какие файлы создаются и что нужно для запуска?",
        "expected": (
            "python index_documents.py (или rag.pipeline). "
            "Создаются: rag_index/fixed.faiss + fixed_metadata.json, structure.faiss + structure_metadata.json. "
            "Два режима: fixed и structure. Требуется OPENAI_API_KEY."
        ),
        "sources": "RAG_QUICKSTART.md §3. Индексирование документов",
    },
    {
        "question": "Как строится итоговый API-запрос к LLM? В каком порядке инжектируются системные сообщения?",
        "expected": (
            "1) базовый контекст стратегии, 2) контекст активной задачи [ACTIVE TASK], "
            "3) агентский промпт или профиль пользователя, 4) RAG-дополнение (если rag_mode.enabled), "
            "5) вызов client.chat.completions.create()."
        ),
        "sources": "schema-graph.md §Итоговая сборка API-запроса, CLAUDE.md §Ключевые паттерны",
    },
    {
        "question": "Как работает хранение с привязкой к профилю? Что происходит при переключении через /profile load?",
        "expected": (
            "Все данные в dialogues/{profile_name}/: сессии, метрики, память, задачи, проекты. "
            "--profile NAME автоматически загружает последнюю сессию. "
            "/profile load Bob сбрасывает active_task_id, agent_mode, plan_dialog_state, токены, ветки."
        ),
        "sources": "CLAUDE.md §Хранение с привязкой к профилю, schema-graph.md §Уровень 0",
    },
]

DIVIDER = "─" * 50


def ask_llm(client: OpenAI, model: str, system: str, question: str) -> str:
    """Задаёт вопрос LLM и возвращает текст ответа."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def run_comparison(output: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("DEFAULT_MODEL", "inception/mercury-coder")

    if not api_key:
        print("Ошибка: переменная OPENAI_API_KEY или API_KEY не задана.", file=sys.stderr)
        sys.exit(1)

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    # Загружаем RAG-чанки
    try:
        from llm_agent.rag.retriever import RAGRetriever
        retriever = RAGRetriever()
        rag_available = True
    except Exception as exc:
        print(f"[Предупреждение] RAGRetriever недоступен: {exc}. Столбец 'С RAG' будет пуст.", file=sys.stderr)
        rag_available = False
        retriever = None  # type: ignore[assignment]

    lines: list[str] = ["# RAG Comparison Report\n"]

    for i, item in enumerate(QUESTIONS, 1):
        q = item["question"]
        expected = item["expected"]
        sources = item["sources"]

        print(f"[{i}/10] {q}")

        # Без RAG
        answer_no_rag = ask_llm(client, model, "", q)

        # С RAG
        if rag_available and retriever is not None:
            try:
                chunks = retriever.search(q, strategy="structure", top_k=3)
                if chunks:
                    from llm_agent.chatbot.context import build_rag_system_addition
                    rag_system = build_rag_system_addition(chunks)
                    chunk_sources = ", ".join(
                        sorted({Path(c.source).name for c in chunks})
                    )
                    answer_rag = ask_llm(client, model, rag_system, q)
                else:
                    chunk_sources = "(чанки не найдены)"
                    answer_rag = "(RAG: индекс пуст)"
            except Exception as exc:
                chunk_sources = f"(ошибка: {exc})"
                answer_rag = f"(RAG: ошибка поиска: {exc})"
        else:
            chunk_sources = "(RAG недоступен)"
            answer_rag = "(RAG недоступен)"

        block = (
            f"## Вопрос {i}: {q}\n\n"
            f"**БЕЗ RAG:**\n{answer_no_rag}\n\n"
            f"**С RAG:**\n{answer_rag}\n\n"
            f"**Источники (RAG):** {chunk_sources}\n\n"
            f"**Ожидание:** {expected}\n\n"
            f"{DIVIDER}\n"
        )
        lines.append(block)

    report = "\n".join(lines)
    output_path = Path(output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nОтчёт сохранён: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение ответов LLM с RAG и без RAG.")
    parser.add_argument(
        "--output",
        default="rag_comparison_report.md",
        help="Путь к выходному файлу (default: rag_comparison_report.md)",
    )
    args = parser.parse_args()
    run_comparison(args.output)


if __name__ == "__main__":
    main()
