"""Скрипт сравнения ответов LLM в четырёх режимах RAG.

Запуск:
    python compare_rag.py --output rag_comparison_report.md
    python compare_rag.py --modes A,B,C --output report.md
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

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

ALL_MODES = ["A", "B", "C", "D"]

MODE_LABELS = {
    "A": "No RAG (baseline)",
    "B": "RAG, no filter/rewrite",
    "C": "RAG + relevance filter (threshold=0.5, top_k_before=10, top_k_after=3)",
    "D": "RAG + filter + query rewrite",
}


def ask_llm(client: OpenAI, model: str, system: str, question: str) -> str:
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


def _build_retrievers(modes: list, llm_client: OpenAI):
    """Build RAGRetriever instances for modes B/C/D."""
    from llm_agent.rag.retriever import RAGRetriever

    retrievers = {}
    if "B" in modes:
        retrievers["B"] = RAGRetriever()
    if "C" in modes:
        retrievers["C"] = RAGRetriever(
            relevance_threshold=0.5,
            top_k_before=10,
            top_k_after=3,
        )
    if "D" in modes:
        retrievers["D"] = RAGRetriever(
            relevance_threshold=0.5,
            top_k_before=10,
            top_k_after=3,
            rewrite_query=True,
            llm_client=llm_client,
        )
    return retrievers


def _rag_answer(client: OpenAI, model: str, retriever, question: str) -> tuple[str, str, str]:
    """Return (answer, chunk_sources, rewritten_query)."""
    from llm_agent.chatbot.context import build_rag_system_addition

    results = retriever.search_with_scores(question, strategy="structure", top_k=3)
    if not results:
        return "(RAG: индекс пуст)", "(чанки не найдены)", question

    chunks = [r.chunk for r in results]
    rewritten = results[0].query if results else question
    rag_system = build_rag_system_addition(chunks)
    chunk_sources = ", ".join(sorted({Path(c.source).name for c in chunks}))
    answer = ask_llm(client, model, rag_system, question)
    return answer, chunk_sources, rewritten


def run_comparison(output: str, modes: list[str]) -> None:
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

    # Try to load RAG retrievers
    rag_available = False
    retrievers: dict = {}
    try:
        retrievers = _build_retrievers(modes, client)
        rag_available = True
    except Exception as exc:
        print(f"[Предупреждение] RAGRetriever недоступен: {exc}. RAG-столбцы будут пусты.", file=sys.stderr)

    # Build header
    mode_cols = " | ".join(modes)
    lines: list[str] = [
        "# RAG Comparison Report\n",
        f"**Modes:** {', '.join(modes)}\n",
        "| Mode | Description |",
        "|------|-------------|",
    ]
    for m in modes:
        lines.append(f"| {m} | {MODE_LABELS[m]} |")
    lines.append("\n")

    # Per-question results
    per_mode_answers: dict[str, list[str]] = {m: [] for m in modes}

    for i, item in enumerate(QUESTIONS, 1):
        q = item["question"]
        expected = item["expected"]
        sources = item["sources"]

        print(f"[{i}/10] {q[:60]}...")

        answers: dict[str, str] = {}
        meta: dict[str, str] = {}

        if "A" in modes:
            answers["A"] = ask_llm(client, model, "", q)
            meta["A"] = "—"

        for mode in ["B", "C", "D"]:
            if mode not in modes:
                continue
            if rag_available and mode in retrievers:
                try:
                    ans, chunk_src, rewritten = _rag_answer(client, model, retrievers[mode], q)
                    answers[mode] = ans
                    meta[mode] = f"sources={chunk_src}; query_used={rewritten}"
                except Exception as exc:
                    answers[mode] = f"(ошибка: {exc})"
                    meta[mode] = f"(ошибка: {exc})"
            else:
                answers[mode] = "(RAG недоступен)"
                meta[mode] = "(RAG недоступен)"

        # Accumulate for summary
        for m in modes:
            per_mode_answers[m].append(answers.get(m, ""))

        # Build block
        block_lines = [f"## Вопрос {i}: {q}\n", f"**Ожидание:** {expected}\n"]
        for m in modes:
            block_lines.append(f"**Режим {m} ({MODE_LABELS[m]}):**\n{answers.get(m, '')}")
            if m != "A":
                block_lines.append(f"*Метаданные:* {meta.get(m, '')}")
            block_lines.append("")
        block_lines.append(f"**Источники (эталон):** {sources}\n")
        block_lines.append(DIVIDER)
        lines.extend(block_lines)

    # Summary table header
    lines.append("\n## Сводная таблица\n")
    header_cols = " | ".join(["**Вопрос**", "**Ожидание**"] + [f"**{m}**" for m in modes])
    lines.append(f"| {header_cols} |")
    sep = "|".join(["---"] * (2 + len(modes)))
    lines.append(f"|{sep}|")

    for i, item in enumerate(QUESTIONS):
        q_short = item["question"][:40].replace("|", "/") + "..."
        exp_short = item["expected"][:50].replace("|", "/") + "..."
        row_cols = [q_short, exp_short] + [
            (per_mode_answers[m][i][:60].replace("\n", " ").replace("|", "/") + "…")
            for m in modes
        ]
        lines.append("| " + " | ".join(row_cols) + " |")

    report = "\n".join(lines)
    output_path = Path(output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nОтчёт сохранён: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение ответов LLM в четырёх режимах RAG.")
    parser.add_argument(
        "--output",
        default="rag_comparison_report.md",
        help="Путь к выходному файлу (default: rag_comparison_report.md)",
    )
    parser.add_argument(
        "--modes",
        default="A,B,C,D",
        help="Режимы через запятую: A,B,C,D (default: все четыре)",
    )
    args = parser.parse_args()
    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ALL_MODES]
    if invalid:
        print(f"Неизвестные режимы: {invalid}. Допустимые: A, B, C, D", file=sys.stderr)
        sys.exit(1)
    run_comparison(args.output, modes)


if __name__ == "__main__":
    main()
