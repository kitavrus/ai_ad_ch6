"""Сравнение LOCAL vs CLOUD в режимах с RAG и без RAG.

Режимы:
  LOCAL_NORAG  — Ollama, без контекста
  LOCAL_RAG    — Ollama + FAISS (structure-индекс)
  CLOUD_NORAG  — OpenAI, без контекста
  CLOUD_RAG    — OpenAI + FAISS (structure-индекс)

Запуск:
    cd /Users/igorpotema/mycode/ai_ad_ch6

    # только локальные режимы (не нужен API-ключ)
    python llm_local/compare_local_cloud.py --modes LOCAL_NORAG,LOCAL_RAG --output local_report.md

    # полное сравнение (требует .env с OPENAI_API_KEY или API_KEY)
    python llm_local/compare_local_cloud.py --output full_report.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from llm_local.rag_local import LocalRAGPipeline

# ---------------------------------------------------------------------------
# 10 контрольных вопросов (из llm_agent/compare_rag.py)
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
    },
    {
        "question": "Какие стратегии управления контекстом поддерживаются и чем они отличаются?",
        "expected": (
            "sliding_window — последние N сообщений + суммаризация старых; "
            "sticky_facts — ключевые факты в системном сообщении, /setfact /delfact; "
            "branching — независимые ветки от чекпоинта, /branch /switch."
        ),
    },
    {
        "question": "Как работает агентский цикл /plan on? Опишите последовательность шагов от ввода до финального ответа.",
        "expected": (
            "Онбординг: awaiting_task → awaiting_invariants → active. "
            "В active: build_agent_system_prompt → LLM-черновик → validate_draft_against_invariants → "
            "retry (до max_retries) → parse_agent_output. "
            "State Update сохраняется в working.preferences."
        ),
    },
    {
        "question": "Какие категории инвариантов поддерживаются? Приведи примеры каждой.",
        "expected": (
            "Стек-ограничение (только Python), техническое решение (использовать PostgreSQL), "
            "бизнес-правило (ответ на русском), архитектурное ограничение (без глобальных переменных), "
            "противоречивые инварианты."
        ),
    },
    {
        "question": "Что происходит при нарушении инварианта в plan builder — как работает retry-цикл и что при исчерпании попыток?",
        "expected": (
            "При нарушении: clarification question + повторный черновик, до max_retries раз. "
            "При исчерпании: _prompt_invariant_resolution предлагает edit <N>, remove <N>, abort. "
            "Шаги нельзя пропустить."
        ),
    },
    {
        "question": "Как устроена иерархия уровней команд (schema graph)? Сколько уровней и что на каждом?",
        "expected": (
            "0: profile_name (глобальное), 1: SessionState/session_*.json, "
            "2: контекстные стратегии (взаимоисключающие), 3: трёхуровневая память, "
            "4: задачи, 5: план/агент + инварианты, 6: проекты."
        ),
    },
    {
        "question": "Как работает Notification Watcher? Опиши сценарий от создания напоминания до автоматического показа.",
        "expected": (
            "Запустить чатбот → создать напоминание через LLM → не вводить ничего → "
            "через ~10 сек бот сам выводит уведомление. "
            "NotificationServer слушает webhook, _start_notification_watcher — фоновый поток."
        ),
    },
    {
        "question": "Как проиндексировать документы для RAG? Какие файлы создаются и что нужно для запуска?",
        "expected": (
            "python index_documents.py (или rag.pipeline). "
            "Создаются: rag_index/fixed.faiss + fixed_metadata.json, structure.faiss + structure_metadata.json. "
            "Два режима: fixed и structure. Требуется OPENAI_API_KEY."
        ),
    },
    {
        "question": "Как строится итоговый API-запрос к LLM? В каком порядке инжектируются системные сообщения?",
        "expected": (
            "1) базовый контекст стратегии, 2) контекст активной задачи [ACTIVE TASK], "
            "3) агентский промпт или профиль пользователя, 4) RAG-дополнение (если rag_mode.enabled), "
            "5) вызов client.chat.completions.create()."
        ),
    },
    {
        "question": "Как работает хранение с привязкой к профилю? Что происходит при переключении через /profile load?",
        "expected": (
            "Все данные в dialogues/{profile_name}/: сессии, метрики, память, задачи, проекты. "
            "--profile NAME автоматически загружает последнюю сессию. "
            "/profile load Bob сбрасывает active_task_id, agent_mode, plan_dialog_state, токены, ветки."
        ),
    },
]

DIVIDER = "─" * 60

ALL_MODES = ["LOCAL_NORAG", "LOCAL_RAG", "CLOUD_NORAG", "CLOUD_RAG"]

MODE_LABELS = {
    "LOCAL_NORAG": "Ollama (без RAG)",
    "LOCAL_RAG":   "Ollama + FAISS (structure)",
    "CLOUD_NORAG": "OpenAI (без RAG)",
    "CLOUD_RAG":   "OpenAI + FAISS (structure)",
}

TIME_COL = {
    "LOCAL_NORAG": "LN_t",
    "LOCAL_RAG":   "LR_t",
    "CLOUD_NORAG": "CN_t",
    "CLOUD_RAG":   "CR_t",
}


def _run_mode(pipeline: LocalRAGPipeline, mode: str, query: str) -> dict:
    """Вызывает нужный метод pipeline по имени режима."""
    dispatch = {
        "LOCAL_NORAG": pipeline.ask_local_norag,
        "LOCAL_RAG":   pipeline.ask_local,
        "CLOUD_NORAG": pipeline.ask_cloud_norag,
        "CLOUD_RAG":   pipeline.ask_cloud,
    }
    try:
        return dispatch[mode](query)
    except Exception as exc:
        return {
            "answer": f"(ошибка: {exc})",
            "search_time": 0.0,
            "generate_time": 0.0,
            "total_time": 0.0,
            "char_count": 0,
            "rag_used": False,
        }


def run_comparison(output: str, modes: list[str]) -> None:
    print(f"Режимы: {', '.join(modes)}")
    print(f"Вопросов: {len(QUESTIONS)}")
    print()

    pipeline = LocalRAGPipeline()

    lines: list[str] = [
        "# Local vs Cloud RAG Comparison\n",
        "## Режимы\n",
        "| Режим | Описание |",
        "|-------|----------|",
    ]
    for m in modes:
        lines.append(f"| `{m}` | {MODE_LABELS[m]} |")
    lines.append("\n")

    # Собираем ответы для итоговой таблицы
    per_mode_results: dict[str, list[dict]] = {m: [] for m in modes}

    for i, item in enumerate(QUESTIONS, 1):
        q = item["question"]
        expected = item["expected"]

        print(f"[{i}/10] {q[:70]}...")

        answers: dict[str, dict] = {}
        for mode in modes:
            answers[mode] = _run_mode(pipeline, mode, q)
            per_mode_results[mode].append(answers[mode])

        # Блок по вопросу
        block: list[str] = [
            f"## Вопрос {i}: {q}\n",
            f"**Ожидаемый ответ:** {expected}\n",
        ]
        for mode in modes:
            r = answers[mode]
            rag_tag = " ✓RAG" if r["rag_used"] else ""
            timing = (
                f"{r['total_time']:.1f}s"
                + (f" (поиск {r['search_time']:.1f}s + генерация {r['generate_time']:.1f}s)"
                   if r["rag_used"] else "")
            )
            block.append(
                f"**{mode}** ({MODE_LABELS[mode]}){rag_tag} — время: {timing}, символов: {r['char_count']}\n"
            )
            block.append(r["answer"])
            block.append("")
        block.append(DIVIDER)
        lines.extend(block)

    # Итоговая таблица
    lines.append("\n## Итоговая таблица\n")
    header_parts = ["**#**", "**Вопрос**"]
    for m in modes:
        header_parts.append(f"**{TIME_COL[m]} (сек)**")
    for m in modes:
        header_parts.append(f"**{TIME_COL[m]}_симв**")
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

    for i, item in enumerate(QUESTIONS):
        q_short = item["question"][:45].replace("|", "/") + "…"
        row = [str(i + 1), q_short]
        for m in modes:
            t = per_mode_results[m][i]["total_time"]
            row.append(f"{t:.1f}" if t > 0 else "—")
        for m in modes:
            c = per_mode_results[m][i]["char_count"]
            row.append(str(c) if c > 0 else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Средние тайминги
    lines.append("\n## Средние тайминги\n")
    lines.append("| Режим | Среднее время (сек) | Средн. символов |")
    lines.append("|-------|---------------------|-----------------|")
    for m in modes:
        valid = [r for r in per_mode_results[m] if r["total_time"] > 0]
        avg_t = sum(r["total_time"] for r in valid) / len(valid) if valid else 0
        avg_c = sum(r["char_count"] for r in valid) / len(valid) if valid else 0
        lines.append(f"| `{m}` | {avg_t:.1f} | {avg_c:.0f} |")

    report = "\n".join(lines)
    output_path = Path(output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nОтчёт сохранён: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение Local LLM vs Cloud LLM с RAG и без RAG."
    )
    parser.add_argument(
        "--output",
        default="local_vs_cloud_report.md",
        help="Путь к выходному файлу (default: local_vs_cloud_report.md)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(ALL_MODES),
        help=f"Режимы через запятую (default: все). Доступные: {', '.join(ALL_MODES)}",
    )
    args = parser.parse_args()
    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ALL_MODES]
    if invalid:
        print(f"Неизвестные режимы: {invalid}. Допустимые: {', '.join(ALL_MODES)}", file=sys.stderr)
        sys.exit(1)
    run_comparison(args.output, modes)


if __name__ == "__main__":
    main()
