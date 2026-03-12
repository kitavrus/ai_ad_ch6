"""Основной модуль: точка входа и цикл интерактивного диалога."""

import contextlib
from datetime import datetime
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
import uuid

from openai import OpenAI

from llm_agent.chatbot.cli import config_from_args, get_resume_flag, parse_args, parse_inline_command
from llm_agent.chatbot.mcp_client import (
    MCPClientManager,
    MCPSchedulerClient,
    MCPWeatherClient,
    _convert_tools_to_openai,
)
from llm_agent.chatbot.notification_server import NotificationServer
from llm_agent.chatbot.config import (
    API_KEY,
    CONTEXT_RECENT_MESSAGES,
    DEFAULT_PROFILE,
    DIALOGUES_DIR,
    RUB_PER_USD,
    USD_PER_1K_TOKENS,
)
from llm_agent.chatbot.context import (
    analyze_invariant_impact,
    build_agent_system_prompt,
    build_builder_step_prompt,
    build_context_by_strategy,
    build_plan_dialog_prompt,
    create_branch,
    create_checkpoint,
    extract_facts_from_llm,
    extract_result_files,
    generate_clarification_question,
    maybe_summarize,
    parse_agent_output,
    parse_draft_plan_block,
    parse_plan_questions,
    switch_branch,
    validate_draft_against_invariants,
)
from llm_agent.chatbot.memory import LongTermMemory, Memory, WorkingMemory
from llm_agent.chatbot.memory_storage import (
    get_memory_stats,
    list_profiles,
    load_long_term,
    load_profile,
    load_working_memory,
    save_long_term,
    save_profile,
    save_short_term,
    save_working_memory,
)
from llm_agent.chatbot.models import (
    AgentMode,
    Branch,
    can_transition,
    ChatMessage,
    ContextStrategy,
    DialogueSession,
    Project,
    RequestMetric,
    SessionState,
    StepStatus,
    StickyFacts,
    TaskPhase,
    TaskPlan,
    TaskStep,
    TokenUsage,
)
from llm_agent.chatbot.project_storage import (
    list_projects,
    load_project,
    save_project,
)
from llm_agent.chatbot.reminders_storage import (
    fetch_reminder,
    fetch_reminders,
    get_reminders_path,
    save_all_reminders,
    update_reminder_in_file,
)
from llm_agent.chatbot.storage import load_last_session, log_request_metric, save_session
from llm_agent.chatbot.task_storage import (
    delete_task_plan,
    find_plan_by_name,
    get_task_result_dir,
    list_task_plans,
    list_task_result_files,
    load_all_steps,
    load_task_plan,
    load_task_step,
    save_task_plan,
    save_task_result_file,
    save_task_step,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DRY-константы и хелперы
# ---------------------------------------------------------------------------

_MSG_NO_ACTIVE_TASK = "[Нет активной задачи]"
_MEMCLEAR_LABELS: dict = {
    "short": "краткосрочная",
    "short_term": "краткосрочная",
    "working": "рабочая",
    "long": "долговременная",
    "long_term": "долговременная",
    "all": "все уровни памяти",
}


def _now_iso() -> str:
    """Возвращает текущее UTC-время в ISO-формате (DRY-замена datetime.utcnow().isoformat())."""
    return datetime.utcnow().isoformat()


def _now_str() -> str:
    """Возвращает текущее UTC-время в формате '%Y-%m-%dT%H:%M:%SZ'."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_plan_flag(arg: str) -> tuple:
    """Извлекает --plan <name> из строки аргумента.

    Возвращает (cleaned_arg, plan_name_or_none).
    """
    m = re.search(r"--plan\s+(\S+)", arg)
    if m:
        plan_name = m.group(1)
        cleaned = (arg[: m.start()] + " " + arg[m.end() :]).strip()
        return cleaned, plan_name
    return arg, None


def _resolve_plan(state: "SessionState", plan_name: Optional[str]) -> Optional["TaskPlan"]:
    """Возвращает план по имени (из active_task_ids) или активный план."""
    if plan_name:
        task_id = find_plan_by_name(plan_name, state.profile_name)
        if not task_id:
            print(f"[План '{plan_name}' не найден. Доступные: /task list]")
            return None
        return load_task_plan(task_id, state.profile_name)
    return _get_active_plan(state)


def _require_active_plan(state: "SessionState") -> Optional["TaskPlan"]:
    """Возвращает активный план или None (с сообщением об ошибке).

    DRY-хелпер: заменяет повторяющийся шаблон:
        plan = _get_active_plan(state)
        if not plan:
            print(_MSG_NO_ACTIVE_TASK)
            return
    """
    plan = _get_active_plan(state)
    if not plan:
        print(_MSG_NO_ACTIVE_TASK)
    return plan


# ---------------------------------------------------------------------------
# Вспомогательные функции отображения
# ---------------------------------------------------------------------------


def _print_loaded_history(messages: List[ChatMessage]) -> None:
    """Выводит последние 5 обменов из истории диалога с метриками токенов."""
    try:
        turns = []
        filtered = [m for m in messages if m.role in {"user", "assistant"}]
        i = 0
        while i < len(filtered):
            msg = filtered[i]
            if msg.role == "user":
                assistant_msg = (
                    filtered[i + 1]
                    if i + 1 < len(filtered) and filtered[i + 1].role == "assistant"
                    else None
                )
                turns.append((
                    msg.content.strip(),
                    assistant_msg.content.strip() if assistant_msg else "",
                    assistant_msg.tokens if assistant_msg else None,
                ))
                i += 2
            else:
                i += 1

        last_turns = turns[-5:]
        if not last_turns:
            return

        print("\nLoaded history (last exchanges):")

        cumulative_prompt = 0
        cumulative_completion = 0
        cumulative_total = 0

        for _, _, td in turns[: -len(last_turns)]:
            if td:
                cumulative_prompt += td.prompt
                cumulative_completion += td.completion
                cumulative_total += td.total

        for user_text, assistant_text, token_data in last_turns:
            print(f"> {user_text}")
            print(assistant_text)
            if token_data:
                cumulative_prompt += token_data.prompt
                cumulative_completion += token_data.completion
                cumulative_total += token_data.total
                cost_rub = (cumulative_total / 1000.0) * USD_PER_1K_TOKENS * RUB_PER_USD
                print(
                    f"[Токены: запрос={token_data.prompt}, "
                    f"ответ={token_data.completion}, "
                    f"итого={token_data.total}]"
                )
                print(
                    f"[История диалога: промпт={cumulative_prompt}, "
                    f"ответ={cumulative_completion}, "
                    f"всего={cumulative_total} | ~{cost_rub:.2f}₽]"
                )
            else:
                print("[Токены: данные недоступны]")
        print()
    except Exception as exc:
        logger.debug("Failed to print loaded history: %s", exc)


def _print_strategy_status(state: SessionState) -> None:
    """Выводит текущую стратегию и связанный статус."""
    strategy = state.context_strategy
    total_msgs = len([m for m in state.messages if m.role != "system"])
    print(f"\n[Стратегия контекста: {strategy.value}]")
    print(f"  Сообщений в истории: {total_msgs}")
    if strategy == ContextStrategy.SLIDING_WINDOW:
        from llm_agent.chatbot.config import CONTEXT_RECENT_MESSAGES, CONTEXT_SUMMARY_INTERVAL
        print(f"  Окно (recent_n): {CONTEXT_RECENT_MESSAGES} | интервал суммаризации: {CONTEXT_SUMMARY_INTERVAL}")
        print(f"  Summary: {'есть' if state.context_summary else 'нет'}")
        if state.context_summary:
            preview = state.context_summary[:120].replace("\n", " ")
            print(f"  Превью: {preview}{'...' if len(state.context_summary) > 120 else ''}")
    elif strategy == ContextStrategy.STICKY_FACTS:
        count = len(state.sticky_facts.facts)
        print(f"  Фактов в памяти: {count}")
        if count:
            for k, v in state.sticky_facts.facts.items():
                print(f"    {k}: {v}")
    elif strategy == ContextStrategy.BRANCHING:
        branch_count = len(state.branches)
        active = state.active_branch_id or "нет"
        print(f"  Веток: {branch_count} | Активная: {active}")
        if state.last_checkpoint:
            print(f"  Последний checkpoint: {state.last_checkpoint.created_at}")
    print()


# ---------------------------------------------------------------------------
# Система управления задачами — вспомогательные функции
# ---------------------------------------------------------------------------

_STEP_ICONS = {
    StepStatus.PENDING: "[ ]",
    StepStatus.IN_PROGRESS: "[>]",
    StepStatus.DONE: "[x]",
    StepStatus.SKIPPED: "[-]",
    StepStatus.FAILED: "[!]",
}


def _build_plan_prompt(description: str) -> str:
    """Формирует промпт для LLM-генерации шагов плана."""
    return (
        "You are a task planning assistant. Given the task description below, "
        "generate a concise step-by-step plan as a JSON array. "
        "Each element must have fields: \"title\" (short action phrase) and "
        "\"description\" (one-sentence detail). Output ONLY the JSON array, "
        "no prose, no markdown fences.\n\n"
        f"Task: {description}"
    )


def _parse_steps_from_llm_response(text: str) -> Optional[List[dict]]:
    """Разбирает ответ LLM в список шагов. Три слоя устойчивости:

    1. Прямой json.loads
    2. Поиск JSON-массива регуляркой
    3. Нумерованный список как fallback
    """
    # Layer 1: direct parse
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: find JSON array in text
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Layer 3: numbered list fallback
    lines = text.splitlines()
    steps = []
    for line in lines:
        m2 = re.match(r"^\s*\d+[\.\)]\s+(.+)", line)
        if m2:
            steps.append({"title": m2.group(1).strip(), "description": ""})
    return steps if steps else None


def _validate_steps(raw_list: List[dict]) -> Optional[List[dict]]:
    """Проверяет и нормализует список шагов. Возвращает None если список пуст."""
    if not raw_list:
        return None
    result = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", item.get("step", ""))).strip()
        if not title:
            continue
        result.append({
            "title": title,
            "description": str(item.get("description", "")).strip(),
        })
    return result if result else None


def _create_task_plan(
    description: str,
    state: SessionState,
    client: OpenAI,
    steps: Optional[List[dict]] = None,
) -> Optional[TaskPlan]:
    """Вызывает LLM для генерации плана (или использует готовые шаги), сохраняет план и шаги, активирует задачу."""
    llm_raw_response = None
    if steps is None:
        print("[Генерация плана задачи...]")
        prompt = _build_plan_prompt(description)
        try:
            response = client.chat.completions.create(
                model=state.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw_text = response.choices[0].message.content or "" if response and response.choices else ""
        except Exception as exc:
            print(f"[Ошибка LLM при генерации плана: {exc}]")
            return None
        raw_steps = _parse_steps_from_llm_response(raw_text)
        if raw_steps is None:
            print("[Не удалось разобрать план из ответа LLM]")
            return None
        validated = _validate_steps(raw_steps)
        llm_raw_response = raw_text
    else:
        validated = _validate_steps(steps)

    if not validated:
        print("[План пуст или не содержит валидных шагов]")
        return None

    now = _now_iso()
    task_id = uuid.uuid4().hex
    plan = TaskPlan(
        task_id=task_id,
        profile_name=state.profile_name,
        name=description[:80],
        description=description,
        phase=TaskPhase.PLANNING,
        total_steps=len(validated),
        current_step_index=0,
        created_at=now,
        updated_at=now,
        llm_raw_response=llm_raw_response,
    )
    step_ids = []
    for i, s in enumerate(validated, start=1):
        step_id = f"{task_id}_step_{i:03d}"
        step = TaskStep(
            step_id=step_id,
            task_id=task_id,
            index=i,
            title=s["title"],
            description=s["description"],
            status=StepStatus.PENDING,
            created_at=now,
        )
        save_task_step(step, state.profile_name)
        step_ids.append(step_id)
    plan.step_ids = step_ids
    save_task_plan(plan, state.profile_name)
    state.active_task_id = task_id
    if task_id not in state.active_task_ids:
        state.active_task_ids.append(task_id)
    # Сохраняем сессию чтобы active_task_id не потерялся при сбое
    try:
        save_session(_build_session_payload(state), state.session_path)
    except Exception as exc:
        logger.debug("Session save after plan creation failed: %s", exc)

    steps = load_all_steps(task_id, state.profile_name)
    print(f"\n[План создан: {plan.name}]")
    _print_task_plan(plan, steps)
    return plan


def _print_task_plan(plan: TaskPlan, steps: List[TaskStep]) -> None:
    """Выводит план задачи с иконками статуса."""
    print(f"\n--- Задача: {plan.name} ---")
    print(f"  Фаза: {plan.phase.value}  |  Шаг {plan.current_step_index + 1}/{plan.total_steps}")
    for step in steps:
        icon = _STEP_ICONS.get(step.status, "[ ]")
        marker = " ◀" if step.index == plan.current_step_index + 1 else ""
        print(f"  {icon} {step.index}. {step.title}{marker}")
        if step.result:
            print(f"       Результат: {step.result}")
        if step.notes:
            print(f"       Заметка: {step.notes}")
    if plan.result:
        print(f"  Итог задачи: {plan.result}")
    print("--- Конец плана ---\n")


def _print_task_result(plan: TaskPlan, steps: List[TaskStep], state: "SessionState") -> None:
    """Выводит только результаты выполнения задачи."""
    print(f"\n=== Результат: {plan.name} ===")
    print(f"Статус: {plan.phase.value}")
    for step in steps:
        if step.result:
            icon = "✓" if step.status == StepStatus.DONE else "○"
            print(f"  [{icon}] Шаг {step.index}. {step.title}")
            print(f"       {step.result}")
    if plan.result:
        print(f"\nИтог задачи:\n{plan.result}")
    else:
        print("\n[Результат не записан. Используйте /task done <текст>]")
    result_files = list_task_result_files(plan.task_id, state.profile_name)
    if result_files:
        result_dir = get_task_result_dir(plan.task_id, state.profile_name)
        print(f"\nФайлы результата ({result_dir}):")
        for f in result_files:
            print(f"  {f}")
    print("=" * 40)


def _get_active_plan(state: SessionState) -> Optional[TaskPlan]:
    """Возвращает активный план или None."""
    if not state.active_task_id:
        return None
    return load_task_plan(state.active_task_id, state.profile_name)


def _transition_plan(
    plan: TaskPlan,
    new_phase: TaskPhase,
    state: SessionState,
    error_msg: str = "",
) -> bool:
    """Переводит план в новую фазу. Возвращает False при недопустимом переходе."""
    if not can_transition(plan.phase, new_phase):
        print(error_msg or f"[Переход {plan.phase.value} → {new_phase.value} недопустим]")
        return False
    plan.phase = new_phase
    plan.updated_at = _now_iso()
    if new_phase == TaskPhase.DONE:
        plan.completed_at = plan.updated_at
    save_task_plan(plan, state.profile_name)
    return True


def _advance_plan(plan: TaskPlan, state: SessionState) -> None:
    """Переходит к следующему шагу. При исчерпании — переводит в validation."""
    plan.current_step_index += 1
    plan.updated_at = _now_iso()
    if plan.current_step_index >= plan.total_steps:
        _transition_plan(plan, TaskPhase.VALIDATION, state)
        print("\n[Все шаги выполнены! Задача переходит в фазу: validation]")
        print("Используйте /task done для завершения или продолжите работу.\n")
    else:
        save_task_plan(plan, state.profile_name)
        next_step = load_task_step(plan.task_id, plan.current_step_index + 1, state.profile_name)
        if next_step:
            print(f"\n[Переход к шагу {plan.current_step_index + 1}: {next_step.title}]\n")


def _handle_step_subcommand(arg: str, state: SessionState) -> None:
    """Обрабатывает /task step done|skip|fail|note <текст> [--plan <name>]."""
    arg, _plan_name = _parse_plan_flag(arg)
    plan = _resolve_plan(state, _plan_name) if _plan_name else _require_active_plan(state)
    if not plan:
        return
    if plan.phase not in (TaskPhase.EXECUTION,):
        print(f"[Команда step доступна только в фазе execution. Текущая: {plan.phase.value}]")
        return

    parts = arg.split(None, 1)
    sub = parts[0].lower() if parts else ""
    text = parts[1].strip() if len(parts) > 1 else ""
    step_num = plan.current_step_index + 1
    step = load_task_step(plan.task_id, step_num, state.profile_name)
    if not step:
        print(f"[Шаг {step_num} не найден]")
        return

    now = _now_iso()
    if sub == "done":
        step.status = StepStatus.DONE
        step.completed_at = now
        if text:
            step.result = text
        save_task_step(step, state.profile_name)
        print(f"[Шаг {step_num} завершён: {step.title}]")
        if text:
            print(f"  Результат: {text}")
        _advance_plan(plan, state)
    elif sub == "skip":
        step.status = StepStatus.SKIPPED
        step.completed_at = now
        save_task_step(step, state.profile_name)
        print(f"[Шаг {step_num} пропущен: {step.title}]")
        _advance_plan(plan, state)
    elif sub == "fail":
        step.status = StepStatus.FAILED
        step.notes = text or step.notes
        step.completed_at = now
        save_task_step(step, state.profile_name)
        plan.failure_reason = text or f"Шаг {step_num} провалился"
        if _transition_plan(plan, TaskPhase.FAILED, state):
            state.active_task_id = None
            if plan.task_id in state.active_task_ids:
                state.active_task_ids.remove(plan.task_id)
            print(f"[Задача помечена как FAILED: {plan.failure_reason}]")
    elif sub == "note":
        step.notes = text
        save_task_step(step, state.profile_name)
        print(f"[Заметка добавлена к шагу {step_num}]")
    else:
        print(f"[Неизвестная подкоманда step: {sub!r}. Доступны: done, skip, fail, note]")


def _build_agent_state_vars(state: SessionState) -> str:
    """Формирует строку текущего состояния для инжекции в системный промпт агента."""
    parts = []
    wm = state.memory.working if state.memory else None
    if wm:
        if wm.current_task:
            parts.append(f"task: {wm.current_task}")
        if wm.task_status:
            parts.append(f"task_status: {wm.task_status}")
        if wm.user_preferences:
            for k, v in wm.user_preferences.items():
                parts.append(f"{k}: {v}")
    if state.active_task_id:
        parts.append(f"active_task_id: {state.active_task_id}")
        plan = load_task_plan(state.active_task_id, state.profile_name)
        if plan and plan.clarifications:
            clarif_lines = "; ".join(
                f"Q: {c['question']} → A: {c['answer']}"
                for c in plan.clarifications
            )
            parts.append(f"clarifications: [{clarif_lines}]")
    return "; ".join(parts) if parts else "(empty)"


def _collect_plan_clarifications(text: str, state: SessionState) -> None:
    """Извлекает вопросы из вывода агента, собирает ответы и сохраняет в план."""
    questions = parse_plan_questions(text)
    if not questions:
        return

    print("\n[Plan: уточняющие вопросы]")
    new_clarifications = []
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
        try:
            answer = input(f"  Ответ {i}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if answer:
            new_clarifications.append({"question": q, "answer": answer})

    if not new_clarifications:
        return

    if state.active_task_id:
        plan = load_task_plan(state.active_task_id, state.profile_name)
        if plan:
            plan.clarifications.extend(new_clarifications)
            plan.updated_at = _now_str()
            save_task_plan(plan, state.profile_name)
            print(f"[Plan: {len(new_clarifications)} уточнений сохранено в план {state.active_task_id}]")
            return

    # Нет активной задачи — сохраняем в рабочую память
    for c in new_clarifications:
        key = "clarif_" + re.sub(r"\W+", "_", c["question"][:30]).strip("_").lower()
        state.memory.working.set_preference(key, c["answer"])
    print(f"[Plan: {len(new_clarifications)} уточнений сохранено в рабочую память]")


_BUILDER_RETRIES_BEFORE_QUESTION = 3
_BUILDER_MAX_CLARIFICATION_ROUNDS = 2


def _call_llm_for_builder_step(
    step: "TaskStep",
    plan: "TaskPlan",
    state: SessionState,
    client: OpenAI,
) -> str:
    """Вызывает LLM для исполнения одного шага плана в режиме builder.

    Returns:
        Текст response из вывода агента.
    """
    all_steps = load_all_steps(plan.task_id, state.profile_name)
    done_steps = [s for s in all_steps if s.status == StepStatus.DONE]
    if done_steps:
        prev_lines = "\n".join(
            f"  {s.index}. [{s.status.value}] {s.title}" for s in done_steps
        )
    else:
        prev_lines = "(нет выполненных шагов)"

    profile_text = state.memory.get_profile_prompt() if state.memory else ""
    state_vars = _build_agent_state_vars(state)
    invariants = state.agent_mode.invariants

    clarif_lines = ""
    if plan.clarifications:
        clarif_lines = "; ".join(
            f"Q: {c['question']} → A: {c['answer']}" for c in plan.clarifications
        )

    step_desc = step.description
    if clarif_lines:
        step_desc = step_desc + f"\n\nClarifications: {clarif_lines}"

    system_prompt = build_builder_step_prompt(
        profile_text=profile_text,
        state_vars=state_vars,
        invariants=invariants,
        step_title=step.title,
        step_description=step_desc,
        previous_steps=prev_lines,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Execute step: {step.title}"},
    ]
    try:
        response = client.chat.completions.create(
            model=state.model,
            messages=messages,
            max_tokens=state.max_tokens,
            temperature=state.temperature,
        )
        raw = (response.choices[0].message.content or "") if response and response.choices else ""
    except Exception as exc:
        logger.warning("Builder LLM error: %s", exc)
        return ""

    response_text, state_update = parse_agent_output(raw)
    if state_update and state.memory:
        for k, v in state_update.items():
            state.memory.working.set_preference(k, v)
    return response_text


def _prompt_invariant_resolution(
    step: "TaskStep",
    violation: str,
    invariants: list,
    state: SessionState,
    client: OpenAI,
) -> str:
    """Интерактивный диалог при исчерпании попыток выполнить шаг.

    Позволяет пользователю отредактировать/удалить нарушенный инвариант.
    Шаг нельзя пропустить — builder обязан его выполнить или остановиться.

    Returns:
        "retry"  — инвариант изменён, продолжить
        "abort"  — остановить builder
    """
    print(f"\n[Builder: все попытки исчерпаны. Нарушение: {violation}]")
    print("\nЧто делать с проблемным инвариантом?")
    for i, inv in enumerate(invariants, 1):
        marker = " ← нарушен" if inv.lower() in violation.lower() or violation.lower() in inv.lower() else ""
        print(f"  {i}. {inv}{marker}")
    print("\nВарианты:")
    print("  edit <N> <новый текст> — изменить инвариант N")
    print("  remove <N>            — удалить инвариант N")
    print("  abort                 — остановить builder")

    while True:
        try:
            raw = input("\nВаш выбор: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Builder: прервано пользователем]")
            return "abort"

        if not raw:
            continue

        parts = raw.split(None, 2)
        cmd = parts[0].lower()

        if cmd == "abort":
            return "abort"

        if cmd == "remove" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
            except ValueError:
                print("[Ожидается номер инварианта]")
                continue
            if not (0 <= idx < len(invariants)):
                print(f"[Нет инварианта с номером {idx + 1}]")
                continue
            removed = invariants[idx]
            others = [inv for i, inv in enumerate(invariants) if i != idx]
            warning = analyze_invariant_impact(client, state.model, removed, None, others)
            if warning:
                print(f"\n{warning}")
                confirm = input("Всё равно удалить? (да/нет): ").strip().lower()
                if confirm not in ("да", "yes", "y"):
                    continue
            invariants.pop(idx)
            print(f"[Инвариант удалён: {removed}]")
            return "retry"

        if cmd == "edit" and len(parts) >= 3:
            try:
                idx = int(parts[1]) - 1
            except ValueError:
                print("[Ожидается номер инварианта]")
                continue
            if not (0 <= idx < len(invariants)):
                print(f"[Нет инварианта с номером {idx + 1}]")
                continue
            old_text = invariants[idx]
            new_text = parts[2].strip()
            others = [inv for i, inv in enumerate(invariants) if i != idx]
            warning = analyze_invariant_impact(client, state.model, old_text, new_text, others)
            if warning:
                print(f"\n{warning}")
                confirm = input("Всё равно изменить? (да/нет): ").strip().lower()
                if confirm not in ("да", "yes", "y"):
                    continue
            invariants[idx] = new_text
            print(f"[Инвариант #{idx + 1} обновлён: {new_text}]")
            return "retry"

        print("[Неверная команда. Доступны: edit <N> <текст>, remove <N>, abort]")


def _execute_builder_step(
    step: "TaskStep",
    plan: "TaskPlan",
    state: SessionState,
    client: OpenAI,
) -> bool:
    """Исполняет один шаг плана с retry и clarification loop.

    Returns:
        True если шаг выполнен успешно, False если не справился.
    """
    invariants = state.agent_mode.invariants
    retry_count = 0
    clarif_rounds = 0

    while True:
        print(f"\n[Builder: выполняю шаг {step.index}. {step.title}...]")
        draft = _call_llm_for_builder_step(step, plan, state, client)

        if not draft:
            passed, violation = False, "пустой ответ от LLM"
        elif invariants:
            passed, violation = validate_draft_against_invariants(
                client, state.model, draft, invariants
            )
        else:
            passed, violation = True, ""

        if passed:
            step.status = StepStatus.DONE
            step.result = draft
            step.completed_at = _now_iso()
            save_task_step(step, state.profile_name)
            # Извлечь файлы из ответа LLM и сохранить в result/
            files = extract_result_files(draft)
            for rel_path, content in files:
                try:
                    saved = save_task_result_file(step.task_id, rel_path, content, state.profile_name)
                    logger.info("Result file saved: %s", saved)
                except Exception as exc:
                    logger.warning("Failed to save result file '%s': %s", rel_path, exc)
            try:
                save_session(_build_session_payload(state), state.session_path)
            except Exception as exc:
                logger.debug("Session save after step completion failed: %s", exc)
            print(f"\n[Builder: шаг {step.index} DONE]\n{draft}")
            return True

        retry_count += 1
        print(
            f"[Builder: инвариант нарушен ({violation}). "
            f"Повтор {retry_count}/{_BUILDER_RETRIES_BEFORE_QUESTION}...]"
        )

        if retry_count < _BUILDER_RETRIES_BEFORE_QUESTION:
            continue  # инварианты уже в системном промпте, retry без фейковых записей

        # 3 fail → спрашиваем пользователя
        if clarif_rounds >= _BUILDER_MAX_CLARIFICATION_ROUNDS:
            result = _prompt_invariant_resolution(step, violation, invariants, state, client)
            if result == "abort":
                print(f"[Builder: шаг {step.index} не удалось выполнить после всех попыток]")
                return False
            # "retry" — инвариант изменён, обновляем локальный список и сбрасываем счётчики
            invariants = state.agent_mode.invariants
            retry_count = 0
            clarif_rounds = 0
            continue

        clarif_rounds += 1
        question = generate_clarification_question(
            client, state.model, step.title, step.description, violation
        )
        print(f"\n[Builder: уточняющий вопрос]\n{question}")
        try:
            answer = input("Ваш ответ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Builder: прервано пользователем]")
            return False

        plan.clarifications.append({"question": question, "answer": answer})
        plan.updated_at = _now_str()
        save_task_plan(plan, state.profile_name)
        retry_count = 0  # reset after clarification


def _run_plan_builder(state: SessionState, client: OpenAI) -> None:
    """Автоматически исполняет все pending-шаги активного TaskPlan."""
    if not state.active_task_id:
        all_plans = list_task_plans(state.profile_name)
        unfinished = [p for p in all_plans if p["phase"] not in ("done", "failed")]
        if not unfinished:
            print("[Plan builder: нет активной задачи. Используйте /task new или /task load]")
            return
        latest = max(unfinished, key=lambda p: p.get("updated_at") or p.get("created_at") or "")
        state.active_task_id = latest["task_id"]
        print(f"[Plan builder: загружена задача '{latest['name']}']")

    plan = load_task_plan(state.active_task_id, state.profile_name)
    if not plan:
        print(f"[Plan builder: план не найден: {state.active_task_id}]")
        return

    steps = load_all_steps(plan.task_id, state.profile_name)
    pending = [s for s in steps if s.status not in (StepStatus.DONE, StepStatus.SKIPPED)]

    if not pending:
        print("[Plan builder: все шаги уже выполнены]")
        return

    print(f"\n[Plan builder: запуск — {len(pending)} шагов к исполнению]")
    try:
        for step in pending:
            success = _execute_builder_step(step, plan, state, client)
            if not success:
                print(f"[Plan builder: остановлен на шаге {step.index}]")
                return
    except KeyboardInterrupt:
        print(f"\n[Plan builder: прерван на шаге {step.index}. Прогресс сохранён. Продолжите через /plan builder]")
        return

    # Проверяем финальный статус
    all_steps = load_all_steps(plan.task_id, state.profile_name)
    all_done = all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in all_steps)
    if all_done:
        # Перезагружаем план с диска (мог обновиться в ходе шагов)
        plan = load_task_plan(plan.task_id, state.profile_name) or plan
        result_parts = []
        for s in all_steps:
            if s.result:
                result_parts.append(f"Шаг {s.index}. {s.title}:\n{s.result}")
        if result_parts:
            plan.result = "\n\n".join(result_parts)
        # EXECUTION → VALIDATION → DONE (соблюдаем граф переходов)
        if plan.phase == TaskPhase.EXECUTION:
            _transition_plan(plan, TaskPhase.VALIDATION, state)
        _transition_plan(plan, TaskPhase.DONE, state)
        if plan.task_id in state.active_task_ids:
            state.active_task_ids.remove(plan.task_id)
        print("\n[Plan builder: ПЛАН ВЫПОЛНЕН]")
        result_files = list_task_result_files(plan.task_id, state.profile_name)
        if result_files:
            result_dir = get_task_result_dir(plan.task_id, state.profile_name)
            print(f"[Plan builder: файлы результата сохранены в {result_dir}]")
            for f in result_files:
                print(f"  {f}")


def _print_draft_plan(steps: List[dict]) -> None:
    """Выводит черновик плана из шагов."""
    print("\n[Черновик плана:]")
    for i, s in enumerate(steps, 1):
        title = s.get("title", f"Шаг {i}")
        desc = s.get("description", "")
        print(f"  {i}. {title}" + (f" — {desc}" if desc else ""))


def _extract_task_description(messages: List[ChatMessage]) -> str:
    """Берёт первое значимое сообщение пользователя из диалога как описание задачи."""
    for m in messages:
        if m.role == "user" and m.content.strip() and m.content != "Начинаем планирование.":
            return m.content.strip()[:120]
    return "Задача из диалога планирования"


def _kick_off_plan_dialog(state: SessionState, client: OpenAI, description: str = "") -> None:
    """Делает первый LLM-вызов, чтобы ассистент задал первый вопрос."""
    system_prompt = build_plan_dialog_prompt(state.agent_mode.invariants)
    if description:
        user_content = f"Задача: {description}\n\nНачинаем планирование."
    else:
        user_content = "Начинаем планирование."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = client.chat.completions.create(
            model=state.model, messages=messages, max_tokens=512, temperature=0.5
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Plan dialog kick-off error: %s", exc)
        return
    display, _ = parse_agent_output(text)
    print(display)
    state.messages.append(ChatMessage(role="user", content=user_content))
    state.messages.append(ChatMessage(role="assistant", content=text))


def _handle_plan_dialog_message(user_input: str, state: SessionState, client: OpenAI) -> None:
    """Обрабатывает сообщение пользователя в активном диалоге планирования."""
    state.messages.append(ChatMessage(role="user", content=user_input))

    system_prompt = build_plan_dialog_prompt(state.agent_mode.invariants)
    api_msgs = [{"role": "system", "content": system_prompt}]
    api_msgs += [m.to_api_dict() for m in state.messages[-CONTEXT_RECENT_MESSAGES:]]

    try:
        resp = client.chat.completions.create(
            model=state.model,
            messages=api_msgs,
            max_tokens=state.max_tokens,
            temperature=state.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"API error: {exc}")
        return

    # Validate against invariants with retry loop
    if state.agent_mode.invariants:
        draft = text
        passed, violation = validate_draft_against_invariants(
            client, state.model, draft, state.agent_mode.invariants
        )
        retry_count = 0
        while not passed and retry_count < state.agent_mode.max_retries:
            retry_count += 1
            print(f"[Plan: инвариант нарушен ({violation}). Повтор {retry_count}/{state.agent_mode.max_retries}...]")
            retry_messages = [
                *api_msgs,
                {"role": "assistant", "content": draft},
                {"role": "user", "content": (
                    f"Твой ответ нарушает инвариант: {violation}. "
                    "Скорректируй ответ с учётом всех инвариантов."
                )},
            ]
            try:
                retry_resp = client.chat.completions.create(
                    model=state.model, messages=retry_messages,
                    max_tokens=state.max_tokens, temperature=state.temperature,
                )
                draft = (retry_resp.choices[0].message.content or "").strip()
            except Exception as exc:
                logger.warning("Plan dialog retry error: %s", exc)
                break
            passed, violation = validate_draft_against_invariants(
                client, state.model, draft, state.agent_mode.invariants
            )

        if not passed:
            state.messages.pop()
            print(f"[Plan: инвариант не пройден — {violation}]")
            print("[Запрос отклонён. Уточните задачу с учётом инвариантов:]")
            return

        text = draft

    state.messages.append(ChatMessage(role="assistant", content=text))
    display, _ = parse_agent_output(text)

    draft_raw = parse_draft_plan_block(text)
    if draft_raw:
        steps = _parse_steps_from_llm_response(draft_raw)
        if steps:
            state.plan_draft_steps = steps
            state.plan_draft_description = _extract_task_description(state.messages)
            state.plan_dialog_state = "confirming"
            print(display)
            _print_draft_plan(steps)
            print("\n[Создать задачи? Введите 'да' для подтверждения или 'нет' для продолжения диалога]")
            return

    print(display)
    _collect_plan_clarifications(text, state)


_AFFIRMATIVE = {"да", "yes", "y", "д", "ок", "ok", "конечно", "создай", "создавай"}
_NEGATIVE = {"нет", "no", "н", "n", "не", "пропустить", "skip"}
_DONE_WORDS = {"готово", "done", "start", "старт", "go", "начать", "begin"}


def _handle_plan_awaiting_task(user_input: str, state: SessionState) -> None:
    """Обрабатывает ввод описания задачи в начале Plan-режима."""
    description = user_input.strip()
    if not description:
        print("[Введите описание задачи (или /plan cancel для отмены):]")
        return
    state.plan_draft_description = description
    print("[Хотите добавить инварианты? (да/нет/готово | /plan cancel для отмены):]")
    state.plan_dialog_state = "awaiting_invariants"


def _handle_plan_awaiting_invariants(user_input: str, state: SessionState, client: Optional[OpenAI]) -> None:
    """Обрабатывает ответ про инварианты перед запуском план-диалога."""
    word = user_input.strip().lower().split()[0] if user_input.strip() else ""
    if word in _NEGATIVE or word in _DONE_WORDS:
        print("[Запускаю диалог планирования...]")
        state.plan_dialog_state = "active"
        if client is not None:
            _kick_off_plan_dialog(state, client, description=state.plan_draft_description)
    elif word in _AFFIRMATIVE:
        print("[Добавляйте инварианты через `/invariant add <текст>`. Введите 'готово' когда закончите.]")
        # state остаётся "awaiting_invariants"
    else:
        print("[Введите 'да' чтобы добавить инварианты, 'нет' чтобы пропустить, 'готово' когда добавили. /plan cancel для отмены.]")


def _confirm_and_create_tasks(user_input: str, state: SessionState, client: OpenAI) -> None:
    """Подтверждение или отказ от создания задач из черновика плана."""
    word = user_input.strip().lower().split()[0] if user_input.strip() else ""
    if word in _AFFIRMATIVE:
        print("[Создаю задачи...]")
        _create_task_plan(
            state.plan_draft_description or "Задача из диалога планирования",
            state,
            client,
            steps=state.plan_draft_steps,
        )
        state.plan_dialog_state = None
        state.plan_draft_steps = []
        state.plan_draft_description = ""
        print("[Задачи созданы! Используйте /plan builder для выполнения]")
    else:
        state.plan_dialog_state = "active"
        print("[Продолжаем диалог. Уточните требования:]")
        _handle_plan_dialog_message(user_input, state, client)


def _handle_agent_command(action: str, arg: str, state: SessionState, client: Optional[OpenAI] = None) -> None:
    """Диспетчер команды /plan on|off|status|builder."""
    if action in ("on", "enable"):
        state.agent_mode.enabled = True
        state.plan_dialog_state = "awaiting_task"
        state.plan_draft_steps = []
        state.plan_draft_description = ""
        print(f"[Plan mode: ON | инвариантов: {len(state.agent_mode.invariants)} | max_retries: {state.agent_mode.max_retries}]")
        if state.memory is not None:
            p = state.memory.long_term.profile
            if p.is_empty():
                print(f"[Профиль: {p.name} | (нет настроек)]")
            else:
                print(f"[Профиль: {p.name}]")
                if p.style:
                    print("  Стиль:       " + ", ".join(f"{k}={v}" for k, v in p.style.items()))
                if p.format:
                    print("  Формат:      " + ", ".join(f"{k}={v}" for k, v in p.format.items()))
                if p.constraints:
                    print("  Ограничения: " + "; ".join(p.constraints))
                if p.custom:
                    print("  Дополнения:  " + ", ".join(f"{k}={v}" for k, v in p.custom.items()))
        print("[Введите описание задачи (или /plan cancel для отмены):]")
    elif action in ("off", "disable"):
        state.agent_mode.enabled = False
        state.plan_dialog_state = None
        state.plan_draft_steps = []
        state.plan_draft_description = ""
        print("[Plan mode: OFF]")
    elif action == "status":
        status = "ON" if state.agent_mode.enabled else "OFF"
        print(f"[Plan mode: {status} | инвариантов: {len(state.agent_mode.invariants)} | max_retries: {state.agent_mode.max_retries}]")
        if state.agent_mode.invariants:
            print("Инварианты:")
            for i, inv in enumerate(state.agent_mode.invariants, 1):
                print(f"  {i}. {inv}")
    elif action == "retries":
        try:
            n = int(arg)
            state.agent_mode.max_retries = max(1, min(10, n))
            print(f"[Plan max_retries: {state.agent_mode.max_retries}]")
        except (ValueError, TypeError):
            print("[plan retries: ожидается целое число]")
    elif action == "builder":
        _, plan_name = _parse_plan_flag(arg)
        if plan_name:
            orig_task_id = state.active_task_id
            task_id = find_plan_by_name(plan_name, state.profile_name)
            if not task_id:
                print(f"[Plan builder: план '{plan_name}' не найден. Доступные: /task list]")
            else:
                state.active_task_id = task_id
                _run_plan_builder(state, client)
                state.active_task_id = orig_task_id
        else:
            _run_plan_builder(state, client)
    elif action == "result":
        _handle_task_command("result", arg, state, client)
    elif action == "cancel":
        state.plan_dialog_state = None
        state.plan_draft_steps = []
        state.plan_draft_description = ""
        print("[Plan dialog: отменён. Состояние сброшено.]")
    else:
        print(f"[Неизвестная подкоманда plan: {action!r}. Доступны: on, off, status, retries <n>, builder, result, cancel]")


def _handle_invariant_command(action: str, arg: str, state: SessionState) -> None:
    """Диспетчер команды /invariant add|del|list."""
    inv = state.agent_mode.invariants
    if action == "add":
        if not arg:
            print("[invariant add: требует текст инварианта]")
            return
        inv.append(arg)
        print(f"[Инвариант добавлен #{len(inv)}: {arg}]")
    elif action in ("del", "delete", "rm"):
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(inv):
                removed = inv.pop(idx)
                print(f"[Инвариант удалён: {removed}]")
            else:
                print(f"[Нет инварианта с номером {int(arg)}]")
        except (ValueError, TypeError):
            print("[invariant del: ожидается номер инварианта]")
    elif action == "list":
        if not inv:
            print("[Инварианты не заданы. Используйте /invariant add <текст>]")
        else:
            print("\n--- Инварианты ---")
            for i, text in enumerate(inv, 1):
                print(f"  {i}. {text}")
            print("--- Конец ---\n")
    elif action in ("edit", "update"):
        parts = arg.split(None, 1) if arg else []
        if len(parts) < 2:
            print("[invariant edit: ожидается 'edit <N> <новый текст>']")
            return
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            print("[invariant edit: ожидается номер инварианта]")
            return
        if not (0 <= idx < len(inv)):
            print(f"[Нет инварианта с номером {idx + 1}]")
            return
        old_text = inv[idx]
        inv[idx] = parts[1].strip()
        print(f"[Инвариант #{idx + 1} изменён: '{old_text}' → '{inv[idx]}']")
    elif action == "clear":
        inv.clear()
        print("[Все инварианты удалены]")
    else:
        print(f"[Неизвестная подкоманда invariant: {action!r}. Доступны: add, del, edit, list, clear]")


def _handle_project_command(action: str, arg: str, state: SessionState, client: Optional[OpenAI] = None) -> None:
    """Диспетчер команды /project."""
    if action == "new":
        if not arg:
            print("[/project new требует название проекта]")
            return
        project_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        project = Project(
            project_id=project_id,
            name=arg,
            profile_name=state.profile_name,
            created_at=now,
            updated_at=now,
        )
        save_project(project, state.profile_name)
        state.active_project_id = project_id
        print(f"[Проект создан: {project.name} | ID: {project_id[:8]}]")

    elif action == "list":
        projects = list_projects(state.profile_name)
        if not projects:
            print("[Проектов нет. Создайте: /project new <название>]")
            return
        print("\n--- Проекты ---")
        for p in projects:
            active_mark = " ◀ активный" if p["project_id"] == state.active_project_id else ""
            print(f"  [{p['project_id'][:8]}] {p['name']} | планов: {p['plan_count']}{active_mark}")
        print("---\n")

    elif action == "switch":
        if not arg:
            print("[/project switch требует название или ID проекта]")
            return
        projects = list_projects(state.profile_name)
        match = next(
            (p for p in projects if p["name"].lower() == arg.lower() or p["project_id"].startswith(arg)),
            None,
        )
        if not match:
            print(f"[Проект не найден: {arg}]")
            return
        state.active_project_id = match["project_id"]
        print(f"[Активный проект: {match['name']}]")

    elif action == "show":
        if not state.active_project_id:
            print("[Нет активного проекта. Выберите: /project switch <название>]")
            return
        project = load_project(state.active_project_id, state.profile_name)
        if not project:
            print(f"[Проект не найден: {state.active_project_id}]")
            return
        print(f"\n--- Проект: {project.name} ---")
        if project.description:
            print(f"  {project.description}")
        if not project.plan_ids:
            print("  (нет планов)")
        else:
            for tid in project.plan_ids:
                plan = load_task_plan(tid, state.profile_name)
                if plan:
                    model_tag = f" [{plan.model}]" if plan.model else ""
                    active_tag = " ◀ активный" if tid == state.active_task_id else ""
                    print(f"  [{plan.phase.value}] {plan.name}{model_tag}{active_tag}")
        print("---\n")

    elif action in ("add_plan", "add-plan"):
        if not arg:
            print("[/project add-plan требует ID задачи]")
            return
        if not state.active_project_id:
            print("[Нет активного проекта. Выберите: /project switch <название>]")
            return
        project = load_project(state.active_project_id, state.profile_name)
        if not project:
            print(f"[Проект не найден: {state.active_project_id}]")
            return
        plan = load_task_plan(arg, state.profile_name)
        if not plan:
            print(f"[Задача не найдена: {arg}]")
            return
        if arg not in project.plan_ids:
            project.plan_ids.append(arg)
            project.updated_at = _now_iso()
            save_project(project, state.profile_name)
        plan.project_id = project.project_id
        save_task_plan(plan, state.profile_name)
        print(f"[План '{plan.name}' добавлен в проект '{project.name}']")

    elif action == "add_plan_name":
        # /project add-plan-name NAME — привязать задачу по имени
        if not arg:
            print("[/project add-plan-name требует имя задачи]")
            return
        if not state.active_project_id:
            print("[Нет активного проекта. Выберите: /project switch <название>]")
            return
        task_id = find_plan_by_name(arg, state.profile_name)
        if not task_id:
            print(f"[Задача с именем '{arg}' не найдена]")
            return
        project = load_project(state.active_project_id, state.profile_name)
        if not project:
            print(f"[Проект не найден: {state.active_project_id}]")
            return
        plan = load_task_plan(task_id, state.profile_name)
        if task_id not in project.plan_ids:
            project.plan_ids.append(task_id)
            project.updated_at = _now_iso()
            save_project(project, state.profile_name)
        if plan:
            plan.project_id = project.project_id
            save_task_plan(plan, state.profile_name)
            print(f"[План '{plan.name}' добавлен в проект '{project.name}']")

    elif action == "tasks":
        # /project tasks — список задач в активном проекте
        if not state.active_project_id:
            print("[Нет активного проекта. Выберите: /project switch <название>]")
            return
        project = load_project(state.active_project_id, state.profile_name)
        if not project:
            print(f"[Проект не найден: {state.active_project_id}]")
            return
        if not project.plan_ids:
            print(f"[Проект '{project.name}': задач нет. Создайте: /project task new <описание>]")
            return
        print(f"\n--- Задачи проекта: {project.name} ---")
        for tid in project.plan_ids:
            plan = load_task_plan(tid, state.profile_name)
            if plan:
                steps = load_all_steps(tid, state.profile_name)
                done_steps = sum(1 for s in steps if s.status.value == "done")
                active_tag = " ◀ активная" if tid == state.active_task_id else ""
                print(f"  [{plan.phase.value}] {plan.name} ({done_steps}/{plan.total_steps} шагов){active_tag}")
            else:
                print(f"  [?] {tid[:8]} (не найдена)")
        print("---\n")

    elif action == "task_new":
        # /project task new DESC или /project plan new DESC
        if not arg:
            print("[/project task new требует описание задачи]")
            return
        if not state.active_project_id:
            print("[Нет активного проекта. Выберите: /project switch <название>]")
            return
        project = load_project(state.active_project_id, state.profile_name)
        if not project:
            print(f"[Проект не найден: {state.active_project_id}]")
            return
        _create_task_plan(arg, state, client)
        if state.active_task_id and state.active_task_id not in project.plan_ids:
            project.plan_ids.append(state.active_task_id)
            project.updated_at = _now_iso()
            save_project(project, state.profile_name)
            new_plan = load_task_plan(state.active_task_id, state.profile_name)
            if new_plan:
                new_plan.project_id = project.project_id
                save_task_plan(new_plan, state.profile_name)

    elif action == "task_rename":
        # /project task rename NEW_NAME [--plan NAME]
        _, plan_name_flag = _parse_plan_flag(arg)
        new_name = arg.replace(f"--plan {plan_name_flag}", "").strip() if plan_name_flag else arg.strip()
        plan = _resolve_plan(state, plan_name_flag)
        if not plan:
            print("[/project task rename: нет активного плана. Укажите --plan <имя> или активируйте задачу]")
            return
        if not new_name:
            print("[/project task rename: требует новое имя]")
            return
        plan.name = new_name
        plan.updated_at = _now_iso()
        save_task_plan(plan, state.profile_name)
        print(f"[Задача переименована: '{new_name}']")

    elif action == "task_describe":
        # /project task describe TEXT [--plan NAME]
        _, plan_name_flag = _parse_plan_flag(arg)
        new_desc = arg.replace(f"--plan {plan_name_flag}", "").strip() if plan_name_flag else arg.strip()
        plan = _resolve_plan(state, plan_name_flag)
        if not plan:
            print("[/project task describe: нет активного плана. Укажите --plan <имя> или активируйте задачу]")
            return
        if not new_desc:
            print("[/project task describe: требует текст описания]")
            return
        plan.description = new_desc
        plan.updated_at = _now_iso()
        save_task_plan(plan, state.profile_name)
        print(f"[Описание задачи обновлено]")

    else:
        print(f"[Неизвестная подкоманда project: {action!r}. Доступны: new, list, switch, show, add-plan, add-plan-name, tasks, task new, task rename, task describe]")


def _handle_task_command(action: str, arg: str, state: SessionState, client: OpenAI) -> None:
    """Диспетчер команды /task."""
    if action == "new":
        if not arg:
            print("[/task new требует описания задачи]")
            return
        _create_task_plan(arg, state, client)

    elif action == "show":
        plan = _require_active_plan(state)
        if not plan:
            return
        steps = load_all_steps(plan.task_id, state.profile_name)
        _print_task_plan(plan, steps)

    elif action == "list":
        plans = list_task_plans(state.profile_name)
        if not plans:
            print("[Задач не найдено]")
            return
        print("\n--- Список задач ---")
        for p in plans:
            active_mark = (
                " ◀ активная" if p["task_id"] == state.active_task_id
                else " ◀ запущена" if p["task_id"] in state.active_task_ids
                else ""
            )
            print(
                f"  [{p['task_id'][:8]}] {p['name']} | {p['phase']} "
                f"| {p['current_step_index']}/{p['total_steps']}{active_mark}"
            )
        print("--- Конец ---\n")

    elif action == "start":
        _arg, _plan_name = _parse_plan_flag(arg)
        plan = _resolve_plan(state, _plan_name) if _plan_name else _require_active_plan(state)
        if not plan:
            return
        if plan.phase != TaskPhase.PLANNING:
            print(f"[Задача уже в фазе: {plan.phase.value}]")
            return
        if _plan_name:
            state.active_task_id = plan.task_id
            if plan.task_id not in state.active_task_ids:
                state.active_task_ids.append(plan.task_id)
        first_step = load_task_step(plan.task_id, 1, state.profile_name)
        if first_step:
            first_step.status = StepStatus.IN_PROGRESS
            first_step.started_at = _now_iso()
            save_task_step(first_step, state.profile_name)
        _transition_plan(plan, TaskPhase.EXECUTION, state)
        steps = load_all_steps(plan.task_id, state.profile_name)
        print(f"\n[Задача запущена: {plan.name}]")
        _print_task_plan(plan, steps)

    elif action == "step":
        _handle_step_subcommand(arg, state)

    elif action == "pause":
        _arg, _plan_name = _parse_plan_flag(arg)
        plan = _resolve_plan(state, _plan_name) if _plan_name else _require_active_plan(state)
        if not plan:
            return
        if plan.phase != TaskPhase.EXECUTION:
            print(
                f"[Нельзя приостановить задачу в фазе {plan.phase.value}. Нужна фаза: execution]"
            )
            return
        _transition_plan(plan, TaskPhase.PAUSED, state)
        print(f"[Задача приостановлена: {plan.name}]")

    elif action == "resume":
        _arg, _plan_name = _parse_plan_flag(arg)
        _lookup = _plan_name or _arg.strip()
        if _lookup:
            plan = load_task_plan(_lookup, state.profile_name)
            if not plan:
                # попробуем найти по имени
                _tid = find_plan_by_name(_lookup, state.profile_name)
                plan = load_task_plan(_tid, state.profile_name) if _tid else None
            if not plan:
                print(f"[Задача не найдена: {_lookup}]")
                return
            state.active_task_id = plan.task_id
            if plan.task_id not in state.active_task_ids:
                state.active_task_ids.append(plan.task_id)
        else:
            plan = _get_active_plan(state)
            if not plan:
                print("[Нет активной задачи для возобновления]")
                return
        if plan.phase not in (TaskPhase.PAUSED, TaskPhase.VALIDATION):
            print(
                f"[Нельзя возобновить задачу в фазе {plan.phase.value}. "
                "Нужна фаза: paused или validation]"
            )
            return
        _transition_plan(plan, TaskPhase.EXECUTION, state)
        steps = load_all_steps(plan.task_id, state.profile_name)
        print(f"\n[Задача возобновлена: {plan.name}]")
        _print_task_plan(plan, steps)

    elif action == "done":
        _arg, _plan_name = _parse_plan_flag(arg)
        plan = _resolve_plan(state, _plan_name) if _plan_name else _require_active_plan(state)
        if not plan:
            return
        if plan.phase != TaskPhase.VALIDATION:
            print(
                f"[Нельзя завершить задачу в фазе {plan.phase.value}. "
                "Сначала выполните все шаги (/task step done) — задача перейдёт в validation.]"
            )
            return
        if _arg:
            plan.result = _arg
        _transition_plan(plan, TaskPhase.DONE, state)
        state.active_task_id = None
        if plan.task_id in state.active_task_ids:
            state.active_task_ids.remove(plan.task_id)
        print(f"[Задача завершена: {plan.name}]")
        if _arg:
            print(f"  Итог: {_arg}")

    elif action == "fail":
        _arg, _plan_name = _parse_plan_flag(arg)
        plan = _resolve_plan(state, _plan_name) if _plan_name else _require_active_plan(state)
        if not plan:
            return
        if plan.phase == TaskPhase.DONE:
            print("[Нельзя провалить завершённую задачу. Фаза: done — терминальное состояние.]")
            return
        if plan.phase == TaskPhase.FAILED:
            print("[Задача уже помечена как FAILED.]")
            return
        plan.failure_reason = _arg or "Задача провалена вручную"
        if _transition_plan(plan, TaskPhase.FAILED, state):
            state.active_task_id = None
            if plan.task_id in state.active_task_ids:
                state.active_task_ids.remove(plan.task_id)
            print(f"[Задача помечена как FAILED: {plan.failure_reason}]")

    elif action == "load":
        if not arg:
            print("[/task load требует ID задачи]")
            return
        plan = load_task_plan(arg, state.profile_name)
        if not plan:
            print(f"[Задача не найдена: {arg}]")
            return
        state.active_task_id = plan.task_id
        steps = load_all_steps(plan.task_id, state.profile_name)
        print(f"[Активирована задача: {plan.name}]")
        _print_task_plan(plan, steps)

    elif action == "delete":
        if not arg:
            print("[/task delete требует ID задачи]")
            return
        ok = delete_task_plan(arg, state.profile_name)
        if ok:
            if state.active_task_id == arg:
                state.active_task_id = None
            print(f"[Задача удалена: {arg}]")
        else:
            print(f"[Задача не найдена: {arg}]")

    elif action == "result":
        task_id = arg.strip() if arg.strip() else state.active_task_id
        if not task_id:
            print("[Нет активной задачи. Укажите ID: /task result <id>]")
            return
        plan = load_task_plan(task_id, state.profile_name)
        if not plan:
            print(f"[Задача не найдена: {task_id}]")
            return
        steps = load_all_steps(plan.task_id, state.profile_name)
        _print_task_result(plan, steps, state)

    else:
        print(f"[Неизвестная подкоманда task: {action!r}]")
        print("Доступны: new, show, list, start, step, pause, resume, done, fail, load, delete, result")


# ---------------------------------------------------------------------------
# Загрузка и применение данных сессии
# ---------------------------------------------------------------------------


def _load_messages_from_dict(raw: list) -> List[ChatMessage]:
    """Преобразует список словарей из JSON-сессии в список ChatMessage."""
    result: List[ChatMessage] = []
    for item in raw:
        tokens = item.get("tokens")
        token_usage: Optional[TokenUsage] = None
        if isinstance(tokens, dict):
            token_usage = TokenUsage(
                prompt=tokens.get("prompt", 0),
                completion=tokens.get("completion", 0),
                total=tokens.get("total", 0),
            )
        result.append(ChatMessage(
            role=item.get("role", "user"),
            content=item.get("content") or "",
            tokens=token_usage,
            tool_call_id=item.get("tool_call_id"),
        ))
    return result


def _apply_session_data(data: dict, state: SessionState) -> None:
    """Применяет данные загруженной сессии к рабочему состоянию."""
    state.messages = _load_messages_from_dict(data.get("messages", []))
    state.model = data.get("model", state.model)
    state.base_url = data.get("base_url", state.base_url)
    state.system_prompt = data.get("system_prompt", state.system_prompt)
    state.initial_prompt = data.get("initial_prompt", state.initial_prompt)
    state.duration = data.get("duration_seconds", 0.0)
    state.total_tokens = data.get("total_tokens", 0)
    state.total_prompt_tokens = data.get("total_prompt_tokens", 0)
    state.total_completion_tokens = data.get("total_completion_tokens", 0)
    state.context_summary = data.get("context_summary", "")
    # Восстанавливаем стратегию и факты если были сохранены
    if "context_strategy" in data:
        with contextlib.suppress(ValueError):
            state.context_strategy = ContextStrategy(data["context_strategy"])
    if "sticky_facts" in data and isinstance(data["sticky_facts"], dict):
        state.sticky_facts = StickyFacts(facts=data["sticky_facts"])
    if "branches" in data and isinstance(data["branches"], list):
        state.branches = [Branch(**b) if isinstance(b, dict) else b for b in data["branches"]]
    if "active_branch_id" in data:
        state.active_branch_id = data["active_branch_id"]
    if "active_task_id" in data:
        state.active_task_id = data["active_task_id"]
    if "active_task_ids" in data and isinstance(data["active_task_ids"], list):
        state.active_task_ids = data["active_task_ids"]
    elif state.active_task_id and state.active_task_id not in state.active_task_ids:
        state.active_task_ids.append(state.active_task_id)
    if "active_project_id" in data:
        state.active_project_id = data["active_project_id"]
    if data.get("agent_mode"):
        state.agent_mode = AgentMode(**data["agent_mode"])
    if data.get("plan_dialog_state"):
        state.plan_dialog_state = data["plan_dialog_state"]


# ---------------------------------------------------------------------------
# Обработчики групп inline-команд (SRP: каждая функция — одна ответственность)
# ---------------------------------------------------------------------------


def _cmd_model_params(key: str, value: Any, state: SessionState) -> None:
    """Применяет изменения параметров модели: model, temperature, top_p, top_k и т.д."""
    if key == "model":
        state.model = value
    elif key == "base_url":
        state.base_url = value
    elif key == "max_tokens":
        state.max_tokens = value
    elif key == "temperature":
        state.temperature = value
    elif key == "top_p":
        state.top_p = value
    elif key == "top_k":
        state.top_k = value
    elif key == "system_prompt":
        state.system_prompt = value
        if state.messages and state.messages[0].role == "system":
            state.messages[0] = ChatMessage(role="system", content=value)
        else:
            state.messages.insert(0, ChatMessage(role="system", content=value))
    elif key == "initial_prompt":
        state.initial_prompt = value
        if len(state.messages) >= 2 and state.messages[1].role == "user":
            state.messages[1] = ChatMessage(role="user", content=value)
        else:
            state.messages.append(ChatMessage(role="user", content=value))


def _cmd_resume(state: SessionState) -> None:
    """Обрабатывает команду /resume — загружает последнюю сессию."""
    loaded = load_last_session(profile_name=state.profile_name)
    if loaded:
        last_path, last_data = loaded
        state.session_path = last_path
        state.session_id_metrics = last_path.replace("/", "_").replace(".json", "")
        _apply_session_data(last_data, state)
        state.dialogue_start_time = time.time() - state.duration
        logger.info("Resumed last session: %s", last_path)
        _print_loaded_history(state.messages)
    else:
        logger.info("No last session found to resume")
        print("No last session found to resume")


def _cmd_strategy(value: str, state: SessionState) -> None:
    """Обрабатывает команду /strategy — меняет стратегию контекста."""
    with contextlib.suppress(ValueError):
        new_strategy = ContextStrategy(value)
        state.context_strategy = new_strategy
        print(f"\n[Стратегия переключена на: {new_strategy.value}]")
        _print_strategy_status(state)
        return
    print(f"Неизвестная стратегия: {value}")


def _cmd_sticky_facts(key: str, value: Any, state: SessionState) -> None:
    """Обрабатывает команды /showfacts, /setfact, /delfact."""
    if key == "showfacts":
        if state.sticky_facts.facts:
            print("\n--- Текущие факты (Sticky Facts) ---")
            for k, v in state.sticky_facts.facts.items():
                print(f"  {k}: {v}")
            print("--- Конец фактов ---\n")
        else:
            print("Факты пока не накоплены.")
    elif key == "setfact" and isinstance(value, dict):
        fact_key = value.get("key", "")
        fact_val = value.get("value", "")
        if fact_key and fact_val:
            state.sticky_facts.set(fact_key, fact_val)
            print(f"[Факт добавлен/обновлён: {fact_key} = {fact_val}]")
    elif key == "delfact" and isinstance(value, str):
        if value in state.sticky_facts.facts:
            del state.sticky_facts.facts[value]
            print(f"[Факт удалён: {value}]")
        else:
            print(f"Факт не найден: {value}")


def _cmd_branching(key: str, value: Any, state: SessionState) -> None:
    """Обрабатывает команды /checkpoint, /branch, /switch, /branches."""
    if key == "checkpoint":
        chk = create_checkpoint(state.messages, state.sticky_facts)
        state.last_checkpoint = chk
        print(f"\n[Checkpoint создан: {chk.created_at}]")
        print(f"  Сообщений в snapshot: {len(chk.messages_snapshot)}")
        print(f"  Фактов в snapshot:    {len(chk.facts_snapshot)}\n")
    elif key == "branch":
        last_chk = state.last_checkpoint
        if last_chk is None:
            last_chk = create_checkpoint(state.messages, state.sticky_facts)
            state.last_checkpoint = last_chk
            print("[Checkpoint создан автоматически для ветвления]")
        branch = create_branch(value, last_chk)
        state.branches.append(branch)
        state.active_branch_id = branch.branch_id
        print(f"\n[Создана ветка '{branch.name}' (ID: {branch.branch_id})]")
        print(f"  Начато от snapshot с {len(last_chk.messages_snapshot)} сообщениями.")
        print(f"  Активная ветка: {branch.branch_id}\n")
    elif key == "switch":
        found = switch_branch(value, state.branches)
        if found:
            state.active_branch_id = found.branch_id
            print(f"\n[Переключено на ветку '{found.name}' (ID: {found.branch_id})]")
            print(f"  Сообщений в ветке: {len(found.messages)}\n")
        else:
            names = [(b.branch_id, b.name) for b in state.branches]
            print(f"Ветка не найдена: '{value}'. Доступны: {names}")
    elif key == "branches":
        if state.branches:
            print("\n--- Ветки диалога ---")
            for b in state.branches:
                active_marker = " ◀ активная" if b.branch_id == state.active_branch_id else ""
                print(f"  [{b.branch_id}] {b.name}{active_marker}  ({len(b.messages)} сообщений)")
            print("--- Конец списка ---\n")
        else:
            print("Веток пока нет. Используйте /checkpoint, затем /branch <имя>.")


def _cmd_memory(key: str, value: Any, state: SessionState) -> None:
    """Обрабатывает команды /memshow, /memstats, /memclear, /memsave, /memload."""
    if key == "memshow":
        mem = state.memory
        print("\n--- Состояние памяти ---")
        print(f"Краткосрочная: {len(mem.short_term.messages)} сообщений")
        print(f"Рабочая: задача={mem.working.current_task!r}, статус={mem.working.task_status}")
        print(
            f"Долговременная: профиль={mem.long_term.profile.name!r}, "
            f"решений={len(mem.long_term.decisions_log)}, знаний={len(mem.long_term.knowledge_base)}"
        )
        print("--- Конец ---\n")
    elif key == "memstats":
        stats = get_memory_stats(profile_name=state.profile_name)
        print("\n--- Статистика памяти ---")
        for mtype, data in stats.items():
            print(f"  {mtype}: {data['files']} файлов, {data['size_bytes']} байт")
        print("--- Конец ---\n")
    elif key == "memclear":
        target = value if isinstance(value, str) else "all"
        if target in ("short", "short_term", "all"):
            state.memory.clear_short_term()
        if target in ("working", "all"):
            state.memory.working = WorkingMemory()
        if target in ("long", "long_term", "all"):
            state.memory.long_term = LongTermMemory()
        print(f"[Память очищена: {_MEMCLEAR_LABELS.get(target, target)}]")
    elif key == "memsave":
        mem = state.memory
        task_name = mem.working.current_task or "current"
        path_w = save_working_memory(mem.working.model_dump(), task_name, profile_name=state.profile_name)
        path_lt = save_long_term(mem.long_term.model_dump(), profile_name=state.profile_name)
        path_st = save_short_term(
            mem.short_term.model_dump(), state.session_path or "default", profile_name=state.profile_name
        )
        print(f"[Память сохранена: рабочая={path_w}, долговременная={path_lt}, краткосрочная={path_st}]")
    elif key == "memload":
        mem = state.memory
        task_name = mem.working.current_task or "current"
        data_w = load_working_memory(task_name, profile_name=state.profile_name)
        if data_w:
            mem.working = WorkingMemory(**data_w)
            print(f"[Рабочая память загружена: задача={mem.working.current_task!r}]")
        else:
            print("[Рабочая память не найдена]")
        data_lt = load_long_term(profile_name=state.profile_name)
        if data_lt:
            mem.long_term = LongTermMemory(**data_lt)
            print(f"[Долговременная память загружена: решений={len(mem.long_term.decisions_log)}]")
        else:
            print("[Долговременная память не найдена]")


def _cmd_knowledge(key: str, value: Any, state: SessionState, client: Optional[OpenAI]) -> None:
    """Обрабатывает /settask, /setpref, /remember."""
    if key == "settask":
        state.memory.working.set_task(value)
        print(f"[Задача установлена: {value!r}]")
        if client is not None:
            _create_task_plan(value, state, client)
    elif key == "setpref":
        if "=" in str(value):
            pref_key, pref_val = str(value).split("=", 1)
            state.memory.working.set_preference(pref_key.strip(), pref_val.strip())
            print(f"[Предпочтение: {pref_key.strip()} = {pref_val.strip()}]")
        else:
            print("[setpref: ожидается формат key=value]")
    elif key == "remember":
        if "=" in str(value):
            fact_key, fact_val = str(value).split("=", 1)
            state.memory.add_to_long_term(knowledge_key=fact_key.strip(), knowledge_value=fact_val.strip())
            print(f"[Знание сохранено: {fact_key.strip()}]")
        else:
            state.memory.add_to_long_term(
                decision=str(value),
                task=state.memory.working.current_task or "untitled",
            )
            print("[Решение сохранено в долговременную память]")


# ---------------------------------------------------------------------------
# Центральный диспетчер inline-команд
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MCP: обработка tool_calls и команды /mcp
# ---------------------------------------------------------------------------


def _tools_for_llm(tools: list) -> list:
    """Strip internal-only parameters from tool schemas before sending to LLM.

    ``webhook_url`` in ``create_reminder`` is injected by the chatbot; the LLM
    must not see it so it focuses on the required fields.
    """
    import copy
    result = []
    for tool in tools:
        tool = copy.deepcopy(tool)
        fn = tool.get("function", {})
        if fn.get("name") == "create_reminder":
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            props.pop("webhook_url", None)
            required = params.get("required", [])
            if "webhook_url" in required:
                required.remove("webhook_url")
        result.append(tool)
    return result


def _handle_tool_calls(
    response: Any,
    api_messages: list,
    state: "SessionState",
    client: OpenAI,
    mcp_client: "MCPClientManager",
    extra: Any,
    notification_server: Optional["NotificationServer"] = None,
) -> tuple:
    """Обрабатывает цепочку tool_calls от LLM через MCP и возвращает (final_response, text)."""
    import json as _json

    while True:
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break

        # Добавляем ответ ассистента с tool_calls в api_messages
        assistant_dict: dict = {"role": "assistant", "content": msg.content or ""}
        assistant_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
        api_messages.append(assistant_dict)

        # Выполняем каждый tool call через MCP
        for tc in tool_calls:
            name = tc.function.name
            try:
                arguments = _json.loads(tc.function.arguments or "{}")
            except Exception:
                arguments = {}
            if name == "create_reminder" and notification_server is not None:
                arguments.setdefault("webhook_url", notification_server.get_url())
            print(f"[MCP: вызов инструмента '{name}' с аргументами {arguments}]")
            result_text = mcp_client.call_tool(name, arguments)
            print(f"[MCP: результат '{name}': {result_text[:200]}]")

            # Добавляем tool-result в api_messages и state.messages
            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })
            state.messages.append(ChatMessage(
                role="tool",
                content=result_text,
                tool_call_id=tc.id,
            ))

        # Продолжаем диалог с результатами инструментов
        try:
            response = client.chat.completions.create(
                model=state.model,
                messages=api_messages,
                max_tokens=state.max_tokens,
                temperature=state.temperature,
                top_p=state.top_p,
                extra_body=extra,
            )
        except Exception as exc:
            logger.warning("MCP follow-up API error: %s", exc)
            break

    text: str = (
        response.choices[0].message.content or ""
        if response and response.choices
        else ""
    )
    return response, text


def _print_reminders_table(reminders: List[Dict]) -> None:
    """Выводит таблицу напоминаний: ID(8), статус, scheduled_at, описание."""
    if not reminders:
        print("[Reminders: список пуст]")
        return
    print(f"\n{'ID':8}  {'Статус':10}  {'Запланировано':20}  Описание")
    print("-" * 72)
    for r in reminders:
        rid = str(r.get("id", ""))[:8]
        status = str(r.get("status", ""))
        scheduled = str(r.get("scheduled_at", "") or "")[:19]
        desc = str(r.get("description", ""))
        print(f"{rid:8}  {status:10}  {scheduled:20}  {desc}")
    print()


def _print_reminder_detail(reminder: Dict) -> None:
    """Выводит все поля одного напоминания."""
    print("\n--- Напоминание ---")
    for key, val in reminder.items():
        print(f"  {key}: {val}")
    print("-------------------\n")


def _handle_reminders_command(action: str, arg: Optional[str], state: "SessionState") -> None:
    """Обрабатывает inline-команды /reminders list|refresh|show."""
    profile = state.profile_name
    if action in ("list", "refresh", ""):
        status_filter = arg or None
        reminders = fetch_reminders(status=status_filter)
        if reminders is None:
            print("[Reminders: не удалось получить данные. Проверьте, что API запущен на :8881]")
            return
        save_all_reminders(reminders, profile)
        _print_reminders_table(reminders)
        print(f"[Сохранено в {get_reminders_path(profile)}]")
    elif action == "show":
        task_id = arg
        if not task_id:
            print("[Reminders: укажите ID задачи: /reminders show <id>]")
            return
        reminder = fetch_reminder(task_id)
        if reminder is None:
            print(f"[Reminders: напоминание '{task_id}' не найдено]")
            return
        update_reminder_in_file(reminder, profile)
        _print_reminder_detail(reminder)
    else:
        print("[Reminders: доступны: list [status], refresh, show <id>]")


def _handle_mcp_command(action: str, arg: str, mcp_client: "MCPClientManager") -> None:
    """Обрабатывает inline-команды /mcp status|tools|reconnect."""
    if action == "status":
        if mcp_client.connected:
            count = len(mcp_client.tools_as_openai_format())
            print(f"[MCP: подключён | инструментов: {count}]")
        else:
            print("[MCP: не подключён]")
    elif action == "tools":
        tools = mcp_client.tools_as_openai_format()
        if not tools:
            print("[MCP: инструменты недоступны]")
        else:
            print("[MCP инструменты:]")
            for t in tools:
                fn = t.get("function", {})
                print(f"  - {fn.get('name')}: {fn.get('description', '')}")
    elif action == "reconnect":
        results = mcp_client.connect_all()
        if any(results.values()):
            count = len(mcp_client.tools_as_openai_format())
            print(f"[MCP: переподключение успешно | инструментов: {count}]")
        else:
            print("[MCP: переподключение не удалось]")
    else:
        print(f"[MCP: неизвестная подкоманда '{action}'. Доступны: status, tools, reconnect]")


_MODEL_PARAM_KEYS = frozenset(
    {"model", "base_url", "max_tokens", "temperature", "top_p", "top_k", "system_prompt", "initial_prompt"}
)
_STICKY_FACT_KEYS = frozenset({"showfacts", "setfact", "delfact"})
_BRANCH_KEYS = frozenset({"checkpoint", "branch", "switch", "branches"})
_MEMORY_KEYS = frozenset({"memshow", "memstats", "memclear", "memsave", "memload"})
_KNOWLEDGE_KEYS = frozenset({"settask", "setpref", "remember"})


def _apply_inline_updates(
    updates: dict,
    state: SessionState,
    client: Optional[OpenAI] = None,
    mcp_client: Optional["MCPClientManager"] = None,
) -> bool:
    """Применяет разобранные inline-команды к рабочему состоянию сессии.

    Диспетчеризует каждую команду в профильный обработчик (SRP).

    Returns:
        True, если была обработана команда /showsummary.
    """
    show_summary = False

    for key, value in updates.items():
        if value is None:
            continue

        if key in _MODEL_PARAM_KEYS:
            _cmd_model_params(key, value, state)
        elif key == "resume" and value:
            _cmd_resume(state)
        elif key == "showsummary":
            show_summary = True
        elif key == "strategy":
            _cmd_strategy(value, state)
        elif key == "strategy_status":
            _print_strategy_status(state)
        elif key in _STICKY_FACT_KEYS:
            _cmd_sticky_facts(key, value, state)
        elif key in _BRANCH_KEYS:
            _cmd_branching(key, value, state)
        elif key in _MEMORY_KEYS:
            _cmd_memory(key, value, state)
        elif key in _KNOWLEDGE_KEYS:
            _cmd_knowledge(key, value, state, client)
        elif key == "task" and isinstance(value, dict):
            _handle_task_command(value["action"], value.get("arg", ""), state, client)
        elif key == "plan" and isinstance(value, dict):
            _handle_agent_command(value["action"], value.get("arg", ""), state, client)
        elif key == "invariant" and isinstance(value, dict):
            _handle_invariant_command(value["action"], value.get("arg", ""), state)
        elif key == "project" and isinstance(value, dict):
            _handle_project_command(value["action"], value.get("arg", ""), state, client)
        elif key == "mcp" and isinstance(value, dict) and mcp_client is not None:
            _handle_mcp_command(value["action"], value.get("arg", ""), mcp_client)
        elif key == "reminders" and isinstance(value, dict):
            _handle_reminders_command(value["action"], value.get("arg"), state)

        elif key == "profile" and isinstance(value, dict):
            _profile_action = value.get("action", "show")
            _profile_arg = value.get("arg", "")
            _lt = state.memory.long_term

            if _profile_action == "show":
                p = _lt.profile
                print(f"\n--- Профиль: {p.name} ---")
                print(f"  Стиль:      {p.style or '(не задан)'}")
                print(f"  Формат:     {p.format or '(не задан)'}")
                print(f"  Ограничения:{p.constraints or '(не заданы)'}")
                print(f"  Прочее:     {p.custom or '(нет)'}")
                print(f"  Модель:     {p.preferred_model or '(не задана)'}")
                print(f"  Обновлён:   {p.updated_at}")
                if not p.is_empty():
                    print(f"\n  Системный промпт:\n  {_lt.get_profile_prompt()}")
                print("--- Конец профиля ---\n")

            elif _profile_action == "list":
                names = list_profiles()
                if names:
                    print("\n--- Сохранённые профили ---")
                    for n in names:
                        print(f"  {n}")
                    print("--- Конец списка ---\n")
                else:
                    print("Профилей не найдено.")

            elif _profile_action == "name":
                if _profile_arg:
                    _lt.profile.name = _profile_arg
                    state.profile_name = _profile_arg
                    try:
                        path = save_profile(_lt.profile, _profile_arg)
                        print(f"[Профиль переименован и сохранён как '{_profile_arg}': {os.path.abspath(path)}]")
                    except Exception as exc:
                        print(f"[Ошибка сохранения профиля: {exc}]")

            elif _profile_action == "style":
                if "=" in _profile_arg:
                    k, v = _profile_arg.split("=", 1)
                    _lt.set_profile_style(k.strip(), v.strip())
                    print(f"[Стиль добавлен: {k.strip()}={v.strip()}]")
                else:
                    print("[profile style: ожидается формат key=value]")

            elif _profile_action == "format":
                if "=" in _profile_arg:
                    k, v = _profile_arg.split("=", 1)
                    _lt.set_profile_format(k.strip(), v.strip())
                    print(f"[Формат добавлен: {k.strip()}={v.strip()}]")
                else:
                    print("[profile format: ожидается формат key=value]")

            elif _profile_action == "constraint":
                sub_parts = _profile_arg.split(None, 1)
                sub_op = sub_parts[0].lower() if sub_parts else ""
                sub_text = sub_parts[1].strip() if len(sub_parts) > 1 else ""
                if sub_op == "add" and sub_text:
                    _lt.add_profile_constraint(sub_text)
                    print(f"[Ограничение добавлено: {sub_text}]")
                elif sub_op == "del" and sub_text:
                    _lt.remove_profile_constraint(sub_text)
                    print(f"[Ограничение удалено: {sub_text}]")
                else:
                    print("[profile constraint: ожидается 'add <текст>' или 'del <текст>']")

            elif _profile_action == "model":
                if _profile_arg:
                    _lt.profile.preferred_model = _profile_arg
                    state.model = _profile_arg
                    try:
                        save_profile(_lt.profile, state.profile_name)
                        print(f"[Модель профиля задана: {_profile_arg}]")
                    except Exception as exc:
                        print(f"[Ошибка сохранения профиля: {exc}]")
                else:
                    current = _lt.profile.preferred_model or "(не задана)"
                    print(f"[Предпочтительная модель профиля: {current}]")

            elif _profile_action == "load":
                load_name = _profile_arg or "default"
                loaded_p = load_profile(load_name)
                if loaded_p:
                    _lt.profile = loaded_p
                    state.profile_name = load_name
                    # Сбрасываем состояние предыдущего профиля
                    state.active_task_id = None
                    state.active_task_ids = []
                    state.active_project_id = None
                    state.agent_mode = AgentMode()
                    state.plan_dialog_state = None
                    state.total_tokens = 0
                    state.total_prompt_tokens = 0
                    state.total_completion_tokens = 0
                    state.branches = []
                    state.active_branch_id = None
                    # Загружаем историю диалога для нового профиля
                    loaded_session = load_last_session(profile_name=load_name)
                    if loaded_session:
                        last_path, last_data = loaded_session
                        state.session_path = last_path
                        state.session_id_metrics = last_path.replace("/", "_").replace(".json", "")
                        _apply_session_data(last_data, state)
                        print(f"[Профиль переключён: {load_name}]")
                        _print_loaded_history(state.messages)
                    else:
                        state.messages = []
                        state.session_path = os.path.join(
                            DIALOGUES_DIR, load_name,
                            f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
                            f"_{state.model.replace('/', '_')}.json",
                        )
                        state.session_id_metrics = state.session_path.replace("/", "_").replace(".json", "")
                        print(f"[Профиль переключён: {load_name}] (история пуста)")
                    # Применяем preferred_model из загруженного профиля
                    if loaded_p.preferred_model:
                        state.model = loaded_p.preferred_model
                        print(f"[Модель переключена: {state.model}]")
                else:
                    print(f"[Профиль не найден: {load_name}]")

            else:
                print(f"[Неизвестная подкоманда профиля: {_profile_action}]")

    # Если системный промпт задан, но сообщения не начинаются с него — добавляем
    if state.system_prompt and not (
        state.messages and state.messages[0].role == "system"
    ):
        state.messages.insert(0, ChatMessage(role="system", content=state.system_prompt))

    return show_summary


# ---------------------------------------------------------------------------
# Сохранение состояния
# ---------------------------------------------------------------------------


def _build_session_payload(
    state: SessionState,
    user_input: Optional[str] = None,
    assistant_text: Optional[str] = None,
) -> DialogueSession:
    """Собирает объект DialogueSession из текущего состояния."""
    return DialogueSession(
        dialogue_session_id=state.session_path or "",
        created_at=_now_str(),
        model=state.model,
        base_url=state.base_url,
        system_prompt=state.system_prompt,
        initial_prompt=state.initial_prompt,
        messages=[m.model_dump() for m in state.messages],
        context_summary=state.context_summary,
        turns=len([m for m in state.messages if m.role in {"user", "assistant"}]),
        last_user_input=user_input,
        last_assistant_content=assistant_text,
        duration_seconds=time.time() - state.dialogue_start_time,
        total_tokens=state.total_tokens,
        total_prompt_tokens=state.total_prompt_tokens,
        total_completion_tokens=state.total_completion_tokens,
        context_strategy=state.context_strategy,
        sticky_facts=dict(state.sticky_facts.facts),
        branches=list(state.branches),
        active_branch_id=state.active_branch_id,
        active_task_id=state.active_task_id,
        active_task_ids=list(state.active_task_ids),
        active_project_id=state.active_project_id,
        agent_mode=state.agent_mode.model_dump(),
        plan_dialog_state=state.plan_dialog_state,
    )


# ---------------------------------------------------------------------------
# Вспомогательная функция: добавить сообщение в нужное место
# ---------------------------------------------------------------------------


def _append_message(state: SessionState, message: ChatMessage) -> None:
    """Добавляет сообщение в активную ветку (если branching) или в state.messages."""
    if (
        state.context_strategy == ContextStrategy.BRANCHING
        and state.active_branch_id
    ):
        branch = switch_branch(state.active_branch_id, state.branches)
        if branch:
            branch.messages.append(message)
            return
    state.messages.append(message)


def _get_active_messages(state: SessionState) -> List[ChatMessage]:
    """Возвращает активный список сообщений (ветка или главная история)."""
    if (
        state.context_strategy == ContextStrategy.BRANCHING
        and state.active_branch_id
    ):
        branch = switch_branch(state.active_branch_id, state.branches)
        if branch:
            return branch.messages
    return state.messages


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------


def _start_notification_watcher(notification_server: "NotificationServer", state: "SessionState") -> None:
    """Запускает daemon-поток, который немедленно печатает уведомления из очереди.

    При получении уведомления автоматически обновляет файл reminders.json.
    """
    import sys
    import threading

    def _watcher() -> None:
        while True:
            notes = notification_server.check_notifications()
            for note in notes:
                sys.stdout.write(f"\r\n⏰ {note}\n> ")
                sys.stdout.flush()
                match = re.search(r'\(id=([^)]+)\)', note)
                if match:
                    task_id = match.group(1)
                    reminder = fetch_reminder(task_id)
                    if reminder:
                        update_reminder_in_file(reminder, state.profile_name)
            time.sleep(0.5)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()


def main() -> None:
    """Основной поток выполнения CLI: инициализация и интерактивный диалог."""
    if not API_KEY:
        raise SystemExit("API_KEY environment variable is not set.")

    args = parse_args()
    cfg = config_from_args(args)

    # Начальная стратегия из CLI-аргумента
    _strategy_raw = getattr(args, "strategy", None)
    try:
        initial_strategy = ContextStrategy(
            _strategy_raw if isinstance(_strategy_raw, str) else ContextStrategy.SLIDING_WINDOW.value
        )
    except (ValueError, TypeError):
        initial_strategy = ContextStrategy.SLIDING_WINDOW

    _profile_name = getattr(args, "profile", None) or DEFAULT_PROFILE

    state = SessionState(
        model=cfg.model,
        base_url=cfg.base_url,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        system_prompt=cfg.system_prompt,
        initial_prompt=cfg.initial_prompt,
        dialogue_start_time=time.time(),
        context_strategy=initial_strategy,
        memory=Memory(),
        profile_name=_profile_name,
    )

    # Загружаем профиль пользователя (авто-создаём если не найден)
    _loaded_profile = load_profile(_profile_name)
    if _loaded_profile:
        state.memory.long_term.profile = _loaded_profile
        print(f"[Профиль загружен: {_profile_name}]")
    else:
        state.memory.long_term.profile.name = _profile_name
        print(f"[Профиль создан: {_profile_name}]")
        save_profile(state.memory.long_term.profile, _profile_name)

    # Загружаем историю: всегда при явном --profile; при дефолтном — только если --resume
    _explicit_profile = getattr(args, "profile", None) is not None
    if get_resume_flag(args) or _explicit_profile:
        loaded = load_last_session(profile_name=_profile_name)
        if loaded:
            last_path, last_data = loaded
            state.session_path = last_path
            state.session_id_metrics = last_path.replace("/", "_").replace(".json", "")
            try:
                _apply_session_data(last_data, state)
                state.dialogue_start_time = time.time() - state.duration
                logger.info("Загрузка последней сессии: %s", state.session_path)
                _print_loaded_history(state.messages)
                # Восстанавливаем активную задачу
                if state.active_task_id:
                    _restored_plan = load_task_plan(state.active_task_id, _profile_name)
                    if _restored_plan:
                        _restored_steps = load_all_steps(state.active_task_id, _profile_name)
                        print(f"\n[Восстановлена активная задача: {_restored_plan.name}]")
                        _print_task_plan(_restored_plan, _restored_steps)
            except Exception as exc:
                logger.warning("Не удалось загрузить последнюю сессию: %s", exc)

    # Применяем preferred_model из профиля, если --model не был указан явно
    _explicit_model_arg = getattr(args, "model", None)
    _pref_model = state.memory.long_term.profile.preferred_model
    if not _explicit_model_arg and _pref_model:
        state.model = _pref_model
        print(f"[Модель из профиля: {state.model}]")

    client = OpenAI(api_key=API_KEY, base_url=state.base_url)

    # Webhook notification server
    _notification_server: Optional[NotificationServer] = None
    try:
        _notification_server = NotificationServer()
        _notification_server.start()
        print(f"[Webhook: {_notification_server.get_url()}]")
        _start_notification_watcher(_notification_server, state)
    except OSError as exc:
        print(f"[Webhook: не удалось запустить ({exc}). Push-уведомления отключены.]")

    # Оба MCP клиента через менеджер
    _mcp_client = MCPClientManager([MCPWeatherClient(), MCPSchedulerClient()])
    for name, ok in _mcp_client.connect_all().items():
        status = "подключён" if ok else "недоступен"
        print(f"[MCP: {os.path.basename(name)} {status}]")
    print(f"[MCP: всего инструментов: {len(_mcp_client.tools_as_openai_format())}]")

    # Создаём путь к файлу сессии (если не восстановлено из предыдущей)
    if state.session_path is None:
        state.session_path = os.path.join(
            DIALOGUES_DIR, state.profile_name,
            f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{state.model.replace('/', '_')}.json",
        )
    state.session_id_metrics = state.session_path.replace("/", "_").replace(".json", "")
    state.dialogue_start_time = time.time()

    # Добавляем системный промпт и начальное сообщение (seed)
    if state.system_prompt:
        state.messages.append(ChatMessage(role="system", content=state.system_prompt))
    if state.initial_prompt:
        state.messages.append(ChatMessage(role="user", content=state.initial_prompt))

    print(f"[Профиль: {state.profile_name}]")
    print(f"Введите запрос (type 'exit' чтобы выйти). Стратегия: {state.context_strategy.value}")
    print("Команды стратегий: /strategy <sliding_window|sticky_facts|branching>")
    print("Ветвление: /checkpoint  /branch <имя>  /switch <имя_или_id>  /branches")
    print("Факты:     /showfacts   /setfact <ключ>: <значение>   /delfact <ключ>")
    print("Память:    /memshow  /memstats  /memsave  /memload  /memclear")
    print("           /settask <задача>  /setpref <ключ>=<значение>  /remember <ключ>=<значение>")
    print("Профиль:   /profile show | list | load <имя>")
    print("           /profile name <имя>  /profile style <к>=<в>  /profile format <к>=<в>")
    print("           /profile constraint add <текст>  /profile constraint del <текст>")
    print("Задачи:    /task new <описание>  /task show  /task list  /task start")
    print("           /task step done|skip|fail|note <текст>  /task pause  /task resume [id]")
    print("           /task done  /task fail [причина]  /task load <id>  /task delete <id>")
    print("Агент:     /plan on|off|status  /plan retries <n>  /plan builder")
    print("           /invariant add <текст>  /invariant del <n>  /invariant list  /invariant clear")
    print("MCP:       /mcp status  /mcp tools  /mcp reconnect\n")

    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        # Inline-команды начинаются с '/'
        if user_input.strip().startswith("/"):
            updates = parse_inline_command(user_input)
            if updates:
                show_summary = _apply_inline_updates(updates, state, client, _mcp_client)
                if show_summary:
                    if state.context_summary:
                        print("\n--- Текущее резюме контекста ---")
                        print(state.context_summary)
                        print("--- Конец резюме ---\n")
                    else:
                        print("Резюме ещё не создано (накопите больше сообщений).")
                elif not any(
                    k in updates for k in (
                        "showfacts", "setfact", "delfact",
                        "checkpoint", "branch", "switch", "branches",
                        "strategy",
                        "memshow", "memstats", "memsave", "memload", "memclear",
                        "settask", "setpref", "remember",
                        "profile", "task",
                        "plan", "invariant",
                        "mcp",
                    )
                ):
                    print(f"Updated config: {updates}")
            else:
                print("Unknown command")
            continue

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            break

        # --- Plan dialog intercept ---
        if state.plan_dialog_state == "awaiting_task":
            _handle_plan_awaiting_task(user_input, state)
            continue

        if state.plan_dialog_state == "awaiting_invariants":
            _handle_plan_awaiting_invariants(user_input, state, client)
            continue

        if state.plan_dialog_state == "confirming":
            _confirm_and_create_tasks(user_input, state, client)
            continue

        if state.plan_dialog_state == "active":
            _handle_plan_dialog_message(user_input, state, client)
            continue

        # Добавляем сообщение пользователя в активный контекст
        _append_message(state, ChatMessage(role="user", content=user_input))
        state.memory.add_user_message(user_input)

        # Суммаризация при необходимости (только для sliding_window)
        if state.context_strategy == ContextStrategy.SLIDING_WINDOW:
            state.messages, state.context_summary, _ = maybe_summarize(
                client, state.model, state.messages, state.context_summary
            )

        # Получаем активную ветку для branching
        active_branch: Optional[Branch] = None
        if state.context_strategy == ContextStrategy.BRANCHING and state.active_branch_id:
            active_branch = switch_branch(state.active_branch_id, state.branches)

        # Собираем контекст по выбранной стратегии
        api_messages: Any = build_context_by_strategy(
            strategy=state.context_strategy,
            messages=state.messages,
            context_summary=state.context_summary,
            facts=state.sticky_facts,
            active_branch=active_branch,
        )

        # Инжектируем активную задачу в системный промпт
        if state.active_task_id:
            _active_plan = load_task_plan(state.active_task_id, state.profile_name)
            if _active_plan and _active_plan.phase in (
                TaskPhase.EXECUTION, TaskPhase.PLANNING, TaskPhase.VALIDATION
            ):
                _cur_step = load_task_step(
                    _active_plan.task_id,
                    _active_plan.current_step_index + 1,
                    state.profile_name,
                )
                _task_ctx = f"[ACTIVE TASK: {_active_plan.name}]\nPhase: {_active_plan.phase.value}\nStep {_active_plan.current_step_index + 1}/{_active_plan.total_steps}"
                if _cur_step:
                    _task_ctx += f": {_cur_step.title}"
                if api_messages and api_messages[0].get("role") == "system":
                    api_messages[0] = {
                        "role": "system",
                        "content": api_messages[0]["content"] + "\n\n" + _task_ctx,
                    }
                else:
                    api_messages.insert(0, {"role": "system", "content": _task_ctx})

        # Инжектируем агентский системный промпт (если режим включён)
        if state.agent_mode.enabled:
            _profile_text = state.memory.get_profile_prompt()
            _state_vars = _build_agent_state_vars(state)
            _agent_sys = build_agent_system_prompt(
                profile_text=_profile_text,
                state_vars=_state_vars,
                invariants=state.agent_mode.invariants,
            )
            if api_messages and api_messages[0].get("role") == "system":
                api_messages[0] = {"role": "system", "content": _agent_sys}
            else:
                api_messages.insert(0, {"role": "system", "content": _agent_sys})
        else:
            # Стандартный инжект профиля
            _profile_text = state.memory.get_profile_prompt()
            if _profile_text:
                if api_messages and api_messages[0].get("role") == "system":
                    api_messages[0] = {
                        "role": "system",
                        "content": api_messages[0]["content"] + "\n\n" + _profile_text,
                    }
                else:
                    api_messages.insert(0, {"role": "system", "content": _profile_text})

        # Выполняем API-запрос
        try:
            start_call = time.perf_counter()
            extra = {"top_k": state.top_k} if state.top_k is not None else None
            _tools_param = _tools_for_llm(_mcp_client.tools_as_openai_format()) if _mcp_client.connected else None
            response = client.chat.completions.create(
                model=state.model,
                messages=api_messages,
                max_tokens=state.max_tokens,
                temperature=state.temperature,
                top_p=state.top_p,
                extra_body=extra,
                **( {"tools": _tools_param} if _tools_param else {}),
            )
        except Exception as exc:
            print("API error:", exc)
            continue

        api_call_time = time.perf_counter() - start_call

        # Обрабатываем tool_calls (MCP)
        response, text = _handle_tool_calls(
            response, api_messages, state, client, _mcp_client, extra,
            notification_server=_notification_server,
        )

        # Stateful Agent: self-correction validation loop
        display_text = text
        if state.agent_mode.enabled and state.agent_mode.invariants:
            draft = text
            passed, violation = validate_draft_against_invariants(
                client, state.model, draft, state.agent_mode.invariants
            )
            retry_count = 0
            while not passed and retry_count < state.agent_mode.max_retries:
                retry_count += 1
                print(
                    f"[Agent: инвариант нарушен ({violation}). "
                    f"Повтор {retry_count}/{state.agent_mode.max_retries}...]"
                )
                retry_messages = [
                    *api_messages,
                    {"role": "assistant", "content": draft},
                    {
                        "role": "user",
                        "content": (
                            f"Your response violates this invariant: {violation}. "
                            "Please correct your response to comply with all invariants "
                            "and output the two required blocks (Response / State Update)."
                        ),
                    },
                ]
                try:
                    retry_resp = client.chat.completions.create(
                        model=state.model,
                        messages=retry_messages,
                        max_tokens=state.max_tokens,
                        temperature=state.temperature,
                        top_p=state.top_p,
                        extra_body=extra,
                    )
                    draft = (
                        retry_resp.choices[0].message.content or ""
                        if retry_resp and retry_resp.choices
                        else draft
                    )
                except Exception as exc:
                    logger.warning("Agent retry error: %s", exc)
                    break
                passed, violation = validate_draft_against_invariants(
                    client, state.model, draft, state.agent_mode.invariants
                )
            text = draft

            # Разбираем вывод агента на Response + State Update
            display_text, state_update = parse_agent_output(text)
            if state_update:
                for su_key, su_val in state_update.items():
                    state.memory.working.set_preference(su_key, su_val)
                print(f"[Agent State Update: {state_update}]")
            if not passed:
                print(f"[Agent: исчерпаны попытки валидации. Последнее нарушение: {violation}]")
            _collect_plan_clarifications(text, state)
        elif state.agent_mode.enabled:
            # Agent mode без инвариантов — только парсим структуру вывода
            display_text, state_update = parse_agent_output(text)
            if state_update:
                for su_key, su_val in state_update.items():
                    state.memory.working.set_preference(su_key, su_val)
                print(f"[Agent State Update: {state_update}]")
            _collect_plan_clarifications(text, state)

        # Метрики токенов
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", None) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", None) or 0)
        total_tokens = int(getattr(usage, "total_tokens", None) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        state.total_tokens += total_tokens
        state.total_prompt_tokens += prompt_tokens
        state.total_completion_tokens += completion_tokens

        total_cost_rub = (state.total_tokens / 1000.0) * USD_PER_1K_TOKENS * RUB_PER_USD
        cost_rub = (total_tokens / 1000.0) * USD_PER_1K_TOKENS * RUB_PER_USD
        total_s = time.time() - state.dialogue_start_time

        print(display_text)
        print(
            f"\n[Токены: запрос={prompt_tokens}, "
            f"ответ={completion_tokens}, итого={total_tokens}]"
        )
        print(
            f"[История диалога: промпт={state.total_prompt_tokens}, "
            f"ответ={state.total_completion_tokens}, "
            f"всего={state.total_tokens} | ~{total_cost_rub:.2f}₽]"
        )
        _agent_tag = " | [Plan]" if state.agent_mode.enabled else ""
        print(f"[Стратегия: {state.context_strategy.value}{_agent_tag}]")

        # Сохраняем ответ ассистента в активный контекст
        assistant_msg = ChatMessage(
            role="assistant",
            content=text,
            tokens=TokenUsage(
                prompt=prompt_tokens,
                completion=completion_tokens,
                total=total_tokens,
            ),
        )
        _append_message(state, assistant_msg)
        state.memory.add_assistant_message(text)

        # Обновляем Sticky Facts после каждого обмена
        if state.context_strategy == ContextStrategy.STICKY_FACTS:
            try:
                new_facts = extract_facts_from_llm(
                    client, state.model,
                    user_input, text,
                    state.sticky_facts.facts,
                )
                if new_facts:
                    for k, v in new_facts.items():
                        state.sticky_facts.set(k, v)
                    logger.info("Sticky facts обновлены: %s", list(new_facts.keys()))
                    print(f"[Facts обновлены: {list(new_facts.keys())}]")
            except Exception as exc:
                logger.debug("Ошибка обновления facts: %s", exc)

        # Метрика запроса
        metric = RequestMetric(
            model=state.model,
            endpoint="chat",
            temp=state.temperature,
            ttft=api_call_time,
            req_time=api_call_time,
            total_time=total_s,
            tokens=total_tokens,
            p_tokens=prompt_tokens,
            c_tokens=completion_tokens,
            cost_rub=cost_rub,
        )

        # Сохраняем сессию
        session = _build_session_payload(state, user_input, text)
        session.requests.append(metric)
        try:
            save_session(session, state.session_path)
            try:
                metric_path = log_request_metric(
                    metric, state.session_id_metrics, state.request_index,
                    profile_name=state.profile_name,
                )
                logger.info("Request metrics logged to: %s", metric_path)
            except Exception as metric_exc:
                logger.debug("Failed to log request metrics: %s", metric_exc)
            state.request_index += 1
            logger.info("Session updated: %s", state.session_path)
        except Exception as save_exc:
            logger.warning("Не удалось обновить сессию: %s", save_exc)

    # Финальное сохранение при выходе
    if state.messages:
        final_session = _build_session_payload(state)
        try:
            save_session(final_session, state.session_path)
            logger.info("Session saved on exit: %s", state.session_path)
        except Exception as exc:
            logger.warning("Failed to save session on exit: %s", exc)
