"""Основной модуль: точка входа и цикл интерактивного диалога."""

import logging
import time
from datetime import datetime
from typing import Any, List, Optional

from openai import OpenAI

from chatbot.cli import config_from_args, get_resume_flag, parse_args, parse_inline_command
from chatbot.config import (
    API_KEY,
    DIALOGUES_DIR,
    RUB_PER_USD,
    USD_PER_1K_TOKENS,
)
from chatbot.context import (
    build_context_by_strategy,
    create_branch,
    create_checkpoint,
    extract_facts_from_llm,
    maybe_summarize,
    switch_branch,
)
from chatbot.memory import Memory, MemoryFactor
from chatbot.memory_storage import (
    get_memory_stats,
    import_memory_state,
    list_long_term_memories,
    list_working_memories,
    load_long_term,
    load_short_term_last,
    load_working_memory,
    save_long_term,
    save_short_term,
    save_working_memory,
)
from chatbot.models import (
    Branch,
    ChatMessage,
    ContextStrategy,
    DialogueSession,
    RequestMetric,
    SessionState,
    StickyFacts,
    TokenUsage,
)
from chatbot.storage import load_last_session, log_request_metric, save_session

logger = logging.getLogger(__name__)


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
    print(f"\n[Стратегия контекста: {strategy.value}]")
    if strategy == ContextStrategy.SLIDING_WINDOW:
        print(f"  Окно: последние N сообщений. Summary: {'есть' if state.context_summary else 'нет'}.")
    elif strategy == ContextStrategy.STICKY_FACTS:
        count = len(state.sticky_facts.facts)
        print(f"  Фактов в памяти: {count}.")
        if count:
            for k, v in state.sticky_facts.facts.items():
                print(f"    {k}: {v}")
    elif strategy == ContextStrategy.BRANCHING:
        branch_count = len(state.branches)
        active = state.active_branch_id or "нет"
        print(f"  Веток: {branch_count}. Активная: {active}.")
    print()


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
        try:
            state.context_strategy = ContextStrategy(data["context_strategy"])
        except ValueError:
            pass
    if "sticky_facts" in data and isinstance(data["sticky_facts"], dict):
        state.sticky_facts = StickyFacts(facts=data["sticky_facts"])


# ---------------------------------------------------------------------------
# Обработка inline-команд
# ---------------------------------------------------------------------------


def _apply_inline_updates(updates: dict, state: SessionState) -> bool:
    """Применяет разобранные inline-команды к рабочему состоянию сессии.

    Returns:
        True, если была обработана команда /showsummary.
    """
    show_summary = False

    for key, value in updates.items():
        if value is None:
            continue

        # --- Базовые параметры ---
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
        elif key == "resume" and value:
            loaded = load_last_session()
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
        elif key == "showsummary":
            show_summary = True

        # --- Стратегия контекста ---
        elif key == "strategy":
            try:
                new_strategy = ContextStrategy(value)
                state.context_strategy = new_strategy
                print(f"\n[Стратегия переключена на: {new_strategy.value}]")
                _print_strategy_status(state)
            except ValueError:
                print(f"Неизвестная стратегия: {value}")

        # --- Sticky Facts ---
        elif key == "showfacts":
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

        # --- Branching ---
        elif key == "checkpoint":
            chk = create_checkpoint(state.messages, state.sticky_facts)
            state.last_checkpoint = chk
            print(f"\n[Checkpoint создан: {chk.created_at}]")
            print(f"  Сообщений в snapshot: {len(chk.messages_snapshot)}")
            print(f"  Фактов в snapshot:    {len(chk.facts_snapshot)}\n")

        elif key == "branch":
            last_chk = state.last_checkpoint
            if last_chk is None:
                # Создаём checkpoint прямо сейчас
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

        # --- Управление памятью ---
        elif key == "memshow":
            mem = state.memory
            print("\n--- Состояние памяти ---")
            print(f"Краткосрочная: {len(mem.short_term.messages)} сообщений")
            print(f"Рабочая: задача={mem.working.current_task!r}, статус={mem.working.task_status}")
            print(f"Долговременная: профиль={mem.long_term.user_profile}, решений={len(mem.long_term.decisions_log)}, знаний={len(mem.long_term.knowledge_base)}")
            print("--- Конец ---\n")

        elif key == "memstats":
            stats = get_memory_stats()
            print("\n--- Статистика памяти ---")
            for mtype, data in stats.items():
                print(f"  {mtype}: {data['files']} файлов, {data['size_bytes']} байт")
            print("--- Конец ---\n")

        elif key == "memclear":
            state.memory.clear_short_term()
            print("[Краткосрочная память очищена]")

        elif key == "memsave":
            mem = state.memory
            task_name = mem.working.current_task or "current"
            path_w = save_working_memory(mem.working.model_dump(), task_name)
            path_lt = save_long_term(mem.long_term.model_dump())
            path_st = save_short_term(mem.short_term.model_dump(), state.session_path or "default")
            print(f"[Память сохранена: рабочая={path_w}, долговременная={path_lt}, краткосрочная={path_st}]")

        elif key == "memload":
            mem = state.memory
            task_name = mem.working.current_task or "current"
            data_w = load_working_memory(task_name)
            if data_w:
                from chatbot.memory import WorkingMemory
                mem.working = WorkingMemory(**data_w)
                print(f"[Рабочая память загружена: задача={mem.working.current_task!r}]")
            else:
                print("[Рабочая память не найдена]")
            data_lt = load_long_term()
            if data_lt:
                from chatbot.memory import LongTermMemory
                mem.long_term = LongTermMemory(**data_lt)
                print(f"[Долговременная память загружена: решений={len(mem.long_term.decisions_log)}]")
            else:
                print("[Долговременная память не найдена]")

        elif key == "settask":
            state.memory.working.set_task(value)
            print(f"[Задача установлена: {value!r}]")

        elif key == "setpref":
            # value expected as "key=val"
            if "=" in str(value):
                pref_key, pref_val = str(value).split("=", 1)
                state.memory.working.set_preference(pref_key.strip(), pref_val.strip())
                print(f"[Предпочтение: {pref_key.strip()} = {pref_val.strip()}]")
            else:
                print("[setpref: ожидается формат key=value]")

        elif key == "remember":
            # Save a fact to long-term knowledge base: "key=value"
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
        created_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    )

    # Опционально загружаем последнюю сессию
    if get_resume_flag(args):
        loaded = load_last_session()
        if loaded:
            last_path, last_data = loaded
            state.session_path = last_path
            state.session_id_metrics = last_path.replace("/", "_").replace(".json", "")
            try:
                _apply_session_data(last_data, state)
                state.dialogue_start_time = time.time() - state.duration
                logger.info("Загрузка последней сессии: %s", state.session_path)
                _print_loaded_history(state.messages)
            except Exception as exc:
                logger.warning("Не удалось загрузить последнюю сессию: %s", exc)

    client = OpenAI(api_key=API_KEY, base_url=state.base_url)

    # Создаём путь к файлу сессии (если не восстановлено из предыдущей)
    if state.session_path is None:
        state.session_path = (
            f"{DIALOGUES_DIR}/session_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_"
            f"{state.model.replace('/', '_')}.json"
        )
    state.session_id_metrics = state.session_path.replace("/", "_").replace(".json", "")
    state.dialogue_start_time = time.time()

    # Добавляем системный промпт и начальное сообщение (seed)
    if state.system_prompt:
        state.messages.append(ChatMessage(role="system", content=state.system_prompt))
    if state.initial_prompt:
        state.messages.append(ChatMessage(role="user", content=state.initial_prompt))

    print(f"Введите запрос (type 'exit' чтобы выйти). Стратегия: {state.context_strategy.value}")
    print("Команды стратегий: /strategy <sliding_window|sticky_facts|branching>")
    print("Ветвление: /checkpoint  /branch <имя>  /switch <имя_или_id>  /branches")
    print("Факты:     /showfacts   /setfact <ключ>: <значение>   /delfact <ключ>")
    print("Память:    /memshow  /memstats  /memsave  /memload  /memclear")
    print("           /settask <задача>  /setpref <ключ>=<значение>  /remember <ключ>=<значение>\n")

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
                show_summary = _apply_inline_updates(updates, state)
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
                    )
                ):
                    print(f"Updated config: {updates}")
            else:
                print("Unknown command")
            continue

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            break

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

        # Выполняем API-запрос
        try:
            start_call = time.perf_counter()
            extra = {"top_k": state.top_k} if state.top_k is not None else None
            response = client.chat.completions.create(
                model=state.model,
                messages=api_messages,
                max_tokens=state.max_tokens,
                temperature=state.temperature,
                top_p=state.top_p,
                extra_body=extra,
            )
        except Exception as exc:
            print("API error:", exc)
            continue

        api_call_time = time.perf_counter() - start_call
        text: str = (
            response.choices[0].message.content or ""
            if response and response.choices
            else ""
        )

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

        print(text)
        print(
            f"\n[Токены: запрос={prompt_tokens}, "
            f"ответ={completion_tokens}, итого={total_tokens}]"
        )
        print(
            f"[История диалога: промпт={state.total_prompt_tokens}, "
            f"ответ={state.total_completion_tokens}, "
            f"всего={state.total_tokens} | ~{total_cost_rub:.2f}₽]"
        )
        print(f"[Стратегия: {state.context_strategy.value}]")

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
                    metric, state.session_id_metrics, state.request_index
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
