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
from chatbot.context import build_context_for_api, maybe_summarize
from chatbot.models import ChatMessage, DialogueSession, RequestMetric, SessionState, TokenUsage
from chatbot.storage import load_last_session, log_request_metric, save_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции отображения
# ---------------------------------------------------------------------------


def _print_loaded_history(messages: List[ChatMessage]) -> None:
    """Выводит последние 5 обменов из истории диалога с метриками токенов.

    Args:
        messages: Полная история сообщений текущей сессии.
    """
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
                turns.append(
                    (
                        msg.content.strip(),
                        assistant_msg.content.strip() if assistant_msg else "",
                        assistant_msg.tokens if assistant_msg else None,
                    )
                )
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


# ---------------------------------------------------------------------------
# Загрузка и применение данных сессии
# ---------------------------------------------------------------------------


def _load_messages_from_dict(raw: list) -> List[ChatMessage]:
    """Преобразует список словарей из JSON-сессии в список ChatMessage.

    Args:
        raw: Список словарей сообщений из сохранённого файла.

    Returns:
        Список объектов ChatMessage.
    """
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
        result.append(
            ChatMessage(
                role=item.get("role", "user"),
                content=item.get("content") or "",
                tokens=token_usage,
            )
        )
    return result


def _apply_session_data(data: dict, state: SessionState) -> None:
    """Применяет данные загруженной сессии к рабочему состоянию.

    Args:
        data: Словарь данных из JSON-файла сессии.
        state: Рабочее состояние сессии (изменяется на месте).
    """
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


# ---------------------------------------------------------------------------
# Обработка inline-команд
# ---------------------------------------------------------------------------


def _apply_inline_updates(updates: dict, state: SessionState) -> bool:
    """Применяет разобранные inline-команды к рабочему состоянию сессии.

    Args:
        updates: Словарь обновлений от parse_inline_command.
        state: Рабочее состояние сессии (изменяется на месте).

    Returns:
        True, если была обработана команда /showsummary.
    """
    show_summary = False

    for key, value in updates.items():
        if value is None:
            continue
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

    # Если системный промпт задан, но сообщения не начинаются с него — добавляем
    if state.system_prompt and not (
        state.messages and state.messages[0].role == "system"
    ):
        state.messages.insert(
            0, ChatMessage(role="system", content=state.system_prompt)
        )

    return show_summary


# ---------------------------------------------------------------------------
# Сохранение состояния
# ---------------------------------------------------------------------------


def _build_session_payload(
    state: SessionState,
    user_input: Optional[str] = None,
    assistant_text: Optional[str] = None,
) -> DialogueSession:
    """Собирает объект DialogueSession из текущего состояния.

    Args:
        state: Рабочее состояние сессии.
        user_input: Последний ввод пользователя (опционально).
        assistant_text: Последний ответ ассистента (опционально).

    Returns:
        Заполненный объект DialogueSession.
    """
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
    )


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------


def main() -> None:
    """Основной поток выполнения CLI: инициализация и интерактивный диалог."""
    if not API_KEY:
        raise SystemExit("API_KEY environment variable is not set.")

    args = parse_args()
    cfg = config_from_args(args)

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

    print("Введите запрос (type 'exit' чтобы выйти):")

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
                else:
                    print(f"Updated config: {updates}")
            else:
                print("Unknown command")
            continue

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            break

        # Добавляем сообщение пользователя
        state.messages.append(ChatMessage(role="user", content=user_input))

        # Суммаризация при необходимости
        state.messages, state.context_summary, _ = maybe_summarize(
            client, state.model, state.messages, state.context_summary
        )

        # Собираем контекст для API
        api_messages: Any = build_context_for_api(state.messages, state.context_summary)

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

        # Извлекаем метрики токенов
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

        # Сохраняем ответ ассистента в историю
        state.messages.append(
            ChatMessage(
                role="assistant",
                content=text,
                tokens=TokenUsage(
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                    total=total_tokens,
                ),
            )
        )

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

        # Перезаписываем файл сессии после каждого шага
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
