#!/usr/bin/env python3
"""CLI чат-бот на основе ИИ (интерактивный режим).

Этот модуль предоставляет простой интерфейс командной строки для общения с AI-моделью:
- настройка параметров через CLI;
- интерактивный диалог с сохранением истории переписки;
- вывод ответа и продолжение диалога.
"""

import glob
import json
import logging
import os
import argparse
import time
from datetime import datetime
from typing import Optional, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

BASE_URL = "https://routerai.ru/api/v1"
API_KEY = os.getenv("API_KEY")

#DEFAULT_MODEL = "deepseek/deepseek-v3.2"
# Эту модель сильно глючит со старта
# API error: Error code: 503 - {'error': "Provider error (status: 400): This endpoint's maximum context length is 32768 tokens.
# However, you requested about 34497 tokens (34497 of text input)."}
#DEFAULT_MODEL = "qwen/qwen2.5-coder-7b-instruct"
#DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1" #
DEFAULT_MODEL = "inception/mercury-coder" #
DEFAULT_MAX_TOKENS = None
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50

# Стоимость токенов
USD_PER_1K_TOKENS = 0.0015
RUB_PER_USD = 100.0

# Управление контекстом
CONTEXT_RECENT_MESSAGES = 10   # сколько последних сообщений держим "как есть"
CONTEXT_SUMMARY_INTERVAL = 10  # каждые N сообщений делаем summary старых


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="CLI chat with AI model (interactive).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", help="Модель для использования")
    parser.add_argument("-u", "--base-url", help="Базовый URL API")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Максимальное число токенов в ответе",
    )
    parser.add_argument(
        "-T", "--temperature", type=float, default=None, help="Температура генерации"
    )
    parser.add_argument(
        "-p", "--top-p", type=float, default=None, help="Top-p (nucleus sampling)"
    )
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Top-k")
    parser.add_argument("--system-prompt", default=None, help="Системный промпт")
    parser.add_argument(
        "--initial-prompt", default=None, help="Начальное сообщение пользователя (seed)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Загрузить последнюю сессию и продолжить диалог",
    )
    return parser.parse_args()


def _parse_inline_command(line: str) -> dict:
    """Разбор интерактивных inline-команд, начинающихся с '/'.

    Поддерживаемые команды (примеры):
      /model=gpt-4
      /base-url https://api.example.com
      /max-tokens 1024
      /temperature 0.8
      /top-p 0.92
      /top-k 50
      /system-prompt Новый системный промпт
      /initial-prompt Начальное сообщение
      /resume true|false
      /showsummary            — показать текущее резюме контекста

    Возвращает словарь обновлений для применения к текущей сессии.
    Ключи нормализованы в стиль Python (подчёркивания).
    """
    cmd = line.strip()
    if not cmd.startswith("/"):
        return {}
    payload = cmd[1:].strip()
    if not payload:
        return {}

    # Разбиваем на ключ и значение
    if "=" in payload:
        key, value = payload.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
    else:
        parts = payload.split(None, 1)
        key = parts[0].strip().lower()
        value = parts[1].strip() if len(parts) > 1 else ""

    # Нормализуем ключ к стилю с подчёркиваниями
    key = key.replace("-", "_")

    updates = {}
    if key == "model":
        updates["model"] = value if value else None
    elif key in {"base_url", "baseurl"}:
        updates["base_url"] = value or None
    elif key == "max_tokens":
        try:
            updates["max_tokens"] = int(value)
        except (ValueError, TypeError):
            updates["max_tokens"] = None
    elif key == "temperature":
        try:
            updates["temperature"] = float(value)
        except (ValueError, TypeError):
            updates["temperature"] = None
    elif key == "top_p":
        try:
            updates["top_p"] = float(value)
        except (ValueError, TypeError):
            updates["top_p"] = None
    elif key == "top_k":
        try:
            updates["top_k"] = int(value)
        except (ValueError, TypeError):
            updates["top_k"] = None
    elif key in {"system_prompt", "system-prompt"}:
        updates["system_prompt"] = value
    elif key in {"initial_prompt", "initial-prompt"}:
        updates["initial_prompt"] = value
    elif key == "resume":
        updates["resume"] = value.lower() in {"true", "1", "yes", "on"}
    elif key == "showsummary":
        updates["showsummary"] = True
    # Неизвестные ключи игнорируются
    return updates


def _get_usage_value(usage, key):
    """Безопасно извлекает значение из объекта или словаря usage."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _print_loaded_history(messages: list) -> None:
    """Выводит историю диалога в формате живого общения с метриками токенов."""
    try:
        # Собираем пары user -> assistant
        turns = []
        i = 0
        filtered = [m for m in messages if m.get("role") in {"user", "assistant"}]
        while i < len(filtered):
            m = filtered[i]
            if m.get("role") == "user":
                user_text = (m.get("content") or "").strip()
                assistant_msg = (
                    filtered[i + 1]
                    if i + 1 < len(filtered)
                    and filtered[i + 1].get("role") == "assistant"
                    else None
                )
                assistant_text = (
                    (assistant_msg.get("content") or "").strip()
                    if assistant_msg
                    else ""
                )
                token_data = assistant_msg.get("tokens") if assistant_msg else None
                turns.append((user_text, assistant_text, token_data))
                i += 2
            else:
                i += 1

        last_turns = turns[-5:]
        if not last_turns:
            return

        print("\nLoaded history (last exchanges):")

        # Накапливаем суммы токенов с начала истории (все turns, не только последние 5)
        cumulative_prompt = 0
        cumulative_completion = 0
        cumulative_total = 0
        # Сначала пройдём все turns чтобы вычислить базу до последних 5
        all_before = turns[: -len(last_turns)]
        for _, _, td in all_before:
            if td:
                cumulative_prompt += td.get("prompt", 0)
                cumulative_completion += td.get("completion", 0)
                cumulative_total += td.get("total", 0)

        for user_text, assistant_text, token_data in last_turns:
            print(f"> {user_text}")
            print(assistant_text)
            if token_data:
                p = token_data.get("prompt", 0)
                c = token_data.get("completion", 0)
                t = token_data.get("total", 0)
                cumulative_prompt += p
                cumulative_completion += c
                cumulative_total += t
                cost_rub = (cumulative_total / 1000.0) * USD_PER_1K_TOKENS * RUB_PER_USD
                print(f"[Токены: запрос={p}, ответ={c}, итого={t}]")
                print(
                    f"[История диалога: промпт={cumulative_prompt}, ответ={cumulative_completion}, всего={cumulative_total} | ~{cost_rub:.2f}₽]"
                )
            else:
                print("[Токены: данные недоступны]")
        print()
    except Exception as err:
        logging.debug(f"Failed to print loaded history: {err}")


def _apply_session_data(
    data: dict,
    *,
    model: str,
    base_url: str,
    system_prompt: Optional[str],
    initial_prompt: Optional[str],
) -> dict:
    """Применяет данные загруженной сессии и возвращает обновлённые значения."""
    return {
        "messages": data.get("messages", []),
        "model": data.get("model", model),
        "base_url": data.get("base_url", base_url),
        "system_prompt": data.get("system_prompt", system_prompt),
        "initial_prompt": data.get("initial_prompt", initial_prompt),
        "duration": data.get("duration_seconds", 0),
        "total_tokens": data.get("total_tokens", 0),
        "total_prompt_tokens": data.get("total_prompt_tokens", 0),
        "total_completion_tokens": data.get("total_completion_tokens", 0),
    }


def _apply_updates(
    updates: dict,
    *,
    model: str,
    base_url: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    top_k: int,
    system_prompt: Optional[str],
    initial_prompt: Optional[str],
    messages: list,
    session_path: str,
    session_id_metrics: str,
    dialogue_start_time: float,
    duration: float,
    context_summary: str,
) -> dict:
    """Применяет inline-команды к текущей конфигурации сессии.

    Возвращает словарь с обновлёнными значениями всех параметров.
    """
    state = {
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "messages": messages,
        "session_path": session_path,
        "session_id_metrics": session_id_metrics,
        "dialogue_start_time": dialogue_start_time,
        "duration": duration,
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "context_summary": context_summary,
        "showsummary": False,
    }

    for k, v in updates.items():
        if v is None:
            continue
        if k == "model":
            state["model"] = v
        elif k == "base_url":
            state["base_url"] = v
        elif k == "max_tokens":
            state["max_tokens"] = v
        elif k == "temperature":
            state["temperature"] = v
        elif k == "top_p":
            state["top_p"] = v
        elif k == "top_k":
            state["top_k"] = v
        elif k == "system_prompt":
            state["system_prompt"] = v
            msgs = state["messages"]
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = v
            else:
                msgs.insert(0, {"role": "system", "content": v})
        elif k == "initial_prompt":
            state["initial_prompt"] = v
            msgs = state["messages"]
            if len(msgs) >= 2 and msgs[1].get("role") == "user":
                msgs[1]["content"] = v
            else:
                msgs.append({"role": "user", "content": v})
        elif k == "resume" and v:
            loaded = _load_last_session()
            if loaded:
                last_path, last_data = loaded
                state["session_path"] = last_path
                state["session_id_metrics"] = last_path.replace("/", "_").replace(
                    ".json", ""
                )
                session_data = _apply_session_data(
                    last_data,
                    model=state["model"],
                    base_url=state["base_url"],
                    system_prompt=state["system_prompt"],
                    initial_prompt=state["initial_prompt"],
                )
                state.update(session_data)
                state["context_summary"] = last_data.get("context_summary", "")
                state["dialogue_start_time"] = time.time() - state["duration"]
                logging.info(f"Resumed last session: {last_path}")
                _print_loaded_history(state["messages"])
            else:
                logging.info("No last session found to resume")
                print("No last session found to resume")
        elif k == "showsummary":
            state["showsummary"] = True

    # Если системный промпт задан, но сообщения не начинаются с него — добавляем
    if state["system_prompt"] and not (
        state["messages"] and state["messages"][0].get("role") == "system"
    ):
        state["messages"].insert(
            0, {"role": "system", "content": state["system_prompt"]}
        )

    return state


# ---------------------------------------------------------------------------
# Слой сохранения данных
# ---------------------------------------------------------------------------


def _save_session_to_path(session: dict, path: str) -> str:
    """Сохраняет всю сессию чата по указанному пути (перезапись)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return path


def _log_request_metric(req_entry: dict, session_id: str, idx: int) -> str:
    """Сохраняет метаданные одного запроса в отдельный лог-файл."""
    dir_path = "dialogues/metrics"
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{dir_path}/session_{session_id}_req_{idx:04d}.log"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(req_entry, f, ensure_ascii=False, indent=2)
    return filename


def _load_last_session() -> Optional[Tuple[str, dict]]:
    """Загружает последнюю сохранённую сессию из dialogues/session_*.json.

    Возвращает кортеж (путь к файлу, данные) или None, если загрузить не удалось.
    """
    try:
        paths = sorted(glob.glob("dialogues/session_*.json"), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return last_path, data
    except Exception as e:
        logging.warning(f"Не удалось загрузить последнюю сессию: {e}")
        return None


# ---------------------------------------------------------------------------
# Управление контекстом: summary + скользящее окно
# ---------------------------------------------------------------------------


def _summarize_messages(client: OpenAI, model: str, messages: list) -> str:
    """Вызывает LLM для получения краткого summary списка сообщений.

    Принимает список сообщений (user/assistant), возвращает строку-резюме.
    При ошибке возвращает пустую строку.
    """
    if not messages:
        return ""

    # Формируем текст диалога для summarization
    dialogue_text = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            label = "User" if role == "user" else "Assistant"
            dialogue_text.append(f"{label}: {content}")

    if not dialogue_text:
        return ""

    joined = "\n".join(dialogue_text)
    prompt = (
        "Ниже приведён фрагмент диалога между пользователем и ИИ-ассистентом. "
        "Составь краткое, ёмкое резюме на русском языке, сохранив все ключевые факты, "
        "решения и договорённости. Резюме будет использоваться как контекст для продолжения диалога.\n\n"
        f"{joined}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logging.warning(f"Не удалось создать summary: {exc}")
        return ""


def _build_context_for_api(
    messages: list,
    context_summary: str,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> list:
    """Собирает список сообщений для отправки в API.

    Структура (порядок важен для модели):
      1. Системное сообщение (если есть) — всегда первым
      2. Сообщение-summary (если есть) — сразу после системного
      3. Последние recent_n сообщений user/assistant "как есть"

    Токены из messages не включаются (поле 'tokens' — внутренняя метрика).
    """
    # Разделяем системное сообщение и остальные
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Берём только последние N не-системных сообщений
    recent = non_system[-recent_n:] if len(non_system) > recent_n else non_system

    # Собираем чистый список (без поля 'tokens', которое не нужно API)
    def _clean(m: dict) -> dict:
        return {"role": m["role"], "content": m.get("content") or ""}

    result = [_clean(m) for m in system_msgs]

    if context_summary:
        result.append(
            {
                "role": "user",
                "content": (
                    f"[Резюме предыдущего диалога для контекста]\n{context_summary}"
                ),
            }
        )
        result.append(
            {
                "role": "assistant",
                "content": "Понял, учту контекст из резюме.",
            }
        )

    result.extend(_clean(m) for m in recent)
    return result


def _maybe_summarize(
    client: OpenAI,
    model: str,
    messages: list,
    context_summary: str,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
    interval: int = CONTEXT_SUMMARY_INTERVAL,
) -> tuple:
    """Проверяет, нужно ли создать новое summary, и если да — делает это.

    Суммаризация происходит каждые `interval` не-системных сообщений.
    Сообщения старше последних `recent_n` сворачиваются в summary и удаляются
    из messages (остаётся только скользящее окно + системное сообщение).

    Возвращает (обновлённые messages, обновлённый context_summary, было_ли_обновление).
    """
    non_system = [m for m in messages if m.get("role") != "system"]
    system_msgs = [m for m in messages if m.get("role") == "system"]

    # Суммаризируем, если накопилось достаточно сообщений сверх окна
    excess = len(non_system) - recent_n
    if excess < interval:
        return messages, context_summary, False

    # Сообщения, которые нужно свернуть (всё, кроме recent_n последних)
    to_summarize = non_system[:-recent_n] if recent_n > 0 else non_system

    # Если есть уже существующее summary — добавляем его как контекст
    summary_context = []
    if context_summary:
        summary_context = [
            {"role": "user", "content": f"[Предыдущее резюме]\n{context_summary}"},
            {"role": "assistant", "content": "Принято."},
        ]

    new_summary_text = _summarize_messages(client, model, summary_context + to_summarize)

    if new_summary_text:
        context_summary = new_summary_text
        logging.info(
            f"Summary обновлён: свёрнуто {len(to_summarize)} сообщений."
        )
        print(
            f"\n[Контекст: {len(to_summarize)} старых сообщений свёрнуто в summary]\n"
        )

    # Оставляем только системные + последние recent_n сообщений
    messages = system_msgs + non_system[-recent_n:]
    return messages, context_summary, True


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------


def main() -> None:
    """Основной поток выполнения CLI: инициализация конфигурации и интерактивный диалог."""
    if not API_KEY:
        raise SystemExit("API_KEY environment variable is not set.")

    args = parse_args()

    model = args.model or DEFAULT_MODEL
    base_url = args.base_url or BASE_URL
    max_tokens = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS
    temperature = (
        args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE
    )
    top_p = args.top_p if args.top_p is not None else DEFAULT_TOP_P
    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    system_prompt = args.system_prompt
    initial_prompt = args.initial_prompt

    session_path = None
    session_id_metrics = ""
    dialogue_start_time = time.time()
    duration = 0.0
    messages = []
    total_tokens_history = 0
    total_prompt_tokens_history = 0
    total_completion_tokens_history = 0
    context_summary: str = ""  # накопленное резюме старых сообщений

    # Опционально загрузить последнюю сессию и продолжить диалог
    if args.resume:
        loaded = _load_last_session()
        if loaded:
            last_path, last_data = loaded
            session_path = last_path
            session_id_metrics = session_path.replace("/", "_").replace(".json", "")
            try:
                session_data = _apply_session_data(
                    last_data,
                    model=model,
                    base_url=base_url,
                    system_prompt=system_prompt,
                    initial_prompt=initial_prompt,
                )
                messages = session_data["messages"]
                model = session_data["model"]
                base_url = session_data["base_url"]
                system_prompt = session_data["system_prompt"]
                initial_prompt = session_data["initial_prompt"]
                duration = session_data["duration"]
                total_tokens_history = session_data.get("total_tokens", 0)
                total_prompt_tokens_history = session_data.get("total_prompt_tokens", 0)
                total_completion_tokens_history = session_data.get(
                    "total_completion_tokens", 0
                )
                context_summary = last_data.get("context_summary", "")
                dialogue_start_time = time.time() - duration
                logging.info(f"Загрузка последней сессии: {session_path}")
                _print_loaded_history(messages)
            except Exception as resume_exc:
                logging.warning(f"Не удалось загрузить последнюю сессию: {resume_exc}")

    client = OpenAI(api_key=API_KEY, base_url=base_url)

    # Файл сессии переиспользуется на протяжении всей сессии
    if session_path is None:
        session_path = (
            f"dialogues/session_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_"
            f"{model.replace('/', '_')}.json"
        )
    session_id_metrics = session_path.replace("/", "_").replace(".json", "")
    request_index = 0

    # Таймер начала диалога для подсчёта общей длительности
    dialogue_start_time = time.time()

    # Добавляем системный промпт и начальное сообщение (seed)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if initial_prompt:
        messages.append({"role": "user", "content": initial_prompt})

    print("Введите запрос (type 'exit' чтобы выйти):")
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        # Inline-команды начинаются с '/'. Они изменяют конфигурацию текущей сессии
        if user_input.strip().startswith("/"):
            updates = _parse_inline_command(user_input)
            if updates:
                state = _apply_updates(
                    updates,
                    model=model,
                    base_url=base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    system_prompt=system_prompt,
                    initial_prompt=initial_prompt,
                    messages=messages,
                    session_path=session_path,
                    session_id_metrics=session_id_metrics,
                    dialogue_start_time=dialogue_start_time,
                    duration=duration,
                    context_summary=context_summary,
                )
                model = state["model"]
                base_url = state["base_url"]
                max_tokens = state["max_tokens"]
                temperature = state["temperature"]
                top_p = state["top_p"]
                top_k = state["top_k"]
                system_prompt = state["system_prompt"]
                initial_prompt = state["initial_prompt"]
                messages = state["messages"]
                session_path = state["session_path"]
                session_id_metrics = state["session_id_metrics"]
                dialogue_start_time = state["dialogue_start_time"]
                duration = state["duration"]
                context_summary = state.get("context_summary", context_summary)
                total_tokens_history = state.get("total_tokens", 0)
                total_prompt_tokens_history = state.get("total_prompt_tokens", 0)
                total_completion_tokens_history = state.get(
                    "total_completion_tokens", 0
                )
                # Обработка /showsummary
                if state.get("showsummary"):
                    if context_summary:
                        print("\n--- Текущее резюме контекста ---")
                        print(context_summary)
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

        # Добавляем запрос пользователя в историю переписки
        messages.append({"role": "user", "content": user_input})

        # Проверяем, нужно ли свернуть старые сообщения в summary
        messages, context_summary, summarized = _maybe_summarize(
            client, model, messages, context_summary
        )

        # Собираем контекст для отправки в API: system + summary + последние N сообщений
        api_messages = _build_context_for_api(messages, context_summary)

        # Выполнение API-запроса с учётом контекста переписки
        try:
            start_call = time.perf_counter()
            extra = {"top_k": top_k} if top_k is not None else None
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra,
            )
        except Exception as exc:
            # Ошибка вызова API — выводим сообщение и продолжаем диалог
            print("API error:", exc)
            continue

        api_call_time = time.perf_counter() - start_call
        text = (
            response.choices[0].message.content if response and response.choices else ""
        )

        # Извлекаем usage, токены и стоимость (если доступны)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(_get_usage_value(usage, "prompt_tokens") or 0)
        completion_tokens = int(_get_usage_value(usage, "completion_tokens") or 0)
        total_tokens = int(_get_usage_value(usage, "total_tokens") or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        total_tokens_history += total_tokens
        total_prompt_tokens_history += prompt_tokens
        total_completion_tokens_history += completion_tokens
        total_cost_rub = (
            (total_tokens_history / 1000.0) * USD_PER_1K_TOKENS * RUB_PER_USD
        )

        usd_cost = (total_tokens / 1000.0) * USD_PER_1K_TOKENS
        cost_rub = usd_cost * RUB_PER_USD
        total_s = time.time() - dialogue_start_time

        print(text)
        print(
            f"\n[Токены: запрос={prompt_tokens}, ответ={completion_tokens}, итого={total_tokens}]"
        )
        print(
            f"[История диалога: промпт={total_prompt_tokens_history}, ответ={total_completion_tokens_history}, всего={total_tokens_history} | ~{total_cost_rub:.2f}₽]"
        )

        # Обновляем историю переписки (сохраняем per-turn метрики токенов)
        messages.append(
            {
                "role": "assistant",
                "content": text,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                },
            }
        )

        # Метаинформация по запросу
        req_entry = {
            "model": model,
            "endpoint": "chat",
            "temp": temperature,
            "ttft": api_call_time,
            "req_time": api_call_time,
            "total_time": total_s,
            "tokens": total_tokens,
            "p_tokens": prompt_tokens,
            "c_tokens": completion_tokens,
            "cost_rub": cost_rub,
        }

        # Перезаписываем файл сессии после каждого шага
        session = {
            "dialogue_session_id": session_path,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "initial_prompt": initial_prompt,
            "messages": messages,
            "context_summary": context_summary,
            "turns": len(
                [m for m in messages if m.get("role") in ("user", "assistant")]
            ),
            "last_user_input": user_input,
            "last_assistant_content": text,
            "duration_seconds": time.time() - dialogue_start_time,
            "total_tokens": total_tokens_history,
            "total_prompt_tokens": total_prompt_tokens_history,
            "total_completion_tokens": total_completion_tokens_history,
        }
        try:
            session.setdefault("requests", []).append(req_entry)
            _save_session_to_path(session, session_path)
            try:
                metric_path = _log_request_metric(
                    req_entry, session_id_metrics, request_index
                )
                logging.info(f"Request metrics logged to: {metric_path}")
            except Exception as metric_exc:
                logging.debug(f"Failed to log request metrics: {metric_exc}")
            request_index += 1
            logging.info(f"Session updated: {session_path}")
        except Exception as save_exc:
            logging.warning(f"Не удалось обновить сессию: {save_exc}")

    # По завершении сессии сохраняем финальное состояние переписки
    if messages:
        session = {
            "dialogue_session_id": (
                f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_"
                f"{model.replace('/', '_')}"
            ),
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "initial_prompt": initial_prompt,
            "messages": messages,
            "context_summary": context_summary,
            "turns": len(
                [m for m in messages if m.get("role") in ("user", "assistant")]
            ),
            "duration_seconds": time.time() - dialogue_start_time,
            "total_tokens": total_tokens_history,
            "total_prompt_tokens": total_prompt_tokens_history,
            "total_completion_tokens": total_completion_tokens_history,
        }
        try:
            _save_session_to_path(session, session_path)
            logging.info(f"Session updated: {session_path}")
        except Exception as sess_exc:
            logging.warning(f"Failed to save session on exit: {sess_exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
