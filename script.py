#!/usr/bin/env python3
"""CLI чат-бот на основе ИИ (интерактивный режим).

Этот модуль предоставляет простой интерфейс командной строки для общения с AI-моделью:
- настройка параметров через CLI;
- интерактивный диалог с сохранением истории переписки;
- вывод ответа и продолжение диалога.

Код структурирован в стиле, близком к PEP 8, с читабельными комментариями на русском языке.
"""

import os
import argparse
import logging
import json
import glob
from typing import Optional, Tuple
import time
from datetime import datetime
from openai import OpenAI

BASE_URL = "https://routerai.ru/api/v1"
API_KEY = os.getenv("API_KEY")

# По умолчанию используются настройки модели и параметров генерации

# Блок конфигурации по умолчанию
# Эти значения используются, если пользователь не переопределил их через CLI

# DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_MAX_TOKENS = None
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50

# Значения по умолчанию применяются, если пользователь не переопределил их через CLI


def parse_args():
    # Разбор аргументов командной строки
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(
        description="CLI chat with AI model (interactive).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Модель, которую использовать по умолчанию можно переопределить через CLI
    parser.add_argument("-m", "--model", help="Model to use")
    # Базовый URL API (по умолчанию взят из BASE_URL)
    parser.add_argument("-u", "--base-url", help="Base URL for API")
    # Максимальное число токенов в ответе (None = без ограничения)
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens in response"
    )
    # Температура генерации
    parser.add_argument(
        "-T", "--temperature", type=float, default=None, help="Temperature"
    )
    # Top-p (nucleus sampling)
    parser.add_argument(
        "-p", "--top-p", type=float, default=None, help="Top-p (nucleus sampling)"
    )
    # Top-k ограничение кандидатов
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Top-k")
    # Системный промпт
    parser.add_argument("--system-prompt", default=None, help="System prompt")
    # Начальное сообщение в переписке
    parser.add_argument(
        "--initial-prompt", default=None, help="Initial user prompt (seed)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Load last session and resume dialog"
    )
    return parser.parse_args()


def _parse_inline_command(line: str) -> dict:
    """Parse interactive inline commands starting with '/'.

    Supported commands (examples):
      /model=gpt-4
      /base-url https://api.example.com
      /max-tokens 1024
      /temperature 0.8
      /top-p 0.92
      /top-k 50
      /system-prompt New system prompt text
      /initial-prompt Seed message for the conversation
      /resume true|false

    Returns a dict of updates suitable for applying to the running session.
    Keys are normalized to pythonic names (underscores).
    """
    cmd = line.strip()
    if not cmd.startswith("/"):
        return {}
    payload = cmd[1:].strip()
    if not payload:
        return {}

    # Split into key and value
    if "=" in payload:
        key, value = payload.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
    else:
        parts = payload.split(None, 1)
        key = parts[0].strip().lower()
        value = parts[1].strip() if len(parts) > 1 else ""

    # Normalize key to underscore style
    key = key.replace("-", "_")

    updates = {}
    # Map known keys to internal variable names
    if key in {"model"}:
        updates["model"] = value if value else None
    elif key in {"base_url", "base_url", "baseurl", "base-url"}:
        updates["base_url"] = value or None
    elif key == "max_tokens":
        try:
            updates["max_tokens"] = int(value)
        except Exception:
            updates["max_tokens"] = None
    elif key == "temperature":
        try:
            updates["temperature"] = float(value)
        except Exception:
            updates["temperature"] = None
    elif key == "top_p":
        try:
            updates["top_p"] = float(value)
        except Exception:
            updates["top_p"] = None
    elif key == "top_k":
        try:
            updates["top_k"] = int(value)
        except Exception:
            updates["top_k"] = None
    elif key in {"system_prompt", "system_prompt", "system-prompt"}:
        updates["system_prompt"] = value
    elif key in {"initial_prompt", "initial_prompt", "initial-prompt"}:
        updates["initial_prompt"] = value
    elif key == "resume":
        updates["resume"] = value.lower() in {"true", "1", "yes", "on"}
    else:
        # Unknown key; ignore
        pass
    return updates


def _get_usage_value(usage, key):
    """Safely extract a token usage value from a usage object or dict."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def main():
    # Основной поток выполнения CLI: инициализация конфигурации и интерактивный диалог
    # Основной вход в CLI: инициализация конфигурации и интерактивный диалог
    # Внешние переменные для интерактивной настройки (декларируем до _apply_updates)
    session_path = None
    session_id_metrics = ""
    dialogue_start_time = time.time()
    duration = 0
    messages = []

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
    # Опционально загрузить последнюю сессию и продолжить диалог
    if args.resume:
        loaded = _load_last_session()
        if loaded:
            last_path, last_data = loaded
            session_path = last_path
            session_id_metrics = session_path.replace("/", "_").replace(".json", "")
            try:
                messages = last_data.get("messages", [])
                model = last_data.get("model", model)
                base_url = last_data.get("base_url", base_url)
                system_prompt = last_data.get("system_prompt", system_prompt)
                initial_prompt = last_data.get("initial_prompt", initial_prompt)
                duration = last_data.get("duration_seconds", 0)
                dialogue_start_time = time.time() - duration
                logging.info(f"Загрузка последней сессии: {session_path}")
                # Print loaded history to help recall context
                try:
                    pairs = []
                    for m in messages:
                        r = m.get("role")
                        if r in {"user", "assistant"}:
                            c = (m.get("content") or "").strip()
                            if c:
                                pairs.append((r, c))
                    last_pairs = pairs[-5:] if len(pairs) >= 5 else pairs
                    if last_pairs:
                        print("\nLoaded history (last exchanges):")
                        for r, t in last_pairs:
                            tag = "You" if r == "user" else "Assistant"
                            print(f"{tag}: {t}")
                        print()
                except Exception as _hist_err:
                    logging.debug(f"Failed to print loaded history: {_hist_err}")
            except Exception as resume_exc:
                logging.warning(f"Не удалось загрузить последнюю сессию: {resume_exc}")

    client = OpenAI(api_key=API_KEY, base_url=base_url)
    # Файл сессии будет переиспользоваться во время всей сессии, переопределяя его на каждом шаге
    if "session_path" not in locals() or session_path is None:
        session_path = f"dialogues/session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{model.replace('/', '_')}.json"
    session_id_metrics = session_path.replace("/", "_").replace(".json", "")
    request_index = 0
    resume_used = False

    # Таймер начала диалога для подсчета общей длительности
    dialogue_start_time = time.time()

    try:
        messages  # type: ignore
    except NameError:
        messages = []
    # Добавляем системный промпт и начальное сообщение в переписку (seed)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if initial_prompt:
        messages.append({"role": "user", "content": initial_prompt})

    def _apply_updates(updates: dict):
        """Apply inline command updates to the running session.

        This updates the in-memory configuration used for subsequent API calls
        and adjusts the current messages to reflect system/initial prompts if
        they have changed.
        """
        nonlocal model, base_url, max_tokens, temperature, top_p, top_k
        nonlocal system_prompt, initial_prompt, messages
        # Session state used during interactive loop
        nonlocal session_path, session_id_metrics, dialogue_start_time, duration
        # Generic updates
        for k, v in updates.items():
            if v is None:
                continue
            if k == "model":
                model = v
            elif k == "base_url":
                base_url = v
            elif k == "max_tokens":
                max_tokens = v
            elif k == "temperature":
                temperature = v
            elif k == "top_p":
                top_p = v
            elif k == "top_k":
                top_k = v
            elif k == "system_prompt":
                system_prompt = v
                # Update existing system message or insert new one
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = system_prompt
                elif system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
            elif k == "initial_prompt":
                initial_prompt = v
                # If there is an initial user message, update it; otherwise append a seed
                if len(messages) >= 2 and messages[1].get("role") == "user":
                    messages[1]["content"] = initial_prompt
                elif initial_prompt:
                    messages.append({"role": "user", "content": initial_prompt})
            elif k == "resume":
                # Handle resume: load the last saved session and continue
                if v:
                    loaded = _load_last_session()
                    if loaded:
                        last_path, last_data = loaded
                        session_path = last_path
                        session_id_metrics = session_path.replace("/", "_").replace(
                            ".json", ""
                        )
                        try:
                            messages = last_data.get("messages", [])
                            model = last_data.get("model", model)
                            base_url = last_data.get("base_url", base_url)
                            system_prompt = last_data.get(
                                "system_prompt", system_prompt
                            )
                            initial_prompt = last_data.get(
                                "initial_prompt", initial_prompt
                            )
                            duration = last_data.get("duration_seconds", 0)
                            dialogue_start_time = time.time() - duration
                            logging.info(f"Resumed last session: {session_path}")
                            # Print loaded history to help recall context
                            try:
                                pairs = []
                                for m in messages:
                                    rr = m.get("role")
                                    if rr in {"user", "assistant"}:
                                        cc = (m.get("content") or "").strip()
                                        if cc:
                                            pairs.append((rr, cc))
                                last_pairs = pairs[-5:] if len(pairs) >= 5 else pairs
                                if last_pairs:
                                    print("\nLoaded history (last exchanges):")
                                    for rr, tt in last_pairs:
                                        tag = "You" if rr == "user" else "Assistant"
                                        print(f"{tag}: {tt}")
                                    print()
                            except Exception as _hist_err:
                                logging.debug(
                                    f"Failed to print loaded history: {_hist_err}"
                                )
                        except Exception as _resume_exc:
                            logging.warning(
                                f"Failed to resume last session: {_resume_exc}"
                            )
                    else:
                        logging.info("No last session found to resume")
                        print("No last session found to resume")
                # If user explicitly sets resume to False, we do nothing special

        # If system prompt changed and no system message exists, ensure one exists.
        if system_prompt and not (messages and messages[0].get("role") == "system"):
            messages.insert(0, {"role": "system", "content": system_prompt})

    print("Введите запрос (type 'exit' чтобы выйти):")  # приглашение к вводу запроса
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input:
            continue
        # Inline commands start with '/'. They modify the running session
        if user_input.strip().startswith("/"):
            updates = _parse_inline_command(user_input)
            if updates:
                _apply_updates(updates)
                print(f"Updated config: {updates}")
                continue
            else:
                # If not a recognized command, ignore the line but do not exit
                print("Unknown command")
                continue
        if user_input.strip().lower() in {"exit", "quit", "q"}:
            break
        # добавляем запрос пользователя в историю переписки
        messages.append({"role": "user", "content": user_input})

        # Выполнение API-запроса с учётом контекста переписки
        try:
            start_call = time.perf_counter()
            extra = {"top_k": top_k} if top_k is not None else None
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra,
            )
        except Exception as exc:
            # Ошибка вызова API — выводим сообщение и продолжаем диалог
            print("API error:", exc)
            continue

        # Ответ API
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

        USD_PER_1K_TOKENS = 0.0015
        RUB_PER_USD = 100.0
        usd_cost = (total_tokens / 1000.0) * USD_PER_1K_TOKENS
        cost_rub = usd_cost * RUB_PER_USD
        total_s = time.time() - dialogue_start_time
        print(text)
        # Обновляем историю переписки
        messages.append({"role": "assistant", "content": text})

        # Вставляем пер‑запросовую метаинформацию в сессию
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
        # Перезаписываем файл сессии после каждого шага (диалог сохраняется целиком)
        session = {
            "dialogue_session_id": session_path,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "initial_prompt": initial_prompt,
            "messages": messages,
            "turns": len(
                [m for m in messages if m.get("role") in ("user", "assistant")]
            ),
            "last_user_input": user_input,
            "last_assistant_content": text,
            "duration_seconds": time.time() - dialogue_start_time,
        }
        try:
            # добавим per-request метадату в сессию
            session.setdefault("requests", []).append(req_entry)
            # сохранение файла сессии после шага
            _save_session_to_path(session, session_path)  # type: ignore[call-arg]
            # логируем per-request метрику в отдельном файле
            try:
                metric_path = _log_request_metric(
                    req_entry, session_id_metrics, request_index
                )
                logging.info(f"Request metrics logged to: {metric_path}")
            except Exception as _e:
                logging.debug(f"Failed to log request metrics: {_e}")
            request_index += 1
            logging.info(f"Session updated: {session_path}")
        except Exception as save_exc:
            logging.warning(f"Не удалось обновить сессию: {save_exc}")

    # По завершении сессии (когда пользователь вводит exit), автоматически сохраняем всю переписку в виде сессии
    if messages:
        session = {
            "dialogue_session_id": f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{model.replace('/', '_')}",
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "initial_prompt": initial_prompt,
            "messages": messages,
            "turns": len(
                [m for m in messages if m.get("role") in ("user", "assistant")]
            ),
            "duration_seconds": time.time() - dialogue_start_time,
        }
        try:
            _save_session_to_path(session, session_path)
            logging.info(f"Session updated: {session_path}")
        except Exception as sess_exc:
            logging.warning(f"Failed to save session on exit: {sess_exc}")


def _save_dialogue(dialogue: dict, dir_path: str = "dialogues") -> str:
    """Сохранение диалога в файл JSON с метаданными."""
    import os

    os.makedirs(dir_path, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_slug = dialogue.get("model", "model").replace("/", "_").replace(" ", "_")
    filename = f"{dir_path}/dialogue_{ts}_{model_slug}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dialogue, f, ensure_ascii=False, indent=2)
    return filename


def _save_session_to_path(session: dict, path: str) -> str:
    """Сохранение всей сессии чата по указанному пути (overwrite)."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return path


def _log_request_metric(req_entry: dict, session_id: str, idx: int) -> str:
    """Сохраняет метаданные одного запроса в файл-лог, чтобы иметь отдельный файл на запрос."""
    import os, json

    dir_path = "dialogues/metrics"
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{dir_path}/session_{session_id}_req_{idx:04d}.log"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(req_entry, f, ensure_ascii=False, indent=2)
    return filename


def _save_session(session: dict, dir_path: str = "dialogues") -> str:
    """Сохранение всей сессии чата в отдельный JSON-файл."""
    import os

    os.makedirs(dir_path, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_slug = session.get("model", "model").replace("/", "_").replace(" ", "_")
    filename = f"{dir_path}/session_{ts}_{model_slug}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return filename


def _load_last_session() -> Optional[Tuple[str, dict]]:
    """Загружает последнюю сохранённую сессию из dialogues/session_*.json.
    Возвращает кортеж (путь к файлу, данные) или None, если загрузить не удалось.
    """
    import os

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


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
