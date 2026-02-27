"""Разбор аргументов командной строки и inline-команд сессии."""

import argparse
import logging
from typing import Optional

from chatbot.config import (
    BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SessionConfig,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Разбор аргументов командной строки.

    Returns:
        Объект Namespace со значениями аргументов.
    """
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
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Температура генерации",
    )
    parser.add_argument(
        "-p",
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus sampling)",
    )
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Top-k")
    parser.add_argument("--system-prompt", default=None, help="Системный промпт")
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Начальное сообщение пользователя (seed)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Загрузить последнюю сессию и продолжить диалог",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SessionConfig:
    """Создаёт SessionConfig из аргументов командной строки.

    Args:
        args: Разобранные аргументы CLI.

    Returns:
        Объект конфигурации сессии.
    """
    return SessionConfig(
        model=args.model or DEFAULT_MODEL,
        max_tokens=args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS,
        temperature=args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE,
        top_p=args.top_p if args.top_p is not None else DEFAULT_TOP_P,
        top_k=args.top_k if args.top_k is not None else DEFAULT_TOP_K,
        system_prompt=args.system_prompt,
        initial_prompt=args.initial_prompt,
    )


def parse_inline_command(line: str) -> dict:
    """Разбор интерактивных inline-команд, начинающихся с '/'.

    Поддерживаемые команды (примеры)::

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

    Args:
        line: Строка ввода пользователя, начинающаяся с '/'.

    Returns:
        Словарь обновлений конфигурации (может быть пустым).
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

    updates: dict = {}
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


def get_resume_flag(args: argparse.Namespace) -> bool:
    """Возвращает флаг --resume из аргументов CLI.

    Args:
        args: Разобранные аргументы CLI.

    Returns:
        True, если передан флаг --resume.
    """
    return bool(getattr(args, "resume", False))
