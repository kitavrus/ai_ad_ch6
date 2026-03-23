"""Разбор аргументов командной строки и inline-команд сессии."""

import argparse
import logging

from llm_agent.chatbot.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SessionConfig,
)
from llm_agent.chatbot.models import ContextStrategy

logger = logging.getLogger(__name__)

# Допустимые значения стратегий для CLI-подсказки
_STRATEGY_VALUES = [s.value for s in ContextStrategy]


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
        "--strategy",
        choices=_STRATEGY_VALUES,
        default=ContextStrategy.SLIDING_WINDOW.value,
        help="Стратегия управления контекстом",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Имя профиля пользователя для загрузки (из dialogues/memory/profiles/)",
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

    Поддерживаемые команды::

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

        # Управление стратегиями контекста:
        /strategy sliding_window   — переключить на Sliding Window
        /strategy sticky_facts     — переключить на Sticky Facts
        /strategy branching        — переключить на Branching
        /showfacts                 — показать текущие sticky facts
        /setfact ключ: значение    — вручную добавить/обновить факт
        /delfact ключ              — удалить факт по ключу

        # Команды ветвления (branching):
        /checkpoint                — создать точку сохранения (checkpoint)
        /branch имя                — создать новую ветку от последнего checkpoint
        /switch имя_или_id         — переключиться на ветку по имени или ID
        /branches                  — список всех веток

        # Команды управления памятью:
        /memory <short_term|working|long_term> <save|load|show|stats|clear>
        /memsave [short|working|long]       — сохранить указанную память
        /memload [short|working|long]       — загрузить указанную память
        /memshow [all|short|working|long]   — показать состояние памяти
        /memstats                           — статистика по всей памяти
        /memclear [all|short|working|long]  — очистить указанную память

        /setpref <ключ>: <значение>         — установить предпочтение (working)
        /getpref <ключ>                     — получить предпочтение (working)
        /decision <описание>               — добавить решение (long_term)
        /knowledge <ключ>: <значение>      — добавить знание (long_term)
        /profile <ключ>: <значение>        — изменить профиль (long_term)

    Args:
        line: Строка ввода пользователя, начинающаяся с '/'.

    Returns:
        Словарь обновлений (может быть пустым).
    """
    cmd = line.strip()
    if not cmd.startswith("/"):
        return {}
    payload = cmd[1:].strip()
    if not payload:
        return {}

    # Специальная обработка /plan
    _first_word = payload.split(None, 1)[0].lower()
    if _first_word == "plan":
        _rest = payload[len("plan"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else "status"
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        return {"plan": {"action": _sub, "arg": _arg}}

    # Специальная обработка /mem (aliases для /settask, /setpref, /remember)
    if _first_word == "mem":
        _rest = payload[len("mem"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else ""
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        if _sub == "task":
            return {"settask": _arg} if _arg else {}
        elif _sub == "pref":
            return {"setpref": _arg} if _arg else {}
        elif _sub == "know":
            return {"remember": _arg} if _arg else {}
        # Для остальных /mem* команд — пропускаем, они разберутся ниже через key-нормализацию
        return {}

    # Специальная обработка /rag
    if _first_word == "rag":
        _rest = payload[len("rag"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else "status"
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        return {"rag": {"action": _sub, "arg": _arg}}

    # Специальная обработка /invariant
    if _first_word == "invariant":
        _rest = payload[len("invariant"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else "list"
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        return {"invariant": {"action": _sub, "arg": _arg}}

    # Специальная обработка /profile — аргумент может содержать '='
    if _first_word == "profile":
        _rest = payload[len("profile"):].strip()
        _sub_parts = _rest.split(None, 1)
        _sub_cmd = _sub_parts[0].lower() if _sub_parts else "show"
        _sub_arg = _sub_parts[1].strip() if len(_sub_parts) > 1 else ""
        return {"profile": {"action": _sub_cmd, "arg": _sub_arg}}

    # Специальная обработка /task
    if _first_word == "task":
        _rest = payload[len("task"):].strip()
        _parts = _rest.split(None, 1)
        _task_action = _parts[0].lower() if _parts else "show"
        _task_arg = _parts[1].strip() if len(_parts) > 1 else ""
        # /task execute → alias для /plan builder
        if _task_action == "execute":
            return {"plan": {"action": "builder", "arg": _task_arg}}
        return {"task": {"action": _task_action, "arg": _task_arg}}

    # Специальная обработка /reminders
    if _first_word == "reminders":
        _rest = payload[len("reminders"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else "list"
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        # /reminders list pending → action=list, arg=pending
        # /reminders show <id> → action=show, arg=<id>
        return {"reminders": {"action": _sub, "arg": _arg}}

    # Специальная обработка /mcp
    if _first_word == "mcp":
        _rest = payload[len("mcp"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else "status"
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        return {"mcp": {"action": _sub, "arg": _arg}}

    # Специальная обработка /project
    if _first_word == "project":
        _rest = payload[len("project"):].strip()
        _parts = _rest.split(None, 1)
        _sub = _parts[0].lower() if _parts else "show"
        _arg = _parts[1].strip() if len(_parts) > 1 else ""
        # Нормализация aliases: /project plans → tasks; /project plan * → task_*
        if _sub == "plans":
            return {"project": {"action": "tasks", "arg": _arg}}
        if _sub == "plan":
            _plan_parts = _arg.split(None, 1)
            _plan_sub = _plan_parts[0].lower() if _plan_parts else ""
            _plan_arg = _plan_parts[1].strip() if len(_plan_parts) > 1 else ""
            _alias_map = {"new": "task_new", "rename": "task_rename", "describe": "task_describe"}
            if _plan_sub in _alias_map:
                return {"project": {"action": _alias_map[_plan_sub], "arg": _plan_arg}}
            return {"project": {"action": f"plan_{_plan_sub}" if _plan_sub else "show", "arg": _plan_arg}}
        # /project task new|rename|describe → нормализуем в task_new|task_rename|task_describe
        if _sub == "task":
            _task_parts = _arg.split(None, 1)
            _task_sub = _task_parts[0].lower() if _task_parts else ""
            _task_arg = _task_parts[1].strip() if len(_task_parts) > 1 else ""
            _alias_map2 = {"new": "task_new", "rename": "task_rename", "describe": "task_describe"}
            if _task_sub in _alias_map2:
                return {"project": {"action": _alias_map2[_task_sub], "arg": _task_arg}}
        # Нормализуем дефисы в action для единообразия с другими подкомандами
        return {"project": {"action": _sub.replace("-", "_"), "arg": _arg}}

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

    # --- Базовые параметры модели ---
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
        updates["resume"] = (not value) or value.lower() in {"true", "1", "yes", "on"}
    elif key == "showsummary":
        updates["showsummary"] = True

    # --- Переключатель стратегий ---
    elif key == "strategy":
        val = value.lower().strip()
        if val == "status":
            updates["strategy_status"] = True
        else:
            # Допускаем короткие псевдонимы: sw, sf, br
            _aliases = {
                "sw": ContextStrategy.SLIDING_WINDOW.value,
                "sf": ContextStrategy.STICKY_FACTS.value,
                "br": ContextStrategy.BRANCHING.value,
            }
            val = _aliases.get(val, val)
            if val in _STRATEGY_VALUES:
                updates["strategy"] = val
            else:
                logger.warning("Неизвестная стратегия: %s. Доступны: %s", val, _STRATEGY_VALUES)

    # --- Sticky Facts ---
    elif key == "showfacts":
        updates["showfacts"] = True
    elif key == "setfact":
        # /setfact ключ: значение  или  /setfact ключ значение
        if ":" in value:
            fact_key, _, fact_val = value.partition(":")
        else:
            parts2 = value.split(None, 1)
            fact_key = parts2[0] if parts2 else ""
            fact_val = parts2[1] if len(parts2) > 1 else ""
        fact_key = fact_key.strip().lower()
        fact_val = fact_val.strip()
        if fact_key and fact_val:
            updates["setfact"] = {"key": fact_key, "value": fact_val}
    elif key == "delfact":
        if value.strip():
            updates["delfact"] = value.strip().lower()

    # --- Branching ---
    elif key == "checkpoint":
        updates["checkpoint"] = True
    elif key == "branch":
        # /branch имя — создать ветку с данным именем от последнего checkpoint
        updates["branch"] = value.strip() if value.strip() else f"branch-{__import__('uuid').uuid4().hex[:4]}"
    elif key == "switch":
        # /switch имя_или_id — переключиться на ветку
        if value.strip():
            updates["switch"] = value.strip()
    elif key == "branches":
        updates["branches"] = True

    # --- Управление памятью ---
    elif key == "memshow":
        updates["memshow"] = value.strip().lower() or "all"
    elif key == "memstats":
        updates["memstats"] = True
    elif key == "memclear":
        updates["memclear"] = value.strip().lower() or "short_term"
    elif key == "memsave":
        updates["memsave"] = value.strip().lower() or "all"
    elif key == "memload":
        updates["memload"] = value.strip().lower() or "all"
    elif key == "settask":
        if value.strip():
            updates["settask"] = value.strip()
    elif key == "setpref":
        # /setpref key=value
        if value.strip():
            updates["setpref"] = value.strip()
    elif key == "remember":
        # /remember key=value  or  /remember decision text
        if value.strip():
            updates["remember"] = value.strip()
    # --- Управление профилем пользователя ---
    elif key == "profile":
        # Субкоманды:
        #   /profile show                  — показать текущий профиль
        #   /profile list                  — список сохранённых профилей
        #   /profile name <val>            — задать имя, создать папку и сохранить профиль
        #   /profile style <k>=<v>         — стиль (тон, краткость, язык)
        #   /profile format <k>=<v>        — формат вывода
        #   /profile constraint add <text> — добавить ограничение
        #   /profile constraint del <text> — удалить ограничение
        #   /profile model <name>          — задать предпочтительную модель
        #   /profile load <name>           — загрузить профиль по имени
        sub_parts = value.strip().split(None, 1)
        sub_cmd = sub_parts[0].lower() if sub_parts else "show"
        sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
        updates["profile"] = {"action": sub_cmd, "arg": sub_arg}

    # Устаревшие варианты — перенаправляем
    elif key == "memory":
        pass  # игнорируем голый /memory без аргументов

    return updates


