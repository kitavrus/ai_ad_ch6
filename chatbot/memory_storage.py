"""Слой сохранения и загрузки данных памяти."""

from __future__ import annotations

from datetime import datetime
import glob
import json
import logging
import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from chatbot.config import DEFAULT_PROFILE, DIALOGUES_DIR

if TYPE_CHECKING:
    from chatbot.models import UserProfile

logger = logging.getLogger(__name__)


# ===========================================================================
# ПУТИ К ФАЙЛАМ ПАМЯТИ (профиль-специфичные)
# ===========================================================================


def get_profile_dir(profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(DIALOGUES_DIR, profile_name)


def get_memory_dir(profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(get_profile_dir(profile_name), "memory")


def get_short_term_dir(profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(get_memory_dir(profile_name), "short_term")


def get_working_dir(profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(get_memory_dir(profile_name), "working")


def get_long_term_dir(profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(get_memory_dir(profile_name), "long_term")


# Aliases pointing to DEFAULT_PROFILE for backward compat
MEMORY_DIR: str = get_memory_dir()
SHORT_TERM_DIR: str = get_short_term_dir()
WORKING_DIR: str = get_working_dir()
LONG_TERM_DIR: str = get_long_term_dir()
PROFILES_DIR: str = os.path.join(MEMORY_DIR, "profiles")  # kept for compat


def _safe_name(name: str) -> str:
    """Превращает произвольную строку в безопасное имя файла.

    Берёт только basename (без пути), убирает расширение .json,
    заменяет все символы кроме букв/цифр/дефиса/подчёркивания на '_',
    обрезает до 64 символов.
    """
    name = os.path.basename(name)
    name = re.sub(r"\.json$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\-]", "_", name)
    return name[:64] or "default"


# ===========================================================================
# КРАТКОСРОЧНАЯ ПАМЯТЬ
# ===========================================================================


def save_short_term(memory: dict, session_id: str, profile_name: str = DEFAULT_PROFILE) -> str:
    """Сохраняет краткосрочную память в файл.

    Args:
        memory: Словарь данных ShortTermMemory.
        session_id: ID сессии.
        profile_name: Имя профиля.

    Returns:
        Путь к сохранённому файлу.
    """
    dir_path = get_short_term_dir(profile_name)
    os.makedirs(dir_path, exist_ok=True)
    filename = f"session_{_safe_name(session_id)}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_short_term_last(session_id: str, profile_name: str = DEFAULT_PROFILE) -> Optional[dict]:
    """Загружает последнюю краткосрочную память для сессии.

    Args:
        session_id: ID сессии.
        profile_name: Имя профиля.

    Returns:
        Словарь данных или None.
    """
    try:
        pattern = os.path.join(get_short_term_dir(profile_name), f"session_{session_id}_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Не удалось загрузить краткосрочную память: %s", exc)
        return None


# ===========================================================================
# РАБОЧАЯ ПАМЯТЬ
# ===========================================================================


def save_working_memory(memory: dict, task_name: Optional[str] = None, profile_name: str = DEFAULT_PROFILE) -> str:
    """Сохраняет рабочую память в файл.

    Args:
        memory: Словарь данных WorkingMemory.
        task_name: Название задачи (опционально).
        profile_name: Имя профиля.

    Returns:
        Путь к сохранённому файлу.
    """
    dir_path = get_working_dir(profile_name)
    os.makedirs(dir_path, exist_ok=True)
    task_part = _safe_name(task_name) if task_name else "current"
    filename = f"task_{task_part}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_working_memory(task_name: str, profile_name: str = DEFAULT_PROFILE) -> Optional[dict]:
    """Загружает рабочую память по названию задачи.

    Args:
        task_name: Название задачи.
        profile_name: Имя профиля.

    Returns:
        Словарь данных или None.
    """
    try:
        pattern = os.path.join(get_working_dir(profile_name), f"task_{task_name}_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Не удалось загрузить рабочую память: %s", exc)
        return None


def list_working_memories(profile_name: str = DEFAULT_PROFILE) -> list:
    """Возвращает список всех сохранённых рабочих памятей.

    Returns:
        Список dict с информацией о файлах.
    """
    try:
        pattern = os.path.join(get_working_dir(profile_name), "task_*.json")
        paths = glob.glob(pattern)
        result = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                result.append({
                    "path": p,
                    "task": data.get("current_task", "unknown"),
                    "status": data.get("task_status", "unknown"),
                    "updated_at": data.get("updated_at", ""),
                })
        return result
    except Exception as exc:
        logger.warning("Не удалось получить список рабочих памятей: %s", exc)
        return []


# ===========================================================================
# ДОЛГОСРОЧНАЯ ПАМЯТЬ
# ===========================================================================


def save_long_term(memory: dict, name: Optional[str] = None, profile_name: str = DEFAULT_PROFILE) -> str:
    """Сохраняет долговременную память в файл.

    Args:
        memory: Словарь данных LongTermMemory.
        name: Название профиля/памяти (по умолчанию "default").
        profile_name: Имя профиля.

    Returns:
        Путь к сохранённому файлу.
    """
    dir_path = get_long_term_dir(profile_name)
    os.makedirs(dir_path, exist_ok=True)
    lt_name = name or "default"
    filename = f"profile_{lt_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_long_term(name: str = "default", profile_name: str = DEFAULT_PROFILE) -> Optional[dict]:
    """Загружает долговременную память по имени профиля.

    Args:
        name: Имя профиля.
        profile_name: Имя профиля (директория).

    Returns:
        Словарь данных или None.
    """
    try:
        pattern = os.path.join(get_long_term_dir(profile_name), f"profile_{name}_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Не удалось загрузить долговременную память: %s", exc)
        return None


def list_long_term_memories(profile_name: str = DEFAULT_PROFILE) -> list:
    """Возвращает список всех сохранённых долговременных памятей.

    Returns:
        Список dict с информацией о файлах.
    """
    try:
        pattern = os.path.join(get_long_term_dir(profile_name), "profile_*.json")
        paths = glob.glob(pattern)
        result = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                profile = data.get("user_profile", {})
                name = profile.get("name", data.get("name", "unknown"))
                result.append({
                    "path": p,
                    "name": name,
                    "profile": profile,
                    "decisions_count": len(data.get("decisions_log", [])),
                    "knowledge_count": len(data.get("knowledge_base", {})),
                })
        return result
    except Exception as exc:
        logger.warning("Не удалось получить список долговременных памятей: %s", exc)
        return []


# ===========================================================================
# ПРОФИЛИ ПОЛЬЗОВАТЕЛЯ
# ===========================================================================


def save_profile(profile: UserProfile, name: Optional[str] = None) -> str:
    """Сохраняет профиль пользователя в файл dialogues/{name}/profile.json.

    Args:
        profile: Объект UserProfile.
        name: Имя профиля. По умолчанию берётся из profile.name.

    Returns:
        Путь к сохранённому файлу.
    """

    safe = _safe_name(name or profile.name or "default")
    profile_dir = os.path.join(DIALOGUES_DIR, safe)
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, "profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info("Профиль сохранён: %s", path)
    return path


def load_profile(name: str = "default") -> Optional[UserProfile]:
    """Загружает профиль пользователя из dialogues/{name}/profile.json.

    Args:
        name: Имя профиля.

    Returns:
        Объект UserProfile или None, если файл не найден.
    """
    from chatbot.models import UserProfile

    safe = _safe_name(name)
    path = os.path.join(DIALOGUES_DIR, safe, "profile.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return UserProfile(**data)
    except Exception as exc:
        logger.warning("Не удалось загрузить профиль '%s': %s", name, exc)
        return None


def list_profiles() -> List[str]:
    """Возвращает список имён профилей (поддиректорий dialogues/ с profile.json).

    Returns:
        Список имён профилей.
    """
    if not os.path.exists(DIALOGUES_DIR):
        return []
    names = []
    for entry in sorted(os.listdir(DIALOGUES_DIR)):
        entry_path = os.path.join(DIALOGUES_DIR, entry)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "profile.json")):
            names.append(entry)
    return names


# ===========================================================================
# ЭКСПОРТ/ИМПОРТ ПОЛНОГО СОСТОЯНИЯ
# ===========================================================================


def export_memory_state(short_term: dict, working: dict, long_term: dict, profile_name: str = DEFAULT_PROFILE) -> str:
    """Экспортирует всё состояние памяти в один файл.

    Args:
        short_term: Данные ShortTermMemory.
        working: Данные WorkingMemory.
        long_term: Данные LongTermMemory.
        profile_name: Имя профиля.

    Returns:
        Путь к сохранённому файлу.
    """
    mem_dir = get_memory_dir(profile_name)
    os.makedirs(mem_dir, exist_ok=True)
    filename = f"state_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(mem_dir, filename)
    state = {
        "exported_at": datetime.utcnow().isoformat(),
        "short_term": short_term,
        "working": working,
        "long_term": long_term,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path


def import_memory_state(path: str) -> Tuple[dict, dict, dict]:
    """Импортирует всё состояние памяти из файла.

    Args:
        path: Путь к файлу состояния.

    Returns:
        Кортеж (short_term, working, long_term) - три словаря.
    """
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    return (
        state.get("short_term", {}),
        state.get("working", {}),
        state.get("long_term", {}),
    )


# ===========================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===========================================================================


def get_memory_stats(profile_name: str = DEFAULT_PROFILE) -> dict:
    """Возвращает статистику по всей памяти.

    Returns:
        dict с количеством файлов и размерами.
    """
    stats = {
        "short_term": {"files": 0, "size_bytes": 0},
        "working": {"files": 0, "size_bytes": 0},
        "long_term": {"files": 0, "size_bytes": 0},
    }

    for directory, key in [
        (get_short_term_dir(profile_name), "short_term"),
        (get_working_dir(profile_name), "working"),
        (get_long_term_dir(profile_name), "long_term"),
    ]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    path = os.path.join(directory, filename)
                    stats[key]["files"] += 1
                    stats[key]["size_bytes"] += os.path.getsize(path)

    return stats
