"""Слой сохранения и загрузки данных памяти."""

import glob
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from chatbot.config import DIALOGUES_DIR

logger = logging.getLogger(__name__)


# ===========================================================================
# ПУТИ К ФАЙЛАМ ПАМЯТИ
# ===========================================================================


MEMORY_DIR: str = os.path.join(DIALOGUES_DIR, "memory")
SHORT_TERM_DIR: str = os.path.join(MEMORY_DIR, "short_term")
WORKING_DIR: str = os.path.join(MEMORY_DIR, "working")
LONG_TERM_DIR: str = os.path.join(MEMORY_DIR, "long_term")


# ===========================================================================
# КРАТКОСРОЧНАЯ ПАМЯТЬ
# ===========================================================================


def save_short_term(memory: dict, session_id: str) -> str:
    """Сохраняет краткосрочную память в файл.
    
    Args:
        memory: Словарь данных ShortTermMemory.
        session_id: ID сессии.
    
    Returns:
        Путь к сохранённому файлу.
    """
    os.makedirs(SHORT_TERM_DIR, exist_ok=True)
    filename = f"session_{session_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(SHORT_TERM_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_short_term_last(session_id: str) -> Optional[dict]:
    """Загружает последнюю краткосрочную память для сессии.
    
    Args:
        session_id: ID сессии.
    
    Returns:
        Словарь данных или None.
    """
    try:
        pattern = os.path.join(SHORT_TERM_DIR, f"session_{session_id}_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Не удалось загрузить краткосрочную память: %s", exc)
        return None


# ===========================================================================
# РАБОЧАЯ ПАМЯТЬ
# ===========================================================================


def save_working_memory(memory: dict, task_name: Optional[str] = None) -> str:
    """Сохраняет рабочую память в файл.
    
    Args:
        memory: Словарь данных WorkingMemory.
        task_name: Название задачи (опционально).
    
    Returns:
        Путь к сохранённому файлу.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    task_part = task_name or "current"
    filename = f"task_{task_part}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(WORKING_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_working_memory(task_name: str) -> Optional[dict]:
    """Загружает рабочую память по названию задачи.
    
    Args:
        task_name: Название задачи.
    
    Returns:
        Словарь данных или None.
    """
    try:
        pattern = os.path.join(WORKING_DIR, f"task_{task_name}_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Не удалось загрузить рабочую память: %s", exc)
        return None


def list_working_memories() -> list:
    """Возвращает список всех сохранённых рабочих памятей.
    
    Returns:
        Список dict с информацией о файлах.
    """
    try:
        pattern = os.path.join(WORKING_DIR, "task_*.json")
        paths = glob.glob(pattern)
        result = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
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


def save_long_term(memory: dict, name: Optional[str] = None) -> str:
    """Сохраняет долговременную память в файл.
    
    Args:
        memory: Словарь данных LongTermMemory.
        name: Название профиля/памяти (по умолчанию "default").
    
    Returns:
        Путь к сохранённому файлу.
    """
    os.makedirs(LONG_TERM_DIR, exist_ok=True)
    profile_name = name or "default"
    filename = f"profile_{profile_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(LONG_TERM_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return path


def load_long_term(name: str = "default") -> Optional[dict]:
    """Загружает долговременную память по имени профиля.
    
    Args:
        name: Имя профиля.
    
    Returns:
        Словарь данных или None.
    """
    try:
        pattern = os.path.join(LONG_TERM_DIR, f"profile_{name}_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Не удалось загрузить долговременную память: %s", exc)
        return None


def list_long_term_memories() -> list:
    """Возвращает список всех сохранённых долговременных памятей.
    
    Returns:
        Список dict с информацией о файлах.
    """
    try:
        pattern = os.path.join(LONG_TERM_DIR, "profile_*.json")
        paths = glob.glob(pattern)
        result = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
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
# ЭКСПОРТ/ИМПОРТ ПОЛНОГО СОСТОЯНИЯ
# ===========================================================================


def export_memory_state(short_term: dict, working: dict, long_term: dict) -> str:
    """Экспортирует всё состояние памяти в один файл.
    
    Args:
        short_term: Данные ShortTermMemory.
        working: Данные WorkingMemory.
        long_term: Данные LongTermMemory.
    
    Returns:
        Путь к сохранённому файлу.
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    filename = f"state_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path = os.path.join(MEMORY_DIR, filename)
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
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return (
        state.get("short_term", {}),
        state.get("working", {}),
        state.get("long_term", {}),
    )


# ===========================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===========================================================================


def get_memory_stats() -> dict:
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
        (SHORT_TERM_DIR, "short_term"),
        (WORKING_DIR, "working"),
        (LONG_TERM_DIR, "long_term"),
    ]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    path = os.path.join(directory, filename)
                    stats[key]["files"] += 1
                    stats[key]["size_bytes"] += os.path.getsize(path)
    
    return stats
