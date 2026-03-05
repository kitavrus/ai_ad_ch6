"""Слой сохранения и загрузки данных планирования задач."""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, List, Optional

from chatbot.config import DEFAULT_PROFILE, DIALOGUES_DIR

if TYPE_CHECKING:
    from chatbot.models import TaskPlan, TaskStep

logger = logging.getLogger(__name__)


# ===========================================================================
# ПУТИ К ФАЙЛАМ ЗАДАЧ (профиль-специфичные)
# ===========================================================================


def get_tasks_dir(profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(DIALOGUES_DIR, profile_name, "tasks")


def get_task_dir(task_id: str, profile_name: str = DEFAULT_PROFILE) -> str:
    return os.path.join(get_tasks_dir(profile_name), task_id)


# ===========================================================================
# ПЛАН ЗАДАЧИ
# ===========================================================================


def save_task_plan(plan: TaskPlan, profile_name: str = DEFAULT_PROFILE) -> str:
    """Сохраняет план задачи в файл plan.json в директории задачи.

    Returns:
        Путь к сохранённому файлу.
    """
    task_dir = get_task_dir(plan.task_id, profile_name)
    os.makedirs(task_dir, exist_ok=True)
    path = os.path.join(task_dir, "plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info("Task plan saved: %s", path)
    return path


def load_task_plan(task_id: str, profile_name: str = DEFAULT_PROFILE) -> Optional[TaskPlan]:
    """Загружает план задачи из файла plan.json.

    Returns:
        Объект TaskPlan или None.
    """
    from chatbot.models import TaskPlan

    path = os.path.join(get_task_dir(task_id, profile_name), "plan.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return TaskPlan(**data)
    except Exception as exc:
        logger.warning("Не удалось загрузить план задачи '%s': %s", task_id, exc)
        return None


# ===========================================================================
# ШАГИ ЗАДАЧИ
# ===========================================================================


def save_task_step(step: TaskStep, profile_name: str = DEFAULT_PROFILE) -> str:
    """Сохраняет шаг задачи в файл step_NNN.json.

    Returns:
        Путь к сохранённому файлу.
    """
    task_dir = get_task_dir(step.task_id, profile_name)
    os.makedirs(task_dir, exist_ok=True)
    filename = f"step_{step.index:03d}.json"
    path = os.path.join(task_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(step.model_dump(), f, ensure_ascii=False, indent=2)
    return path


def load_task_step(
    task_id: str,
    step_index: int,
    profile_name: str = DEFAULT_PROFILE,
) -> Optional[TaskStep]:
    """Загружает шаг задачи по индексу (1-based).

    Returns:
        Объект TaskStep или None.
    """
    from chatbot.models import TaskStep

    path = os.path.join(get_task_dir(task_id, profile_name), f"step_{step_index:03d}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return TaskStep(**data)
    except Exception as exc:
        logger.warning("Не удалось загрузить шаг %d задачи '%s': %s", step_index, task_id, exc)
        return None


def load_all_steps(task_id: str, profile_name: str = DEFAULT_PROFILE) -> List[TaskStep]:
    """Загружает все шаги задачи, отсортированные по индексу.

    Returns:
        Список объектов TaskStep.
    """
    from chatbot.models import TaskStep

    task_dir = get_task_dir(task_id, profile_name)
    pattern = os.path.join(task_dir, "step_*.json")
    paths = sorted(glob.glob(pattern))
    steps: List[TaskStep] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            steps.append(TaskStep(**data))
        except Exception as exc:
            logger.warning("Не удалось загрузить шаг из '%s': %s", p, exc)
    steps.sort(key=lambda s: s.index)
    return steps


# ===========================================================================
# СПИСОК ЗАДАЧ
# ===========================================================================


def list_task_plans(profile_name: str = DEFAULT_PROFILE) -> List[dict]:
    """Возвращает краткие сводки всех планов задач профиля (без загрузки шагов).

    Returns:
        Список dict с полями: task_id, name, phase, total_steps,
        current_step_index, created_at, updated_at.
    """
    tasks_dir = get_tasks_dir(profile_name)
    if not os.path.exists(tasks_dir):
        return []
    result = []
    for entry in sorted(os.listdir(tasks_dir)):
        plan_path = os.path.join(tasks_dir, entry, "plan.json")
        if not os.path.isfile(plan_path):
            continue
        try:
            with open(plan_path, encoding="utf-8") as f:
                data = json.load(f)
            result.append({
                "task_id": data.get("task_id", entry),
                "name": data.get("name", ""),
                "phase": data.get("phase", ""),
                "total_steps": data.get("total_steps", 0),
                "current_step_index": data.get("current_step_index", 0),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception as exc:
            logger.warning("Не удалось прочитать план из '%s': %s", plan_path, exc)
    return result


# ===========================================================================
# УДАЛЕНИЕ ЗАДАЧИ
# ===========================================================================


def delete_task_plan(task_id: str, profile_name: str = DEFAULT_PROFILE) -> bool:
    """Удаляет директорию задачи со всем содержимым.

    Returns:
        True если удалено успешно, False если директория не найдена.
    """
    task_dir = get_task_dir(task_id, profile_name)
    if not os.path.exists(task_dir):
        logger.warning("Директория задачи не найдена: %s", task_dir)
        return False
    try:
        shutil.rmtree(task_dir)
        logger.info("Задача удалена: %s", task_dir)
        return True
    except Exception as exc:
        logger.warning("Не удалось удалить задачу '%s': %s", task_id, exc)
        return False
