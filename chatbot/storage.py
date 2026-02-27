"""Слой сохранения и загрузки данных сессий."""

import glob
import json
import logging
import os
from typing import Optional, Tuple

from chatbot.config import DIALOGUES_DIR, METRICS_DIR
from chatbot.models import DialogueSession, RequestMetric

logger = logging.getLogger(__name__)


def save_session(session: DialogueSession, path: str) -> str:
    """Сохраняет всю сессию чата по указанному пути (перезапись).

    Args:
        session: Объект сессии диалога.
        path: Путь к файлу для записи.

    Returns:
        Путь к сохранённому файлу.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            session.model_dump(exclude_none=False),
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return path


def log_request_metric(metric: RequestMetric, session_id: str, idx: int) -> str:
    """Сохраняет метаданные одного запроса в отдельный лог-файл.

    Args:
        metric: Объект метрики запроса.
        session_id: Идентификатор сессии (используется в имени файла).
        idx: Порядковый номер запроса.

    Returns:
        Путь к записанному лог-файлу.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    filename = f"{METRICS_DIR}/session_{session_id}_req_{idx:04d}.log"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metric.model_dump(), f, ensure_ascii=False, indent=2)
    return filename


def load_last_session() -> Optional[Tuple[str, dict]]:
    """Загружает последнюю сохранённую сессию из dialogues/session_*.json.

    Returns:
        Кортеж (путь к файлу, словарь данных) или None при ошибке/отсутствии файлов.
    """
    try:
        pattern = os.path.join(DIALOGUES_DIR, "session_*.json")
        paths = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not paths:
            return None
        last_path = paths[-1]
        with open(last_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return last_path, data
    except Exception as exc:
        logger.warning("Не удалось загрузить последнюю сессию: %s", exc)
        return None
