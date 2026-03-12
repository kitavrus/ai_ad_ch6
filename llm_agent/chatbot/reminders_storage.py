"""Хранение и получение напоминаний из Scheduler API."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

SCHEDULER_BASE_URL = os.getenv("SCHEDULER_BASE_URL", "http://localhost:8881")
SCHEDULER_API_KEY = os.getenv("SCHEDULER_API_KEY", "secret-token")
REMINDERS_FILE = "reminders.json"

_HEADERS = {"X-API-Key": SCHEDULER_API_KEY}


def fetch_reminders(status: Optional[str] = None) -> Optional[List[Dict]]:
    """GET /reminders[?status=...] → список dict или None при ошибке."""
    if httpx is None:
        return None
    params = {}
    if status:
        params["status"] = status
    try:
        resp = httpx.get(
            f"{SCHEDULER_BASE_URL}/reminders",
            headers=_HEADERS,
            params=params,
            timeout=5.0,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def fetch_reminder(task_id: str) -> Optional[Dict]:
    """GET /reminders/{task_id} → dict или None."""
    if httpx is None:
        return None
    try:
        resp = httpx.get(
            f"{SCHEDULER_BASE_URL}/reminders/{task_id}",
            headers=_HEADERS,
            timeout=5.0,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def get_reminders_path(profile_name: str) -> str:
    """→ dialogues/{profile}/reminders.json"""
    return os.path.join("dialogues", profile_name, REMINDERS_FILE)


def load_reminders_file(profile_name: str) -> Dict:
    """Загружает файл напоминаний. Если файл отсутствует — возвращает пустую структуру."""
    path = get_reminders_path(profile_name)
    if not os.path.exists(path):
        return {"last_updated": None, "reminders": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "reminders" not in data:
            data["reminders"] = []
        if "last_updated" not in data:
            data["last_updated"] = None
        return data
    except (json.JSONDecodeError, OSError):
        return {"last_updated": None, "reminders": []}


def save_all_reminders(reminders: List[Dict], profile_name: str) -> None:
    """Перезаписывает весь список напоминаний, проставляет last_updated."""
    path = get_reminders_path(profile_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reminders": reminders,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_reminder_in_file(reminder: Dict, profile_name: str) -> None:
    """Обновляет/добавляет одно напоминание по полю 'id'."""
    data = load_reminders_file(profile_name)
    task_id = reminder.get("id")
    reminders = data["reminders"]
    for i, r in enumerate(reminders):
        if r.get("id") == task_id:
            reminders[i] = reminder
            break
    else:
        reminders.append(reminder)
    data["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = get_reminders_path(profile_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
