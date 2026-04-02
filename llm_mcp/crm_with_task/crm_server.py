import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

TASKS_FILE = Path(os.getenv("CRM_TASKS_FILE", Path(__file__).parent / "tasks.json"))
VALID_STATUSES = {"NEW", "IN_PROCESS", "DONE"}
VALID_PRIORITIES = {"LOW", "MEDIUM", "HIGH"}

mcp = FastMCP("crm-tasks")


def _seed_data() -> dict:
    now = datetime.now(timezone.utc)

    def _ts(seconds_ago: int) -> str:
        from datetime import timedelta
        return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()

    tasks = {}
    seed = [
        {"title": "Связаться с клиентом Иванов", "description": "Перезвонить по вопросу договора", "assignee": "Анна", "status": "NEW", "priority": "HIGH", "offset": 3600},
        {"title": "Подготовить коммерческое предложение", "description": "КП для компании ООО Ромашка", "assignee": "Петр", "status": "NEW", "priority": "MEDIUM", "offset": 7200},
        {"title": "Провести демо продукта", "description": "Онлайн-демонстрация для клиента Сидоров", "assignee": "Анна", "status": "IN_PROCESS", "priority": "HIGH", "offset": 10800},
        {"title": "Обновить CRM-карточки клиентов", "description": "Актуализировать контакты за Q1", "assignee": "Мария", "status": "IN_PROCESS", "priority": "LOW", "offset": 14400},
        {"title": "Закрыть сделку с Петров-Водкин", "description": "Подписать договор и выставить счёт", "assignee": "Петр", "status": "DONE", "priority": "HIGH", "offset": 86400},
    ]
    for item in seed:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "id": task_id,
            "title": item["title"],
            "description": item["description"],
            "assignee": item["assignee"],
            "status": item["status"],
            "priority": item["priority"],
            "created_at": _ts(item["offset"]),
        }
    return tasks


def _load() -> dict:
    if not TASKS_FILE.exists():
        data = _seed_data()
        _save(data)
        return data
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@mcp.tool()
def create_task(
    title: str,
    description: Optional[str] = None,
    assignee: Optional[str] = None,
    priority: str = "MEDIUM",
) -> dict:
    """Create a new CRM task with status NEW.

    Args:
        title: Task title (required)
        description: Optional task description
        assignee: Optional assignee name or email
        priority: Task priority — LOW, MEDIUM, or HIGH (default: MEDIUM)

    Returns:
        The created task object
    """
    if priority not in VALID_PRIORITIES:
        return {"error": f"Invalid priority '{priority}'. Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"}

    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "title": title,
        "description": description,
        "assignee": assignee,
        "status": "NEW",
        "priority": priority,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data = _load()
    data[task_id] = task
    _save(data)
    return task


@mcp.tool()
def get_tasks(status: Optional[str] = None) -> list:
    """Return CRM tasks, optionally filtered by status.

    Args:
        status: Filter by status — NEW, IN_PROCESS, or DONE. If omitted, returns all tasks.

    Returns:
        List of task objects sorted by created_at descending
    """
    if status is not None and status not in VALID_STATUSES:
        return [{"error": f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"}]

    data = _load()
    tasks = list(data.values())
    if status is not None:
        tasks = [t for t in tasks if t["status"] == status]
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return tasks


@mcp.tool()
def update_status(task_id: str, status: str) -> dict:
    """Update the status of an existing task.

    Args:
        task_id: The UUID of the task to update
        status: New status — NEW, IN_PROCESS, or DONE

    Returns:
        The updated task object
    """
    if status not in VALID_STATUSES:
        return {"error": f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"}

    data = _load()
    if task_id not in data:
        return {"error": f"Task '{task_id}' not found"}

    data[task_id]["status"] = status
    _save(data)
    return data[task_id]


@mcp.tool()
def delete_task(task_id: str) -> dict:
    """Delete a task by its ID.

    Args:
        task_id: The UUID of the task to delete

    Returns:
        Confirmation object with deleted task id
    """
    data = _load()
    if task_id not in data:
        return {"error": f"Task '{task_id}' not found"}

    del data[task_id]
    _save(data)
    return {"deleted": task_id, "ok": True}


if __name__ == "__main__":
    mcp.run()
