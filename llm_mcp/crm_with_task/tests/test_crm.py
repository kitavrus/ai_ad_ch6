import pytest
from llm_mcp.crm_with_task.crm_server import (
    create_task,
    get_tasks,
    update_status,
    delete_task,
)


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------

def test_create_task_defaults():
    task = create_task(title="Test task")
    assert task["id"]
    assert task["title"] == "Test task"
    assert task["status"] == "NEW"
    assert task["priority"] == "MEDIUM"
    assert task["description"] is None
    assert task["assignee"] is None
    assert task["created_at"]


def test_create_task_full():
    task = create_task(
        title="Full task",
        description="Some description",
        assignee="Alice",
        priority="HIGH",
    )
    assert task["description"] == "Some description"
    assert task["assignee"] == "Alice"
    assert task["priority"] == "HIGH"


def test_create_task_invalid_priority():
    result = create_task(title="Bad", priority="URGENT")
    assert "error" in result


# ---------------------------------------------------------------------------
# get_tasks
# ---------------------------------------------------------------------------

def test_get_tasks_returns_all_after_seed():
    # First call triggers seed (5 tasks)
    tasks = get_tasks()
    assert len(tasks) == 5


def test_get_tasks_filter_new():
    tasks = get_tasks(status="NEW")
    assert all(t["status"] == "NEW" for t in tasks)
    assert len(tasks) == 2


def test_get_tasks_filter_in_process():
    tasks = get_tasks(status="IN_PROCESS")
    assert all(t["status"] == "IN_PROCESS" for t in tasks)
    assert len(tasks) == 2


def test_get_tasks_filter_done():
    tasks = get_tasks(status="DONE")
    assert all(t["status"] == "DONE" for t in tasks)
    assert len(tasks) == 1


def test_get_tasks_invalid_status():
    result = get_tasks(status="INVALID")
    assert "error" in result[0]


def test_get_tasks_sorted_by_created_at_desc():
    create_task(title="First")
    create_task(title="Second")
    tasks = get_tasks(status="NEW")
    # After seed (2 NEW) + 2 new creates = 4 NEW tasks; newest first
    assert tasks[0]["created_at"] >= tasks[1]["created_at"]


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------

def test_update_status_ok():
    task = create_task(title="To update")
    updated = update_status(task["id"], "IN_PROCESS")
    assert updated["status"] == "IN_PROCESS"
    assert updated["id"] == task["id"]


def test_update_status_persists():
    task = create_task(title="Persist check")
    update_status(task["id"], "DONE")
    tasks = get_tasks(status="DONE")
    ids = [t["id"] for t in tasks]
    assert task["id"] in ids


def test_update_status_invalid_status():
    task = create_task(title="Bad status")
    result = update_status(task["id"], "UNKNOWN")
    assert "error" in result


def test_update_status_nonexistent_id():
    result = update_status("nonexistent-id", "DONE")
    assert "error" in result


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------

def test_delete_task_ok():
    task = create_task(title="To delete")
    result = delete_task(task["id"])
    assert result == {"deleted": task["id"], "ok": True}


def test_delete_task_removed_from_list():
    task = create_task(title="Will be gone")
    delete_task(task["id"])
    all_tasks = get_tasks()
    ids = [t["id"] for t in all_tasks]
    assert task["id"] not in ids


def test_delete_task_nonexistent_id():
    result = delete_task("no-such-id")
    assert "error" in result
