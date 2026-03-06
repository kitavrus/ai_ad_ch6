"""Покрытие ранее непокрытых веток chatbot/task_storage.py."""

import json
import os
import pytest
from unittest.mock import patch

from chatbot.task_storage import (
    delete_task_plan,
    get_task_result_dir,
    list_task_plans,
    list_task_result_files,
    load_task_step,
    save_task_plan,
    save_task_result_file,
    save_task_step,
)
from chatbot.models import TaskPlan, TaskPhase, TaskStep, StepStatus


def _make_plan(task_id="task_abc"):
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    return TaskPlan(
        task_id=task_id,
        name="Test plan",
        description="desc",
        phase=TaskPhase.PLANNING,
        total_steps=2,
        current_step_index=0,
        created_at=now,
        updated_at=now,
    )


def _make_step(task_id="task_abc", index=1):
    from datetime import datetime
    return TaskStep(
        step_id=f"{task_id}_step_{index:03d}",
        task_id=task_id,
        index=index,
        title=f"Step {index}",
        description="do it",
        status=StepStatus.PENDING,
        created_at=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# load_task_step — exception path
# ---------------------------------------------------------------------------

class TestLoadTaskStepException:
    def test_returns_none_on_corrupt_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from chatbot.task_storage import get_task_dir
        d = get_task_dir("task_bad", "test")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "step_001.json")
        with open(path, "w") as f:
            f.write("{{broken json")
        result = load_task_step("task_bad", 1, "test")
        assert result is None


# ---------------------------------------------------------------------------
# list_task_plans — edge cases
# ---------------------------------------------------------------------------

class TestListTaskPlansEdgeCases:
    def test_skips_entry_without_plan_json(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from chatbot.task_storage import get_tasks_dir
        tasks_dir = get_tasks_dir("test")
        os.makedirs(os.path.join(tasks_dir, "orphan_dir"), exist_ok=True)
        # No plan.json in orphan_dir
        result = list_task_plans("test")
        assert result == []

    def test_skips_corrupt_plan_json(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from chatbot.task_storage import get_tasks_dir
        tasks_dir = get_tasks_dir("test")
        task_dir = os.path.join(tasks_dir, "corrupt_task")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "plan.json"), "w") as f:
            f.write("{{broken")
        result = list_task_plans("test")
        assert result == []

    def test_happy_path_returns_plan_info(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan("task_xyz")
        save_task_plan(plan, "test")
        result = list_task_plans("test")
        assert len(result) == 1
        assert result[0]["task_id"] == "task_xyz"
        assert result[0]["name"] == "Test plan"


# ---------------------------------------------------------------------------
# delete_task_plan — exception path
# ---------------------------------------------------------------------------

class TestDeleteTaskPlanException:
    def test_returns_false_on_rmtree_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan("task_del")
        save_task_plan(plan, "test")

        import shutil
        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            result = delete_task_plan("task_del", "test")
        assert result is False

    def test_returns_false_when_not_found(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = delete_task_plan("nonexistent_task", "test")
        assert result is False

    def test_returns_true_on_success(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan("task_ok")
        save_task_plan(plan, "test")
        result = delete_task_plan("task_ok", "test")
        assert result is True


# ---------------------------------------------------------------------------
# get_task_result_dir / save_task_result_file / list_task_result_files
# ---------------------------------------------------------------------------

class TestTaskResultFiles:
    def test_get_task_result_dir_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from chatbot.task_storage import get_task_dir
        expected = os.path.join(get_task_dir("tid", "test"), "result")
        assert get_task_result_dir("tid", "test") == expected

    def test_save_creates_file_and_dirs(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        saved = save_task_result_file("tid", "src/main.py", "print('hi')", "test")
        assert os.path.isfile(saved)
        with open(saved, encoding="utf-8") as f:
            assert f.read() == "print('hi')"

    def test_save_nested_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        saved = save_task_result_file("tid", "a/b/c.txt", "content", "test")
        assert os.path.isfile(saved)
        assert "a" + os.sep + "b" in saved

    def test_list_empty_when_no_result_dir(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = list_task_result_files("no_such_task", "test")
        assert result == []

    def test_list_returns_sorted_relative_paths(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        save_task_result_file("tid", "z.txt", "z", "test")
        save_task_result_file("tid", "a.txt", "a", "test")
        save_task_result_file("tid", "sub/b.txt", "b", "test")
        files = list_task_result_files("tid", "test")
        assert files == sorted(files)
        assert any("z.txt" in f for f in files)
        assert any("a.txt" in f for f in files)
