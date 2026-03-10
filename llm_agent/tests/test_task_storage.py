"""Тесты для chatbot/task_storage.py."""

import json
import os

from llm_agent.chatbot.models import StepStatus, TaskPhase, TaskPlan, TaskStep
from llm_agent.chatbot.task_storage import (
    delete_task_plan,
    get_task_dir,
    get_tasks_dir,
    list_task_plans,
    load_all_steps,
    load_task_plan,
    load_task_step,
    save_task_plan,
    save_task_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plan(**kwargs) -> TaskPlan:
    defaults = {
        "task_id": "testid001",
        "name": "Test task",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    defaults.update(kwargs)
    return TaskPlan(**defaults)


def _make_step(task_id: str = "testid001", index: int = 1, **kwargs) -> TaskStep:
    defaults = {
        "step_id": f"{task_id}_step_{index:03d}",
        "task_id": task_id,
        "index": index,
        "title": f"Step {index}",
        "created_at": "2024-01-01T00:00:00",
    }
    defaults.update(kwargs)
    return TaskStep(**defaults)


# ---------------------------------------------------------------------------
# get_tasks_dir / get_task_dir
# ---------------------------------------------------------------------------


class TestDirHelpers:
    def test_get_tasks_dir(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        d = get_tasks_dir("test")
        assert d.endswith(os.path.join("test", "tasks"))

    def test_get_task_dir(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        d = get_task_dir("myid", "test")
        assert d.endswith(os.path.join("test", "tasks", "myid"))


# ---------------------------------------------------------------------------
# save_task_plan / load_task_plan
# ---------------------------------------------------------------------------


class TestSaveLoadTaskPlan:
    def test_save_creates_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        path = save_task_plan(plan, "test")
        assert os.path.isfile(path)

    def test_save_content_valid_json(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        path = save_task_plan(plan, "test")
        with open(path) as f:
            data = json.load(f)
        assert data["task_id"] == "testid001"
        assert data["name"] == "Test task"

    def test_load_returns_plan(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(phase=TaskPhase.EXECUTION, total_steps=3)
        save_task_plan(plan, "test")
        loaded = load_task_plan("testid001", "test")
        assert loaded is not None
        assert loaded.task_id == "testid001"
        assert loaded.phase == TaskPhase.EXECUTION
        assert loaded.total_steps == 3

    def test_load_missing_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = load_task_plan("nonexistent", "test")
        assert result is None

    def test_save_creates_nested_dirs(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(task_id="deepid")
        save_task_plan(plan, "test")
        assert os.path.isdir(get_task_dir("deepid", "test"))

    def test_load_corrupt_json_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        task_dir = get_task_dir("badid", "test")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "plan.json"), "w") as f:
            f.write("not json{{{{")
        result = load_task_plan("badid", "test")
        assert result is None

    def test_roundtrip_with_step_ids(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(step_ids=["s1", "s2"], total_steps=2)
        save_task_plan(plan, "test")
        loaded = load_task_plan("testid001", "test")
        assert loaded.step_ids == ["s1", "s2"]

    def test_roundtrip_llm_raw_response(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(llm_raw_response="[{\"title\": \"Step 1\"}]")
        save_task_plan(plan, "test")
        loaded = load_task_plan("testid001", "test")
        assert loaded.llm_raw_response == "[{\"title\": \"Step 1\"}]"


# ---------------------------------------------------------------------------
# save_task_step / load_task_step
# ---------------------------------------------------------------------------


class TestSaveLoadTaskStep:
    def test_save_creates_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # Need plan dir to exist
        plan = _make_plan()
        save_task_plan(plan, "test")
        step = _make_step()
        path = save_task_step(step, "test")
        assert os.path.isfile(path)
        assert "step_001.json" in path

    def test_step_filename_padded(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        step = _make_step(index=5)
        path = save_task_step(step, "test")
        assert "step_005.json" in path

    def test_load_returns_step(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        step = _make_step(title="My step")
        save_task_step(step, "test")
        loaded = load_task_step("testid001", 1, "test")
        assert loaded is not None
        assert loaded.title == "My step"
        assert loaded.index == 1

    def test_load_missing_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = load_task_step("nonexistent", 1, "test")
        assert result is None

    def test_load_wrong_index_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        step = _make_step(index=1)
        save_task_step(step, "test")
        result = load_task_step("testid001", 99, "test")
        assert result is None

    def test_step_status_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        step = _make_step(status=StepStatus.DONE, completed_at="2024-01-01T10:00:00")
        save_task_step(step, "test")
        loaded = load_task_step("testid001", 1, "test")
        assert loaded.status == StepStatus.DONE
        assert loaded.completed_at == "2024-01-01T10:00:00"


# ---------------------------------------------------------------------------
# load_all_steps
# ---------------------------------------------------------------------------


class TestLoadAllSteps:
    def test_empty_task_dir(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        steps = load_all_steps("nonexistent", "test")
        assert steps == []

    def test_returns_sorted_steps(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        for i in [3, 1, 2]:
            save_task_step(_make_step(index=i, title=f"Step {i}"), "test")
        steps = load_all_steps("testid001", "test")
        assert [s.index for s in steps] == [1, 2, 3]

    def test_returns_all_steps(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        for i in range(1, 6):
            save_task_step(_make_step(index=i), "test")
        steps = load_all_steps("testid001", "test")
        assert len(steps) == 5

    def test_corrupt_step_file_skipped(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        save_task_step(_make_step(index=1), "test")
        # Corrupt step 2
        task_dir = get_task_dir("testid001", "test")
        with open(os.path.join(task_dir, "step_002.json"), "w") as f:
            f.write("invalid")
        steps = load_all_steps("testid001", "test")
        assert len(steps) == 1
        assert steps[0].index == 1


# ---------------------------------------------------------------------------
# list_task_plans
# ---------------------------------------------------------------------------


class TestListTaskPlans:
    def test_empty_profile(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = list_task_plans("test")
        assert result == []

    def test_lists_plans(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan1 = _make_plan(task_id="id1", name="Task One")
        plan2 = _make_plan(task_id="id2", name="Task Two")
        save_task_plan(plan1, "test")
        save_task_plan(plan2, "test")
        result = list_task_plans("test")
        names = [p["name"] for p in result]
        assert "Task One" in names
        assert "Task Two" in names

    def test_summary_fields(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(task_id="id1", total_steps=5, current_step_index=2)
        save_task_plan(plan, "test")
        result = list_task_plans("test")
        assert len(result) == 1
        p = result[0]
        assert p["task_id"] == "id1"
        assert p["total_steps"] == 5
        assert p["current_step_index"] == 2
        assert "phase" in p
        assert "created_at" in p
        assert "updated_at" in p

    def test_no_steps_loaded(self, monkeypatch, tmp_path):
        """list_task_plans must not load steps — just summary."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(task_id="id1")
        save_task_plan(plan, "test")
        # No step files exist — should still work
        result = list_task_plans("test")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# delete_task_plan
# ---------------------------------------------------------------------------


class TestDeleteTaskPlan:
    def test_delete_existing(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        assert delete_task_plan("testid001", "test") is True
        assert not os.path.exists(get_task_dir("testid001", "test"))

    def test_delete_nonexistent_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = delete_task_plan("nonexistent", "test")
        assert result is False

    def test_delete_removes_steps_too(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        plan = _make_plan()
        save_task_plan(plan, "test")
        save_task_step(_make_step(), "test")
        delete_task_plan("testid001", "test")
        assert load_task_step("testid001", 1, "test") is None
