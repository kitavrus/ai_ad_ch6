"""Тесты Project-хранилища и команд /project."""

import pytest
from datetime import datetime
import uuid

from chatbot.models import Project, SessionState, TaskPhase, TaskPlan


def _now():
    return datetime.utcnow().isoformat()


def _make_state(profile="default"):
    return SessionState(
        model="gpt-4",
        base_url="https://api.example.com",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        profile_name=profile,
    )


def _make_project(profile="default", name="Test Project") -> Project:
    return Project(
        project_id=uuid.uuid4().hex,
        name=name,
        profile_name=profile,
        created_at=_now(),
        updated_at=_now(),
    )


# ===========================================================================
# Project model
# ===========================================================================

class TestProjectModel:
    def test_defaults(self):
        p = _make_project()
        assert p.description == ""
        assert p.plan_ids == []

    def test_plan_ids(self):
        p = _make_project()
        p.plan_ids.append("tid-001")
        assert "tid-001" in p.plan_ids

    def test_serialization(self):
        p = _make_project()
        data = p.model_dump()
        assert "project_id" in data
        assert data["name"] == "Test Project"


# ===========================================================================
# project_storage: save / load / list / delete
# ===========================================================================

class TestProjectStorage:
    def test_save_and_load(self, monkeypatch, tmp_path):
        from chatbot.project_storage import save_project, load_project
        monkeypatch.chdir(tmp_path)
        p = _make_project(profile="test")
        save_project(p, "test")
        loaded = load_project(p.project_id, "test")
        assert loaded is not None
        assert loaded.name == p.name

    def test_load_nonexistent_returns_none(self, monkeypatch, tmp_path):
        from chatbot.project_storage import load_project
        monkeypatch.chdir(tmp_path)
        assert load_project("no-such-id", "test") is None

    def test_list_projects(self, monkeypatch, tmp_path):
        from chatbot.project_storage import save_project, list_projects
        monkeypatch.chdir(tmp_path)
        p1 = _make_project("test", "Alpha")
        p2 = _make_project("test", "Beta")
        save_project(p1, "test")
        save_project(p2, "test")
        projects = list_projects("test")
        names = [p["name"] for p in projects]
        assert "Alpha" in names
        assert "Beta" in names

    def test_list_empty(self, monkeypatch, tmp_path):
        from chatbot.project_storage import list_projects
        monkeypatch.chdir(tmp_path)
        assert list_projects("test") == []

    def test_delete_project(self, monkeypatch, tmp_path):
        from chatbot.project_storage import save_project, load_project, delete_project
        monkeypatch.chdir(tmp_path)
        p = _make_project("test")
        save_project(p, "test")
        assert load_project(p.project_id, "test") is not None
        result = delete_project(p.project_id, "test")
        assert result is True
        assert load_project(p.project_id, "test") is None

    def test_delete_nonexistent_returns_false(self, monkeypatch, tmp_path):
        from chatbot.project_storage import delete_project
        monkeypatch.chdir(tmp_path)
        assert delete_project("no-such-id", "test") is False

    def test_load_corrupt_json_returns_none(self, monkeypatch, tmp_path):
        from chatbot.project_storage import load_project, _project_path
        monkeypatch.chdir(tmp_path)
        pid = uuid.uuid4().hex
        path = _project_path(pid, "test")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{invalid json", encoding="utf-8")
        assert load_project(pid, "test") is None


# ===========================================================================
# /project commands via _handle_project_command
# ===========================================================================

class TestHandleProjectCommand:
    def test_new_creates_project(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("new", "My App", state)
        out = capsys.readouterr().out
        assert "My App" in out
        assert state.active_project_id is not None

    def test_new_without_name_prints_error(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("new", "", state)
        out = capsys.readouterr().out
        assert "требует" in out
        assert state.active_project_id is None

    def test_list_empty(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("list", "", state)
        out = capsys.readouterr().out
        assert "нет" in out.lower() or "Проектов нет" in out

    def test_list_shows_projects(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        p = _make_project()
        save_project(p, state.profile_name)
        _handle_project_command("list", "", state)
        out = capsys.readouterr().out
        assert p.name in out

    def test_switch_by_name(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        p = _make_project(name="Backend")
        save_project(p, state.profile_name)
        _handle_project_command("switch", "backend", state)
        assert state.active_project_id == p.project_id

    def test_switch_not_found(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("switch", "nonexistent", state)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_switch_without_arg(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("switch", "", state)
        out = capsys.readouterr().out
        assert "требует" in out

    def test_show_no_active_project(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("show", "", state)
        out = capsys.readouterr().out
        assert "нет активного" in out.lower()

    def test_show_active_project(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        p = _make_project(name="Frontend")
        save_project(p, state.profile_name)
        state.active_project_id = p.project_id
        _handle_project_command("show", "", state)
        out = capsys.readouterr().out
        assert "Frontend" in out

    def test_add_plan_links_to_project(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project, load_project
        from chatbot.task_storage import save_task_plan, load_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id

        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="API Design",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)

        _handle_project_command("add-plan", plan.task_id, state)
        out = capsys.readouterr().out
        assert "добавлен" in out

        updated_proj = load_project(proj.project_id, state.profile_name)
        assert plan.task_id in updated_proj.plan_ids

        updated_plan = load_task_plan(plan.task_id, state.profile_name)
        assert updated_plan.project_id == proj.project_id

    def test_unknown_action(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("bogus", "", state)
        out = capsys.readouterr().out
        assert "неизвестная" in out.lower()


# ===========================================================================
# SessionState: active_task_ids и active_project_id
# ===========================================================================

class TestSessionStateParallelTasks:
    def test_active_task_ids_default_empty(self):
        state = _make_state()
        assert state.active_task_ids == []

    def test_active_project_id_default_none(self):
        state = _make_state()
        assert state.active_project_id is None

    def test_task_creation_adds_to_active_ids(self, monkeypatch, tmp_path):
        from chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        steps = [{"title": "Step 1", "description": "Do it"}]
        _create_task_plan("Test task", state, client=None, steps=steps)
        assert state.active_task_id is not None
        assert state.active_task_id in state.active_task_ids

    def test_multiple_active_tasks(self, monkeypatch, tmp_path):
        from chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        steps = [{"title": "Step 1", "description": "Do it"}]
        _create_task_plan("Frontend task", state, client=None, steps=steps)
        first_id = state.active_task_id
        _create_task_plan("Backend task", state, client=None, steps=steps)
        assert len(state.active_task_ids) == 2
        assert first_id in state.active_task_ids

    def test_session_persists_active_task_ids(self, monkeypatch, tmp_path):
        from chatbot.main import _build_session_payload, _apply_session_data
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_task_ids = ["id1", "id2"]
        state.active_project_id = "proj-abc"

        payload = _build_session_payload(state, user_input="", assistant_text="")
        data = payload.model_dump()

        state2 = _make_state()
        _apply_session_data(data, state2)
        assert state2.active_task_ids == ["id1", "id2"]
        assert state2.active_project_id == "proj-abc"


# ===========================================================================
# --plan flag parsing
# ===========================================================================

class TestPlanFlagParsing:
    def test_parse_plan_flag_present(self):
        from chatbot.main import _parse_plan_flag
        cleaned, name = _parse_plan_flag("done result --plan frontend")
        assert name == "frontend"
        assert "--plan" not in cleaned

    def test_parse_plan_flag_absent(self):
        from chatbot.main import _parse_plan_flag
        cleaned, name = _parse_plan_flag("done result")
        assert name is None
        assert cleaned == "done result"

    def test_parse_plan_flag_only(self):
        from chatbot.main import _parse_plan_flag
        cleaned, name = _parse_plan_flag("--plan backend")
        assert name == "backend"

    def test_builder_with_plan_flag(self, monkeypatch, tmp_path, capsys):
        """_handle_agent_command('builder', '--plan backend', ...) должен запустить builder
        на нужном плане и восстановить active_task_id после завершения."""
        from unittest.mock import patch
        from chatbot.main import _handle_agent_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        frontend_plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="frontend",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=1,
            current_step_index=0,
            created_at=_now(),
            updated_at=_now(),
        )
        backend_plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="backend",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=1,
            current_step_index=0,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(frontend_plan, state.profile_name)
        save_task_plan(backend_plan, state.profile_name)
        state.active_task_id = frontend_plan.task_id
        state.active_task_ids = [frontend_plan.task_id, backend_plan.task_id]

        captured_task_ids = []

        def mock_builder(s, client):
            captured_task_ids.append(s.active_task_id)

        with patch("chatbot.main._run_plan_builder", side_effect=mock_builder):
            _handle_agent_command("builder", "--plan backend", state, client=None)

        assert captured_task_ids == [backend_plan.task_id], "builder должен использовать backend"
        assert state.active_task_id == frontend_plan.task_id, "active_task_id должен восстановиться"

    def test_builder_with_plan_flag_not_found(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_agent_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_agent_command("builder", "--plan nonexistent", state, client=None)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_step_with_plan_flag(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan, save_task_step
        from chatbot.models import StepStatus, TaskStep
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="frontend",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=1,
            current_step_index=0,
            created_at=_now(),
            updated_at=_now(),
        )
        step = TaskStep(
            step_id=f"{plan.task_id}_step_001",
            task_id=plan.task_id,
            index=1,
            title="Step 1",
            status=StepStatus.PENDING,
            created_at=_now(),
        )
        plan.step_ids = [step.step_id]
        save_task_plan(plan, state.profile_name)
        save_task_step(step, state.profile_name)
        state.active_task_id = plan.task_id

        _handle_task_command("step", f"done Готово --plan frontend", state, client=None)
        out = capsys.readouterr().out
        assert "завершён" in out or "VALIDATION" in out or "Шаг 1" in out
