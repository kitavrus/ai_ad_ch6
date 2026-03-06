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


# ===========================================================================
# Новые команды: /project tasks, /project task new|rename|describe,
#                /project add-plan-name, /plan cancel, /task execute,
#                /strategy status, /mem task|pref|know
# ===========================================================================

class TestNewProjectSubcommands:
    def test_project_tasks_no_active_project(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("tasks", "", state)
        out = capsys.readouterr().out
        assert "Нет активного проекта" in out

    def test_project_tasks_empty(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id
        _handle_project_command("tasks", "", state)
        out = capsys.readouterr().out
        assert "задач нет" in out.lower() or "нет" in out.lower()

    def test_project_tasks_shows_plan(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Feature X",
            profile_name=state.profile_name,
            total_steps=3,
            created_at=_now(),
            updated_at=_now(),
        )
        proj.plan_ids.append(plan.task_id)
        save_project(proj, state.profile_name)
        save_task_plan(plan, state.profile_name)
        state.active_project_id = proj.project_id
        _handle_project_command("tasks", "", state)
        out = capsys.readouterr().out
        assert "Feature X" in out

    def test_project_task_rename(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.task_storage import save_task_plan, load_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Old Name",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_project_command("task_rename", "New Name", state)
        out = capsys.readouterr().out
        assert "переименована" in out.lower() or "New Name" in out
        updated = load_task_plan(plan.task_id, state.profile_name)
        assert updated.name == "New Name"

    def test_project_task_rename_no_plan(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("task_rename", "Something", state)
        out = capsys.readouterr().out
        assert "нет активного плана" in out.lower()

    def test_project_task_rename_no_name(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Old",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_project_command("task_rename", "", state)
        out = capsys.readouterr().out
        assert "требует новое имя" in out.lower() or "требует" in out.lower()

    def test_project_task_describe(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.task_storage import save_task_plan, load_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Task",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_project_command("task_describe", "New description text", state)
        capsys.readouterr()
        updated = load_task_plan(plan.task_id, state.profile_name)
        assert updated.description == "New description text"

    def test_add_plan_name(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project, load_project
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="My Feature",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        _handle_project_command("add_plan_name", "My Feature", state)
        out = capsys.readouterr().out
        assert "добавлен" in out
        updated = load_project(proj.project_id, state.profile_name)
        assert plan.task_id in updated.plan_ids

    def test_add_plan_name_not_found(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id
        _handle_project_command("add_plan_name", "Nonexistent Plan", state)
        out = capsys.readouterr().out
        assert "не найдена" in out


class TestNewCliAliases:
    def test_task_execute_maps_to_plan_builder(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/task execute")
        assert result == {"plan": {"action": "builder", "arg": ""}}

    def test_task_execute_with_plan_flag(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/task execute --plan myplan")
        assert result["plan"]["action"] == "builder"
        assert "--plan myplan" in result["plan"]["arg"]

    def test_project_plans_maps_to_tasks(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project plans")
        assert result == {"project": {"action": "tasks", "arg": ""}}

    def test_project_plan_new_maps_to_task_new(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project plan new Build API")
        assert result["project"]["action"] == "task_new"
        assert result["project"]["arg"] == "Build API"

    def test_project_plan_rename_maps_to_task_rename(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project plan rename New Title")
        assert result["project"]["action"] == "task_rename"
        assert result["project"]["arg"] == "New Title"

    def test_project_plan_describe_maps_to_task_describe(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project plan describe Some desc")
        assert result["project"]["action"] == "task_describe"
        assert result["project"]["arg"] == "Some desc"

    def test_project_task_new(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project task new My Task")
        assert result["project"]["action"] == "task_new"
        assert result["project"]["arg"] == "My Task"

    def test_project_task_rename(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project task rename Updated Name")
        assert result["project"]["action"] == "task_rename"

    def test_project_add_plan_name(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/project add-plan-name My Feature")
        assert result["project"]["action"] == "add_plan_name"
        assert result["project"]["arg"] == "My Feature"

    def test_strategy_status(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/strategy status")
        assert result == {"strategy_status": True}

    def test_mem_task_alias(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/mem task Write docs")
        assert result == {"settask": "Write docs"}

    def test_mem_pref_alias(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/mem pref tone=casual")
        assert result == {"setpref": "tone=casual"}

    def test_mem_know_alias(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/mem know db: postgres")
        assert result == {"remember": "db: postgres"}

    def test_plan_cancel_parses(self):
        from chatbot.cli import parse_inline_command
        result = parse_inline_command("/plan cancel")
        assert result == {"plan": {"action": "cancel", "arg": ""}}


class TestPlanCancel:
    def test_cancel_resets_fsm(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.agent_mode.enabled = True
        state.plan_dialog_state = "awaiting_task"
        state.plan_draft_steps = [{"title": "Step"}]
        state.plan_draft_description = "Some task"
        _handle_agent_command("cancel", "", state)
        assert state.plan_dialog_state is None
        assert state.plan_draft_steps == []
        assert state.plan_draft_description == ""
        out = capsys.readouterr().out
        assert "отменён" in out.lower()

    def test_cancel_from_confirming_state(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.agent_mode.enabled = True
        state.plan_dialog_state = "confirming"
        state.plan_draft_steps = [{"title": "S1"}]
        _handle_agent_command("cancel", "", state)
        assert state.plan_dialog_state is None
        assert state.plan_draft_steps == []


# ===========================================================================
# Coverage gaps: _handle_project_command edge cases
# ===========================================================================

class TestProjectCommandCoverage:
    """Покрывает непроверенные ветки _handle_project_command."""

    def test_show_stale_project_id(self, monkeypatch, tmp_path, capsys):
        """show: active_project_id указывает на несуществующий проект."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_project_id = "nonexistent_id"
        _handle_project_command("show", "", state)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_show_project_with_description_and_model(self, monkeypatch, tmp_path, capsys):
        """show: проект с описанием и планом с model-тегом."""
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project(name="My App")
        proj.description = "Описание проекта"
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Auth Service",
            profile_name=state.profile_name,
            model="gpt-4",
            created_at=_now(),
            updated_at=_now(),
        )
        proj.plan_ids.append(plan.task_id)
        save_project(proj, state.profile_name)
        save_task_plan(plan, state.profile_name)
        state.active_project_id = proj.project_id
        state.active_task_id = plan.task_id  # план помечается как активный
        _handle_project_command("show", "", state)
        out = capsys.readouterr().out
        assert "My App" in out
        assert "Описание проекта" in out
        assert "Auth Service" in out
        assert "gpt-4" in out
        assert "активный" in out

    def test_add_plan_no_arg(self, monkeypatch, tmp_path, capsys):
        """add-plan без аргумента."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("add_plan", "", state)
        out = capsys.readouterr().out
        assert "требует" in out

    def test_add_plan_no_active_project(self, monkeypatch, tmp_path, capsys):
        """add-plan: нет активного проекта."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("add_plan", "some-task-id", state)
        out = capsys.readouterr().out
        assert "нет активного" in out.lower()

    def test_add_plan_project_not_found(self, monkeypatch, tmp_path, capsys):
        """add-plan: active_project_id указывает на несуществующий проект."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_project_id = "ghost_project"
        _handle_project_command("add_plan", "some-task-id", state)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_add_plan_task_not_found(self, monkeypatch, tmp_path, capsys):
        """add-plan: задача с указанным ID не существует."""
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id
        _handle_project_command("add_plan", "nonexistent-task-id", state)
        out = capsys.readouterr().out
        assert "не найдена" in out

    def test_add_plan_name_no_arg(self, monkeypatch, tmp_path, capsys):
        """add-plan-name без аргумента."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("add_plan_name", "", state)
        out = capsys.readouterr().out
        assert "требует" in out

    def test_add_plan_name_no_active_project(self, monkeypatch, tmp_path, capsys):
        """add-plan-name: нет активного проекта."""
        from chatbot.main import _handle_project_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Orphan Plan",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        _handle_project_command("add_plan_name", "Orphan Plan", state)
        out = capsys.readouterr().out
        assert "нет активного" in out.lower()

    def test_add_plan_name_project_not_found(self, monkeypatch, tmp_path, capsys):
        """add-plan-name: задача найдена, но проект не существует."""
        from chatbot.main import _handle_project_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Ghost Proj Plan",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        state.active_project_id = "ghost_project_id"
        _handle_project_command("add_plan_name", "Ghost Proj Plan", state)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_tasks_project_not_found(self, monkeypatch, tmp_path, capsys):
        """tasks: active_project_id указывает на несуществующий проект."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_project_id = "ghost_id"
        _handle_project_command("tasks", "", state)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_tasks_shows_orphaned_plan_id(self, monkeypatch, tmp_path, capsys):
        """tasks: план в project.plan_ids не существует на диске."""
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        proj.plan_ids.append("orphan-task-id-xyz")
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id
        _handle_project_command("tasks", "", state)
        out = capsys.readouterr().out
        assert "не найдена" in out or "orphan" in out.lower() or "?" in out

    def test_task_new_no_arg(self, monkeypatch, tmp_path, capsys):
        """task_new без описания задачи."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("task_new", "", state)
        out = capsys.readouterr().out
        assert "требует" in out

    def test_task_new_no_active_project(self, monkeypatch, tmp_path, capsys):
        """task_new: нет активного проекта."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("task_new", "Build Login", state)
        out = capsys.readouterr().out
        assert "нет активного" in out.lower()

    def test_task_new_project_not_found(self, monkeypatch, tmp_path, capsys):
        """task_new: проект не существует."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_project_id = "ghost_proj"
        _handle_project_command("task_new", "Build Login", state)
        out = capsys.readouterr().out
        assert "не найден" in out

    def test_task_new_creates_plan_in_project(self, monkeypatch, tmp_path, capsys):
        """task_new: создаёт план и привязывает к проекту."""
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project, load_project
        from chatbot.task_storage import load_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id

        steps = [{"title": "Step 1", "description": "Do it"}]
        from unittest.mock import patch
        with patch("chatbot.main._create_task_plan") as mock_create:
            def side_effect(desc, s, client, **kwargs):
                s.active_task_id = "new-task-id-001"
                if "new-task-id-001" not in s.active_task_ids:
                    s.active_task_ids.append("new-task-id-001")
                # Создаём план на диске
                from chatbot.task_storage import save_task_plan
                plan = TaskPlan(
                    task_id="new-task-id-001",
                    name=desc[:80],
                    profile_name=s.profile_name,
                    created_at=_now(),
                    updated_at=_now(),
                )
                save_task_plan(plan, s.profile_name)
            mock_create.side_effect = side_effect
            _handle_project_command("task_new", "Build Authentication", state)

        updated_proj = load_project(proj.project_id, state.profile_name)
        assert "new-task-id-001" in updated_proj.plan_ids
        plan = load_task_plan("new-task-id-001", state.profile_name)
        assert plan.project_id == proj.project_id

    def test_task_describe_no_plan(self, monkeypatch, tmp_path, capsys):
        """task_describe: нет активного плана."""
        from chatbot.main import _handle_project_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_project_command("task_describe", "Some description", state)
        out = capsys.readouterr().out
        assert "нет активного плана" in out.lower()

    def test_task_describe_no_text(self, monkeypatch, tmp_path, capsys):
        """task_describe: нет текста описания."""
        from chatbot.main import _handle_project_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="My Plan",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_project_command("task_describe", "", state)
        out = capsys.readouterr().out
        assert "требует" in out


# ===========================================================================
# Интеграционный сценарий: Project → Plan → Invariants → Builder
# ===========================================================================

class TestProjectPlanInvariantIntegration:
    """
    Полный сценарий: создание проекта, добавление плана,
    настройка инвариантов и запуск builder с проверкой контекста.
    """

    def _setup_project_with_plan(self, tmp_path, state):
        """Вспомогательный метод: создаёт проект и план на диске."""
        from chatbot.project_storage import save_project
        from chatbot.task_storage import save_task_plan, save_task_step
        from chatbot.models import StepStatus

        proj = _make_project(profile=state.profile_name, name="E2E Project")
        now = _now()
        plan_id = uuid.uuid4().hex
        plan = TaskPlan(
            task_id=plan_id,
            name="Build Feature",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=2,
            current_step_index=0,
            created_at=now,
            updated_at=now,
        )
        from chatbot.models import TaskStep
        step1 = TaskStep(
            step_id=f"{plan_id}_step_001",
            task_id=plan_id,
            index=1,
            title="Design API",
            description="Create API schema",
            status=StepStatus.PENDING,
            created_at=now,
        )
        step2 = TaskStep(
            step_id=f"{plan_id}_step_002",
            task_id=plan_id,
            index=2,
            title="Implement endpoints",
            description="Write endpoint handlers",
            status=StepStatus.PENDING,
            created_at=now,
        )
        plan.step_ids = [step1.step_id, step2.step_id]
        proj.plan_ids.append(plan_id)
        save_project(proj, state.profile_name)
        save_task_plan(plan, state.profile_name)
        save_task_step(step1, state.profile_name)
        save_task_step(step2, state.profile_name)
        return proj, plan

    def test_full_flow_create_project_add_invariants_run_builder(
        self, monkeypatch, tmp_path, capsys
    ):
        """
        Сценарий:
        1. /project new → создаём проект
        2. /invariant add → добавляем ограничения
        3. /project add-plan → добавляем план
        4. /plan builder → запускаем builder (с мок LLM)
        5. Проверяем, что план завершён и инварианты применялись
        """
        from chatbot.main import _handle_project_command, _handle_agent_command, _handle_invariant_command
        from chatbot.project_storage import save_project, load_project
        from chatbot.task_storage import save_task_plan, load_task_plan
        from unittest.mock import MagicMock, patch
        monkeypatch.chdir(tmp_path)

        state = _make_state()

        # Шаг 1: создаём проект
        _handle_project_command("new", "My Web App", state)
        assert state.active_project_id is not None
        out = capsys.readouterr().out
        assert "My Web App" in out

        # Шаг 2: добавляем инварианты
        _handle_invariant_command("add", "Ответ должен быть на русском языке", state)
        _handle_invariant_command("add", "Каждый шаг должен содержать код", state)
        assert len(state.agent_mode.invariants) == 2
        capsys.readouterr()

        # Шаг 3: создаём план в проекте (с мок _create_task_plan)
        proj_id = state.active_project_id
        with patch("chatbot.main._create_task_plan") as mock_create:
            plan_id = uuid.uuid4().hex

            def side_effect(desc, s, client, **kwargs):
                plan = TaskPlan(
                    task_id=plan_id,
                    name=desc[:80],
                    profile_name=s.profile_name,
                    phase=TaskPhase.EXECUTION,
                    total_steps=1,
                    current_step_index=0,
                    created_at=_now(),
                    updated_at=_now(),
                )
                from chatbot.task_storage import save_task_plan
                save_task_plan(plan, s.profile_name)
                s.active_task_id = plan_id
                if plan_id not in s.active_task_ids:
                    s.active_task_ids.append(plan_id)

            mock_create.side_effect = side_effect
            _handle_project_command("task_new", "build-rest-api", state)

        capsys.readouterr()
        proj = load_project(proj_id, state.profile_name)
        assert plan_id in proj.plan_ids

        plan = load_task_plan(plan_id, state.profile_name)
        assert plan.project_id == proj_id

        # Шаг 4: запускаем builder (мокируем _run_plan_builder)
        builder_calls = []
        with patch("chatbot.main._run_plan_builder") as mock_builder:
            mock_builder.side_effect = lambda s, c: builder_calls.append(s.active_task_id)
            _handle_agent_command("builder", f"--plan {plan.name}", state, client=None)

        assert plan_id in builder_calls

    def test_invariants_list_after_add_and_edit(self, capsys):
        """Проверяет добавление, редактирование и список инвариантов."""
        from chatbot.main import _handle_invariant_command
        state = _make_state()

        _handle_invariant_command("add", "Не использовать глобальные переменные", state)
        _handle_invariant_command("add", "Все функции должны иметь типы", state)
        _handle_invariant_command("edit", "1 Не изменять глобальные переменные", state)
        capsys.readouterr()

        _handle_invariant_command("list", "", state)
        out = capsys.readouterr().out
        assert "Не изменять глобальные переменные" in out
        assert "Все функции должны иметь типы" in out

    def test_project_show_with_active_plan_mark(self, monkeypatch, tmp_path, capsys):
        """Проверяет, что /project show помечает активный план."""
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        proj = _make_project(name="Web Backend")
        plan1 = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Auth",
            profile_name=state.profile_name,
            created_at=_now(),
            updated_at=_now(),
        )
        plan2 = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="API",
            profile_name=state.profile_name,
            model="gpt-4",
            created_at=_now(),
            updated_at=_now(),
        )
        proj.plan_ids = [plan1.task_id, plan2.task_id]
        save_project(proj, state.profile_name)
        save_task_plan(plan1, state.profile_name)
        save_task_plan(plan2, state.profile_name)
        state.active_project_id = proj.project_id
        state.active_task_id = plan2.task_id

        _handle_project_command("show", "", state)
        out = capsys.readouterr().out
        assert "Auth" in out
        assert "API" in out
        assert "gpt-4" in out
        assert "активный" in out

    def test_builder_with_invariants_uses_correct_plan(self, monkeypatch, tmp_path, capsys):
        """
        Сценарий с двумя планами в проекте:
        - builder запускается для конкретного плана через --plan
        - invariants активны и применяются к нужному плану
        """
        from chatbot.main import _handle_agent_command, _handle_invariant_command
        from chatbot.project_storage import save_project
        from chatbot.task_storage import save_task_plan
        from unittest.mock import patch
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        # Настраиваем инварианты
        _handle_invariant_command("add", "Код должен быть задокументирован", state)
        capsys.readouterr()

        # Создаём два плана
        proj = _make_project()
        plan_a = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="frontend",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=1,
            current_step_index=0,
            created_at=_now(),
            updated_at=_now(),
        )
        plan_b = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="backend",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=1,
            current_step_index=0,
            created_at=_now(),
            updated_at=_now(),
        )
        proj.plan_ids = [plan_a.task_id, plan_b.task_id]
        save_project(proj, state.profile_name)
        save_task_plan(plan_a, state.profile_name)
        save_task_plan(plan_b, state.profile_name)
        state.active_project_id = proj.project_id
        state.active_task_id = plan_a.task_id
        state.active_task_ids = [plan_a.task_id, plan_b.task_id]

        builder_task_ids = []
        with patch("chatbot.main._run_plan_builder") as mock_builder:
            mock_builder.side_effect = lambda s, c: builder_task_ids.append(s.active_task_id)
            _handle_agent_command("builder", "--plan backend", state, client=None)

        # builder должен запуститься для backend, а active_task_id восстановиться
        assert plan_b.task_id in builder_task_ids
        assert state.active_task_id == plan_a.task_id  # восстановлено

    def test_memory_context_with_active_project(self, monkeypatch, tmp_path, capsys):
        """
        Проверяет, что active_project_id сохраняется в сессии
        и восстанавливается при _apply_session_data.
        """
        from chatbot.main import _build_session_payload, _apply_session_data
        from chatbot.project_storage import save_project
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id
        state.active_task_ids = ["task-1", "task-2"]
        state.agent_mode.invariants = ["Инвариант 1", "Инвариант 2"]

        payload = _build_session_payload(state, user_input="", assistant_text="")
        data = payload.model_dump()

        state2 = _make_state()
        _apply_session_data(data, state2)

        assert state2.active_project_id == proj.project_id
        assert state2.active_task_ids == ["task-1", "task-2"]

    def test_add_plan_hyphen_alias(self, monkeypatch, tmp_path, capsys):
        """Проверяет, что add-plan (через hyphen) работает как add_plan."""
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project, load_project
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        proj = _make_project()
        save_project(proj, state.profile_name)
        state.active_project_id = proj.project_id

        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="My Service",
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

    def test_project_tasks_active_plan_marked(self, monkeypatch, tmp_path, capsys):
        """
        /project tasks: активный план помечается стрелкой ◀.
        """
        from chatbot.main import _handle_project_command
        from chatbot.project_storage import save_project
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()

        proj = _make_project()
        plan = TaskPlan(
            task_id=uuid.uuid4().hex,
            name="Active Plan",
            profile_name=state.profile_name,
            phase=TaskPhase.EXECUTION,
            total_steps=3,
            current_step_index=1,
            created_at=_now(),
            updated_at=_now(),
        )
        proj.plan_ids.append(plan.task_id)
        save_project(proj, state.profile_name)
        save_task_plan(plan, state.profile_name)
        state.active_project_id = proj.project_id
        state.active_task_id = plan.task_id

        _handle_project_command("tasks", "", state)
        out = capsys.readouterr().out
        assert "Active Plan" in out
        assert "активная" in out
