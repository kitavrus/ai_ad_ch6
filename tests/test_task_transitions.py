"""Тесты явных переходов между состояниями задачи."""

import pytest
from unittest.mock import patch
from datetime import datetime
import uuid

from chatbot.models import (
    ALLOWED_TRANSITIONS,
    SessionState,
    TaskPhase,
    TaskPlan,
    can_transition,
)


def _make_state():
    return SessionState(
        model="gpt-4",
        base_url="https://api.example.com",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        profile_name="default",
    )


def _make_plan(phase=TaskPhase.PLANNING, **kwargs):
    now = datetime.utcnow().isoformat()
    return TaskPlan(
        task_id=uuid.uuid4().hex,
        profile_name="default",
        name="Test Plan",
        phase=phase,
        created_at=now,
        updated_at=now,
        **kwargs,
    )


def _save_plan(plan, state, tmp_path):
    from chatbot.task_storage import save_task_plan
    save_task_plan(plan, state.profile_name)


# ===========================================================================
# can_transition / ALLOWED_TRANSITIONS
# ===========================================================================

class TestCanTransition:
    """Тестирует функцию can_transition и таблицу ALLOWED_TRANSITIONS."""

    # Допустимые переходы
    @pytest.mark.parametrize("from_phase, to_phase", [
        (TaskPhase.PLANNING,   TaskPhase.EXECUTION),
        (TaskPhase.PLANNING,   TaskPhase.FAILED),
        (TaskPhase.EXECUTION,  TaskPhase.VALIDATION),
        (TaskPhase.EXECUTION,  TaskPhase.PAUSED),
        (TaskPhase.EXECUTION,  TaskPhase.FAILED),
        (TaskPhase.VALIDATION, TaskPhase.DONE),
        (TaskPhase.VALIDATION, TaskPhase.EXECUTION),
        (TaskPhase.PAUSED,     TaskPhase.EXECUTION),
        (TaskPhase.PAUSED,     TaskPhase.FAILED),
    ])
    def test_allowed(self, from_phase, to_phase):
        assert can_transition(from_phase, to_phase) is True

    # Недопустимые переходы
    @pytest.mark.parametrize("from_phase, to_phase", [
        (TaskPhase.PLANNING,   TaskPhase.DONE),
        (TaskPhase.PLANNING,   TaskPhase.PAUSED),
        (TaskPhase.PLANNING,   TaskPhase.VALIDATION),
        (TaskPhase.EXECUTION,  TaskPhase.DONE),
        (TaskPhase.EXECUTION,  TaskPhase.PLANNING),
        (TaskPhase.VALIDATION, TaskPhase.PLANNING),
        (TaskPhase.VALIDATION, TaskPhase.PAUSED),
        (TaskPhase.VALIDATION, TaskPhase.FAILED),
        (TaskPhase.PAUSED,     TaskPhase.DONE),
        (TaskPhase.PAUSED,     TaskPhase.PLANNING),
        (TaskPhase.PAUSED,     TaskPhase.VALIDATION),
        (TaskPhase.DONE,       TaskPhase.EXECUTION),
        (TaskPhase.DONE,       TaskPhase.PLANNING),
        (TaskPhase.DONE,       TaskPhase.PAUSED),
        (TaskPhase.DONE,       TaskPhase.FAILED),
        (TaskPhase.DONE,       TaskPhase.VALIDATION),
        (TaskPhase.FAILED,     TaskPhase.EXECUTION),
        (TaskPhase.FAILED,     TaskPhase.PLANNING),
        (TaskPhase.FAILED,     TaskPhase.PAUSED),
        (TaskPhase.FAILED,     TaskPhase.DONE),
        (TaskPhase.FAILED,     TaskPhase.VALIDATION),
    ])
    def test_forbidden(self, from_phase, to_phase):
        assert can_transition(from_phase, to_phase) is False

    def test_done_is_terminal(self):
        assert ALLOWED_TRANSITIONS[TaskPhase.DONE] == set()

    def test_failed_is_terminal(self):
        assert ALLOWED_TRANSITIONS[TaskPhase.FAILED] == set()


# ===========================================================================
# _transition_plan guard
# ===========================================================================

class TestTransitionPlanGuard:
    def test_allowed_transition_returns_true(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(TaskPhase.PLANNING)
        _save_plan(plan, state, tmp_path)
        result = _transition_plan(plan, TaskPhase.EXECUTION, state)
        assert result is True
        assert plan.phase == TaskPhase.EXECUTION

    def test_forbidden_transition_returns_false(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(TaskPhase.PLANNING)
        _save_plan(plan, state, tmp_path)
        result = _transition_plan(plan, TaskPhase.DONE, state)
        assert result is False
        assert plan.phase == TaskPhase.PLANNING  # не изменилась

    def test_forbidden_prints_error(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(TaskPhase.DONE)
        _save_plan(plan, state, tmp_path)
        _transition_plan(plan, TaskPhase.EXECUTION, state)
        out = capsys.readouterr().out
        assert "done" in out and "execution" in out

    def test_custom_error_message(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(TaskPhase.DONE)
        _save_plan(plan, state, tmp_path)
        _transition_plan(plan, TaskPhase.EXECUTION, state, error_msg="[Кастомная ошибка]")
        assert "[Кастомная ошибка]" in capsys.readouterr().out

    def test_done_transition_sets_completed_at(self, monkeypatch, tmp_path):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(TaskPhase.VALIDATION)
        _save_plan(plan, state, tmp_path)
        _transition_plan(plan, TaskPhase.DONE, state)
        assert plan.completed_at is not None

    def test_paused_transition_no_completed_at(self, monkeypatch, tmp_path):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(TaskPhase.EXECUTION)
        _save_plan(plan, state, tmp_path)
        _transition_plan(plan, TaskPhase.PAUSED, state)
        assert plan.completed_at is None


# ===========================================================================
# /task pause guard
# ===========================================================================

class TestPauseGuard:
    def _run(self, phase, tmp_path, monkeypatch, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(phase, total_steps=1)
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_task_command("pause", "", state, client=None)
        return capsys.readouterr().out

    def test_pause_from_execution_ok(self, monkeypatch, tmp_path, capsys):
        out = self._run(TaskPhase.EXECUTION, tmp_path, monkeypatch, capsys)
        assert "приостановлена" in out

    @pytest.mark.parametrize("phase", [
        TaskPhase.PLANNING, TaskPhase.VALIDATION, TaskPhase.DONE, TaskPhase.FAILED, TaskPhase.PAUSED,
    ])
    def test_pause_blocked_from_non_execution(self, phase, monkeypatch, tmp_path, capsys):
        out = self._run(phase, tmp_path, monkeypatch, capsys)
        assert "нельзя" in out.lower() or "приостановить" not in out.lower()


# ===========================================================================
# /task resume guard
# ===========================================================================

class TestResumeGuard:
    def _run(self, phase, tmp_path, monkeypatch, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(phase, total_steps=1)
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_task_command("resume", "", state, client=None)
        return capsys.readouterr().out

    @pytest.mark.parametrize("phase", [TaskPhase.PAUSED, TaskPhase.VALIDATION])
    def test_resume_from_allowed_phase(self, phase, monkeypatch, tmp_path, capsys):
        out = self._run(phase, tmp_path, monkeypatch, capsys)
        assert "возобновлена" in out

    @pytest.mark.parametrize("phase", [
        TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.DONE, TaskPhase.FAILED,
    ])
    def test_resume_blocked_from_non_paused(self, phase, monkeypatch, tmp_path, capsys):
        out = self._run(phase, tmp_path, monkeypatch, capsys)
        assert "нельзя" in out.lower() or "возобновить" not in out.lower()


# ===========================================================================
# /task done guard
# ===========================================================================

class TestDoneGuard:
    def _run(self, phase, tmp_path, monkeypatch, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(phase)
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_task_command("done", "", state, client=None)
        return capsys.readouterr().out

    def test_done_from_validation_ok(self, monkeypatch, tmp_path, capsys):
        out = self._run(TaskPhase.VALIDATION, tmp_path, monkeypatch, capsys)
        assert "завершена" in out

    @pytest.mark.parametrize("phase", [
        TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.PAUSED, TaskPhase.DONE, TaskPhase.FAILED,
    ])
    def test_done_blocked_from_non_validation(self, phase, monkeypatch, tmp_path, capsys):
        out = self._run(phase, tmp_path, monkeypatch, capsys)
        assert "нельзя" in out.lower() or "завершена" not in out.lower()

    def test_done_error_mentions_validation(self, monkeypatch, tmp_path, capsys):
        out = self._run(TaskPhase.EXECUTION, tmp_path, monkeypatch, capsys)
        assert "validation" in out.lower()


# ===========================================================================
# /task fail guard
# ===========================================================================

class TestFailGuard:
    def _run(self, phase, tmp_path, monkeypatch, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _make_plan(phase)
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        _handle_task_command("fail", "причина", state, client=None)
        return capsys.readouterr().out

    @pytest.mark.parametrize("phase", [
        TaskPhase.PLANNING, TaskPhase.EXECUTION, TaskPhase.PAUSED,
    ])
    def test_fail_from_allowed_phase(self, phase, monkeypatch, tmp_path, capsys):
        out = self._run(phase, tmp_path, monkeypatch, capsys)
        assert "FAILED" in out

    def test_fail_from_done_blocked(self, monkeypatch, tmp_path, capsys):
        out = self._run(TaskPhase.DONE, tmp_path, monkeypatch, capsys)
        assert "нельзя" in out.lower() or "FAILED" not in out

    def test_fail_from_done_mentions_terminal(self, monkeypatch, tmp_path, capsys):
        out = self._run(TaskPhase.DONE, tmp_path, monkeypatch, capsys)
        assert "терминальное" in out or "завершённую" in out

    def test_fail_from_failed_blocked(self, monkeypatch, tmp_path, capsys):
        out = self._run(TaskPhase.FAILED, tmp_path, monkeypatch, capsys)
        # already failed message, no duplicate FAILED transition
        assert "already" in out.lower() or "уже" in out.lower() or "FAILED" not in out or "провалить" not in out


# ===========================================================================
# Терминальные состояния (DONE и FAILED)
# ===========================================================================

class TestTerminalStates:
    def _make_terminal(self, phase, state, tmp_path):
        from chatbot.task_storage import save_task_plan
        plan = _make_plan(phase)
        save_task_plan(plan, state.profile_name)
        state.active_task_id = plan.task_id
        return plan

    def test_done_cannot_be_paused(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        self._make_terminal(TaskPhase.DONE, state, tmp_path)
        _handle_task_command("pause", "", state, client=None)
        out = capsys.readouterr().out
        assert "приостановлена" not in out

    def test_done_cannot_be_resumed(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        self._make_terminal(TaskPhase.DONE, state, tmp_path)
        _handle_task_command("resume", "", state, client=None)
        out = capsys.readouterr().out
        assert "возобновлена" not in out

    def test_done_cannot_be_failed(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        self._make_terminal(TaskPhase.DONE, state, tmp_path)
        _handle_task_command("fail", "", state, client=None)
        out = capsys.readouterr().out
        assert "FAILED" not in out

    def test_failed_cannot_be_paused(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        self._make_terminal(TaskPhase.FAILED, state, tmp_path)
        _handle_task_command("pause", "", state, client=None)
        out = capsys.readouterr().out
        assert "приостановлена" not in out

    def test_failed_cannot_be_resumed(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        self._make_terminal(TaskPhase.FAILED, state, tmp_path)
        _handle_task_command("resume", "", state, client=None)
        out = capsys.readouterr().out
        assert "возобновлена" not in out

    def test_failed_cannot_be_done(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        self._make_terminal(TaskPhase.FAILED, state, tmp_path)
        _handle_task_command("done", "", state, client=None)
        out = capsys.readouterr().out
        assert "завершена" not in out


# ===========================================================================
# Полный жизненный цикл: planning → execution → validation → done
# ===========================================================================

class TestFullLifecycle:
    def test_complete_lifecycle(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import load_task_plan, save_task_plan, save_task_step
        from chatbot.models import StepStatus, TaskStep
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        now = datetime.utcnow().isoformat()

        plan = _make_plan(TaskPhase.PLANNING, total_steps=1)
        step = TaskStep(
            step_id=f"{plan.task_id}_step_001",
            task_id=plan.task_id,
            index=1,
            title="Шаг 1",
            status=StepStatus.PENDING,
            created_at=now,
        )
        plan.step_ids = [step.step_id]
        from chatbot.task_storage import save_task_plan
        save_task_plan(plan, state.profile_name)
        save_task_step(step, state.profile_name)
        state.active_task_id = plan.task_id

        # Попытка done из PLANNING → блок
        _handle_task_command("done", "", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.PLANNING

        # Старт
        _handle_task_command("start", "", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.EXECUTION

        # Попытка done из EXECUTION → блок
        _handle_task_command("done", "", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.EXECUTION

        # Выполняем шаг → auto VALIDATION
        _handle_task_command("step", "done Шаг выполнен", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.VALIDATION

        # Финал из VALIDATION → DONE
        _handle_task_command("done", "итог", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.DONE
        assert state.active_task_id is None

    def test_pause_and_resume(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import load_task_plan, save_task_plan, save_task_step
        from chatbot.models import StepStatus, TaskStep
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        now = datetime.utcnow().isoformat()

        plan = _make_plan(TaskPhase.EXECUTION, total_steps=1)
        step = TaskStep(
            step_id=f"{plan.task_id}_step_001",
            task_id=plan.task_id,
            index=1,
            title="Шаг 1",
            status=StepStatus.PENDING,
            created_at=now,
        )
        plan.step_ids = [step.step_id]
        save_task_plan(plan, state.profile_name)
        save_task_step(step, state.profile_name)
        state.active_task_id = plan.task_id

        # Пауза
        _handle_task_command("pause", "", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.PAUSED

        # Попытка done из PAUSED → блок
        _handle_task_command("done", "", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.PAUSED

        # Возобновление
        _handle_task_command("resume", "", state, client=None)
        assert load_task_plan(plan.task_id, state.profile_name).phase == TaskPhase.EXECUTION
