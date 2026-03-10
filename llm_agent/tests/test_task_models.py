"""Тесты для новых моделей задач в chatbot/models.py."""

import pytest
from pydantic import ValidationError

from llm_agent.chatbot.models import (
    DialogueSession,
    SessionState,
    StepStatus,
    TaskPhase,
    TaskPlan,
    TaskStep,
)


# ---------------------------------------------------------------------------
# TaskPhase
# ---------------------------------------------------------------------------


class TestTaskPhase:
    def test_all_values(self):
        assert TaskPhase.PLANNING == "planning"
        assert TaskPhase.EXECUTION == "execution"
        assert TaskPhase.VALIDATION == "validation"
        assert TaskPhase.DONE == "done"
        assert TaskPhase.PAUSED == "paused"
        assert TaskPhase.FAILED == "failed"

    def test_is_str_enum(self):
        assert isinstance(TaskPhase.PLANNING, str)

    def test_from_string(self):
        assert TaskPhase("planning") == TaskPhase.PLANNING
        assert TaskPhase("done") == TaskPhase.DONE


# ---------------------------------------------------------------------------
# StepStatus
# ---------------------------------------------------------------------------


class TestStepStatus:
    def test_all_values(self):
        assert StepStatus.PENDING == "pending"
        assert StepStatus.IN_PROGRESS == "in_progress"
        assert StepStatus.DONE == "done"
        assert StepStatus.SKIPPED == "skipped"
        assert StepStatus.FAILED == "failed"

    def test_is_str_enum(self):
        assert isinstance(StepStatus.PENDING, str)

    def test_from_string(self):
        assert StepStatus("done") == StepStatus.DONE


# ---------------------------------------------------------------------------
# TaskStep
# ---------------------------------------------------------------------------


class TestTaskStep:
    def _make(self, **kwargs) -> TaskStep:
        defaults = {
            "step_id": "step1",
            "task_id": "task1",
            "index": 1,
            "title": "Do something",
            "created_at": "2024-01-01T00:00:00",
        }
        defaults.update(kwargs)
        return TaskStep(**defaults)

    def test_basic_creation(self):
        step = self._make()
        assert step.step_id == "step1"
        assert step.task_id == "task1"
        assert step.index == 1
        assert step.title == "Do something"
        assert step.status == StepStatus.PENDING
        assert step.description == ""
        assert step.notes == ""
        assert step.started_at is None
        assert step.completed_at is None

    def test_index_must_be_ge_1(self):
        with pytest.raises(ValidationError):
            self._make(index=0)

    def test_custom_status(self):
        step = self._make(status=StepStatus.DONE)
        assert step.status == StepStatus.DONE

    def test_with_optional_fields(self):
        step = self._make(
            description="Details here",
            notes="Some notes",
            started_at="2024-01-01T01:00:00",
            completed_at="2024-01-01T02:00:00",
        )
        assert step.description == "Details here"
        assert step.notes == "Some notes"
        assert step.started_at == "2024-01-01T01:00:00"
        assert step.completed_at == "2024-01-01T02:00:00"

    def test_serialization(self):
        step = self._make()
        data = step.model_dump()
        assert data["step_id"] == "step1"
        assert data["status"] == "pending"

    def test_roundtrip(self):
        step = self._make(notes="test note")
        data = step.model_dump()
        restored = TaskStep(**data)
        assert restored.notes == "test note"
        assert restored.status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# TaskPlan
# ---------------------------------------------------------------------------


class TestTaskPlan:
    def _make(self, **kwargs) -> TaskPlan:
        defaults = {
            "task_id": "abc123",
            "name": "My task",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        defaults.update(kwargs)
        return TaskPlan(**defaults)

    def test_basic_creation(self):
        plan = self._make()
        assert plan.task_id == "abc123"
        assert plan.name == "My task"
        assert plan.profile_name == "default"
        assert plan.phase == TaskPhase.PLANNING
        assert plan.step_ids == []
        assert plan.total_steps == 0
        assert plan.current_step_index == 0
        assert plan.completed_at is None
        assert plan.failure_reason is None
        assert plan.llm_raw_response is None
        assert plan.description == ""

    def test_custom_profile(self):
        plan = self._make(profile_name="Igor")
        assert plan.profile_name == "Igor"

    def test_with_steps(self):
        plan = self._make(step_ids=["s1", "s2", "s3"], total_steps=3, current_step_index=1)
        assert len(plan.step_ids) == 3
        assert plan.total_steps == 3
        assert plan.current_step_index == 1

    def test_phase_transition(self):
        plan = self._make()
        assert plan.phase == TaskPhase.PLANNING
        plan.phase = TaskPhase.EXECUTION
        assert plan.phase == TaskPhase.EXECUTION

    def test_failure_reason(self):
        plan = self._make(phase=TaskPhase.FAILED, failure_reason="Step 2 failed")
        assert plan.failure_reason == "Step 2 failed"

    def test_serialization(self):
        plan = self._make(phase=TaskPhase.EXECUTION, total_steps=5)
        data = plan.model_dump()
        assert data["phase"] == "execution"
        assert data["total_steps"] == 5

    def test_roundtrip(self):
        plan = self._make(llm_raw_response="raw text")
        data = plan.model_dump()
        restored = TaskPlan(**data)
        assert restored.llm_raw_response == "raw text"
        assert restored.phase == TaskPhase.PLANNING


# ---------------------------------------------------------------------------
# SessionState.active_task_id
# ---------------------------------------------------------------------------


class TestSessionStateActiveTaskId:
    def test_default_is_none(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert state.active_task_id is None

    def test_can_be_set(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            active_task_id="task123",
        )
        assert state.active_task_id == "task123"

    def test_can_be_mutated(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        state.active_task_id = "newtask"
        assert state.active_task_id == "newtask"


# ---------------------------------------------------------------------------
# DialogueSession.active_task_id
# ---------------------------------------------------------------------------


class TestDialogueSessionActiveTaskId:
    def test_default_is_none(self):
        session = DialogueSession(
            dialogue_session_id="s1",
            created_at="2024-01-01T00:00:00",
            model="gpt-4",
            base_url="https://api.example.com",
        )
        assert session.active_task_id is None

    def test_can_be_set(self):
        session = DialogueSession(
            dialogue_session_id="s1",
            created_at="2024-01-01T00:00:00",
            model="gpt-4",
            base_url="https://api.example.com",
            active_task_id="task999",
        )
        assert session.active_task_id == "task999"

    def test_serialization_includes_task_id(self):
        session = DialogueSession(
            dialogue_session_id="s1",
            created_at="2024-01-01T00:00:00",
            model="gpt-4",
            base_url="https://api.example.com",
            active_task_id="myid",
        )
        data = session.model_dump()
        assert "active_task_id" in data
        assert data["active_task_id"] == "myid"
