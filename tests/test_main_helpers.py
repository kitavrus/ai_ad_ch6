"""Покрытие ранее непокрытых хелперов chatbot/main.py."""

import time
import pytest
from unittest.mock import MagicMock, patch

from chatbot.models import (
    ChatMessage,
    ContextStrategy,
    SessionState,
    StickyFacts,
    TaskPhase,
    StepStatus,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> SessionState:
    from chatbot.memory import Memory
    defaults = {
        "model": "gpt-4",
        "base_url": "https://api.example.com",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "dialogue_start_time": time.time(),
    }
    defaults.update(kwargs)
    state = SessionState(**defaults)
    state.memory = Memory()
    return state


def _make_plan_and_steps(state, tmp_path, title="Test task", n_steps=2):
    from chatbot.task_storage import save_task_plan, save_task_step
    from chatbot.models import TaskPlan, TaskStep
    from datetime import datetime
    import uuid

    now = datetime.utcnow().isoformat()
    task_id = uuid.uuid4().hex
    plan = TaskPlan(
        task_id=task_id,
        profile_name=state.profile_name,
        name=title,
        description=title,
        phase=TaskPhase.EXECUTION,
        total_steps=n_steps,
        current_step_index=0,
        created_at=now,
        updated_at=now,
    )
    step_ids = []
    for i in range(1, n_steps + 1):
        step = TaskStep(
            step_id=f"{task_id}_step_{i:03d}",
            task_id=task_id,
            index=i,
            title=f"Step {i}",
            description=f"Do step {i}",
            status=StepStatus.PENDING,
            created_at=now,
        )
        save_task_step(step, state.profile_name)
        step_ids.append(step.step_id)
    plan.step_ids = step_ids
    save_task_plan(plan, state.profile_name)
    state.active_task_id = task_id
    return plan, task_id


# ===========================================================================
# _print_strategy_status
# ===========================================================================

class TestPrintStrategyStatus:
    def test_sliding_window_no_summary(self, capsys):
        from chatbot.main import _print_strategy_status
        state = _make_state()
        state.context_strategy = ContextStrategy.SLIDING_WINDOW
        state.context_summary = ""
        _print_strategy_status(state)
        out = capsys.readouterr().out
        assert "sliding_window" in out
        assert "Summary: нет" in out

    def test_sliding_window_with_summary(self, capsys):
        from chatbot.main import _print_strategy_status
        state = _make_state()
        state.context_strategy = ContextStrategy.SLIDING_WINDOW
        state.context_summary = "Some summary"
        _print_strategy_status(state)
        out = capsys.readouterr().out
        assert "Summary: есть" in out

    def test_sticky_facts_empty(self, capsys):
        from chatbot.main import _print_strategy_status
        state = _make_state()
        state.context_strategy = ContextStrategy.STICKY_FACTS
        _print_strategy_status(state)
        out = capsys.readouterr().out
        assert "sticky_facts" in out
        assert "Фактов в памяти: 0" in out

    def test_sticky_facts_with_items(self, capsys):
        from chatbot.main import _print_strategy_status
        state = _make_state()
        state.context_strategy = ContextStrategy.STICKY_FACTS
        state.sticky_facts = StickyFacts(facts={"lang": "Python", "style": "formal"})
        _print_strategy_status(state)
        out = capsys.readouterr().out
        assert "Фактов в памяти: 2" in out
        assert "lang: Python" in out

    def test_branching(self, capsys):
        from chatbot.main import _print_strategy_status
        state = _make_state()
        state.context_strategy = ContextStrategy.BRANCHING
        _print_strategy_status(state)
        out = capsys.readouterr().out
        assert "branching" in out
        assert "Веток: 0" in out


# ===========================================================================
# _build_plan_prompt
# ===========================================================================

class TestBuildPlanPrompt:
    def test_contains_description(self):
        from chatbot.main import _build_plan_prompt
        result = _build_plan_prompt("Build REST API")
        assert "Build REST API" in result
        assert "JSON array" in result


# ===========================================================================
# _parse_steps_from_llm_response
# ===========================================================================

class TestParseStepsFromLlmResponse:
    def test_layer1_clean_json_array(self):
        from chatbot.main import _parse_steps_from_llm_response
        text = '[{"title": "Step 1", "description": "do it"}]'
        result = _parse_steps_from_llm_response(text)
        assert result is not None
        assert result[0]["title"] == "Step 1"

    def test_layer2_json_in_prose(self):
        from chatbot.main import _parse_steps_from_llm_response
        text = 'Here is the plan:\n[{"title": "Step 1", "description": "do it"}]\nDone.'
        result = _parse_steps_from_llm_response(text)
        assert result is not None
        assert result[0]["title"] == "Step 1"

    def test_layer3_numbered_list_dot(self):
        from chatbot.main import _parse_steps_from_llm_response
        text = "1. Setup environment\n2. Write tests\n3. Deploy"
        result = _parse_steps_from_llm_response(text)
        assert result is not None
        assert len(result) == 3
        assert result[0]["title"] == "Setup environment"

    def test_layer3_numbered_list_paren(self):
        from chatbot.main import _parse_steps_from_llm_response
        text = "1) First step\n2) Second step"
        result = _parse_steps_from_llm_response(text)
        assert result is not None
        assert len(result) == 2

    def test_returns_none_when_all_fail(self):
        from chatbot.main import _parse_steps_from_llm_response
        result = _parse_steps_from_llm_response("no valid data here")
        assert result is None


# ===========================================================================
# _validate_steps
# ===========================================================================

class TestValidateSteps:
    def test_empty_list_returns_none(self):
        from chatbot.main import _validate_steps
        assert _validate_steps([]) is None

    def test_non_dict_items_skipped(self):
        from chatbot.main import _validate_steps
        result = _validate_steps(["string", 42, {"title": "Valid"}])
        assert result is not None
        assert len(result) == 1

    def test_step_key_fallback(self):
        from chatbot.main import _validate_steps
        result = _validate_steps([{"step": "Do something", "description": "desc"}])
        assert result[0]["title"] == "Do something"

    def test_empty_title_skipped(self):
        from chatbot.main import _validate_steps
        result = _validate_steps([{"title": "", "description": "desc"}])
        assert result is None

    def test_all_empty_returns_none(self):
        from chatbot.main import _validate_steps
        result = _validate_steps([{"title": "   ", "description": "d"}])
        assert result is None

    def test_valid_steps_normalized(self):
        from chatbot.main import _validate_steps
        steps = [
            {"title": "  Step 1  ", "description": "  do it  "},
            {"title": "Step 2", "description": ""},
        ]
        result = _validate_steps(steps)
        assert result[0]["title"] == "Step 1"
        assert result[0]["description"] == "do it"


# ===========================================================================
# _print_task_plan
# ===========================================================================

class TestPrintTaskPlan:
    def test_prints_current_step_marker(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _print_task_plan
        from chatbot.models import TaskPlan, TaskStep
        from datetime import datetime

        monkeypatch.chdir(tmp_path)
        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="t1", name="My Task", description="d",
            phase=TaskPhase.EXECUTION, total_steps=2,
            current_step_index=0, created_at=now, updated_at=now,
        )
        steps = [
            TaskStep(step_id="s1", task_id="t1", index=1, title="First",
                     description="", status=StepStatus.IN_PROGRESS, created_at=now),
            TaskStep(step_id="s2", task_id="t1", index=2, title="Second",
                     description="", status=StepStatus.PENDING, created_at=now),
        ]
        _print_task_plan(plan, steps)
        out = capsys.readouterr().out
        assert "◀" in out
        assert "First" in out
        assert "Second" in out


# ===========================================================================
# _get_active_plan
# ===========================================================================

class TestGetActivePlan:
    def test_returns_none_when_no_active_task(self):
        from chatbot.main import _get_active_plan
        state = _make_state()
        assert _get_active_plan(state) is None


# ===========================================================================
# _transition_plan
# ===========================================================================

class TestTransitionPlan:
    def test_done_sets_completed_at(self, monkeypatch, tmp_path):
        from chatbot.main import _transition_plan
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        plan.phase = TaskPhase.VALIDATION
        save_task_plan(plan, state.profile_name)
        _transition_plan(plan, TaskPhase.DONE, state)
        assert plan.completed_at is not None
        assert plan.phase == TaskPhase.DONE

    def test_non_done_no_completed_at(self, monkeypatch, tmp_path):
        from chatbot.main import _transition_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        _transition_plan(plan, TaskPhase.PAUSED, state)
        assert plan.phase == TaskPhase.PAUSED


# ===========================================================================
# _advance_plan
# ===========================================================================

class TestAdvancePlan:
    def test_advances_to_next_step(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _advance_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path, n_steps=3)
        plan.current_step_index = 0
        _advance_plan(plan, state)
        assert plan.current_step_index == 1

    def test_all_steps_done_transitions_to_validation(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _advance_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path, n_steps=2)
        plan.current_step_index = 1  # last step
        _advance_plan(plan, state)
        assert plan.phase == TaskPhase.VALIDATION
        out = capsys.readouterr().out
        assert "validation" in out.lower()


# ===========================================================================
# _handle_step_subcommand
# ===========================================================================

class TestHandleStepSubcommand:
    def test_no_active_task(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_step_subcommand("done", state)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_wrong_phase(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        plan.phase = TaskPhase.PLANNING
        save_task_plan(plan, state.profile_name)
        _handle_step_subcommand("done", state)
        out = capsys.readouterr().out
        assert "доступна только в фазе execution" in out

    def test_done_advances_plan(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path, n_steps=3)
        _handle_step_subcommand("done", state)
        out = capsys.readouterr().out
        assert "завершён" in out

    def test_skip(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path, n_steps=3)
        _handle_step_subcommand("skip", state)
        out = capsys.readouterr().out
        assert "пропущен" in out

    def test_fail(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        _handle_step_subcommand("fail reason text", state)
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert state.active_task_id is None

    def test_note(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        _handle_step_subcommand("note important detail", state)
        out = capsys.readouterr().out
        assert "Заметка" in out

    def test_unknown_subcommand(self, monkeypatch, tmp_path, capsys):
        from chatbot.main import _handle_step_subcommand
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        _handle_step_subcommand("unknown_cmd", state)
        out = capsys.readouterr().out
        assert "Неизвестная подкоманда step" in out


# ===========================================================================
# _build_agent_state_vars
# ===========================================================================

class TestBuildAgentStateVars:
    def test_empty_state_no_memory(self):
        from chatbot.main import _build_agent_state_vars
        state = _make_state()
        state.memory = None
        result = _build_agent_state_vars(state)
        assert result == "(empty)"

    def test_with_task_in_working_memory(self):
        from chatbot.main import _build_agent_state_vars
        state = _make_state()
        state.memory.working.set_task("Write tests")
        result = _build_agent_state_vars(state)
        assert "task: Write tests" in result

    def test_with_task_status(self):
        from chatbot.main import _build_agent_state_vars
        state = _make_state()
        state.memory.working.set_task("Write tests")
        state.memory.working.task_status = "in_progress"
        result = _build_agent_state_vars(state)
        assert "task_status: in_progress" in result

    def test_with_preferences(self):
        from chatbot.main import _build_agent_state_vars
        state = _make_state()
        state.memory.working.set_preference("lang", "Python")
        result = _build_agent_state_vars(state)
        assert "lang: Python" in result

    def test_with_active_task_id(self, monkeypatch, tmp_path):
        from chatbot.main import _build_agent_state_vars
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, tmp_path)
        result = _build_agent_state_vars(state)
        assert f"active_task_id: {task_id}" in result

    def test_with_clarifications(self, monkeypatch, tmp_path):
        from chatbot.main import _build_agent_state_vars
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        plan.clarifications = [{"question": "Q1", "answer": "A1"}]
        save_task_plan(plan, state.profile_name)
        result = _build_agent_state_vars(state)
        assert "clarifications" in result
        assert "Q1" in result


# ===========================================================================
# _handle_agent_command
# ===========================================================================

class TestHandleAgentCommand:
    def test_off(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.agent_mode.enabled = True
        _handle_agent_command("off", "", state)
        assert state.agent_mode.enabled is False
        assert "OFF" in capsys.readouterr().out

    def test_status_on(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.agent_mode.enabled = True
        state.agent_mode.invariants = ["rule one"]
        _handle_agent_command("status", "", state)
        out = capsys.readouterr().out
        assert "ON" in out
        assert "rule one" in out

    def test_status_off(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.agent_mode.enabled = False
        _handle_agent_command("status", "", state)
        assert "OFF" in capsys.readouterr().out

    def test_retries_valid(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        _handle_agent_command("retries", "5", state)
        assert state.agent_mode.max_retries == 5

    def test_retries_invalid(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        _handle_agent_command("retries", "abc", state)
        assert "целое число" in capsys.readouterr().out

    def test_retries_clamped_min(self):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        _handle_agent_command("retries", "0", state)
        assert state.agent_mode.max_retries == 1

    def test_retries_clamped_max(self):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        _handle_agent_command("retries", "99", state)
        assert state.agent_mode.max_retries == 10

    def test_unknown_action(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        _handle_agent_command("unknown_action", "", state)
        assert "Неизвестная подкоманда plan" in capsys.readouterr().out

    def test_on_with_non_empty_profile(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.memory.long_term.profile.style = {"tone": "formal"}
        _handle_agent_command("on", "", state)
        out = capsys.readouterr().out
        assert "tone=formal" in out

    def test_on_with_format_and_constraints(self, capsys):
        from chatbot.main import _handle_agent_command
        state = _make_state()
        state.memory.long_term.profile.format = {"output": "markdown"}
        state.memory.long_term.profile.constraints = ["no emojis"]
        state.memory.long_term.profile.custom = {"focus": "backend"}
        _handle_agent_command("on", "", state)
        out = capsys.readouterr().out
        assert "output=markdown" in out
        assert "no emojis" in out
        assert "focus=backend" in out


# ===========================================================================
# _handle_invariant_command
# ===========================================================================

class TestHandleInvariantCommand:
    def test_add(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        _handle_invariant_command("add", "reply in Russian", state)
        assert "reply in Russian" in state.agent_mode.invariants
        assert "добавлен" in capsys.readouterr().out

    def test_add_empty_arg(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        _handle_invariant_command("add", "", state)
        assert "требует текст" in capsys.readouterr().out

    def test_del_valid(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        state.agent_mode.invariants = ["rule one", "rule two"]
        _handle_invariant_command("del", "1", state)
        assert "rule one" not in state.agent_mode.invariants
        assert "удалён" in capsys.readouterr().out

    def test_del_out_of_range(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        state.agent_mode.invariants = ["rule one"]
        _handle_invariant_command("del", "99", state)
        assert "Нет инварианта" in capsys.readouterr().out

    def test_del_invalid_arg(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        _handle_invariant_command("del", "abc", state)
        assert "ожидается номер" in capsys.readouterr().out

    def test_list_empty(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        _handle_invariant_command("list", "", state)
        assert "не заданы" in capsys.readouterr().out

    def test_list_with_invariants(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        state.agent_mode.invariants = ["rule one", "rule two"]
        _handle_invariant_command("list", "", state)
        out = capsys.readouterr().out
        assert "rule one" in out
        assert "rule two" in out

    def test_clear(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        state.agent_mode.invariants = ["r1", "r2"]
        _handle_invariant_command("clear", "", state)
        assert state.agent_mode.invariants == []
        assert "удалены" in capsys.readouterr().out

    def test_unknown_action(self, capsys):
        from chatbot.main import _handle_invariant_command
        state = _make_state()
        _handle_invariant_command("unknown", "", state)
        assert "Неизвестная подкоманда invariant" in capsys.readouterr().out


# ===========================================================================
# _handle_task_command
# ===========================================================================

class TestHandleTaskCommand:
    def test_new_without_arg(self, capsys):
        from chatbot.main import _handle_task_command
        state = _make_state()
        _handle_task_command("new", "", state, client=None)
        assert "требует описания" in capsys.readouterr().out

    def test_show_no_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("show", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_list_empty(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("list", "", state, client=None)
        assert "не найдено" in capsys.readouterr().out.lower()

    def test_start_no_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("start", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_start_wrong_phase(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)  # phase=EXECUTION
        _handle_task_command("start", "", state, client=None)
        out = capsys.readouterr().out
        assert "уже в фазе" in out

    def test_pause_no_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("pause", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_pause_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        _handle_task_command("pause", "", state, client=None)
        assert "приостановлена" in capsys.readouterr().out

    def test_resume_no_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("resume", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_resume_with_explicit_id(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, tmp_path)
        plan.phase = TaskPhase.PAUSED
        save_task_plan(plan, state.profile_name)
        state.active_task_id = None  # clear it, resume by id
        _handle_task_command("resume", task_id, state, client=None)
        assert state.active_task_id == task_id

    def test_resume_with_invalid_id(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("resume", "nonexistent_id", state, client=None)
        assert "не найдена" in capsys.readouterr().out

    def test_done_no_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("done", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_done_completes_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        from chatbot.task_storage import save_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        plan.phase = TaskPhase.VALIDATION
        save_task_plan(plan, state.profile_name)
        _handle_task_command("done", "", state, client=None)
        assert state.active_task_id is None
        assert "завершена" in capsys.readouterr().out

    def test_fail_no_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("fail", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_fail_with_reason(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, tmp_path)
        _handle_task_command("fail", "API error", state, client=None)
        assert state.active_task_id is None
        assert "FAILED" in capsys.readouterr().out

    def test_load_without_arg(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("load", "", state, client=None)
        assert "требует ID" in capsys.readouterr().out

    def test_load_not_found(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("load", "nonexistent", state, client=None)
        assert "не найдена" in capsys.readouterr().out

    def test_load_found(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, tmp_path)
        state.active_task_id = None
        _handle_task_command("load", task_id, state, client=None)
        assert state.active_task_id == task_id

    def test_delete_without_arg(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("delete", "", state, client=None)
        assert "требует ID" in capsys.readouterr().out

    def test_delete_clears_active_task_id(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, tmp_path)
        _handle_task_command("delete", task_id, state, client=None)
        assert state.active_task_id is None

    def test_delete_not_found(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("delete", "nonexistent", state, client=None)
        assert "не найдена" in capsys.readouterr().out

    def test_unknown_action(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("unknown_cmd", "", state, client=None)
        assert "Неизвестная подкоманда task" in capsys.readouterr().out

    def test_result_no_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_task_id = None
        _handle_task_command("result", "", state, client=None)
        assert "Нет активной задачи" in capsys.readouterr().out

    def test_result_not_found(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _handle_task_command("result", "nonexistent_id", state, client=None)
        assert "не найдена" in capsys.readouterr().out

    def test_result_active_task(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, tmp_path)
        _handle_task_command("result", "", state, client=None)
        out = capsys.readouterr().out
        assert "Результат" in out

    def test_result_by_explicit_id(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, tmp_path)
        state.active_task_id = None
        _handle_task_command("result", task_id, state, client=None)
        out = capsys.readouterr().out
        assert "Результат" in out


# ===========================================================================
# _apply_session_data — ValueError path
# ===========================================================================

class TestApplySessionDataValueError:
    def test_invalid_strategy_kept(self):
        from chatbot.main import _apply_session_data
        state = _make_state()
        state.context_strategy = ContextStrategy.SLIDING_WINDOW
        _apply_session_data({"context_strategy": "invalid_strategy_xyz"}, state)
        assert state.context_strategy == ContextStrategy.SLIDING_WINDOW

    def test_branches_restored(self):
        from chatbot.main import _apply_session_data
        from chatbot.context import create_checkpoint, create_branch
        state = _make_state()
        cp = create_checkpoint([])
        b = create_branch("my-fork", cp)
        data = {
            "branches": [b.model_dump()],
            "active_branch_id": b.branch_id,
        }
        _apply_session_data(data, state)
        assert len(state.branches) == 1
        assert state.branches[0].name == "my-fork"
        assert state.active_branch_id == b.branch_id


# ===========================================================================
# _apply_inline_updates — various branches
# ===========================================================================

class TestApplyInlineUpdatesExtra:
    def test_strategy_valid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"strategy": "sticky_facts"}, state)
        assert state.context_strategy == ContextStrategy.STICKY_FACTS

    def test_strategy_invalid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"strategy": "bogus"}, state)
        assert "Неизвестная стратегия" in capsys.readouterr().out

    def test_showfacts_empty(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"showfacts": True}, state)
        assert "Факты пока не накоплены" in capsys.readouterr().out

    def test_showfacts_with_facts(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        state.sticky_facts = StickyFacts(facts={"lang": "Python"})
        _apply_inline_updates({"showfacts": True}, state)
        assert "lang: Python" in capsys.readouterr().out

    def test_setfact_valid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"setfact": {"key": "lang", "value": "Python"}}, state)
        assert state.sticky_facts.facts.get("lang") == "Python"

    def test_delfact_found(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        state.sticky_facts = StickyFacts(facts={"lang": "Python"})
        _apply_inline_updates({"delfact": "lang"}, state)
        assert "lang" not in state.sticky_facts.facts

    def test_delfact_not_found(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"delfact": "nonexistent"}, state)
        assert "Факт не найден" in capsys.readouterr().out

    def test_checkpoint_creates(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"checkpoint": True}, state)
        assert state.last_checkpoint is not None
        assert "Checkpoint создан" in capsys.readouterr().out

    def test_branch_auto_creates_checkpoint(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        assert state.last_checkpoint is None
        _apply_inline_updates({"branch": "my-fork"}, state)
        assert state.last_checkpoint is not None
        assert len(state.branches) == 1
        assert state.branches[0].name == "my-fork"

    def test_branch_uses_existing_checkpoint(self, capsys):
        from chatbot.main import _apply_inline_updates
        from chatbot.context import create_checkpoint
        state = _make_state()
        state.last_checkpoint = create_checkpoint([])
        _apply_inline_updates({"branch": "fork2"}, state)
        assert len(state.branches) == 1
        out = capsys.readouterr().out
        assert "Checkpoint создан автоматически" not in out

    def test_switch_found(self, capsys):
        from chatbot.main import _apply_inline_updates
        from chatbot.context import create_checkpoint, create_branch
        state = _make_state()
        cp = create_checkpoint([])
        b = create_branch("alpha", cp)
        state.branches.append(b)
        _apply_inline_updates({"switch": "alpha"}, state)
        assert state.active_branch_id == b.branch_id

    def test_switch_not_found(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"switch": "ghost"}, state)
        assert "не найдена" in capsys.readouterr().out

    def test_branches_empty(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"branches": True}, state)
        assert "Веток пока нет" in capsys.readouterr().out

    def test_branches_with_items(self, capsys):
        from chatbot.main import _apply_inline_updates
        from chatbot.context import create_checkpoint, create_branch
        state = _make_state()
        cp = create_checkpoint([])
        b = create_branch("alpha", cp)
        state.branches.append(b)
        _apply_inline_updates({"branches": True}, state)
        assert "alpha" in capsys.readouterr().out

    def test_setpref_valid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"setpref": "lang=Python"}, state)
        assert state.memory.working.user_preferences.get("lang") == "Python"

    def test_setpref_no_equals(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"setpref": "no_equals_sign"}, state)
        assert "ожидается формат key=value" in capsys.readouterr().out

    def test_remember_with_equals(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"remember": "framework=FastAPI"}, state)
        assert "framework" in capsys.readouterr().out

    def test_remember_without_equals(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"remember": "use async everywhere"}, state)
        assert "Решение сохранено" in capsys.readouterr().out

    def test_profile_show(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "show", "arg": ""}}, state)
        assert "Профиль" in capsys.readouterr().out

    def test_profile_list_empty(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "list", "arg": ""}}, state)
        assert "Профилей не найдено" in capsys.readouterr().out

    def test_profile_name(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "name", "arg": "Igor"}}, state)
        assert state.memory.long_term.profile.name == "Igor"

    def test_profile_style_valid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "style", "arg": "tone=formal"}}, state)
        assert state.memory.long_term.profile.style.get("tone") == "formal"

    def test_profile_style_invalid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "style", "arg": "no_equals"}}, state)
        assert "ожидается формат key=value" in capsys.readouterr().out

    def test_profile_format_invalid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "format", "arg": "no_equals"}}, state)
        assert "ожидается формат key=value" in capsys.readouterr().out

    def test_profile_constraint_add(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "constraint", "arg": "add no emojis"}}, state)
        assert "no emojis" in state.memory.long_term.profile.constraints

    def test_profile_constraint_del(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        state.memory.long_term.profile.constraints = ["no emojis"]
        _apply_inline_updates({"profile": {"action": "constraint", "arg": "del no emojis"}}, state)
        assert "no emojis" not in state.memory.long_term.profile.constraints

    def test_profile_constraint_invalid(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "constraint", "arg": ""}}, state)
        assert "ожидается" in capsys.readouterr().out

    def test_profile_name_saves(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "name", "arg": "TestProfile"}}, state)
        assert "сохранён" in capsys.readouterr().out

    def test_profile_name_error(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        with patch("chatbot.main.save_profile", side_effect=Exception("disk full")):
            _apply_inline_updates({"profile": {"action": "name", "arg": "X"}}, state)
        assert "Ошибка сохранения" in capsys.readouterr().out

    def test_profile_unknown_action(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "unknown_cmd", "arg": ""}}, state)
        assert "Неизвестная подкоманда профиля" in capsys.readouterr().out

    def test_memclear(self, capsys):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        state.memory.short_term.messages.append(ChatMessage(role="user", content="hi"))
        _apply_inline_updates({"memclear": "short_term"}, state)
        out = capsys.readouterr().out
        assert "очищена" in out
        assert len(state.memory.short_term.messages) == 0

    def test_memclear_all(self, capsys):
        from chatbot.main import _apply_inline_updates
        from chatbot.memory import WorkingMemory, LongTermMemory
        state = _make_state()
        state.memory.short_term.messages.append(ChatMessage(role="user", content="hi"))
        state.memory.working.set_task("some task")
        state.memory.long_term.add_knowledge("key", "val")
        _apply_inline_updates({"memclear": "all"}, state)
        out = capsys.readouterr().out
        assert "очищена" in out
        assert len(state.memory.short_term.messages) == 0
        assert state.memory.working.current_task is None
        assert len(state.memory.long_term.knowledge_base) == 0

    def test_memsave(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"memsave": "all"}, state)
        assert "сохранена" in capsys.readouterr().out

    def test_memload_not_found(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"memload": "all"}, state)
        out = capsys.readouterr().out
        assert "не найдена" in out

    def test_none_value_skipped(self):
        from chatbot.main import _apply_inline_updates
        state = _make_state()
        original_model = state.model
        _apply_inline_updates({"model": None}, state)
        assert state.model == original_model
