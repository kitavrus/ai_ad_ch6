"""Дополнительное покрытие main.py — вторая волна."""

import time
from unittest.mock import MagicMock, patch

from llm_agent.chatbot.models import (
    ChatMessage,
    ContextStrategy,
    SessionState,
    TaskPhase,
    StepStatus,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> SessionState:
    from llm_agent.chatbot.memory import Memory
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


def _make_plan_and_steps(state, n_steps=2, phase=TaskPhase.EXECUTION):
    from llm_agent.chatbot.task_storage import save_task_plan, save_task_step
    from llm_agent.chatbot.models import TaskPlan, TaskStep
    from datetime import datetime
    import uuid

    now = datetime.utcnow().isoformat()
    task_id = uuid.uuid4().hex
    plan = TaskPlan(
        task_id=task_id,
        profile_name=state.profile_name,
        name="Test task",
        description="desc",
        phase=phase,
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
# _create_task_plan
# ===========================================================================

class TestCreateTaskPlan:
    def test_with_prebuilt_steps(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        steps = [
            {"title": "Step 1", "description": "do it"},
            {"title": "Step 2", "description": "check it"},
        ]
        plan = _create_task_plan("My task", state, client=None, steps=steps)
        assert plan is not None
        assert state.active_task_id == plan.task_id
        out = capsys.readouterr().out
        assert "План создан" in out

    def test_llm_generates_steps(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='[{"title": "Step A", "description": "desc A"}]'
            ))]
        )
        plan = _create_task_plan("My task", state, client=mock_client)
        assert plan is not None
        assert state.active_task_id == plan.task_id

    def test_llm_error_returns_none(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        plan = _create_task_plan("My task", state, client=mock_client)
        assert plan is None
        assert "Ошибка LLM" in capsys.readouterr().out

    def test_unparseable_llm_response_returns_none(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="no valid plan here at all"))]
        )
        plan = _create_task_plan("My task", state, client=mock_client)
        assert plan is None

    def test_empty_steps_returns_none(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _create_task_plan
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan = _create_task_plan("My task", state, client=None, steps=[])
        assert plan is None
        assert "пуст" in capsys.readouterr().out


# ===========================================================================
# _handle_step_subcommand — step not found
# ===========================================================================

class TestHandleStepNotFound:
    def test_step_not_found(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_step_subcommand
        from llm_agent.chatbot.task_storage import save_task_plan
        from llm_agent.chatbot.models import TaskPlan
        from datetime import datetime
        import uuid

        monkeypatch.chdir(tmp_path)
        state = _make_state()
        now = datetime.utcnow().isoformat()
        task_id = uuid.uuid4().hex
        plan = TaskPlan(
            task_id=task_id, name="T", description="d",
            phase=TaskPhase.EXECUTION, total_steps=1,
            current_step_index=0, created_at=now, updated_at=now,
        )
        save_task_plan(plan, state.profile_name)
        state.active_task_id = task_id
        # No step file saved → step not found
        _handle_step_subcommand("done", state)
        assert "не найден" in capsys.readouterr().out


# ===========================================================================
# _collect_plan_clarifications
# ===========================================================================

class TestCollectPlanClarifications:
    def test_no_questions_returns_early(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _collect_plan_clarifications
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _collect_plan_clarifications("No questions block here", state)
        # No output, no crash
        assert capsys.readouterr().out == ""

    def test_questions_saved_to_active_plan(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _collect_plan_clarifications
        from llm_agent.chatbot.context import parse_plan_questions
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state)

        text = "**Questions:**\n1. What language?\n2. What framework?\n**End Questions**"
        # Determine how many questions the parser finds and supply that many answers
        n_questions = len(parse_plan_questions(text))
        answers = ["Python", "FastAPI", "none"][:n_questions]
        with patch("builtins.input", side_effect=answers):
            _collect_plan_clarifications(text, state)

        from llm_agent.chatbot.task_storage import load_task_plan
        saved = load_task_plan(state.active_task_id, state.profile_name)
        assert len(saved.clarifications) >= 2

    def test_questions_saved_to_working_memory_when_no_active_task(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _collect_plan_clarifications
        from llm_agent.chatbot.context import parse_plan_questions
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        # No active task
        text = "**Questions:**\n1. What language?\n**End Questions**"
        n_questions = len(parse_plan_questions(text))
        answers = ["Python", "none"][:n_questions]
        with patch("builtins.input", side_effect=answers):
            _collect_plan_clarifications(text, state)
        out = capsys.readouterr().out
        assert "рабочую память" in out

    def test_eoferror_breaks_input_loop(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _collect_plan_clarifications
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        text = "**Questions:**\n1. What language?\n**End Questions**"
        with patch("builtins.input", side_effect=EOFError):
            _collect_plan_clarifications(text, state)
        # No crash, no clarifications saved
        assert state.memory.working.user_preferences == {}


# ===========================================================================
# _call_llm_for_builder_step
# ===========================================================================

class TestCallLlmForBuilderStep:
    def test_with_done_steps(self, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _call_llm_for_builder_step
        from llm_agent.chatbot.task_storage import save_task_step

        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, n_steps=2)

        # Mark step 1 as done
        from llm_agent.chatbot.task_storage import load_task_step
        step1 = load_task_step(task_id, 1, state.profile_name)
        step1.status = StepStatus.DONE
        save_task_step(step1, state.profile_name)

        step2 = load_task_step(task_id, 2, state.profile_name)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="**Response:**\nDone step 2"))]
        )

        result = _call_llm_for_builder_step(step2, plan, state, mock_client)
        assert "Done step 2" in result

    def test_no_done_steps(self, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _call_llm_for_builder_step
        from llm_agent.chatbot.task_storage import load_task_step

        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, n_steps=1)
        step = load_task_step(task_id, 1, state.profile_name)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="**Response:**\nResult"))]
        )

        # prev_lines = "(нет выполненных шагов)" branch
        result = _call_llm_for_builder_step(step, plan, state, mock_client)
        assert isinstance(result, str)

    def test_llm_error_returns_empty(self, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _call_llm_for_builder_step
        from llm_agent.chatbot.task_storage import load_task_step

        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, n_steps=1)
        step = load_task_step(task_id, 1, state.profile_name)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("timeout")

        result = _call_llm_for_builder_step(step, plan, state, mock_client)
        assert result == ""

    def test_with_clarifications(self, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _call_llm_for_builder_step
        from llm_agent.chatbot.task_storage import load_task_step, save_task_plan

        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, n_steps=1)
        plan.clarifications = [{"question": "Q?", "answer": "A"}]
        save_task_plan(plan, state.profile_name)
        step = load_task_step(task_id, 1, state.profile_name)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="**Response:**\nok"))]
        )
        result = _call_llm_for_builder_step(step, plan, state, mock_client)
        # verify clarifications were included in prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        sys_msg = messages[0]["content"]
        assert "Q?" in sys_msg or "Clarifications" in sys_msg


# ===========================================================================
# _run_plan_builder
# ===========================================================================

class TestRunPlanBuilder:
    def test_no_active_task(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _run_plan_builder
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _run_plan_builder(state, client=None)
        assert "нет активной задачи" in capsys.readouterr().out.lower()

    def test_plan_not_found(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _run_plan_builder
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.active_task_id = "nonexistent_id"
        _run_plan_builder(state, client=None)
        assert "не найден" in capsys.readouterr().out

    def test_all_steps_already_done(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _run_plan_builder
        from llm_agent.chatbot.task_storage import load_task_step, save_task_step
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, n_steps=1)
        step = load_task_step(task_id, 1, state.profile_name)
        step.status = StepStatus.DONE
        save_task_step(step, state.profile_name)
        _run_plan_builder(state, client=None)
        assert "уже выполнены" in capsys.readouterr().out

    def test_successful_run_marks_plan_done(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _run_plan_builder
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state, n_steps=1)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="**Response:**\nDone"))]
        )

        with patch("chatbot.main.validate_draft_against_invariants", return_value=(True, "")):
            _run_plan_builder(state, client=mock_client)

        from llm_agent.chatbot.task_storage import load_task_plan
        saved = load_task_plan(task_id, state.profile_name)
        assert saved.phase == TaskPhase.DONE
        assert "ПЛАН ВЫПОЛНЕН" in capsys.readouterr().out


# ===========================================================================
# _print_draft_plan + _extract_task_description + _kick_off_plan_dialog
# ===========================================================================

class TestPrintDraftPlan:
    def test_prints_all_steps(self, capsys):
        from llm_agent.chatbot.main import _print_draft_plan
        steps = [
            {"title": "Step A", "description": "do A"},
            {"title": "Step B", "description": "do B"},
        ]
        _print_draft_plan(steps)
        out = capsys.readouterr().out
        assert "Step A" in out
        assert "Step B" in out


class TestExtractTaskDescription:
    def test_skips_planning_message(self):
        from llm_agent.chatbot.main import _extract_task_description
        messages = [
            ChatMessage(role="user", content="Начинаем планирование."),
            ChatMessage(role="user", content="Build REST API"),
        ]
        result = _extract_task_description(messages)
        assert result == "Build REST API"

    def test_returns_fallback_when_empty(self):
        from llm_agent.chatbot.main import _extract_task_description
        result = _extract_task_description([])
        assert result == "Задача из диалога планирования"


class TestKickOffPlanDialog:
    def test_calls_llm_with_description(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _kick_off_plan_dialog
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="First question?"))]
        )
        _kick_off_plan_dialog(state, mock_client, description="Build API")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Build API" in user_msg["content"]

    def test_llm_error_no_crash(self, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _kick_off_plan_dialog
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("err")
        _kick_off_plan_dialog(state, mock_client)
        assert len(state.messages) == 0


# ===========================================================================
# _handle_plan_dialog_message
# ===========================================================================

class TestHandlePlanDialogMessage:
    def test_draft_plan_triggers_confirming(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_plan_dialog_message
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.plan_dialog_state = "active"
        state.plan_draft_description = "Build API"

        draft_content = (
            "Let me plan this.\n"
            "**Draft Plan:**\n"
            "```json\n"
            '[{"title": "Step 1", "description": "do it"}]\n'
            "```\n"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=draft_content))]
        )
        _handle_plan_dialog_message("Please create a plan", state, mock_client)
        assert state.plan_dialog_state == "confirming"
        assert len(state.plan_draft_steps) > 0

    def test_no_draft_stays_active(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_plan_dialog_message
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.plan_dialog_state = "active"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="**Response:**\nWhat is the scope?"))]
        )
        _handle_plan_dialog_message("Tell me about the task", state, mock_client)
        assert state.plan_dialog_state == "active"


# ===========================================================================
# _handle_task_command — show/list/start/step
# ===========================================================================

class TestHandleTaskCommandExtra:
    def test_show_active_task(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state)
        _handle_task_command("show", "", state, client=None)
        out = capsys.readouterr().out
        assert "Test task" in out

    def test_list_with_tasks(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, task_id = _make_plan_and_steps(state)
        _handle_task_command("list", "", state, client=None)
        out = capsys.readouterr().out
        assert "Test task" in out
        assert "активная" in out

    def test_start_planning_phase(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, phase=TaskPhase.PLANNING)
        _handle_task_command("start", "", state, client=None)
        out = capsys.readouterr().out
        assert "запущена" in out

    def test_step_delegates(self, monkeypatch, tmp_path, capsys):
        from llm_agent.chatbot.main import _handle_task_command
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        plan, _ = _make_plan_and_steps(state, n_steps=3)
        _handle_task_command("step", "done", state, client=None)
        out = capsys.readouterr().out
        assert "завершён" in out


# ===========================================================================
# _apply_inline_updates — memshow, memstats, memload found, settask, task/plan/invariant
# ===========================================================================

class TestApplyInlineUpdatesExtra2:
    def test_memshow(self, capsys):
        from llm_agent.chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"memshow": "all"}, state)
        out = capsys.readouterr().out
        assert "Долговременная" in out
        assert "Краткосрочная" in out

    def test_memstats(self, capsys, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"memstats": True}, state)
        assert "Статистика" in capsys.readouterr().out

    def test_memload_working_found(self, capsys, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _apply_inline_updates
        from llm_agent.chatbot.memory_storage import save_working_memory
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        # Save under "current" (no current_task → task_name="current")
        save_working_memory(
            {"current_task": None, "task_status": "saved", "updated_at": ""},
            task_name="current",
            profile_name=state.profile_name,
        )
        # current_task is None → load_working_memory("current") will find the file
        _apply_inline_updates({"memload": "all"}, state)
        out = capsys.readouterr().out
        assert "загружена" in out.lower()

    def test_settask_with_client(self, capsys, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='[{"title": "S1", "description": "d"}]'
            ))]
        )
        _apply_inline_updates({"settask": "Write tests"}, state, client=mock_client)
        assert state.memory.working.current_task == "Write tests"

    def test_task_command_dispatched(self, capsys, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"task": {"action": "list", "arg": ""}}, state, client=None)
        assert "не найдено" in capsys.readouterr().out.lower()

    def test_plan_command_dispatched(self, capsys):
        from llm_agent.chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"plan": {"action": "status", "arg": ""}}, state)
        assert "Plan mode" in capsys.readouterr().out

    def test_invariant_command_dispatched(self, capsys):
        from llm_agent.chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"invariant": {"action": "add", "arg": "reply in Russian"}}, state)
        assert "reply in Russian" in state.agent_mode.invariants

    def test_profile_list_nonempty(self, capsys, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _apply_inline_updates
        from llm_agent.chatbot.memory_storage import save_profile
        from llm_agent.chatbot.models import UserProfile
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        save_profile(UserProfile(name="Alice"), "Alice")
        _apply_inline_updates({"profile": {"action": "list", "arg": ""}}, state)
        out = capsys.readouterr().out
        assert "alice" in out.lower() or "Alice" in out

    def test_profile_load_not_found(self, capsys, monkeypatch, tmp_path):
        from llm_agent.chatbot.main import _apply_inline_updates
        monkeypatch.chdir(tmp_path)
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "load", "arg": "Ghost"}}, state)
        assert "не найден" in capsys.readouterr().out

    def test_profile_format_valid(self, capsys):
        from llm_agent.chatbot.main import _apply_inline_updates
        state = _make_state()
        _apply_inline_updates({"profile": {"action": "format", "arg": "output=markdown"}}, state)
        assert state.memory.long_term.profile.format.get("output") == "markdown"


# ===========================================================================
# _append_message / _get_active_messages — branching paths
# ===========================================================================

class TestAppendAndGetMessages:
    def _make_branching_state(self):
        from llm_agent.chatbot.context import create_checkpoint, create_branch
        state = _make_state()
        state.context_strategy = ContextStrategy.BRANCHING
        cp = create_checkpoint([])
        b = create_branch("test-branch", cp)
        state.branches.append(b)
        state.active_branch_id = b.branch_id
        return state, b

    def test_append_message_to_branch(self):
        from llm_agent.chatbot.main import _append_message
        state, b = self._make_branching_state()
        msg = ChatMessage(role="user", content="hello branch")
        _append_message(state, msg)
        assert b.messages[-1].content == "hello branch"
        assert msg not in state.messages

    def test_append_message_no_branch_found(self):
        from llm_agent.chatbot.main import _append_message
        state = _make_state()
        state.context_strategy = ContextStrategy.BRANCHING
        state.active_branch_id = "nonexistent"
        msg = ChatMessage(role="user", content="fallback")
        _append_message(state, msg)
        assert state.messages[-1].content == "fallback"

    def test_append_message_non_branching(self):
        from llm_agent.chatbot.main import _append_message
        state = _make_state()
        state.context_strategy = ContextStrategy.SLIDING_WINDOW
        msg = ChatMessage(role="user", content="sliding")
        _append_message(state, msg)
        assert state.messages[-1].content == "sliding"

    def test_get_active_messages_from_branch(self):
        from llm_agent.chatbot.main import _get_active_messages
        state, b = self._make_branching_state()
        b.messages.append(ChatMessage(role="user", content="branch msg"))
        result = _get_active_messages(state)
        assert result is b.messages

    def test_get_active_messages_no_branch(self):
        from llm_agent.chatbot.main import _get_active_messages
        state = _make_state()
        state.context_strategy = ContextStrategy.BRANCHING
        state.active_branch_id = "nonexistent"
        state.messages.append(ChatMessage(role="user", content="main"))
        result = _get_active_messages(state)
        assert result is state.messages

    def test_get_active_messages_non_branching(self):
        from llm_agent.chatbot.main import _get_active_messages
        state = _make_state()
        result = _get_active_messages(state)
        assert result is state.messages
