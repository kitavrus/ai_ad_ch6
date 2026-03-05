"""Тесты для Stateful AI Agent: модели, CLI, context-функции."""

import pytest
from unittest.mock import MagicMock, patch

from chatbot.models import AgentMode, SessionState
from chatbot.cli import parse_inline_command
from chatbot.context import (
    build_agent_system_prompt,
    build_plan_dialog_prompt,
    parse_agent_output,
    parse_draft_plan_block,
    parse_plan_questions,
    validate_draft_against_invariants,
)


# ---------------------------------------------------------------------------
# AgentMode model
# ---------------------------------------------------------------------------


class TestAgentMode:
    def test_defaults(self):
        m = AgentMode()
        assert m.enabled is False
        assert m.invariants == []
        assert m.max_retries == 3

    def test_enable(self):
        m = AgentMode(enabled=True, invariants=["no SQL"], max_retries=5)
        assert m.enabled is True
        assert m.invariants == ["no SQL"]
        assert m.max_retries == 5

    def test_max_retries_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentMode(max_retries=0)
        with pytest.raises(ValidationError):
            AgentMode(max_retries=11)

    def test_session_state_has_agent_mode(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert isinstance(state.agent_mode, AgentMode)
        assert state.agent_mode.enabled is False


# ---------------------------------------------------------------------------
# CLI /plan parsing
# ---------------------------------------------------------------------------


class TestAgentCommandParsing:
    def test_agent_on(self):
        result = parse_inline_command("/plan on")
        assert result == {"plan": {"action": "on", "arg": ""}}

    def test_agent_off(self):
        result = parse_inline_command("/plan off")
        assert result == {"plan": {"action": "off", "arg": ""}}

    def test_agent_status(self):
        result = parse_inline_command("/plan status")
        assert result == {"plan": {"action": "status", "arg": ""}}

    def test_agent_retries(self):
        result = parse_inline_command("/plan retries 5")
        assert result == {"plan": {"action": "retries", "arg": "5"}}

    def test_agent_bare(self):
        result = parse_inline_command("/plan")
        assert result == {"plan": {"action": "status", "arg": ""}}


# ---------------------------------------------------------------------------
# CLI /invariant parsing
# ---------------------------------------------------------------------------


class TestInvariantCommandParsing:
    def test_invariant_add(self):
        result = parse_inline_command("/invariant add no SQL queries allowed")
        assert result == {"invariant": {"action": "add", "arg": "no SQL queries allowed"}}

    def test_invariant_del(self):
        result = parse_inline_command("/invariant del 2")
        assert result == {"invariant": {"action": "del", "arg": "2"}}

    def test_invariant_list(self):
        result = parse_inline_command("/invariant list")
        assert result == {"invariant": {"action": "list", "arg": ""}}

    def test_invariant_clear(self):
        result = parse_inline_command("/invariant clear")
        assert result == {"invariant": {"action": "clear", "arg": ""}}

    def test_invariant_bare(self):
        result = parse_inline_command("/invariant")
        assert result == {"invariant": {"action": "list", "arg": ""}}


# ---------------------------------------------------------------------------
# build_agent_system_prompt
# ---------------------------------------------------------------------------


class TestBuildAgentSystemPrompt:
    def test_contains_required_sections(self):
        prompt = build_agent_system_prompt("tone=formal", "task: test", ["no SQL"])
        assert "# ROLE" in prompt
        assert "# CONTEXT VARIABLES" in prompt
        assert "# INSTRUCTIONS" in prompt
        assert "**Response:**" in prompt
        assert "**State Update:**" in prompt

    def test_invariants_injected(self):
        prompt = build_agent_system_prompt("", "", ["no SQL", "respond in Russian"])
        assert "no SQL" in prompt
        assert "respond in Russian" in prompt

    def test_profile_injected(self):
        prompt = build_agent_system_prompt("tone=formal; verbosity=concise", "", [])
        assert "tone=formal" in prompt

    def test_state_injected(self):
        prompt = build_agent_system_prompt("", "task: build API; status: planning", [])
        assert "task: build API" in prompt

    def test_empty_invariants_placeholder(self):
        prompt = build_agent_system_prompt("", "", [])
        assert "(не заданы)" in prompt

    def test_empty_profile_placeholder(self):
        prompt = build_agent_system_prompt("", "", [])
        assert "(не задан)" in prompt


# ---------------------------------------------------------------------------
# parse_agent_output
# ---------------------------------------------------------------------------


class TestParseAgentOutput:
    def test_parses_response_block(self):
        text = "**Response:**\nHere is the answer.\n\n**State Update:**\n(none)"
        response, updates = parse_agent_output(text)
        assert response == "Here is the answer."
        assert updates == {}

    def test_parses_state_update(self):
        text = "**Response:**\nDone.\n\n**State Update:**\nlanguage: Python\nstatus: done"
        response, updates = parse_agent_output(text)
        assert response == "Done."
        assert updates == {"language": "Python", "status": "done"}

    def test_no_blocks_returns_full_text(self):
        text = "Just a plain response with no special blocks."
        response, updates = parse_agent_output(text)
        assert response == text
        assert updates == {}

    def test_state_update_none_keyword(self):
        text = "**Response:**\nOK\n\n**State Update:**\n(none)"
        _, updates = parse_agent_output(text)
        assert updates == {}

    def test_state_update_with_dashes(self):
        text = "**Response:**\nResult.\n\n**State Update:**\n- key1: value1\n- key2: value2"
        _, updates = parse_agent_output(text)
        assert updates["key1"] == "value1"
        assert updates["key2"] == "value2"

    def test_case_insensitive_blocks(self):
        text = "**RESPONSE:**\nAnswer.\n\n**STATE UPDATE:**\n(none)"
        response, _ = parse_agent_output(text)
        assert response == "Answer."


# ---------------------------------------------------------------------------
# validate_draft_against_invariants
# ---------------------------------------------------------------------------


class TestValidateDraftAgainstInvariants:
    def _make_client(self, reply: str) -> MagicMock:
        client = MagicMock()
        msg = MagicMock()
        msg.content = reply
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        return client

    def test_pass(self):
        client = self._make_client("PASS")
        passed, reason = validate_draft_against_invariants(client, "gpt-4", "Hello", ["be polite"])
        assert passed is True
        assert reason == ""

    def test_fail(self):
        client = self._make_client("FAIL: contains SQL query")
        passed, reason = validate_draft_against_invariants(client, "gpt-4", "SELECT * FROM users", ["no SQL"])
        assert passed is False
        assert "SQL" in reason

    def test_empty_invariants_always_pass(self):
        client = MagicMock()
        passed, reason = validate_draft_against_invariants(client, "gpt-4", "anything", [])
        assert passed is True
        client.chat.completions.create.assert_not_called()

    def test_api_error_returns_pass(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("network error")
        passed, reason = validate_draft_against_invariants(client, "gpt-4", "text", ["inv"])
        assert passed is True


# ---------------------------------------------------------------------------
# parse_plan_questions
# ---------------------------------------------------------------------------


class TestParsePlanQuestions:
    def test_numbered_questions(self):
        text = (
            "**Response:**\nHere is my plan.\n\n"
            "**Questions:**\n1. What is the deadline?\n2. Which stack?\n\n"
            "**State Update:**\n(none)"
        )
        questions = parse_plan_questions(text)
        assert questions == ["What is the deadline?", "Which stack?"]

    def test_no_questions_block(self):
        text = "**Response:**\nDone.\n\n**State Update:**\n(none)"
        assert parse_plan_questions(text) == []

    def test_empty_questions_block(self):
        text = "**Response:**\nOK\n\n**Questions:**\n\n**State Update:**\n(none)"
        assert parse_plan_questions(text) == []

    def test_dash_prefixed_questions(self):
        text = "**Response:**\nPlan.\n\n**Questions:**\n- Budget?\n- Team size?\n\n**State Update:**\n(none)"
        questions = parse_plan_questions(text)
        assert questions == ["Budget?", "Team size?"]

    def test_questions_without_state_update(self):
        text = "**Response:**\nPlan.\n\n**Questions:**\n1. Timeline?\n2. Budget?"
        questions = parse_plan_questions(text)
        assert questions == ["Timeline?", "Budget?"]

    def test_parse_agent_output_skips_questions_in_response(self):
        """Response блок не должен захватывать Questions."""
        text = (
            "**Response:**\nMy answer.\n\n"
            "**Questions:**\n1. Q1?\n\n"
            "**State Update:**\nkey: val"
        )
        response, updates = parse_agent_output(text)
        assert response == "My answer."
        assert updates == {"key": "val"}

    def test_taskplan_has_clarifications_field(self):
        from chatbot.models import TaskPlan
        from datetime import datetime
        plan = TaskPlan(
            task_id="t1",
            name="Test",
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        assert plan.clarifications == []
        plan.clarifications.append({"question": "Q?", "answer": "A"})
        assert plan.clarifications[0]["answer"] == "A"


# ---------------------------------------------------------------------------
# Plan Builder
# ---------------------------------------------------------------------------


class TestPlanBuilderCLI:
    def test_plan_builder_cli_parsing(self):
        result = parse_inline_command("/plan builder")
        assert result is not None
        assert "plan" in result
        assert result["plan"]["action"] == "builder"


class TestGenerateClarificationQuestion:
    def test_returns_question_from_llm(self):
        from chatbot.context import generate_clarification_question

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "What technology stack should be used?"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        result = generate_clarification_question(
            mock_client, "gpt-4", "Design API", "Create REST endpoints", "no SQL violation"
        )
        assert result == "What technology stack should be used?"
        mock_client.chat.completions.create.assert_called_once()

    def test_fallback_on_api_error(self):
        from chatbot.context import generate_clarification_question

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = generate_clarification_question(
            mock_client, "gpt-4", "Step Title", "Step description", "some violation"
        )
        assert "Step Title" in result or "some violation" in result


class TestRunPlanBuilderNoActiveTask:
    def test_no_active_task(self, capsys):
        from chatbot.main import _run_plan_builder

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        state.active_task_id = None
        mock_client = MagicMock()

        _run_plan_builder(state, mock_client)

        captured = capsys.readouterr()
        assert "нет активной задачи" in captured.out.lower()


class TestRunPlanBuilderAllDone:
    def test_all_steps_done(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _run_plan_builder
        from chatbot.models import StepStatus, TaskPlan, TaskStep, TaskPhase
        from datetime import datetime

        monkeypatch.chdir(tmp_path)

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.active_task_id = "task-001"

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-001",
            profile_name="test",
            name="Test Plan",
            created_at=now,
            updated_at=now,
        )

        step = TaskStep(
            step_id="s1",
            task_id="task-001",
            index=1,
            title="Step 1",
            status=StepStatus.DONE,
            created_at=now,
            completed_at=now,
        )

        with patch("chatbot.main.load_task_plan", return_value=plan), \
             patch("chatbot.main.load_all_steps", return_value=[step]):
            _run_plan_builder(state, MagicMock())

        captured = capsys.readouterr()
        assert "все шаги уже выполнены" in captured.out.lower()


class TestExecuteBuilderStepPassFirstTry:
    def test_pass_first_try(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _execute_builder_step
        from chatbot.models import StepStatus, TaskPlan, TaskStep, TaskPhase
        from datetime import datetime

        monkeypatch.chdir(tmp_path)

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.active_task_id = "task-001"
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}
        state.agent_mode.invariants = []

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-001",
            profile_name="test",
            name="Test Plan",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s1",
            task_id="task-001",
            index=1,
            title="Implement endpoint",
            status=StepStatus.PENDING,
            created_at=now,
        )

        with patch("chatbot.main.load_all_steps", return_value=[step]), \
             patch("chatbot.main.save_task_step") as mock_save, \
             patch("chatbot.main._call_llm_for_builder_step", return_value="Done result"):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is True
        assert step.status == StepStatus.DONE
        mock_save.assert_called_once()
        captured = capsys.readouterr()
        assert "Done result" in captured.out


class TestExecuteBuilderStepRetryThenPass:
    def test_retry_then_pass(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _execute_builder_step
        from chatbot.models import StepStatus, TaskPlan, TaskStep
        from datetime import datetime

        monkeypatch.chdir(tmp_path)

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.active_task_id = "task-001"
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}
        state.agent_mode.invariants = ["no SQL"]

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-001",
            profile_name="test",
            name="Test Plan",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s1",
            task_id="task-001",
            index=1,
            title="Implement endpoint",
            status=StepStatus.PENDING,
            created_at=now,
        )

        # First call fails validation, second passes
        validate_results = [(False, "uses SQL"), (True, "")]

        with patch("chatbot.main.load_all_steps", return_value=[step]), \
             patch("chatbot.main.save_task_step"), \
             patch("chatbot.main.save_task_plan"), \
             patch("chatbot.main._call_llm_for_builder_step", return_value="Good result"), \
             patch("chatbot.main.validate_draft_against_invariants",
                   side_effect=validate_results):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is True
        assert step.status == StepStatus.DONE
        captured = capsys.readouterr()
        assert "Good result" in captured.out


class TestExecuteBuilderStepAsksClarification:
    def test_asks_clarification_after_3_fails(self, capsys, monkeypatch, tmp_path):
        from chatbot.main import _execute_builder_step
        from chatbot.models import StepStatus, TaskPlan, TaskStep
        from datetime import datetime

        monkeypatch.chdir(tmp_path)

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.active_task_id = "task-001"
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}
        state.agent_mode.invariants = ["no SQL"]

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-001",
            profile_name="test",
            name="Test Plan",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s1",
            task_id="task-001",
            index=1,
            title="Implement endpoint",
            status=StepStatus.PENDING,
            created_at=now,
        )

        # First 3 calls fail, then pass after clarification
        validate_results = [
            (False, "uses SQL"),
            (False, "uses SQL"),
            (False, "uses SQL"),
            (True, ""),
        ]

        inputs = iter(["Use ORM instead of raw SQL"])

        with patch("chatbot.main.load_all_steps", return_value=[step]), \
             patch("chatbot.main.save_task_step"), \
             patch("chatbot.main.save_task_plan"), \
             patch("chatbot.main._call_llm_for_builder_step", return_value="ORM result"), \
             patch("chatbot.main.validate_draft_against_invariants",
                   side_effect=validate_results), \
             patch("chatbot.main.generate_clarification_question",
                   return_value="How should data be persisted without SQL?"), \
             patch("builtins.input", side_effect=inputs):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is True
        captured = capsys.readouterr()
        assert "How should data be persisted without SQL?" in captured.out
        # clarification saved to plan
        assert any(c["answer"] == "Use ORM instead of raw SQL" for c in plan.clarifications)


# ---------------------------------------------------------------------------
# Plan Dialog: parse_draft_plan_block
# ---------------------------------------------------------------------------


class TestParseDraftPlanBlock:
    def test_json_block_extracted(self):
        text = (
            "**Response:**\nHere is my plan.\n\n"
            '**Draft Plan:**\n[{"title": "Step 1", "description": "Do X"}]\n'
        )
        block = parse_draft_plan_block(text)
        assert block is not None
        assert "Step 1" in block

    def test_no_block_returns_none(self):
        text = "**Response:**\nStill gathering information.\n\n**State Update:**\n(none)"
        assert parse_draft_plan_block(text) is None

    def test_empty_block_returns_none(self):
        text = "**Draft Plan:**\n\n**Response:**\nOK"
        result = parse_draft_plan_block(text)
        assert result is None

    def test_multiline_json_extracted(self):
        text = (
            "**Response:**\nReady.\n\n"
            "**Draft Plan:**\n"
            '[\n  {"title": "A", "description": "a"},\n  {"title": "B", "description": "b"}\n]\n'
        )
        block = parse_draft_plan_block(text)
        assert block is not None
        assert '"title"' in block


# ---------------------------------------------------------------------------
# Plan Dialog: build_plan_dialog_prompt
# ---------------------------------------------------------------------------


class TestBuildPlanDialogPrompt:
    def test_contains_draft_plan(self):
        prompt = build_plan_dialog_prompt([])
        assert "Draft Plan" in prompt

    def test_contains_role_section(self):
        prompt = build_plan_dialog_prompt([])
        assert "# ROLE" in prompt

    def test_invariants_injected(self):
        prompt = build_plan_dialog_prompt(["no SQL", "respond in Russian"])
        assert "no SQL" in prompt
        assert "respond in Russian" in prompt

    def test_empty_invariants_placeholder(self):
        prompt = build_plan_dialog_prompt([])
        assert "(не заданы)" in prompt


# ---------------------------------------------------------------------------
# Plan Dialog: _confirm_and_create_tasks
# ---------------------------------------------------------------------------


class TestConfirmAndCreateTasks:
    def _make_state(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.plan_dialog_state = "confirming"
        state.plan_draft_steps = [{"title": "Step 1", "description": "Do X"}]
        state.plan_draft_description = "Build REST API"
        return state

    def test_affirmative_creates_tasks(self, monkeypatch, tmp_path):
        from chatbot.main import _confirm_and_create_tasks

        state = self._make_state()
        monkeypatch.chdir(tmp_path)

        with patch("chatbot.main._create_task_plan") as mock_create:
            mock_create.return_value = None
            _confirm_and_create_tasks("да", state, MagicMock())

        mock_create.assert_called_once()
        assert state.plan_dialog_state is None
        assert state.plan_draft_steps == []
        assert state.plan_draft_description == ""

    def test_negative_returns_to_active(self, monkeypatch, tmp_path):
        from chatbot.main import _confirm_and_create_tasks

        state = self._make_state()
        monkeypatch.chdir(tmp_path)

        with patch("chatbot.main._handle_plan_dialog_message") as mock_handle:
            _confirm_and_create_tasks("нет", state, MagicMock())

        assert state.plan_dialog_state == "active"
        mock_handle.assert_called_once()

    def test_yes_english_creates_tasks(self, monkeypatch, tmp_path):
        from chatbot.main import _confirm_and_create_tasks

        state = self._make_state()
        monkeypatch.chdir(tmp_path)

        with patch("chatbot.main._create_task_plan") as mock_create:
            mock_create.return_value = None
            _confirm_and_create_tasks("yes", state, MagicMock())

        mock_create.assert_called_once()
        assert state.plan_dialog_state is None


# ---------------------------------------------------------------------------
# SessionState plan dialog fields
# ---------------------------------------------------------------------------


class TestSessionStatePlanDialogFields:
    def test_default_plan_dialog_fields(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert state.plan_dialog_state is None
        assert state.plan_draft_steps == []
        assert state.plan_draft_description == ""

    def test_fields_assignable(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        state.plan_dialog_state = "active"
        state.plan_draft_steps = [{"title": "T", "description": "D"}]
        state.plan_draft_description = "Build API"
        assert state.plan_dialog_state == "active"
        assert len(state.plan_draft_steps) == 1
        assert state.plan_draft_description == "Build API"


# ---------------------------------------------------------------------------
# TestHandleAgentCommandOn — /plan on sets awaiting_task
# ---------------------------------------------------------------------------


class TestHandleAgentCommandOn:
    def _make_state(self):
        return SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    def test_plan_on_sets_awaiting_task(self, capsys):
        from chatbot.main import _handle_agent_command

        state = self._make_state()
        _handle_agent_command("on", "", state, client=None)

        assert state.plan_dialog_state == "awaiting_task"
        assert state.agent_mode.enabled is True
        assert state.plan_draft_steps == []
        assert state.plan_draft_description == ""

    def test_plan_on_prints_prompt(self, capsys):
        from chatbot.main import _handle_agent_command

        state = self._make_state()
        _handle_agent_command("on", "", state, client=None)

        out = capsys.readouterr().out
        assert "Plan mode: ON" in out
        assert "Введите описание задачи" in out

    def test_plan_on_does_not_call_kick_off(self, capsys):
        from chatbot.main import _handle_agent_command

        state = self._make_state()
        mock_client = MagicMock()
        _handle_agent_command("on", "", state, client=mock_client)

        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# TestHandlePlanAwaitingTask
# ---------------------------------------------------------------------------


class TestHandlePlanAwaitingTask:
    def _make_state(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        state.plan_dialog_state = "awaiting_task"
        return state

    def test_empty_input_stays_awaiting_task(self, capsys):
        from chatbot.main import _handle_plan_awaiting_task

        state = self._make_state()
        _handle_plan_awaiting_task("", state)

        assert state.plan_dialog_state == "awaiting_task"
        assert state.plan_draft_description == ""
        out = capsys.readouterr().out
        assert "Введите описание задачи" in out

    def test_whitespace_only_stays_awaiting_task(self, capsys):
        from chatbot.main import _handle_plan_awaiting_task

        state = self._make_state()
        _handle_plan_awaiting_task("   ", state)

        assert state.plan_dialog_state == "awaiting_task"

    def test_nonempty_input_sets_description(self, capsys):
        from chatbot.main import _handle_plan_awaiting_task

        state = self._make_state()
        _handle_plan_awaiting_task("Разработать REST API", state)

        assert state.plan_draft_description == "Разработать REST API"
        assert state.plan_dialog_state == "awaiting_invariants"

    def test_nonempty_input_prints_invariants_prompt(self, capsys):
        from chatbot.main import _handle_plan_awaiting_task

        state = self._make_state()
        _handle_plan_awaiting_task("Some task", state)

        out = capsys.readouterr().out
        assert "инварианты" in out.lower() or "Хотите" in out


# ---------------------------------------------------------------------------
# TestHandlePlanAwaitingInvariants
# ---------------------------------------------------------------------------


class TestHandlePlanAwaitingInvariants:
    def _make_state(self):
        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        state.plan_dialog_state = "awaiting_invariants"
        state.plan_draft_description = "Разработать REST API"
        return state

    def test_no_transitions_to_active_and_kicks_off(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="First question?"))]
        )

        _handle_plan_awaiting_invariants("нет", state, mock_client)

        assert state.plan_dialog_state == "active"
        mock_client.chat.completions.create.assert_called_once()

    def test_skip_english_transitions_to_active(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="First question?"))]
        )

        _handle_plan_awaiting_invariants("skip", state, mock_client)

        assert state.plan_dialog_state == "active"

    def test_yes_stays_awaiting_invariants(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        _handle_plan_awaiting_invariants("да", state, None)

        assert state.plan_dialog_state == "awaiting_invariants"
        out = capsys.readouterr().out
        assert "/invariant add" in out

    def test_done_transitions_to_active(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="First question?"))]
        )

        _handle_plan_awaiting_invariants("готово", state, mock_client)

        assert state.plan_dialog_state == "active"
        mock_client.chat.completions.create.assert_called_once()

    def test_start_english_transitions_to_active(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="First question?"))]
        )

        _handle_plan_awaiting_invariants("start", state, mock_client)

        assert state.plan_dialog_state == "active"

    def test_unknown_word_prints_hint(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        _handle_plan_awaiting_invariants("может быть", state, None)

        assert state.plan_dialog_state == "awaiting_invariants"
        out = capsys.readouterr().out
        assert "да" in out and "нет" in out

    def test_no_client_no_crash(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        _handle_plan_awaiting_invariants("нет", state, None)

        assert state.plan_dialog_state == "active"

    def test_kick_off_passes_description(self, capsys):
        from chatbot.main import _handle_plan_awaiting_invariants

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="?"))]
        )

        _handle_plan_awaiting_invariants("нет", state, mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Разработать REST API" in user_msg["content"]


# ---------------------------------------------------------------------------
# _run_plan_builder — result aggregation
# ---------------------------------------------------------------------------


class TestRunPlanBuilderResultAggregation:
    def test_aggregates_step_results_into_plan(self, capsys, monkeypatch, tmp_path):
        """После завершения всех шагов plan.result содержит результаты шагов."""
        from chatbot.main import _run_plan_builder
        from chatbot.models import StepStatus, TaskPlan, TaskStep, TaskPhase
        from datetime import datetime

        monkeypatch.chdir(tmp_path)

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.active_task_id = "task-agg"

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-agg",
            profile_name="test",
            name="Aggregation Plan",
            created_at=now,
            updated_at=now,
        )

        step1 = TaskStep(
            step_id="s1",
            task_id="task-agg",
            index=1,
            title="Step One",
            status=StepStatus.PENDING,
            created_at=now,
        )
        step2 = TaskStep(
            step_id="s2",
            task_id="task-agg",
            index=2,
            title="Step Two",
            status=StepStatus.PENDING,
            created_at=now,
        )

        # After _execute_builder_step the steps become DONE with results
        def fake_execute(step, plan, state, client):
            step.status = StepStatus.DONE
            step.result = f"Result of {step.title}"
            return True

        done_steps = [
            TaskStep(step_id="s1", task_id="task-agg", index=1, title="Step One",
                     status=StepStatus.DONE, result="Result of Step One", created_at=now),
            TaskStep(step_id="s2", task_id="task-agg", index=2, title="Step Two",
                     status=StepStatus.DONE, result="Result of Step Two", created_at=now),
        ]

        saved_plans = []

        with patch("chatbot.main.load_task_plan", return_value=plan), \
             patch("chatbot.main.load_all_steps", side_effect=[[step1, step2], done_steps]), \
             patch("chatbot.main._execute_builder_step", side_effect=fake_execute), \
             patch("chatbot.main.save_task_plan", side_effect=lambda p, _: saved_plans.append(p)):
            _run_plan_builder(state, MagicMock())

        assert len(saved_plans) == 1
        saved = saved_plans[0]
        assert saved.phase == TaskPhase.DONE
        assert "Step One" in saved.result
        assert "Result of Step One" in saved.result
        assert "Step Two" in saved.result
        assert "Result of Step Two" in saved.result

    def test_no_results_plan_result_stays_empty(self, capsys, monkeypatch, tmp_path):
        """Если шаги без result — plan.result не устанавливается."""
        from chatbot.main import _run_plan_builder
        from chatbot.models import StepStatus, TaskPlan, TaskStep, TaskPhase
        from datetime import datetime

        monkeypatch.chdir(tmp_path)

        state = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            profile_name="test",
        )
        state.active_task_id = "task-norez"

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-norez",
            profile_name="test",
            name="No Result Plan",
            created_at=now,
            updated_at=now,
        )

        step = TaskStep(
            step_id="s1",
            task_id="task-norez",
            index=1,
            title="Step One",
            status=StepStatus.PENDING,
            created_at=now,
        )
        done_step = TaskStep(
            step_id="s1",
            task_id="task-norez",
            index=1,
            title="Step One",
            status=StepStatus.DONE,
            created_at=now,
        )

        saved_plans = []

        def fake_execute(step, plan, state, client):
            step.status = StepStatus.DONE
            return True

        with patch("chatbot.main.load_task_plan", return_value=plan), \
             patch("chatbot.main.load_all_steps", side_effect=[[step], [done_step]]), \
             patch("chatbot.main._execute_builder_step", side_effect=fake_execute), \
             patch("chatbot.main.save_task_plan", side_effect=lambda p, _: saved_plans.append(p)):
            _run_plan_builder(state, MagicMock())

        assert saved_plans[0].phase == TaskPhase.DONE
        assert not saved_plans[0].result  # stays empty

