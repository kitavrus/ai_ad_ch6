"""Интеграционный тест: профиль (style/format/constraints) + /plan on + инварианты."""

import pytest
from unittest.mock import MagicMock

from chatbot.models import AgentMode, SessionState, UserProfile
from chatbot.context import build_agent_system_prompt, validate_draft_against_invariants
from chatbot.main import _handle_plan_awaiting_task, _handle_plan_awaiting_invariants


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


def make_state() -> SessionState:
    return SessionState(
        model="gpt-4",
        base_url="https://api.example.com",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )


def make_profile() -> UserProfile:
    return UserProfile(
        name="Igor",
        style={"tone": "formal", "language": "ru"},
        format={"output": "markdown"},
        constraints=["Всегда отвечай на русском", "Не используй эмодзи"],
    )


def make_mock_client(llm_response: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = llm_response
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# 1. UserProfile.to_system_prompt() с style + format + constraints
# ---------------------------------------------------------------------------


class TestProfileToSystemPrompt:
    def test_style_appears(self):
        profile = make_profile()
        text = profile.to_system_prompt()
        assert "Style:" in text
        assert "tone=formal" in text
        assert "language=ru" in text

    def test_format_appears(self):
        profile = make_profile()
        text = profile.to_system_prompt()
        assert "Format:" in text
        assert "output=markdown" in text

    def test_constraints_appear(self):
        profile = make_profile()
        text = profile.to_system_prompt()
        assert "Constraints:" in text
        assert "Всегда отвечай на русском" in text
        assert "Не используй эмодзи" in text

    def test_all_three_sections_present(self):
        profile = make_profile()
        text = profile.to_system_prompt()
        lines = text.splitlines()
        sections = [l.split(":")[0] for l in lines if ":" in l]
        assert "Style" in sections
        assert "Format" in sections
        assert "Constraints" in sections

    def test_empty_profile_returns_empty_string(self):
        profile = UserProfile(name="empty")
        assert profile.to_system_prompt() == ""

    def test_only_constraints_no_style_format(self):
        profile = UserProfile(name="strict", constraints=["Только русский"])
        text = profile.to_system_prompt()
        assert "Constraints:" in text
        assert "Style:" not in text
        assert "Format:" not in text


# ---------------------------------------------------------------------------
# 2. FSM /plan on: awaiting_task → awaiting_invariants → active
# ---------------------------------------------------------------------------


class TestPlanOnFSMWithProfile:
    def test_awaiting_task_captures_description(self, capsys):
        state = make_state()
        state.plan_dialog_state = "awaiting_task"

        _handle_plan_awaiting_task("Написать план проекта по ML", state)

        assert state.plan_draft_description == "Написать план проекта по ML"
        assert state.plan_dialog_state == "awaiting_invariants"

    def test_awaiting_task_empty_input_stays(self, capsys):
        state = make_state()
        state.plan_dialog_state = "awaiting_task"

        _handle_plan_awaiting_task("", state)

        assert state.plan_dialog_state == "awaiting_task"
        assert not state.plan_draft_description

    def test_awaiting_invariants_yes_stays_in_state(self, capsys):
        """Ответ 'да' — пользователь хочет добавить инварианты, состояние не меняется."""
        state = make_state()
        state.plan_dialog_state = "awaiting_invariants"

        _handle_plan_awaiting_invariants("да", state, client=None)

        assert state.plan_dialog_state == "awaiting_invariants"

    def test_awaiting_invariants_no_transitions_to_active(self, capsys):
        """Ответ 'нет' — пропускаем инварианты, переходим в active."""
        state = make_state()
        state.plan_dialog_state = "awaiting_invariants"
        state.plan_draft_description = "Задача"

        _handle_plan_awaiting_invariants("нет", state, client=None)

        assert state.plan_dialog_state == "active"

    def test_awaiting_invariants_done_transitions_to_active(self, capsys):
        """Ответ 'готово' после добавления инвариантов — переходим в active."""
        state = make_state()
        state.plan_dialog_state = "awaiting_invariants"
        state.plan_draft_description = "Задача"
        state.agent_mode.invariants = ["Ответ на русском", "Без таблиц"]

        _handle_plan_awaiting_invariants("готово", state, client=None)

        assert state.plan_dialog_state == "active"

    def test_profile_constraints_visible_when_plan_starts(self):
        """Профиль с constraints корректно сериализуется до старта /plan on."""
        profile = make_profile()
        profile_text = profile.to_system_prompt()

        # В profile_text есть constraints — они будут видны в системном промпте плана
        assert "Всегда отвечай на русском" in profile_text
        assert "Не используй эмодзи" in profile_text

    def test_invariants_added_before_done(self):
        """Инварианты добавляются вручную перед переходом в active."""
        state = make_state()
        state.plan_dialog_state = "awaiting_invariants"
        state.plan_draft_description = "ML-проект"

        # Пользователь добавляет инварианты
        state.agent_mode.invariants.append("Сроки должны быть указаны явно")
        state.agent_mode.invariants.append("Использовать только Python")

        assert len(state.agent_mode.invariants) == 2

        # Затем вводит 'готово'
        _handle_plan_awaiting_invariants("готово", state, client=None)

        assert state.plan_dialog_state == "active"
        assert len(state.agent_mode.invariants) == 2  # инварианты сохранены


# ---------------------------------------------------------------------------
# 3. build_agent_system_prompt: оба слоя в одном промпте
# ---------------------------------------------------------------------------


class TestBuildAgentSystemPromptIntegration:
    def test_profile_text_appears_in_prompt(self):
        profile = make_profile()
        profile_text = profile.to_system_prompt()
        invariants = ["Ответ должен содержать конкретные сроки"]

        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="task=Написать план ML-проекта",
            invariants=invariants,
        )

        assert "tone=formal" in prompt
        assert "output=markdown" in prompt
        assert "Всегда отвечай на русском" in prompt

    def test_invariants_appear_in_prompt(self):
        profile = make_profile()
        profile_text = profile.to_system_prompt()
        invariants = ["Ответ должен содержать конкретные сроки", "Не используй таблицы"]

        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="task=Проект",
            invariants=invariants,
        )

        assert "Ответ должен содержать конкретные сроки" in prompt
        assert "Не используй таблицы" in prompt

    def test_profile_and_invariants_are_separate(self):
        """Profile и Invariants — разные секции в промпте."""
        profile = make_profile()
        profile_text = profile.to_system_prompt()
        invariants = ["Только Python"]

        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="task=Проект",
            invariants=invariants,
        )

        profile_pos = prompt.find("tone=formal")
        invariant_pos = prompt.find("Только Python")
        # Оба присутствуют и не перепутаны
        assert profile_pos != -1
        assert invariant_pos != -1
        assert profile_pos != invariant_pos

    def test_state_vars_appear_in_prompt(self):
        profile = make_profile()
        profile_text = profile.to_system_prompt()

        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="task=ML-проект",
            invariants=[],
        )

        assert "ML-проект" in prompt

    def test_empty_profile_still_builds_prompt(self):
        empty_profile = UserProfile(name="anon")
        profile_text = empty_profile.to_system_prompt()  # ""

        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="task=Задача",
            invariants=["Без эмодзи"],
        )

        assert "Без эмодзи" in prompt
        assert "Задача" in prompt

    def test_multiple_invariants_all_present(self):
        profile_text = make_profile().to_system_prompt()
        invariants = ["Rule A", "Rule B", "Rule C"]

        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="",
            invariants=invariants,
        )

        for rule in invariants:
            assert rule in prompt


# ---------------------------------------------------------------------------
# 4. Валидация: invariants vs profile constraints — разные механизмы
# ---------------------------------------------------------------------------


class TestInvariantValidationVsProfileConstraints:
    def test_validate_pass_when_draft_ok(self):
        """Черновик без нарушений → PASS."""
        client = make_mock_client("PASS")
        passed, violation = validate_draft_against_invariants(
            client, "gpt-4",
            draft="Срок выполнения: 2 недели. Реализация на Python.",
            invariants=["Сроки должны быть указаны", "Использовать Python"],
        )
        assert passed is True
        assert violation == ""

    def test_validate_fail_when_invariant_violated(self):
        """Черновик нарушает инвариант → FAIL с причиной."""
        client = make_mock_client('FAIL: Invariant 1 (Сроки должны быть указаны): no deadline found')
        passed, violation = validate_draft_against_invariants(
            client, "gpt-4",
            draft="Реализация на Python без указания сроков.",
            invariants=["Сроки должны быть указаны"],
        )
        assert passed is False
        assert violation != ""

    def test_validate_empty_invariants_always_passes(self):
        """Без инвариантов валидация всегда проходит без вызова LLM."""
        client = MagicMock()
        passed, violation = validate_draft_against_invariants(
            client, "gpt-4",
            draft="Любой текст",
            invariants=[],
        )
        assert passed is True
        assert violation == ""
        client.chat.completions.create.assert_not_called()

    def test_profile_constraints_not_passed_to_validate(self):
        """Profile constraints НЕ передаются в validate_draft_against_invariants.

        Проверяем, что validate получает только agent_mode.invariants,
        а profile.constraints живут отдельно в системном промпте.
        """
        profile = make_profile()
        state = make_state()
        state.agent_mode.invariants = ["Результат в JSON"]

        # Profile constraints — в профиле, не в invariants
        assert "Всегда отвечай на русском" in profile.constraints
        assert "Всегда отвечай на русском" not in state.agent_mode.invariants

        # validate вызывается ТОЛЬКО с agent_mode.invariants
        client = make_mock_client("PASS")
        passed, _ = validate_draft_against_invariants(
            client, "gpt-4",
            draft='{"result": "ok"}',
            invariants=state.agent_mode.invariants,  # только invariants!
        )
        assert passed is True

        # Проверяем что LLM получил именно agent_mode.invariants, а не profile.constraints
        call_args = client.chat.completions.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Результат в JSON" in prompt_content
        assert "Всегда отвечай на русском" not in prompt_content

    def test_two_layers_work_together(self):
        """Оба слоя (profile + invariants) работают совместно в системном промпте."""
        profile = make_profile()
        state = make_state()
        state.agent_mode.invariants = ["Результат в JSON"]

        profile_text = profile.to_system_prompt()

        # Системный промпт агента объединяет оба источника
        prompt = build_agent_system_prompt(
            profile_text=profile_text,
            state_vars="task=Анализ данных",
            invariants=state.agent_mode.invariants,
        )

        # Profile constraints из профиля — в промпте
        assert "Всегда отвечай на русском" in prompt
        assert "tone=formal" in prompt

        # Invariants из agent_mode — тоже в промпте (и будут валидироваться отдельно)
        assert "Результат в JSON" in prompt

        # Валидация — только по invariants (не по profile.constraints)
        client = make_mock_client("PASS")
        passed, _ = validate_draft_against_invariants(
            client, "gpt-4",
            draft='{"result": "ok"}',
            invariants=state.agent_mode.invariants,
        )
        assert passed is True
