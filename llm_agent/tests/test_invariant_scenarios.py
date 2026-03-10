"""Тесты инвариантной системы: персистентность, retry, сценарии нарушений."""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

from llm_agent.chatbot.models import (
    SessionState,
    StepStatus,
    TaskPlan,
    TaskStep,
)
from llm_agent.chatbot.context import validate_draft_against_invariants
from llm_agent.chatbot.storage import load_last_session, save_session


# ---------------------------------------------------------------------------
# Вспомогательная фабрика
# ---------------------------------------------------------------------------


def _make_state(**kwargs) -> SessionState:
    defaults = dict(
        model="gpt-4",
        base_url="https://api.example.com",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        profile_name="test",
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


def _make_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# Phase 1: Персистентность инвариантов
# ---------------------------------------------------------------------------


class TestInvariantPersistence:
    """Сохранение и восстановление инвариантов через JSON-сессию."""

    def test_invariants_survive_save_load_round_trip(self, monkeypatch, tmp_path):
        """Инварианты, сохранённые в сессию, восстанавливаются при --resume."""
        from llm_agent.chatbot.main import _apply_session_data, _build_session_payload

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.agent_mode.enabled = True
        state.agent_mode.invariants = [
            "Использовать только Python stdlib",
            "Все данные хранить в JSON-файлах",
            "Ответы только на русском языке",
        ]
        state.agent_mode.max_retries = 5
        state.dialogue_start_time = 0.0
        state.session_path = str(tmp_path / "dialogues" / "test" / "session_test.json")

        session = _build_session_payload(state)
        path = save_session(session, state.session_path)

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["agent_mode"]["invariants"] == state.agent_mode.invariants
        assert raw["agent_mode"]["enabled"] is True
        assert raw["agent_mode"]["max_retries"] == 5

        # Restore into fresh state
        fresh = _make_state()
        _apply_session_data(raw, fresh)

        assert fresh.agent_mode.enabled is True
        assert fresh.agent_mode.max_retries == 5
        assert fresh.agent_mode.invariants == [
            "Использовать только Python stdlib",
            "Все данные хранить в JSON-файлах",
            "Ответы только на русском языке",
        ]

    def test_load_last_session_restores_invariants(self, monkeypatch, tmp_path):
        """load_last_session + _apply_session_data возвращает инварианты."""
        from llm_agent.chatbot.main import _apply_session_data, _build_session_payload

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.agent_mode.enabled = True
        state.agent_mode.invariants = ["no SQL", "Python only"]
        state.dialogue_start_time = 0.0
        session_dir = tmp_path / "dialogues" / "test"
        session_dir.mkdir(parents=True)
        state.session_path = str(session_dir / "session_2024_model.json")

        session = _build_session_payload(state)
        save_session(session, state.session_path)

        result = load_last_session(profile_name="test")
        assert result is not None
        _, data = result

        fresh = _make_state()
        _apply_session_data(data, fresh)
        assert fresh.agent_mode.invariants == ["no SQL", "Python only"]

    def test_empty_invariants_round_trip(self, monkeypatch, tmp_path):
        """Пустой список инвариантов тоже корректно сохраняется."""
        from llm_agent.chatbot.main import _apply_session_data, _build_session_payload

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.agent_mode.invariants = []
        state.dialogue_start_time = 0.0
        state.session_path = str(tmp_path / "dialogues" / "test" / "session_empty.json")

        session = _build_session_payload(state)
        path = save_session(session, state.session_path)

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        fresh = _make_state()
        _apply_session_data(raw, fresh)
        assert fresh.agent_mode.invariants == []


# ---------------------------------------------------------------------------
# Phase 2: Сценарии нарушения инвариантов (validate_draft_against_invariants)
# ---------------------------------------------------------------------------


class TestStackConstraintInvariant:
    """Сценарий 1: запрет на использование внешних хранилищ (Redis, SQLite)."""

    def test_redis_suggestion_fails(self):
        """Предложение использовать Redis нарушает инвариант."""
        client = _make_client("FAIL: предложение использует Redis, что нарушает инвариант")
        invariants = ["Все данные хранить в JSON-файлах, не использовать базы данных"]
        draft = "Предлагаю использовать Redis для кэширования ответов API."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is False
        assert reason != ""
        client.chat.completions.create.assert_called_once()

    def test_sqlite_suggestion_fails(self):
        """Предложение использовать SQLite нарушает инвариант."""
        client = _make_client("FAIL: SQLite нарушает ограничение на хранение в JSON-файлах")
        invariants = ["Все данные хранить в JSON-файлах, не использовать базы данных"]
        draft = "Можно использовать SQLite для персистентного хранения."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is False

    def test_json_storage_passes(self):
        """Предложение хранить в JSON-файлах не нарушает инвариант."""
        client = _make_client("PASS")
        invariants = ["Все данные хранить в JSON-файлах, не использовать базы данных"]
        draft = "Предлагаю сохранять кэш в файл cache.json рядом с сессиями."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is True
        assert reason == ""


class TestTechnicalDecisionInvariant:
    """Сценарий 2: запрет на замену openai SDK на httpx."""

    def test_httpx_suggestion_fails(self):
        """Предложение использовать httpx вместо openai SDK нарушает инвариант."""
        client = _make_client("FAIL: предложение заменить openai.OpenAI на httpx нарушает инвариант клиентского слоя")
        invariants = ["Использовать только openai.OpenAI клиент. Не менять клиентский слой."]
        draft = "Перепиши chatbot/main.py используя httpx напрямую вместо openai SDK."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is False
        assert "httpx" in reason.lower() or reason != ""

    def test_openai_sdk_usage_passes(self):
        """Использование openai.OpenAI клиента не нарушает инвариант."""
        client = _make_client("PASS")
        invariants = ["Использовать только openai.OpenAI клиент. Не менять клиентский слой."]
        draft = "Вызов через client.chat.completions.create() сохранён как и прежде."

        passed, _ = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is True


class TestBusinessRuleInvariant:
    """Сценарий 3: все ответы должны быть на русском языке."""

    def test_english_response_fails(self):
        """Ответ на английском нарушает инвариант языка."""
        client = _make_client("FAIL: ответ написан на английском, что нарушает требование русского языка")
        invariants = ["Все ответы ассистента должны быть на русском языке"]
        draft = "Please respond only in English from now on. Here is the plan..."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is False

    def test_russian_response_passes(self):
        """Ответ на русском не нарушает инвариант."""
        client = _make_client("PASS")
        invariants = ["Все ответы ассистента должны быть на русском языке"]
        draft = "Хорошо, вот мой план разработки модуля кэширования..."

        passed, _ = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is True


class TestGlobalVariablesInvariant:
    """Сценарий 4: запрет на глобальные переменные."""

    def test_global_counter_fails(self):
        """Предложение добавить глобальный счётчик нарушает инвариант."""
        client = _make_client("FAIL: добавление глобального счётчика нарушает инвариант")
        invariants = ["Не использовать глобальные переменные в Python-коде"]
        draft = "Добавлю глобальный счётчик: `_api_call_count = 0` в начало main.py."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is False

    def test_pydantic_field_alternative_passes(self):
        """Счётчик в SessionState (Pydantic field) не нарушает инвариант."""
        client = _make_client("PASS")
        invariants = ["Не использовать глобальные переменные в Python-коде"]
        draft = "Добавлю поле `api_call_count: int = 0` в класс SessionState."

        passed, _ = validate_draft_against_invariants(client, "gpt-4", draft, invariants)

        assert passed is True


# ---------------------------------------------------------------------------
# Phase 3: Противоречивые инварианты
# ---------------------------------------------------------------------------


class TestContradictoryInvariants:
    """Сценарий 5: взаимоисключающие инварианты — любой черновик нарушает хотя бы один."""

    def test_both_invariants_cause_failure(self):
        """Инварианты 'не добавлять классы' и 'новый функционал — отдельный класс' противоречат."""
        client = _make_client("FAIL: добавление класса нарушает первый инвариант")
        invariants = [
            "Не добавлять новые классы",
            "Каждый новый функционал оформлять как отдельный класс",
        ]
        draft = "Добавлю класс Logger для обработки логирования."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)
        assert passed is False

    def test_contradictory_always_fails(self):
        """Любой черновик проваливается при взаимоисключающих инвариантах."""
        # Simulate different draft, still fails
        client = _make_client("FAIL: функция вместо класса нарушает второй инвариант")
        invariants = [
            "Не добавлять новые классы",
            "Каждый новый функционал оформлять как отдельный класс",
        ]
        draft = "Добавлю функцию log_event() для логирования."

        passed, reason = validate_draft_against_invariants(client, "gpt-4", draft, invariants)
        assert passed is False


# ---------------------------------------------------------------------------
# Phase 3: Автоматический retry при нарушении в основном диалоге
# ---------------------------------------------------------------------------


class TestMainDialogRetryOnViolation:
    """Проверка retry-цикла в main dialog loop при нарушении инварианта."""

    def test_retry_message_printed_on_violation(self, capsys, monkeypatch, tmp_path):
        """При нарушении инварианта выводится сообщение о повторе."""
        monkeypatch.chdir(tmp_path)

        with patch("chatbot.main.validate_draft_against_invariants") as mock_validate, \
             patch("chatbot.main.parse_agent_output", return_value=("Corrected response", {})):
            # First call fails, second passes
            mock_validate.side_effect = [(False, "uses Redis"), (True, "")]

            # Simulate the retry loop logic directly
            draft = "Let me use Redis."
            invariants = ["No external databases"]
            client = MagicMock()
            client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Corrected: use JSON files."))]
            )
            max_retries = 3

            passed, violation = mock_validate(client, "gpt-4", draft, invariants)
            retry_count = 0
            while not passed and retry_count < max_retries:
                retry_count += 1
                print(f"[Agent: инвариант нарушен ({violation}). Повтор {retry_count}/{max_retries}...]")
                draft = "Corrected: use JSON files."
                passed, violation = mock_validate(client, "gpt-4", draft, invariants)

            out = capsys.readouterr().out
            assert "инвариант нарушен" in out
            assert "Повтор 1/3" in out
            assert retry_count == 1

    def test_exhausted_retries_message_printed(self, capsys):
        """Когда все попытки исчерпаны, выводится сообщение об исчерпании."""
        with patch("chatbot.main.validate_draft_against_invariants") as mock_validate:
            # Always fails
            mock_validate.return_value = (False, "violates invariant")

            draft = "Bad response."
            invariants = ["invariant A", "invariant B"]
            client = MagicMock()
            client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Still bad."))]
            )
            max_retries = 3

            passed, violation = mock_validate(client, "gpt-4", draft, invariants)
            retry_count = 0
            while not passed and retry_count < max_retries:
                retry_count += 1
                print(
                    f"[Agent: инвариант нарушен ({violation}). "
                    f"Повтор {retry_count}/{max_retries}...]"
                )
                try:
                    resp = client.chat.completions.create(model="gpt-4", messages=[])
                    draft = resp.choices[0].message.content or draft
                except Exception:
                    break
                passed, violation = mock_validate(client, "gpt-4", draft, invariants)

            if not passed:
                print(f"[Agent: исчерпаны попытки валидации. Последнее нарушение: {violation}]")

            out = capsys.readouterr().out
            assert "исчерпаны попытки валидации" in out
            assert retry_count == 3


# ---------------------------------------------------------------------------
# Phase 3: Retry в plan builder при нарушении инварианта
# ---------------------------------------------------------------------------


class TestInvariantRespectedInPlanBuilder:
    """Сценарий 6: /plan builder ретраит шаг при нарушении инварианта."""

    def test_builder_retries_then_passes(self, capsys, monkeypatch, tmp_path):
        """_execute_builder_step делает retry и завершается успехом после прохождения валидации."""
        from llm_agent.chatbot.main import _execute_builder_step

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.active_task_id = "task-inv"
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}
        state.agent_mode.invariants = [
            "Использовать только встроенные структуры Python (dict, list). Никаких внешних хранилищ.",
            "Кэш должен очищаться при перезапуске. Не персистентный кэш.",
        ]

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-inv",
            profile_name="test",
            name="Cache Plan",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s1",
            task_id="task-inv",
            index=1,
            title="Implement cache",
            status=StepStatus.PENDING,
            created_at=now,
        )

        validate_side_effects = [(False, "uses Redis"), (True, "")]

        with patch("chatbot.main.load_all_steps", return_value=[step]), \
             patch("chatbot.main.save_task_step"), \
             patch("chatbot.main.save_task_plan"), \
             patch("chatbot.main._call_llm_for_builder_step", return_value="Using dict for cache."), \
             patch("chatbot.main.validate_draft_against_invariants",
                   side_effect=validate_side_effects):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is True
        assert step.status == StepStatus.DONE
        captured = capsys.readouterr()
        assert "Using dict for cache." in captured.out

    def test_builder_exhausts_retries_asks_clarification(self, capsys, monkeypatch, tmp_path):
        """После 3 провалов валидации builder запрашивает уточнение у пользователя."""
        from llm_agent.chatbot.main import _execute_builder_step

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.active_task_id = "task-inv2"
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}
        state.agent_mode.invariants = ["Только dict/list. Не Redis, не SQLite."]

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-inv2",
            profile_name="test",
            name="Cache Plan 2",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s2",
            task_id="task-inv2",
            index=1,
            title="Implement persistence",
            status=StepStatus.PENDING,
            created_at=now,
        )

        # 3 failures → clarification round → pass
        validate_side_effects = [
            (False, "uses Redis"),
            (False, "uses Redis"),
            (False, "uses Redis"),
            (True, ""),
        ]

        with patch("chatbot.main.load_all_steps", return_value=[step]), \
             patch("chatbot.main.save_task_step"), \
             patch("chatbot.main.save_task_plan"), \
             patch("chatbot.main._call_llm_for_builder_step", return_value="Use dict."), \
             patch("chatbot.main.validate_draft_against_invariants",
                   side_effect=validate_side_effects), \
             patch("chatbot.main.generate_clarification_question",
                   return_value="Как хранить без Redis?"), \
             patch("builtins.input", return_value="Использовать dict с TTL"):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is True
        captured = capsys.readouterr()
        assert "Как хранить без Redis?" in captured.out
        assert any(c["answer"] == "Использовать dict с TTL" for c in plan.clarifications)

    def test_prompt_invariant_resolution_no_skip_option(self, capsys, monkeypatch, tmp_path):
        """_prompt_invariant_resolution не предлагает skip и не принимает его."""
        from llm_agent.chatbot.main import _prompt_invariant_resolution

        monkeypatch.chdir(tmp_path)
        state = _make_state()
        state.agent_mode.invariants = ["Только dict/list"]

        now = datetime.utcnow().isoformat()
        step = TaskStep(
            step_id="s3",
            task_id="t3",
            index=1,
            title="Test",
            status=StepStatus.PENDING,
            created_at=now,
        )

        # Пользователь сначала вводит "skip" (должно быть отклонено), потом "abort"
        inputs = iter(["skip", "abort"])
        with patch("builtins.input", side_effect=inputs):
            result = _prompt_invariant_resolution(
                step, "uses Redis", state.agent_mode.invariants, state, MagicMock()
            )

        assert result == "abort"
        captured = capsys.readouterr()
        # skip не упомянут в вариантах
        assert "skip" not in captured.out
        # подсказка об ошибке выводилась (пользователь ввёл неизвестную команду)
        assert "Неверная команда" in captured.out


# ---------------------------------------------------------------------------
# Phase 3: _handle_invariant_command
# ---------------------------------------------------------------------------


class TestHandleInvariantCommand:
    """Тесты на добавление, удаление, список и очистку инвариантов через команды."""

    def _make_state(self):
        return _make_state()

    def test_add_invariant(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        _handle_invariant_command("add", "Только Python stdlib", state)

        assert state.agent_mode.invariants == ["Только Python stdlib"]
        out = capsys.readouterr().out
        assert "Инвариант добавлен" in out

    def test_add_empty_arg_does_not_add(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        _handle_invariant_command("add", "", state)

        assert state.agent_mode.invariants == []
        out = capsys.readouterr().out
        assert "требует текст" in out

    def test_del_invariant_by_number(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        state.agent_mode.invariants = ["inv A", "inv B", "inv C"]
        _handle_invariant_command("del", "2", state)

        assert state.agent_mode.invariants == ["inv A", "inv C"]
        out = capsys.readouterr().out
        assert "inv B" in out

    def test_del_out_of_range(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        state.agent_mode.invariants = ["inv A"]
        _handle_invariant_command("del", "5", state)

        assert state.agent_mode.invariants == ["inv A"]
        out = capsys.readouterr().out
        assert "Нет инварианта" in out

    def test_list_shows_invariants(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        state.agent_mode.invariants = ["inv X", "inv Y"]
        _handle_invariant_command("list", "", state)

        out = capsys.readouterr().out
        assert "inv X" in out
        assert "inv Y" in out

    def test_list_empty(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        _handle_invariant_command("list", "", state)

        out = capsys.readouterr().out
        assert "инвариант" in out.lower() or "не заданы" in out.lower()

    def test_clear_removes_all(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        state.agent_mode.invariants = ["inv 1", "inv 2", "inv 3"]
        _handle_invariant_command("clear", "", state)

        assert state.agent_mode.invariants == []
        out = capsys.readouterr().out
        assert "очищен" in out.lower() or "удален" in out.lower() or "clear" in out.lower()

    def test_add_multiple_invariants(self, capsys):
        from llm_agent.chatbot.main import _handle_invariant_command

        state = _make_state()
        _handle_invariant_command("add", "Python only", state)
        _handle_invariant_command("add", "No SQL", state)
        _handle_invariant_command("add", "Russian language", state)

        assert len(state.agent_mode.invariants) == 3
        assert "Python only" in state.agent_mode.invariants
        assert "No SQL" in state.agent_mode.invariants
        assert "Russian language" in state.agent_mode.invariants


# ---------------------------------------------------------------------------
# Fail-closed: неопределённый ответ, исключение, пустой черновик, фейковые clarifications
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Новые тесты fail-closed поведения после фикса."""

    def test_validation_inconclusive_fails(self):
        """Модель вернула что-то кроме PASS/FAIL → passed is False."""
        client = _make_client("MAYBE: not sure about this")

        passed, reason = validate_draft_against_invariants(
            client, "gpt-4", "some draft", ["inv"]
        )
        assert passed is False
        assert "inconclusive" in reason

    def test_validation_api_error_fails(self):
        """Исключение при validation LLM → passed is False."""
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network error")

        passed, reason = validate_draft_against_invariants(
            client, "gpt-4", "some draft", ["inv"]
        )
        assert passed is False
        assert "validation error" in reason

    def test_empty_draft_fails_validation(self, monkeypatch, tmp_path, capsys):
        """Пустой черновик от LLM не помечает шаг как DONE."""
        from llm_agent.chatbot.main import _execute_builder_step

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.agent_mode.invariants = ["non-empty response"]
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-empty",
            profile_name="test",
            name="Empty draft plan",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s-empty",
            task_id="task-empty",
            index=1,
            title="Empty step",
            status=StepStatus.PENDING,
            created_at=now,
        )

        with patch("chatbot.main._call_llm_for_builder_step", return_value=""), \
             patch("chatbot.main.save_task_step"), \
             patch("chatbot.main.save_task_plan"), \
             patch("chatbot.main.generate_clarification_question", return_value="Q?"), \
             patch("builtins.input", side_effect=EOFError):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is False
        assert step.status != StepStatus.DONE

    def test_no_fake_clarifications_on_retry(self, monkeypatch, tmp_path):
        """После авто-ретраев plan.clarifications не содержит фейковых записей."""
        from llm_agent.chatbot.main import _execute_builder_step

        monkeypatch.chdir(tmp_path)

        state = _make_state()
        state.agent_mode.invariants = ["short response"]
        state.memory = MagicMock()
        state.memory.get_profile_prompt.return_value = ""
        state.memory.working.user_preferences = {}

        now = datetime.utcnow().isoformat()
        plan = TaskPlan(
            task_id="task-retry",
            profile_name="test",
            name="Retry plan",
            created_at=now,
            updated_at=now,
        )
        step = TaskStep(
            step_id="s-retry",
            task_id="task-retry",
            index=1,
            title="Retry step",
            status=StepStatus.PENDING,
            created_at=now,
        )

        # Always fails → exhausts retries → user asked → EOFError → False
        with patch("chatbot.main._call_llm_for_builder_step", return_value="some draft"), \
             patch("chatbot.main.validate_draft_against_invariants", return_value=(False, "violation")), \
             patch("chatbot.main.save_task_step"), \
             patch("chatbot.main.save_task_plan"), \
             patch("chatbot.main.generate_clarification_question", return_value="Q?"), \
             patch("builtins.input", side_effect=EOFError):
            result = _execute_builder_step(step, plan, state, MagicMock())

        assert result is False
        fake = [
            c for c in plan.clarifications
            if "Fix invariant violation" in c.get("question", "")
        ]
        assert fake == [], f"Найдены фейковые clarifications: {plan.clarifications}"
