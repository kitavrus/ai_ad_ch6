"""Тесты для модуля chatbot.main."""

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_agent.chatbot.main import (
    _apply_inline_updates,
    _apply_session_data,
    _build_session_payload,
    _load_messages_from_dict,
    _print_loaded_history,
)
from llm_agent.chatbot.models import ChatMessage, SessionState, TokenUsage


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------


def _make_state(**kwargs) -> SessionState:
    defaults = {
        "model": "gpt-4",
        "base_url": "https://api.example.com",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "dialogue_start_time": time.time(),
    }
    defaults.update(kwargs)
    return SessionState(**defaults)


# ---------------------------------------------------------------------------
# _load_messages_from_dict
# ---------------------------------------------------------------------------


class TestLoadMessagesFromDict:
    def test_empty_list(self):
        result = _load_messages_from_dict([])
        assert result == []

    def test_user_message(self):
        raw = [{"role": "user", "content": "Hello"}]
        result = _load_messages_from_dict(raw)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Hello"
        assert result[0].tokens is None

    def test_assistant_with_tokens(self):
        raw = [
            {
                "role": "assistant",
                "content": "Hi",
                "tokens": {"prompt": 5, "completion": 10, "total": 15},
            }
        ]
        result = _load_messages_from_dict(raw)
        assert result[0].tokens is not None
        assert result[0].tokens.prompt == 5
        assert result[0].tokens.completion == 10
        assert result[0].tokens.total == 15

    def test_none_content_becomes_empty_string(self):
        raw = [{"role": "user", "content": None}]
        result = _load_messages_from_dict(raw)
        assert result[0].content == ""

    def test_missing_content_becomes_empty(self):
        raw = [{"role": "system"}]
        result = _load_messages_from_dict(raw)
        assert result[0].content == ""

    def test_missing_role_defaults_to_user(self):
        raw = [{"content": "text"}]
        result = _load_messages_from_dict(raw)
        assert result[0].role == "user"

    def test_tokens_not_dict_ignored(self):
        raw = [{"role": "assistant", "content": "hi", "tokens": "not-a-dict"}]
        result = _load_messages_from_dict(raw)
        assert result[0].tokens is None

    def test_multiple_messages(self):
        raw = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = _load_messages_from_dict(raw)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"


# ---------------------------------------------------------------------------
# _apply_session_data
# ---------------------------------------------------------------------------


class TestApplySessionData:
    def test_applies_model(self):
        state = _make_state()
        _apply_session_data({"model": "claude-3"}, state)
        assert state.model == "claude-3"

    def test_applies_messages(self):
        state = _make_state()
        data = {"messages": [{"role": "user", "content": "Hi"}]}
        _apply_session_data(data, state)
        assert len(state.messages) == 1
        assert state.messages[0].content == "Hi"

    def test_applies_token_counts(self):
        state = _make_state()
        data = {
            "total_tokens": 500,
            "total_prompt_tokens": 300,
            "total_completion_tokens": 200,
        }
        _apply_session_data(data, state)
        assert state.total_tokens == 500
        assert state.total_prompt_tokens == 300
        assert state.total_completion_tokens == 200

    def test_applies_context_summary(self):
        state = _make_state()
        _apply_session_data({"context_summary": "Previous talk."}, state)
        assert state.context_summary == "Previous talk."

    def test_applies_duration(self):
        state = _make_state()
        _apply_session_data({"duration_seconds": 120.5}, state)
        assert state.duration == 120.5

    def test_missing_keys_keep_state_defaults(self):
        state = _make_state(model="original-model")
        _apply_session_data({}, state)
        assert state.model == "original-model"
        assert state.total_tokens == 0

    def test_applies_system_and_initial_prompt(self):
        state = _make_state()
        data = {
            "system_prompt": "Be helpful.",
            "initial_prompt": "Hello!",
        }
        _apply_session_data(data, state)
        assert state.system_prompt == "Be helpful."
        assert state.initial_prompt == "Hello!"


# ---------------------------------------------------------------------------
# _apply_inline_updates
# ---------------------------------------------------------------------------


class TestApplyInlineUpdates:
    def test_update_model(self):
        state = _make_state()
        _apply_inline_updates({"model": "gpt-3.5"}, state)
        assert state.model == "gpt-3.5"

    def test_update_base_url(self):
        state = _make_state()
        _apply_inline_updates({"base_url": "https://new.api.com"}, state)
        assert state.base_url == "https://new.api.com"

    def test_update_temperature(self):
        state = _make_state()
        _apply_inline_updates({"temperature": 0.3}, state)
        assert state.temperature == 0.3

    def test_update_top_p(self):
        state = _make_state()
        _apply_inline_updates({"top_p": 0.5}, state)
        assert state.top_p == 0.5

    def test_update_top_k(self):
        state = _make_state()
        _apply_inline_updates({"top_k": 20}, state)
        assert state.top_k == 20

    def test_update_max_tokens(self):
        state = _make_state()
        _apply_inline_updates({"max_tokens": 512}, state)
        assert state.max_tokens == 512

    def test_none_values_are_skipped(self):
        state = _make_state()
        original_model = state.model
        _apply_inline_updates({"model": None}, state)
        assert state.model == original_model

    def test_update_system_prompt_inserts_if_missing(self):
        state = _make_state()
        _apply_inline_updates({"system_prompt": "New sys."}, state)
        assert state.messages[0].role == "system"
        assert state.messages[0].content == "New sys."

    def test_update_system_prompt_replaces_existing(self):
        state = _make_state()
        state.messages = [ChatMessage(role="system", content="Old sys.")]
        _apply_inline_updates({"system_prompt": "New sys."}, state)
        assert state.messages[0].content == "New sys."
        assert len([m for m in state.messages if m.role == "system"]) == 1

    def test_update_initial_prompt_appends_if_no_user(self):
        state = _make_state()
        state.messages = [ChatMessage(role="system", content="sys")]
        _apply_inline_updates({"initial_prompt": "Start!"}, state)
        assert state.messages[-1].content == "Start!"

    def test_update_initial_prompt_replaces_second_message(self):
        state = _make_state()
        state.messages = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="Old start"),
        ]
        _apply_inline_updates({"initial_prompt": "New start"}, state)
        assert state.messages[1].content == "New start"

    def test_showsummary_returns_true(self):
        state = _make_state()
        result = _apply_inline_updates({"showsummary": True}, state)
        assert result is True

    def test_no_showsummary_returns_false(self):
        state = _make_state()
        result = _apply_inline_updates({"model": "x"}, state)
        assert result is False

    def test_system_prompt_in_state_auto_prepended(self):
        state = _make_state(system_prompt="Auto sys.")
        state.messages = []
        _apply_inline_updates({}, state)
        assert state.messages[0].role == "system"
        assert state.messages[0].content == "Auto sys."

    def test_resume_no_session_prints_message(self, capsys):
        state = _make_state()
        with patch("chatbot.main.load_last_session", return_value=None):
            _apply_inline_updates({"resume": True}, state)
        out = capsys.readouterr().out
        assert "No last session found" in out

    def test_resume_loads_session(self, capsys):
        state = _make_state()
        mock_data = {
            "model": "loaded-model",
            "base_url": "https://loaded.url",
            "messages": [{"role": "user", "content": "Hi"}],
            "total_tokens": 42,
            "total_prompt_tokens": 30,
            "total_completion_tokens": 12,
            "duration_seconds": 10.0,
            "context_summary": "summary",
        }
        with patch(
            "chatbot.main.load_last_session",
            return_value=("dialogues/session_test.json", mock_data),
        ):
            with patch("chatbot.main._print_loaded_history"):
                _apply_inline_updates({"resume": True}, state)
        assert state.model == "loaded-model"
        assert state.total_tokens == 42


# ---------------------------------------------------------------------------
# _build_session_payload
# ---------------------------------------------------------------------------


class TestBuildSessionPayload:
    def test_basic_fields(self):
        state = _make_state(session_path="dialogues/session_test.json")
        session = _build_session_payload(state)
        assert session.model == "gpt-4"
        assert session.base_url == "https://api.example.com"
        assert session.dialogue_session_id == "dialogues/session_test.json"

    def test_with_user_and_assistant(self):
        state = _make_state(session_path="dialogues/s.json")
        state.messages = [
            ChatMessage(role="user", content="Q"),
            ChatMessage(role="assistant", content="A"),
        ]
        session = _build_session_payload(state, user_input="Q", assistant_text="A")
        assert session.last_user_input == "Q"
        assert session.last_assistant_content == "A"
        assert session.turns == 2

    def test_token_counts(self):
        state = _make_state(
            session_path="dialogues/s.json",
            total_tokens=200,
            total_prompt_tokens=150,
            total_completion_tokens=50,
        )
        session = _build_session_payload(state)
        assert session.total_tokens == 200
        assert session.total_prompt_tokens == 150
        assert session.total_completion_tokens == 50

    def test_messages_serialized_as_list_of_dicts(self):
        state = _make_state(session_path="dialogues/s.json")
        state.messages = [ChatMessage(role="user", content="Hi")]
        session = _build_session_payload(state)
        assert isinstance(session.messages, list)
        assert isinstance(session.messages[0], dict)

    def test_context_summary_included(self):
        state = _make_state(
            session_path="dialogues/s.json", context_summary="Some summary"
        )
        session = _build_session_payload(state)
        assert session.context_summary == "Some summary"

    def test_duration_seconds_positive(self):
        state = _make_state(session_path="dialogues/s.json")
        state.dialogue_start_time = time.time() - 10
        session = _build_session_payload(state)
        assert session.duration_seconds >= 10


# ---------------------------------------------------------------------------
# _print_loaded_history
# ---------------------------------------------------------------------------


class TestPrintLoadedHistory:
    def test_empty_messages_prints_nothing(self, capsys):
        _print_loaded_history([])
        out = capsys.readouterr().out
        assert out == ""

    def test_prints_user_and_assistant(self, capsys):
        msgs = [
            ChatMessage(role="user", content="Question?"),
            ChatMessage(
                role="assistant",
                content="Answer!",
                tokens=TokenUsage(prompt=5, completion=10, total=15),
            ),
        ]
        _print_loaded_history(msgs)
        out = capsys.readouterr().out
        assert "Question?" in out
        assert "Answer!" in out
        assert "Токены" in out

    def test_no_tokens_shows_unavailable(self, capsys):
        msgs = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]
        _print_loaded_history(msgs)
        out = capsys.readouterr().out
        assert "недоступны" in out

    def test_only_last_five_turns_shown(self, capsys):
        msgs = []
        for i in range(10):
            msgs.append(ChatMessage(role="user", content=f"q{i}"))
            msgs.append(ChatMessage(
                role="assistant",
                content=f"a{i}",
                tokens=TokenUsage(prompt=1, completion=1, total=2),
            ))
        _print_loaded_history(msgs)
        out = capsys.readouterr().out
        # Последние 5 должны быть видны
        for i in range(5, 10):
            assert f"q{i}" in out
        # Первые 5 — нет
        for i in range(5):
            assert f"q{i}" not in out

    def test_system_messages_skipped(self, capsys):
        msgs = [
            ChatMessage(role="system", content="System instruction"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]
        _print_loaded_history(msgs)
        out = capsys.readouterr().out
        assert "System instruction" not in out
        assert "Hello" in out

    def test_exception_does_not_raise(self, capsys):
        # Передаём некорректные данные, чтобы спровоцировать ошибку внутри функции
        from typing import List, cast
        bad_msgs = cast(List[ChatMessage], [MagicMock(role="user", content=None)])
        # Не должно поднимать исключение
        _print_loaded_history(bad_msgs)


# ---------------------------------------------------------------------------
# main() — smoke-test с заглушками
# ---------------------------------------------------------------------------


class TestMain:
    def test_raises_systemexit_without_api_key(self):
        with patch("chatbot.main.API_KEY", None):
            with pytest.raises(SystemExit, match="API_KEY"):
                from llm_agent.chatbot.main import main
                main()

    def test_main_exits_on_eof(self, monkeypatch, tmp_path):
        """main() должна завершиться без ошибок при EOFError на вводе."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=EOFError):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()  # не должно поднимать исключений

    def test_main_processes_one_message(self, monkeypatch, tmp_path, capsys):
        """main() должна обработать одно сообщение и выйти по команде exit."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI response"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        inputs = iter(["Hello!", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()

        out = capsys.readouterr().out
        assert "AI response" in out

    def test_main_handles_api_error(self, monkeypatch, tmp_path, capsys):
        """main() должна показать ошибку и продолжить диалог при сбое API."""
        monkeypatch.chdir(tmp_path)
        inputs = iter(["Hello!", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("Network error")
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()

    def test_main_resume_with_session(self, monkeypatch, tmp_path):
        """--resume при наличии сессии загружает её без падений."""
        monkeypatch.chdir(tmp_path)

        mock_data = {
            "model": "resumed-model",
            "base_url": "https://resumed.url",
            "messages": [{"role": "user", "content": "Hi"}],
            "total_tokens": 10,
            "total_prompt_tokens": 8,
            "total_completion_tokens": 2,
            "duration_seconds": 5.0,
            "context_summary": "",
        }

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.load_last_session",
                   return_value=("dialogues/session_x.json", mock_data)), \
             patch("chatbot.main._print_loaded_history"), \
             patch("builtins.input", side_effect=EOFError):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = True
            mock_args.profile = None
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()

    def test_main_resume_apply_session_exception(self, monkeypatch, tmp_path):
        """Исключение при применении данных сессии не роняет main()."""
        monkeypatch.chdir(tmp_path)

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.load_last_session",
                   return_value=("dialogues/session_x.json", {})), \
             patch("chatbot.main._apply_session_data",
                   side_effect=RuntimeError("corrupt")), \
             patch("builtins.input", side_effect=EOFError):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = True
            mock_args.profile = None
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()  # не должно падать

    def test_main_showsummary_with_existing_summary(self, monkeypatch, tmp_path, capsys):
        """/showsummary выводит резюме, если оно уже есть."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI reply"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        # Первый запрос — сообщение, устанавливаем context_summary через патч
        inputs = iter(["Hello!", "/showsummary", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.maybe_summarize",
                   return_value=(
                       [ChatMessage(role="user", content="Hello!")],
                       "Existing summary text",
                       False,
                   )), \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()

        out = capsys.readouterr().out
        assert "Existing summary text" in out
        assert "Текущее резюме контекста" in out

    def test_main_total_tokens_zero_fallback(self, monkeypatch, tmp_path, capsys):
        """Если total_tokens=0, считается как prompt+completion."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Reply"
        # total_tokens=0 → должно вычислиться как 10+5=15
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=0
        )

        inputs = iter(["Hello!", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()

        out = capsys.readouterr().out
        assert "итого=15" in out

    def test_main_save_session_exception(self, monkeypatch, tmp_path):
        """Исключение при сохранении сессии не роняет main()."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage = MagicMock(
            prompt_tokens=5, completion_tokens=5, total_tokens=10
        )

        inputs = iter(["Hi", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.save_session", side_effect=OSError("disk full")), \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()  # не должно падать

    def test_main_metric_log_exception(self, monkeypatch, tmp_path):
        """Исключение при логировании метрики не роняет main()."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage = MagicMock(
            prompt_tokens=5, completion_tokens=5, total_tokens=10
        )

        inputs = iter(["Hi", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.log_request_metric",
                   side_effect=OSError("no space")), \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()  # не должно падать

    def test_main_final_save_exception(self, monkeypatch, tmp_path):
        """Исключение при финальном сохранении не роняет main()."""
        monkeypatch.chdir(tmp_path)

        call_count = {"n": 0}

        def flaky_save(session, path):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise OSError("disk full on exit")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage = MagicMock(
            prompt_tokens=5, completion_tokens=5, total_tokens=10
        )

        inputs = iter(["Hi", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.save_session", side_effect=flaky_save), \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()  # не должно падать


# ---------------------------------------------------------------------------
# _print_loaded_history: ветка assistant-first (i += 1)
# ---------------------------------------------------------------------------


class TestPrintLoadedHistoryEdgeCases:
    def test_assistant_without_preceding_user(self, capsys):
        """Сообщение assistant без предшествующего user — шагаем через i += 1."""
        msgs = [
            ChatMessage(role="assistant", content="Stray assistant msg"),
            ChatMessage(role="user", content="Now user"),
            ChatMessage(role="assistant", content="Reply",
                        tokens=TokenUsage(prompt=1, completion=1, total=2)),
        ]
        _print_loaded_history(msgs)
        out = capsys.readouterr().out
        assert "Now user" in out

    def test_main_inline_command_updates_model(self, monkeypatch, tmp_path, capsys):
        """Inline-команда /model= должна обновить модель."""
        monkeypatch.chdir(tmp_path)
        inputs = iter(["/model=claude-3", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()

        out = capsys.readouterr().out
        assert "claude-3" in out

    def test_main_unknown_slash_command(self, monkeypatch, tmp_path, capsys):
        """Неизвестная /команда выводит 'Unknown command'."""
        monkeypatch.chdir(tmp_path)
        inputs = iter(["/unknownxyz", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()

        out = capsys.readouterr().out
        assert "Unknown command" in out

    def test_main_showsummary_with_no_summary(self, monkeypatch, tmp_path, capsys):
        """/showsummary при пустом резюме выводит соответствующее сообщение."""
        monkeypatch.chdir(tmp_path)
        inputs = iter(["/showsummary", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()

        out = capsys.readouterr().out
        assert "Резюме ещё не создано" in out

    def test_main_empty_input_skipped(self, monkeypatch, tmp_path):
        """Пустой ввод не отправляется в API."""
        monkeypatch.chdir(tmp_path)
        inputs = iter(["", "exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()

        mock_client.chat.completions.create.assert_not_called()

    def test_main_resume_flag(self, monkeypatch, tmp_path, capsys):
        """--resume при отсутствии сохранённых сессий не падает."""
        monkeypatch.chdir(tmp_path)
        inputs = iter([EOFError()])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.load_last_session", return_value=None), \
             patch("builtins.input", side_effect=EOFError):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = True
            mock_args.profile = None
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()

    def test_main_with_system_and_initial_prompt(self, monkeypatch, tmp_path):
        """Системный промпт и initial_prompt добавляются в messages перед диалогом."""
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Resp"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        inputs = iter(["exit"])

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("builtins.input", side_effect=inputs):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = "Be helpful."
            mock_args.initial_prompt = "Hello AI!"
            mock_args.resume = False
            mock_args.profile = None
            mock_parse_args.return_value = mock_args

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from llm_agent.chatbot.main import main
            main()


# ---------------------------------------------------------------------------
# Profile personalization tests
# ---------------------------------------------------------------------------


class TestProfileInlineCommands:
    """Tests for /profile subcommands in _apply_inline_updates."""

    def _make_state_with_memory(self, **kwargs) -> SessionState:
        from llm_agent.chatbot.memory import Memory

        defaults = {
            "model": "gpt-4",
            "base_url": "https://api.example.com",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "dialogue_start_time": time.time(),
            "memory": Memory(),
        }
        defaults.update(kwargs)
        return SessionState(**defaults)

    def test_profile_style_sets_style(self):
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "style", "arg": "tone=formal"}}, state)
        assert state.memory.long_term.profile.style["tone"] == "formal"

    def test_profile_format_sets_format(self):
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "format", "arg": "output=markdown"}}, state)
        assert state.memory.long_term.profile.format["output"] == "markdown"

    def test_profile_constraint_add(self):
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "constraint", "arg": "add no emojis"}}, state)
        assert "no emojis" in state.memory.long_term.profile.constraints

    def test_profile_constraint_del(self):
        state = self._make_state_with_memory()
        state.memory.long_term.add_profile_constraint("no emojis")
        _apply_inline_updates({"profile": {"action": "constraint", "arg": "del no emojis"}}, state)
        assert "no emojis" not in state.memory.long_term.profile.constraints

    def test_profile_name_sets_name(self):
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "name", "arg": "Alice"}}, state)
        assert state.memory.long_term.profile.name == "Alice"

    def test_profile_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = self._make_state_with_memory()
        state.memory.long_term.set_profile_style("tone", "formal")
        _apply_inline_updates({"profile": {"action": "name", "arg": "test_profile"}}, state)

        state2 = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "load", "arg": "test_profile"}}, state2)
        assert state2.memory.long_term.profile.style["tone"] == "formal"

    def test_profile_load_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "load", "arg": "nonexistent"}}, state)
        out = capsys.readouterr().out
        assert "nonexistent" in out

    def test_profile_show_prints_profile(self, capsys):
        state = self._make_state_with_memory()
        state.memory.long_term.set_profile_style("tone", "formal")
        _apply_inline_updates({"profile": {"action": "show", "arg": ""}}, state)
        out = capsys.readouterr().out
        assert "tone" in out
        assert "formal" in out

    def test_profile_load_switches_profile_name(self, tmp_path, monkeypatch):
        """При /profile load state.profile_name обновляется."""
        monkeypatch.chdir(tmp_path)
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "name", "arg": "bob"}}, state)

        state2 = self._make_state_with_memory()
        assert state2.profile_name == "default"
        _apply_inline_updates({"profile": {"action": "load", "arg": "bob"}}, state2)
        assert state2.profile_name == "bob"

    def test_profile_load_restores_session_history(self, tmp_path, monkeypatch):
        """При /profile load загружается история диалога выбранного профиля."""
        monkeypatch.chdir(tmp_path)
        from llm_agent.chatbot.models import DialogueSession
        from llm_agent.chatbot.storage import save_session

        # Создаём профиль и кладём сессию для "alice"
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "name", "arg": "alice"}}, state)
        profile_dir = tmp_path / "dialogues" / "alice"
        profile_dir.mkdir(parents=True, exist_ok=True)
        alice_session = DialogueSession(
            dialogue_session_id="alice_test",
            created_at="2026-01-01T00:00:00Z",
            model="gpt-4",
            base_url="https://api.example.com",
            messages=[{"role": "user", "content": "Сообщение Алисы"}],
        )
        save_session(alice_session, str(profile_dir / "session_alice.json"))

        # Переключаем профиль из другой сессии
        state2 = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "load", "arg": "alice"}}, state2)
        assert state2.profile_name == "alice"
        assert any(m.content == "Сообщение Алисы" for m in state2.messages)

    def test_profile_load_empty_history_clears_messages(self, tmp_path, monkeypatch):
        """При /profile load нового профиля (без сессий) история очищается."""
        monkeypatch.chdir(tmp_path)
        state = self._make_state_with_memory()
        from llm_agent.chatbot.models import ChatMessage
        state.messages = [ChatMessage(role="user", content="старое сообщение")]
        _apply_inline_updates({"profile": {"action": "name", "arg": "newuser"}}, state)

        state2 = self._make_state_with_memory()
        state2.messages = [ChatMessage(role="user", content="чужое сообщение")]
        _apply_inline_updates({"profile": {"action": "load", "arg": "newuser"}}, state2)
        # История должна очиститься — нет сессии для newuser
        assert all(m.content != "чужое сообщение" for m in state2.messages)

    def test_profile_model_set(self, tmp_path, monkeypatch, capsys):
        """'/profile model gpt-4' задаёт preferred_model и меняет state.model."""
        monkeypatch.chdir(tmp_path)
        state = self._make_state_with_memory()
        state.profile_name = "default"
        _apply_inline_updates({"profile": {"action": "model", "arg": "gpt-4"}}, state)
        assert state.memory.long_term.profile.preferred_model == "gpt-4"
        assert state.model == "gpt-4"
        out = capsys.readouterr().out
        assert "gpt-4" in out

    def test_profile_model_show_empty(self, capsys):
        """'/profile model' без аргумента показывает '(не задана)'."""
        state = self._make_state_with_memory()
        _apply_inline_updates({"profile": {"action": "model", "arg": ""}}, state)
        out = capsys.readouterr().out
        assert "(не задана)" in out

    def test_profile_model_show_set(self, capsys):
        """'/profile model' без аргумента показывает заданную модель."""
        state = self._make_state_with_memory()
        state.memory.long_term.profile.preferred_model = "claude-opus-4"
        _apply_inline_updates({"profile": {"action": "model", "arg": ""}}, state)
        out = capsys.readouterr().out
        assert "claude-opus-4" in out

    def test_profile_show_displays_model(self, capsys):
        """'/profile show' отображает строку 'Модель'."""
        state = self._make_state_with_memory()
        state.memory.long_term.profile.preferred_model = "gpt-4"
        _apply_inline_updates({"profile": {"action": "show", "arg": ""}}, state)
        out = capsys.readouterr().out
        assert "Модель" in out
        assert "gpt-4" in out

    def test_profile_load_applies_preferred_model(self, tmp_path, monkeypatch, capsys):
        """'/profile load' применяет preferred_model из профиля."""
        monkeypatch.chdir(tmp_path)
        from llm_agent.chatbot.models import UserProfile
        from llm_agent.chatbot.memory_storage import save_profile
        p = UserProfile(name="moduser", preferred_model="gpt-4-turbo")
        save_profile(p, "moduser")

        state = self._make_state_with_memory()
        assert state.model == "gpt-4"  # default из _make_state_with_memory
        _apply_inline_updates({"profile": {"action": "load", "arg": "moduser"}}, state)
        assert state.model == "gpt-4-turbo"
        out = capsys.readouterr().out
        assert "gpt-4-turbo" in out

    def test_profile_load_no_preferred_model_keeps_model(self, tmp_path, monkeypatch):
        """'/profile load' профиля без preferred_model не меняет state.model."""
        monkeypatch.chdir(tmp_path)
        from llm_agent.chatbot.models import UserProfile
        from llm_agent.chatbot.memory_storage import save_profile
        p = UserProfile(name="nomoduser")
        save_profile(p, "nomoduser")

        state = self._make_state_with_memory()
        original_model = state.model
        _apply_inline_updates({"profile": {"action": "load", "arg": "nomoduser"}}, state)
        assert state.model == original_model

    def test_main_profile_flag_auto_resumes(self, monkeypatch, tmp_path, capsys):
        """--profile автоматически загружает последнюю сессию профиля без --resume."""
        monkeypatch.chdir(tmp_path)

        mock_data = {
            "model": "gpt-4",
            "base_url": "https://api.example.com",
            "messages": [{"role": "user", "content": "История Игоря"}],
            "total_tokens": 5,
            "total_prompt_tokens": 3,
            "total_completion_tokens": 2,
            "duration_seconds": 1.0,
            "context_summary": "",
        }

        with patch("chatbot.main.API_KEY", "fake-key"), \
             patch("chatbot.main.parse_args") as mock_parse_args, \
             patch("chatbot.main.OpenAI") as mock_openai, \
             patch("chatbot.main.load_last_session",
                   return_value=("dialogues/Igor/session_x.json", mock_data)), \
             patch("chatbot.main._print_loaded_history") as mock_print_hist, \
             patch("builtins.input", side_effect=EOFError):

            mock_args = MagicMock()
            mock_args.model = None
            mock_args.base_url = None
            mock_args.max_tokens = None
            mock_args.temperature = None
            mock_args.top_p = None
            mock_args.top_k = None
            mock_args.system_prompt = None
            mock_args.initial_prompt = None
            mock_args.resume = False
            mock_args.profile = "Igor"  # явно задан профиль
            mock_parse_args.return_value = mock_args
            mock_openai.return_value = MagicMock()

            from llm_agent.chatbot.main import main
            main()

        # _print_loaded_history должен был вызваться — история загружена
        mock_print_hist.assert_called_once()
