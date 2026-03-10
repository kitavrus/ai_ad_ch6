"""Тесты для модуля chatbot.context."""

from unittest.mock import MagicMock

from llm_agent.chatbot.context import build_context_for_api, maybe_summarize, summarize_messages
from llm_agent.chatbot.models import ChatMessage


def _make_messages(*role_content_pairs):
    """Вспомогательная функция: создаёт список ChatMessage из пар (role, content)."""
    return [ChatMessage(role=r, content=c) for r, c in role_content_pairs]


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------


class TestSummarizeMessages:
    def _mock_client(self, reply: str) -> MagicMock:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = reply
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def test_empty_messages_returns_empty(self):
        client = MagicMock()
        result = summarize_messages(client, "model", [])
        assert result == ""
        client.chat.completions.create.assert_not_called()

    def test_only_system_messages_returns_empty(self):
        client = MagicMock()
        msgs = _make_messages(("system", "You are helpful."))
        result = summarize_messages(client, "model", msgs)
        assert result == ""
        client.chat.completions.create.assert_not_called()

    def test_blank_content_messages_returns_empty(self):
        client = MagicMock()
        msgs = _make_messages(("user", "   "), ("assistant", ""))
        result = summarize_messages(client, "model", msgs)
        assert result == ""
        client.chat.completions.create.assert_not_called()

    def test_returns_summary_text(self):
        client = self._mock_client("Summary of the dialogue.")
        msgs = _make_messages(("user", "Hello"), ("assistant", "Hi"))
        result = summarize_messages(client, "gpt-4", msgs)
        assert result == "Summary of the dialogue."

    def test_strips_whitespace_from_response(self):
        client = self._mock_client("  Summary  \n")
        msgs = _make_messages(("user", "test"))
        result = summarize_messages(client, "m", msgs)
        assert result == "Summary"

    def test_api_exception_returns_empty(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        msgs = _make_messages(("user", "Hello"), ("assistant", "Hi"))
        result = summarize_messages(client, "m", msgs)
        assert result == ""

    def test_none_response_content_returns_empty(self):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        msgs = _make_messages(("user", "Hi"))
        result = summarize_messages(client, "m", msgs)
        assert result == ""

    def test_passes_correct_model(self):
        client = self._mock_client("ok")
        msgs = _make_messages(("user", "test"))
        summarize_messages(client, "my-model", msgs)
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-model"

    def test_uses_low_temperature(self):
        client = self._mock_client("ok")
        msgs = _make_messages(("user", "test"))
        summarize_messages(client, "m", msgs)
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3

    def test_includes_user_and_assistant_in_prompt(self):
        client = self._mock_client("ok")
        msgs = _make_messages(("user", "Question?"), ("assistant", "Answer!"))
        summarize_messages(client, "m", msgs)
        call_kwargs = client.chat.completions.create.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "User: Question?" in prompt_content
        assert "Assistant: Answer!" in prompt_content


# ---------------------------------------------------------------------------
# build_context_for_api
# ---------------------------------------------------------------------------


class TestBuildContextForApi:
    def test_empty_messages_no_summary(self):
        result = build_context_for_api([], "")
        assert result == []

    def test_system_message_first(self):
        msgs = _make_messages(
            ("system", "Be helpful."),
            ("user", "Hi"),
        )
        result = build_context_for_api(msgs, "")
        assert result[0] == {"role": "system", "content": "Be helpful."}

    def test_no_tokens_in_output(self):
        from llm_agent.chatbot.models import TokenUsage
        msgs = [ChatMessage(
            role="assistant",
            content="Hi",
            tokens=TokenUsage(prompt=5, completion=5, total=10),
        )]
        result = build_context_for_api(msgs, "")
        assert all("tokens" not in m for m in result)

    def test_summary_inserted_after_system(self):
        msgs = _make_messages(("system", "sys"), ("user", "q"), ("assistant", "a"))
        result = build_context_for_api(msgs, "Previous summary.")
        # system, summary_user, summary_assistant, user, assistant
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "Резюме" in result[1]["content"]
        assert result[2]["role"] == "assistant"

    def test_no_summary_if_empty_string(self):
        msgs = _make_messages(("user", "Hi"), ("assistant", "Hello"))
        result = build_context_for_api(msgs, "")
        # Только user и assistant, без summary-блока
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant"]

    def test_sliding_window_recent_n(self):
        # 5 пар user/assistant, recent_n=3 — должны войти только последние 3 non-system
        msgs = []
        for i in range(5):
            msgs.append(ChatMessage(role="user", content=f"q{i}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i}"))
        result = build_context_for_api(msgs, "", recent_n=3)
        assert len(result) == 3

    def test_all_messages_within_window(self):
        msgs = _make_messages(("user", "A"), ("assistant", "B"))
        result = build_context_for_api(msgs, "", recent_n=10)
        assert len(result) == 2

    def test_summary_content_in_user_message(self):
        msgs = _make_messages(("user", "q"))
        result = build_context_for_api(msgs, "My summary text", recent_n=10)
        summary_msg = result[0]
        assert "My summary text" in summary_msg["content"]

    def test_multiple_system_messages(self):
        msgs = _make_messages(
            ("system", "sys1"),
            ("system", "sys2"),
            ("user", "hi"),
        )
        result = build_context_for_api(msgs, "")
        assert result[0] == {"role": "system", "content": "sys1"}
        assert result[1] == {"role": "system", "content": "sys2"}


# ---------------------------------------------------------------------------
# maybe_summarize
# ---------------------------------------------------------------------------


class TestMaybeSummarize:
    def _mock_client(self, reply: str = "New summary") -> MagicMock:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = reply
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def _make_excess_messages(self, total: int):
        """Создаёт total non-system сообщений."""
        msgs = []
        for i in range(total):
            msgs.append(ChatMessage(role="user", content=f"q{i}"))
        return msgs

    def test_no_summarize_when_below_threshold(self):
        client = self._mock_client()
        # recent_n=10, interval=10 → нужно excess >= 10, т.е. >20 сообщений
        msgs = self._make_excess_messages(15)
        result_msgs, result_summary, updated = maybe_summarize(
            client, "m", msgs, "", recent_n=10, interval=10
        )
        assert updated is False
        assert result_msgs is msgs
        client.chat.completions.create.assert_not_called()

    def test_summarize_when_above_threshold(self, capsys):
        client = self._mock_client("Compact summary")
        msgs = self._make_excess_messages(25)  # 25 - 10 = 15 >= 10
        result_msgs, result_summary, updated = maybe_summarize(
            client, "m", msgs, "", recent_n=10, interval=10
        )
        assert updated is True
        assert result_summary == "Compact summary"
        assert len(result_msgs) == 10  # только recent_n остаётся

    def test_system_messages_preserved_after_summarize(self, capsys):
        client = self._mock_client("summary")
        sys_msg = ChatMessage(role="system", content="Be helpful.")
        non_sys = self._make_excess_messages(25)
        msgs = [sys_msg] + non_sys
        result_msgs, _, updated = maybe_summarize(
            client, "m", msgs, "", recent_n=10, interval=10
        )
        assert updated is True
        assert result_msgs[0].role == "system"
        assert result_msgs[0].content == "Be helpful."

    def test_existing_summary_prepended(self, capsys):
        client = self._mock_client("Updated summary")
        msgs = self._make_excess_messages(25)
        result_msgs, result_summary, updated = maybe_summarize(
            client, "m", msgs, "Old summary", recent_n=10, interval=10
        )
        assert updated is True
        # Проверяем, что старое резюме было передано в API
        call_kwargs = client.chat.completions.create.call_args.kwargs
        msgs_sent = call_kwargs["messages"]
        first_content = msgs_sent[0]["content"]
        assert "Old summary" in first_content or "Предыдущее резюме" in first_content

    def test_failed_summary_keeps_old(self, capsys):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")
        msgs = self._make_excess_messages(25)
        result_msgs, result_summary, updated = maybe_summarize(
            client, "m", msgs, "old", recent_n=10, interval=10
        )
        # updated=True, но summary остаётся старым (summarize вернул "")
        assert updated is True
        assert result_summary == "old"

    def test_exact_threshold_triggers(self, capsys):
        client = self._mock_client("s")
        # recent_n=5, interval=5 → нужно 5 сверх окна, т.е. 10
        msgs = self._make_excess_messages(10)
        _, _, updated = maybe_summarize(
            client, "m", msgs, "", recent_n=5, interval=5
        )
        assert updated is True

    def test_one_below_threshold_no_trigger(self):
        client = self._mock_client("s")
        msgs = self._make_excess_messages(9)  # 9 - 5 = 4 < 5
        _, _, updated = maybe_summarize(
            client, "m", msgs, "", recent_n=5, interval=5
        )
        assert updated is False
