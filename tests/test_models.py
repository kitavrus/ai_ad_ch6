"""Тесты для модуля chatbot.models."""

import pytest
from pydantic import ValidationError

from chatbot.models import (
    ChatMessage,
    DialogueSession,
    RequestMetric,
    SessionState,
    TokenUsage,
)


class TestTokenUsage:
    def test_defaults(self):
        tu = TokenUsage()
        assert tu.prompt == 0
        assert tu.completion == 0
        assert tu.total == 0

    def test_custom_values(self):
        tu = TokenUsage(prompt=10, completion=20, total=30)
        assert tu.prompt == 10
        assert tu.completion == 20
        assert tu.total == 30

    def test_negative_prompt_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(prompt=-1)

    def test_negative_completion_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(completion=-1)

    def test_negative_total_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(total=-1)


class TestChatMessage:
    def test_basic_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tokens is None

    def test_assistant_message_with_tokens(self):
        tu = TokenUsage(prompt=5, completion=10, total=15)
        msg = ChatMessage(role="assistant", content="Hi there", tokens=tu)
        assert msg.tokens is not None
        assert msg.tokens.total == 15

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_default_content_empty(self):
        msg = ChatMessage(role="user")
        assert msg.content == ""

    def test_to_api_dict_excludes_tokens(self):
        tu = TokenUsage(prompt=5, completion=10, total=15)
        msg = ChatMessage(role="assistant", content="Hello", tokens=tu)
        d = msg.to_api_dict()
        assert d == {"role": "assistant", "content": "Hello"}
        assert "tokens" not in d

    def test_to_api_dict_user(self):
        msg = ChatMessage(role="user", content="What is 2+2?")
        assert msg.to_api_dict() == {"role": "user", "content": "What is 2+2?"}

    def test_to_api_dict_no_tokens_field(self):
        msg = ChatMessage(role="user", content="test")
        d = msg.to_api_dict()
        assert set(d.keys()) == {"role", "content"}


class TestRequestMetric:
    def test_create(self):
        m = RequestMetric(
            model="gpt-4",
            temp=0.7,
            ttft=0.5,
            req_time=0.5,
            total_time=1.2,
            tokens=100,
            p_tokens=80,
            c_tokens=20,
            cost_rub=0.015,
        )
        assert m.model == "gpt-4"
        assert m.endpoint == "chat"
        assert m.tokens == 100
        assert m.cost_rub == 0.015

    def test_custom_endpoint(self):
        m = RequestMetric(
            model="x",
            endpoint="completions",
            temp=0.5,
            ttft=0.1,
            req_time=0.1,
            total_time=0.2,
            tokens=50,
            p_tokens=40,
            c_tokens=10,
            cost_rub=0.005,
        )
        assert m.endpoint == "completions"


class TestDialogueSession:
    def test_minimal(self):
        s = DialogueSession(
            dialogue_session_id="sess_001",
            created_at="2026-01-01T00:00:00Z",
            model="gpt-4",
            base_url="https://api.example.com",
        )
        assert s.dialogue_session_id == "sess_001"
        assert s.messages == []
        assert s.requests == []
        assert s.total_tokens == 0
        assert s.context_summary == ""

    def test_with_messages(self):
        s = DialogueSession(
            dialogue_session_id="s",
            created_at="2026-01-01T00:00:00Z",
            model="m",
            base_url="u",
            messages=[{"role": "user", "content": "Hi"}],
            turns=1,
        )
        assert len(s.messages) == 1
        assert s.turns == 1

    def test_requests_list(self):
        metric = RequestMetric(
            model="m",
            temp=0.7,
            ttft=0.1,
            req_time=0.1,
            total_time=0.2,
            tokens=10,
            p_tokens=8,
            c_tokens=2,
            cost_rub=0.001,
        )
        s = DialogueSession(
            dialogue_session_id="s",
            created_at="2026-01-01T00:00:00Z",
            model="m",
            base_url="u",
            requests=[metric],
        )
        assert len(s.requests) == 1


class TestSessionState:
    def test_create(self):
        st = SessionState(
            model="gpt-4",
            base_url="https://api.example.com",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert st.model == "gpt-4"
        assert st.messages == []
        assert st.total_tokens == 0
        assert st.context_summary == ""
        assert st.request_index == 0

    def test_messages_mutation(self):
        st = SessionState(
            model="m",
            base_url="u",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        st.messages.append(ChatMessage(role="user", content="Hello"))
        assert len(st.messages) == 1

    def test_optional_fields_default_none(self):
        st = SessionState(
            model="m",
            base_url="u",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert st.max_tokens is None
        assert st.system_prompt is None
        assert st.initial_prompt is None
        assert st.session_path is None
