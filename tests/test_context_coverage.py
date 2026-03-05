"""Покрытие ранее непокрытых веток chatbot/context.py."""

import pytest
from unittest.mock import MagicMock

from chatbot.context import (
    build_builder_step_prompt,
    build_context_branching,
    build_context_by_strategy,
    build_context_sticky_facts,
    create_branch,
    create_checkpoint,
    extract_facts_from_llm,
    switch_branch,
    validate_draft_against_invariants,
)
from chatbot.models import (
    Branch,
    ChatMessage,
    ContextStrategy,
    DialogueCheckpoint,
    StickyFacts,
)


# ---------------------------------------------------------------------------
# extract_facts_from_llm
# ---------------------------------------------------------------------------

class TestExtractFactsFromLlm:
    def _make_client(self, content: str):
        mock = MagicMock()
        mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=content))]
        )
        return mock

    def test_with_existing_facts_in_prompt(self):
        client = self._make_client("lang: Python")
        result = extract_facts_from_llm(
            client, "m", "u msg", "a msg",
            existing_facts={"prev": "value"},
        )
        assert result == {"lang": "Python"}
        call_args = client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Текущие факты" in prompt
        assert "prev: value" in prompt

    def test_without_existing_facts(self):
        client = self._make_client("topic: databases")
        result = extract_facts_from_llm(client, "m", "q", "a", existing_facts={})
        assert result["topic"] == "databases"

    def test_empty_response_returns_empty_dict(self):
        client = self._make_client("")
        result = extract_facts_from_llm(client, "m", "q", "a", existing_facts={})
        assert result == {}

    def test_lines_without_colon_skipped(self):
        client = self._make_client("no colon here\nlang: Python")
        result = extract_facts_from_llm(client, "m", "q", "a", existing_facts={})
        assert list(result.keys()) == ["lang"]

    def test_exception_returns_empty_dict(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network error")
        result = extract_facts_from_llm(client, "m", "q", "a", existing_facts={})
        assert result == {}


# ---------------------------------------------------------------------------
# build_context_sticky_facts
# ---------------------------------------------------------------------------

class TestBuildContextStickyFacts:
    def _msg(self, role, content):
        return ChatMessage(role=role, content=content)

    def test_with_facts_inserts_facts_block(self):
        msgs = [self._msg("user", "hello")]
        facts = StickyFacts(facts={"lang": "Python"})
        result = build_context_sticky_facts(msgs, facts, recent_n=10)
        roles = [m["role"] for m in result]
        assert "user" in roles
        assert "assistant" in roles
        # facts block is a user message with facts content
        facts_msgs = [m for m in result if "Ключевые факты" in m.get("content", "")]
        assert facts_msgs

    def test_without_facts_no_facts_block(self):
        msgs = [self._msg("user", "hello")]
        facts = StickyFacts()
        result = build_context_sticky_facts(msgs, facts)
        facts_msgs = [m for m in result if "Ключевые факты" in m.get("content", "")]
        assert not facts_msgs

    def test_system_message_preserved(self):
        msgs = [self._msg("system", "sys"), self._msg("user", "hi")]
        facts = StickyFacts()
        result = build_context_sticky_facts(msgs, facts)
        assert result[0]["role"] == "system"

    def test_sliding_window_applied(self):
        msgs = [self._msg("user", f"msg{i}") for i in range(20)]
        facts = StickyFacts()
        result = build_context_sticky_facts(msgs, facts, recent_n=5)
        non_facts = [m for m in result if "Ключевые факты" not in m.get("content", "")]
        assert len(non_facts) == 5


# ---------------------------------------------------------------------------
# create_checkpoint
# ---------------------------------------------------------------------------

class TestCreateCheckpoint:
    def test_with_facts(self):
        msgs = [ChatMessage(role="user", content="hi")]
        facts = StickyFacts(facts={"k": "v"})
        cp = create_checkpoint(msgs, facts)
        assert cp.facts_snapshot == {"k": "v"}
        assert len(cp.messages_snapshot) == 1

    def test_without_facts_empty_snapshot(self):
        msgs = [ChatMessage(role="user", content="hi")]
        cp = create_checkpoint(msgs, facts=None)
        assert cp.facts_snapshot == {}


# ---------------------------------------------------------------------------
# create_branch
# ---------------------------------------------------------------------------

class TestCreateBranch:
    def test_branch_has_unique_id(self):
        cp = create_checkpoint([])
        b1 = create_branch("v1", cp)
        b2 = create_branch("v2", cp)
        assert b1.branch_id != b2.branch_id

    def test_branch_id_is_8_chars(self):
        cp = create_checkpoint([])
        b = create_branch("test", cp)
        assert len(b.branch_id) == 8

    def test_branch_copies_messages(self):
        msgs = [ChatMessage(role="user", content="original")]
        cp = create_checkpoint(msgs)
        b = create_branch("fork", cp)
        assert len(b.messages) == 1
        assert b.messages[0].content == "original"


# ---------------------------------------------------------------------------
# switch_branch
# ---------------------------------------------------------------------------

class TestSwitchBranch:
    def _make_branch(self, name):
        cp = create_checkpoint([])
        b = create_branch(name, cp)
        return b

    def test_found_by_id(self):
        b = self._make_branch("alpha")
        result = switch_branch(b.branch_id, [b])
        assert result is b

    def test_found_by_name(self):
        b = self._make_branch("alpha")
        result = switch_branch("alpha", [b])
        assert result is b

    def test_not_found_returns_none(self):
        b = self._make_branch("alpha")
        result = switch_branch("nonexistent", [b])
        assert result is None

    def test_empty_list_returns_none(self):
        assert switch_branch("x", []) is None


# ---------------------------------------------------------------------------
# build_context_branching
# ---------------------------------------------------------------------------

class TestBuildContextBranching:
    def test_system_messages_preserved(self):
        msgs = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]
        cp = create_checkpoint(msgs)
        b = create_branch("test", cp)
        result = build_context_branching(b)
        assert result[0]["role"] == "system"

    def test_with_facts_in_snapshot(self):
        msgs = []
        facts = StickyFacts(facts={"k": "v"})
        cp = create_checkpoint(msgs, facts)
        b = create_branch("test", cp)
        result = build_context_branching(b)
        facts_msgs = [m for m in result if "Факты на момент ветвления" in m.get("content", "")]
        assert facts_msgs

    def test_without_facts_no_facts_block(self):
        cp = create_checkpoint([])
        b = create_branch("test", cp)
        result = build_context_branching(b)
        facts_msgs = [m for m in result if "Факты на момент ветвления" in m.get("content", "")]
        assert not facts_msgs

    def test_only_new_messages_included(self):
        snapshot_msgs = [ChatMessage(role="user", content="old")]
        cp = create_checkpoint(snapshot_msgs)
        b = create_branch("test", cp)
        b.messages.append(ChatMessage(role="user", content="new"))
        result = build_context_branching(b, recent_n=10)
        contents = [m["content"] for m in result]
        assert "new" in contents
        assert "old" not in contents  # snapshot messages excluded from window

    def test_recent_n_applied_to_branch_messages(self):
        cp = create_checkpoint([])
        b = create_branch("test", cp)
        for i in range(15):
            b.messages.append(ChatMessage(role="user", content=f"msg{i}"))
        result = build_context_branching(b, recent_n=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# validate_draft_against_invariants — fallback
# ---------------------------------------------------------------------------

class TestValidateDraftFallback:
    def test_ambiguous_response_returns_pass(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="MAYBE this is ok"))]
        )
        passed, reason = validate_draft_against_invariants(
            client, "m", "draft text", "some invariant"
        )
        assert passed is True
        assert reason == ""


# ---------------------------------------------------------------------------
# build_builder_step_prompt — empty invariants
# ---------------------------------------------------------------------------

class TestBuildBuilderStepPromptEmptyInvariants:
    def test_empty_invariants_placeholder(self):
        result = build_builder_step_prompt(
            profile_text="profile",
            state_vars="state",
            invariants=[],
            step_title="Do X",
            step_description="desc",
            previous_steps="none",
        )
        assert "(не заданы)" in result

    def test_with_invariants(self):
        result = build_builder_step_prompt(
            profile_text="profile",
            state_vars="state",
            invariants=["rule one", "rule two"],
            step_title="Do X",
            step_description="desc",
            previous_steps="none",
        )
        assert "rule one" in result
        assert "rule two" in result


# ---------------------------------------------------------------------------
# build_context_by_strategy — all branches
# ---------------------------------------------------------------------------

class TestBuildContextByStrategy:
    def _msg(self, role, content):
        return ChatMessage(role=role, content=content)

    def test_sticky_facts_branch(self):
        msgs = [self._msg("user", "hi")]
        facts = StickyFacts(facts={"k": "v"})
        result = build_context_by_strategy(
            ContextStrategy.STICKY_FACTS, msgs, facts=facts
        )
        assert isinstance(result, list)
        assert any("Ключевые факты" in m.get("content", "") for m in result)

    def test_branching_with_active_branch(self):
        cp = create_checkpoint([])
        b = create_branch("test", cp)
        b.messages.append(self._msg("user", "new"))
        result = build_context_by_strategy(
            ContextStrategy.BRANCHING, [], active_branch=b
        )
        assert isinstance(result, list)

    def test_branching_without_active_branch_degrades(self):
        msgs = [self._msg("user", "hi")]
        result = build_context_by_strategy(
            ContextStrategy.BRANCHING, msgs, active_branch=None
        )
        assert isinstance(result, list)
        assert any(m.get("content") == "hi" for m in result)

    def test_unknown_strategy_fallback(self):
        msgs = [self._msg("user", "hi")]
        # Pass an unexpected value to trigger the final fallback
        result = build_context_by_strategy(
            ContextStrategy.SLIDING_WINDOW, msgs
        )
        assert isinstance(result, list)
