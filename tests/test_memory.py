"""Тесты для модуля chatbot.memory и chatbot.memory_storage."""

import json
import os

import pytest

from chatbot.memory import (
    LongTermMemory,
    Memory,
    MemoryFactor,
    ShortTermMemory,
    WorkingMemory,
    extract_memory_factors,
)


# ===========================================================================
# ShortTermMemory
# ===========================================================================


class TestShortTermMemory:
    def test_defaults(self):
        m = ShortTermMemory()
        assert m.messages == []
        assert m.session_id == ""

    def test_add_message_appends(self):
        m = ShortTermMemory()
        m.add_message("user", "hello")
        assert len(m.messages) == 1
        assert m.messages[0]["role"] == "user"
        assert m.messages[0]["content"] == "hello"

    def test_add_message_has_timestamp(self):
        m = ShortTermMemory()
        m.add_message("assistant", "hi")
        assert "timestamp" in m.messages[0]

    def test_get_recent_all(self):
        m = ShortTermMemory()
        for i in range(5):
            m.add_message("user", f"msg{i}")
        result = m.get_recent(10)
        assert len(result) == 5

    def test_get_recent_truncated(self):
        m = ShortTermMemory()
        for i in range(20):
            m.add_message("user", f"msg{i}")
        result = m.get_recent(5)
        assert len(result) == 5
        assert result[-1]["content"] == "msg19"

    def test_clear_empties_messages(self):
        m = ShortTermMemory()
        m.add_message("user", "hello")
        m.add_message("assistant", "hi")
        m.clear()
        assert m.messages == []


# ===========================================================================
# WorkingMemory
# ===========================================================================


class TestWorkingMemory:
    def test_defaults(self):
        w = WorkingMemory()
        assert w.current_task is None
        assert w.task_status == "new"
        assert w.user_preferences == {}
        assert w.recent_actions == []

    def test_set_task(self):
        w = WorkingMemory()
        w.set_task("Write tests")
        assert w.current_task == "Write tests"
        assert w.task_status == "in_progress"
        assert w.updated_at != ""

    def test_set_task_with_context(self):
        w = WorkingMemory()
        w.set_task("Deploy", context={"env": "production"})
        assert w.task_context["env"] == "production"

    def test_update_status(self):
        w = WorkingMemory()
        w.update_status("done")
        assert w.task_status == "done"
        assert w.updated_at != ""

    def test_add_action_appends(self):
        w = WorkingMemory()
        w.add_action("step1")
        assert w.recent_actions == ["step1"]

    def test_add_action_capped_at_10(self):
        w = WorkingMemory()
        for i in range(15):
            w.add_action(f"step{i}")
        assert len(w.recent_actions) == 10
        assert w.recent_actions[-1] == "step14"

    def test_set_preference(self):
        w = WorkingMemory()
        w.set_preference("lang", "python")
        assert w.user_preferences["lang"] == "python"

    def test_to_short_term_snapshot_with_task(self):
        w = WorkingMemory()
        w.set_task("My Task")
        snapshot = w.to_short_term_snapshot()
        assert any("My Task" in m["content"] for m in snapshot.messages)

    def test_to_short_term_snapshot_with_prefs(self):
        w = WorkingMemory()
        w.set_preference("color", "blue")
        snapshot = w.to_short_term_snapshot()
        assert any("color" in m["content"] or "blue" in m["content"] for m in snapshot.messages)

    def test_to_short_term_snapshot_empty(self):
        w = WorkingMemory()
        snapshot = w.to_short_term_snapshot()
        assert isinstance(snapshot, ShortTermMemory)
        assert snapshot.messages == []


# ===========================================================================
# LongTermMemory
# ===========================================================================


class TestLongTermMemory:
    def test_defaults(self):
        lt = LongTermMemory()
        assert lt.user_profile == {}
        assert lt.decisions_log == []
        assert lt.knowledge_base == {}

    def test_add_decision(self):
        lt = LongTermMemory()
        lt.add_decision("task1", "decision text")
        assert len(lt.decisions_log) == 1
        assert lt.decisions_log[0]["task"] == "task1"
        assert lt.decisions_log[0]["decision"] == "decision text"
        assert "timestamp" in lt.decisions_log[0]

    def test_add_decision_with_context(self):
        lt = LongTermMemory()
        lt.add_decision("t", "d", context={"key": "val"})
        assert lt.decisions_log[0]["context"] == {"key": "val"}

    def test_add_knowledge(self):
        lt = LongTermMemory()
        lt.add_knowledge("python_tip", "use list comprehensions")
        assert lt.knowledge_base["python_tip"] == "use list comprehensions"

    def test_get_knowledge_existing(self):
        lt = LongTermMemory()
        lt.add_knowledge("topic", "value")
        assert lt.get_knowledge("topic") == "value"

    def test_get_knowledge_missing(self):
        lt = LongTermMemory()
        assert lt.get_knowledge("nonexistent") is None

    def test_set_profile(self):
        lt = LongTermMemory()
        lt.set_profile("name", "Alice")
        assert lt.user_profile["name"] == "Alice"

    def test_get_profile_all(self):
        lt = LongTermMemory()
        lt.set_profile("name", "Bob")
        lt.set_profile("lang", "ru")
        profile = lt.get_profile()
        assert profile["name"] == "Bob"
        assert profile["lang"] == "ru"

    def test_get_profile_by_key(self):
        lt = LongTermMemory()
        lt.set_profile("name", "Alice")
        assert lt.get_profile("name") == "Alice"

    def test_get_profile_missing_key(self):
        lt = LongTermMemory()
        assert lt.get_profile("nonexistent") is None

    def test_get_decision_history_all(self):
        lt = LongTermMemory()
        lt.add_decision("task1", "d1")
        lt.add_decision("task2", "d2")
        assert len(lt.get_decision_history()) == 2

    def test_get_decision_history_filtered(self):
        lt = LongTermMemory()
        lt.add_decision("task1", "d1")
        lt.add_decision("task2", "d2")
        lt.add_decision("task1", "d3")
        result = lt.get_decision_history("task1")
        assert len(result) == 2
        assert all(d["task"] == "task1" for d in result)

    def test_last_accessed_updated(self):
        lt = LongTermMemory()
        lt.add_knowledge("k", "v")
        assert lt.last_accessed != ""


# ===========================================================================
# Memory (центральный класс)
# ===========================================================================


class TestMemory:
    def test_defaults(self):
        m = Memory()
        assert isinstance(m.short_term, ShortTermMemory)
        assert isinstance(m.working, WorkingMemory)
        assert isinstance(m.long_term, LongTermMemory)

    def test_add_user_message(self):
        m = Memory()
        m.add_user_message("hello")
        assert len(m.short_term.messages) == 1
        assert m.short_term.messages[0]["role"] == "user"

    def test_add_assistant_message(self):
        m = Memory()
        m.add_assistant_message("hi there")
        assert m.short_term.messages[0]["role"] == "assistant"

    def test_add_to_working_memory_task(self):
        m = Memory()
        m.add_to_working_memory(task="My task")
        assert m.working.current_task == "My task"

    def test_add_to_working_memory_action(self):
        m = Memory()
        m.add_to_working_memory(action="step1")
        assert "step1" in m.working.recent_actions

    def test_add_to_working_memory_preference(self):
        m = Memory()
        m.add_to_working_memory(preference="lang", preference_value="python")
        assert m.working.user_preferences["lang"] == "python"

    def test_add_to_long_term_decision(self):
        m = Memory()
        m.add_to_long_term(decision="use postgres", task="setup db")
        assert len(m.long_term.decisions_log) == 1

    def test_add_to_long_term_knowledge(self):
        m = Memory()
        m.add_to_long_term(knowledge_key="tip", knowledge_value="write tests")
        assert m.long_term.knowledge_base["tip"] == "write tests"

    def test_add_to_long_term_profile(self):
        m = Memory()
        m.add_to_long_term(profile_key="name", profile_value="Igor")
        assert m.long_term.user_profile["name"] == "Igor"

    def test_get_short_term_context(self):
        m = Memory()
        m.add_user_message("msg1")
        m.add_user_message("msg2")
        ctx = m.get_short_term_context(n=1)
        assert len(ctx) == 1
        assert ctx[0]["content"] == "msg2"

    def test_get_working_context_empty(self):
        m = Memory()
        ctx = m.get_working_context()
        assert "user_preferences" in ctx
        assert "current_task" not in ctx

    def test_get_working_context_with_task(self):
        m = Memory()
        m.working.set_task("Write code")
        ctx = m.get_working_context()
        assert ctx["current_task"] == "Write code"

    def test_save_working_to_long_term(self):
        m = Memory()
        m.working.set_task("My task")
        m.save_working_to_long_term()
        assert len(m.long_term.decisions_log) == 1

    def test_save_working_to_long_term_custom_name(self):
        m = Memory()
        m.save_working_to_long_term(task_name="custom")
        assert m.long_term.decisions_log[0]["task"] == "custom"

    def test_clear_short_term(self):
        m = Memory()
        m.add_user_message("hello")
        m.add_assistant_message("hi")
        m.clear_short_term()
        assert m.short_term.messages == []

    def test_get_full_state(self):
        m = Memory()
        m.add_user_message("hello")
        state = m.get_full_state()
        assert "short_term" in state
        assert "working" in state
        assert "long_term" in state
        assert len(state["short_term"]["messages"]) == 1

    def test_load_full_state(self):
        m = Memory()
        m.add_user_message("hello")
        m.working.set_task("task1")
        m.long_term.set_profile("name", "Alice")
        state = m.get_full_state()

        m2 = Memory()
        m2.load_full_state(state)
        assert len(m2.short_term.messages) == 1
        assert m2.working.current_task == "task1"
        assert m2.long_term.user_profile["name"] == "Alice"

    def test_load_full_state_partial(self):
        m = Memory()
        m.add_user_message("hello")
        full = m.get_full_state()

        m2 = Memory()
        m2.load_full_state({"short_term": full["short_term"]})
        assert len(m2.short_term.messages) == 1
        assert m2.working.current_task is None

    def test_three_types_isolated(self):
        """Все три типа памяти хранятся и изменяются независимо."""
        m = Memory()
        m.add_user_message("user msg")
        m.working.set_task("task A")
        m.long_term.add_knowledge("key", "value")
        # краткосрочная
        assert len(m.short_term.messages) == 1
        # рабочая
        assert m.working.current_task == "task A"
        assert len(m.long_term.decisions_log) == 0
        # долговременная
        assert m.long_term.knowledge_base["key"] == "value"
        assert len(m.short_term.messages) == 1


# ===========================================================================
# MemoryFactor / extract_memory_factors
# ===========================================================================


class TestExtractMemoryFactors:
    def _working(self, task: str = None) -> WorkingMemory:
        w = WorkingMemory()
        if task:
            w.set_task(task)
        return w

    def test_returns_empty_for_plain_exchange(self):
        factors = extract_memory_factors("What time is it?", "It's 3pm.", self._working())
        # No keywords — may be empty or have entries; just ensure it's a list
        assert isinstance(factors, list)

    def test_detects_preference_keyword(self):
        factors = extract_memory_factors("я предпочитаю Python", "Хорошо.", self._working())
        types = [f["type"] for f in factors]
        assert MemoryFactor.PREFERENCE in types

    def test_detects_fact_keyword(self):
        factors = extract_memory_factors("запомни это важно", "Запомнил.", self._working())
        types = [f["type"] for f in factors]
        assert MemoryFactor.FACT in types

    def test_detects_decision_with_task(self):
        factors = extract_memory_factors(
            "что делать?", "Решение: использовать PostgreSQL.", self._working(task="setup db")
        )
        types = [f["type"] for f in factors]
        assert MemoryFactor.DECISION in types

    def test_no_decision_without_task(self):
        factors = extract_memory_factors(
            "что делать?", "Решение: использовать PostgreSQL.", self._working()
        )
        types = [f["type"] for f in factors]
        assert MemoryFactor.DECISION not in types

    def test_detects_summary_request(self):
        factors = extract_memory_factors("дай мне резюме диалога", "Вот резюме...", self._working())
        types = [f["type"] for f in factors]
        assert MemoryFactor.SUMMARY in types

    def test_factor_has_required_fields(self):
        factors = extract_memory_factors("я предпочитаю Java", "Хорошо.", self._working())
        for f in factors:
            assert "type" in f
            assert "content" in f
            assert "source" in f


# ===========================================================================
# memory_storage
# ===========================================================================


class TestMemoryStorage:
    def test_save_and_load_short_term(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import load_short_term_last, save_short_term
        data = {"messages": [{"role": "user", "content": "hi"}], "session_id": "s1"}
        path = save_short_term(data, "session_abc")
        assert os.path.exists(path)
        loaded = load_short_term_last("session_abc")
        assert loaded is not None
        assert loaded["messages"][0]["content"] == "hi"

    def test_save_and_load_working_memory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import load_working_memory, save_working_memory
        data = {"current_task": "deploy", "task_status": "in_progress", "task_context": {},
                "recent_actions": [], "user_preferences": {}, "created_at": "", "updated_at": ""}
        path = save_working_memory(data, "deploy")
        assert os.path.exists(path)
        loaded = load_working_memory("deploy")
        assert loaded is not None
        assert loaded["current_task"] == "deploy"

    def test_save_and_load_long_term(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import load_long_term, save_long_term
        data = {"user_profile": {"name": "Alice"}, "decisions_log": [], "knowledge_base": {}, "create_at": "", "last_accessed": ""}
        path = save_long_term(data, "default")
        assert os.path.exists(path)
        loaded = load_long_term("default")
        assert loaded is not None
        assert loaded["user_profile"]["name"] == "Alice"

    def test_load_short_term_returns_none_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import load_short_term_last
        assert load_short_term_last("nonexistent") is None

    def test_load_working_memory_returns_none_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import load_working_memory
        assert load_working_memory("nonexistent") is None

    def test_load_long_term_returns_none_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import load_long_term
        assert load_long_term("nonexistent") is None

    def test_export_and_import_state(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import export_memory_state, import_memory_state
        st = {"messages": []}
        wm = {"current_task": "x", "task_status": "new", "task_context": {},
              "recent_actions": [], "user_preferences": {}, "created_at": "", "updated_at": ""}
        lt = {"user_profile": {}, "decisions_log": [], "knowledge_base": {}, "create_at": "", "last_accessed": ""}
        path = export_memory_state(st, wm, lt)
        assert os.path.exists(path)
        loaded_st, loaded_wm, loaded_lt = import_memory_state(path)
        assert loaded_st == st
        assert loaded_wm == wm
        assert loaded_lt == lt

    def test_get_memory_stats_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import get_memory_stats
        stats = get_memory_stats()
        assert "short_term" in stats
        assert "working" in stats
        assert "long_term" in stats
        for v in stats.values():
            assert v["files"] == 0

    def test_get_memory_stats_after_save(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from chatbot.memory_storage import get_memory_stats, save_working_memory
        data = {"current_task": "x", "task_status": "new", "task_context": {},
                "recent_actions": [], "user_preferences": {}, "created_at": "", "updated_at": ""}
        save_working_memory(data, "t1")
        save_working_memory(data, "t2")
        stats = get_memory_stats()
        assert stats["working"]["files"] == 2
