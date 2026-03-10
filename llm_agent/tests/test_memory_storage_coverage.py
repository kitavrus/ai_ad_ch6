"""Покрытие ранее непокрытых веток chatbot/memory_storage.py."""

import os

from llm_agent.chatbot.memory_storage import (
    list_long_term_memories,
    list_profiles,
    list_working_memories,
    load_long_term,
    load_profile,
    load_short_term_last,
    load_working_memory,
    save_long_term,
    save_profile,
    save_working_memory,
)
from llm_agent.chatbot.models import UserProfile


# ---------------------------------------------------------------------------
# load_short_term_memory — exception path
# ---------------------------------------------------------------------------

class TestLoadShortTermLastException:
    def test_returns_none_on_corrupt_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import llm_agent.chatbot.memory_storage as ms
        d = ms.get_short_term_dir("test")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "session_bad_20240101T000000Z.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")
        result = load_short_term_last("bad", profile_name="test")
        assert result is None


# ---------------------------------------------------------------------------
# load_working_memory — exception path
# ---------------------------------------------------------------------------

class TestLoadWorkingMemoryException:
    def test_returns_none_on_corrupt_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import llm_agent.chatbot.memory_storage as ms
        d = ms.get_working_dir("test")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "task_bad_20240101T000000Z.json")
        with open(path, "w") as f:
            f.write("{{broken")
        result = load_working_memory("bad", profile_name="test")
        assert result is None


# ---------------------------------------------------------------------------
# list_working_memories — happy path + exception
# ---------------------------------------------------------------------------

class TestListWorkingMemories:
    def test_happy_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        save_working_memory(
            {"current_task": "write tests", "task_status": "in_progress", "updated_at": "2024-01-01"},
            task_name="write_tests",
            profile_name="test",
        )
        result = list_working_memories(profile_name="test")
        assert len(result) == 1
        assert result[0]["task"] == "write tests"
        assert result[0]["status"] == "in_progress"

    def test_exception_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import llm_agent.chatbot.memory_storage as ms
        d = ms.get_working_dir("test")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "task_bad_20240101T000000Z.json")
        with open(path, "w") as f:
            f.write("{{broken")
        result = list_working_memories(profile_name="test")
        # exception in reading → returns []
        assert result == []


# ---------------------------------------------------------------------------
# load_long_term — exception path
# ---------------------------------------------------------------------------

class TestLoadLongTermException:
    def test_returns_none_on_corrupt_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import llm_agent.chatbot.memory_storage as ms
        d = ms.get_long_term_dir("test")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "profile_bad_20240101T000000Z.json")
        with open(path, "w") as f:
            f.write("{{broken")
        result = load_long_term("bad", profile_name="test")
        assert result is None


# ---------------------------------------------------------------------------
# list_long_term_memories — happy path + exception
# ---------------------------------------------------------------------------

class TestListLongTermMemories:
    def test_happy_path_with_user_profile(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        save_long_term(
            {
                "user_profile": {"name": "Alice"},
                "decisions_log": ["d1", "d2"],
                "knowledge_base": {"k1": "v1"},
            },
            name="alice",
            profile_name="test",
        )
        result = list_long_term_memories(profile_name="test")
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["decisions_count"] == 2
        assert result[0]["knowledge_count"] == 1

    def test_fallback_to_data_name(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        save_long_term(
            {"name": "Bob", "decisions_log": [], "knowledge_base": {}},
            name="bob",
            profile_name="test",
        )
        result = list_long_term_memories(profile_name="test")
        assert result[0]["name"] == "Bob"

    def test_exception_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import llm_agent.chatbot.memory_storage as ms
        d = ms.get_long_term_dir("test")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "profile_bad_20240101T000000Z.json")
        with open(path, "w") as f:
            f.write("{{broken")
        result = list_long_term_memories(profile_name="test")
        assert result == []


# ---------------------------------------------------------------------------
# load_profile — exception path
# ---------------------------------------------------------------------------

class TestLoadProfileException:
    def test_returns_none_on_corrupt_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        profile_dir = os.path.join("dialogues", "corrupted")
        os.makedirs(profile_dir, exist_ok=True)
        path = os.path.join(profile_dir, "profile.json")
        with open(path, "w") as f:
            f.write("{{broken json")
        result = load_profile("corrupted")
        assert result is None


# ---------------------------------------------------------------------------
# list_profiles — all branches
# ---------------------------------------------------------------------------

class TestListProfiles:
    def test_no_dialogues_dir_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # dialogues/ does not exist
        result = list_profiles()
        assert result == []

    def test_skips_entries_without_profile_json(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        d = os.path.join("dialogues", "no_profile")
        os.makedirs(d, exist_ok=True)
        result = list_profiles()
        assert result == []

    def test_returns_name_with_profile_json(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        profile = UserProfile(name="TestUser")
        save_profile(profile, name="TestUser")
        result = list_profiles()
        assert "testuser" in result or "TestUser" in result

    def test_skips_files_not_directories(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        dialogues_dir = os.path.join(str(tmp_path), "dialogues")
        os.makedirs(dialogues_dir, exist_ok=True)
        # create a file, not a directory
        open(os.path.join(dialogues_dir, "not_a_dir.txt"), "w").close()
        result = list_profiles()
        assert result == []
