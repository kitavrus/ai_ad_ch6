"""Тесты для модуля chatbot.storage."""

import json
import os

import pytest

from chatbot.models import DialogueSession, RequestMetric
from chatbot.storage import load_last_session, log_request_metric, save_session


@pytest.fixture
def sample_session():
    return DialogueSession(
        dialogue_session_id="test_session",
        created_at="2026-01-01T00:00:00Z",
        model="gpt-4",
        base_url="https://api.example.com",
        messages=[{"role": "user", "content": "Hello"}],
        turns=1,
        total_tokens=100,
    )


@pytest.fixture
def sample_metric():
    return RequestMetric(
        model="gpt-4",
        temp=0.7,
        ttft=0.5,
        req_time=0.5,
        total_time=1.0,
        tokens=100,
        p_tokens=80,
        c_tokens=20,
        cost_rub=0.015,
    )


class TestSaveSession:
    def test_creates_file(self, tmp_path, sample_session):
        path = str(tmp_path / "dialogues" / "session_test.json")
        result = save_session(sample_session, path)
        assert result == path
        assert os.path.exists(path)

    def test_valid_json_content(self, tmp_path, sample_session):
        path = str(tmp_path / "session.json")
        save_session(sample_session, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["dialogue_session_id"] == "test_session"
        assert data["model"] == "gpt-4"
        assert data["total_tokens"] == 100

    def test_creates_nested_dirs(self, tmp_path, sample_session):
        path = str(tmp_path / "a" / "b" / "c" / "session.json")
        save_session(sample_session, path)
        assert os.path.exists(path)

    def test_overwrites_existing_file(self, tmp_path, sample_session):
        path = str(tmp_path / "session.json")
        save_session(sample_session, path)
        # Изменяем и перезаписываем
        sample_session.total_tokens = 999
        save_session(sample_session, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_tokens"] == 999

    def test_ensure_ascii_false(self, tmp_path, sample_session):
        sample_session.context_summary = "Привет мир"
        path = str(tmp_path / "session.json")
        save_session(sample_session, path)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        assert "Привет мир" in raw

    def test_messages_serialized(self, tmp_path, sample_session):
        path = str(tmp_path / "session.json")
        save_session(sample_session, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data["messages"], list)
        assert data["messages"][0]["role"] == "user"


class TestLogRequestMetric:
    def test_creates_file(self, tmp_path, sample_metric, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = log_request_metric(sample_metric, "sess123", 0)
        assert os.path.exists(path)
        assert "sess123" in path
        assert "_req_0000.log" in path

    def test_index_padding(self, tmp_path, sample_metric, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = log_request_metric(sample_metric, "s", 42)
        assert "_req_0042.log" in path

    def test_valid_json_content(self, tmp_path, sample_metric, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = log_request_metric(sample_metric, "s", 0)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["model"] == "gpt-4"
        assert data["tokens"] == 100
        assert data["cost_rub"] == 0.015


class TestLoadLastSession:
    def test_returns_none_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_last_session()
        assert result is None

    def test_loads_single_session(self, tmp_path, monkeypatch, sample_session):
        monkeypatch.chdir(tmp_path)
        dialogues = tmp_path / "dialogues"
        dialogues.mkdir()
        path = dialogues / "session_20260101T000000Z_gpt-4.json"
        save_session(sample_session, str(path))

        result = load_last_session()
        assert result is not None
        loaded_path, data = result
        assert "session_" in loaded_path
        assert data["model"] == "gpt-4"

    def test_loads_latest_of_multiple(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dialogues = tmp_path / "dialogues"
        dialogues.mkdir()

        for i, name in enumerate(["session_A.json", "session_B.json", "session_C.json"]):
            p = dialogues / name
            data = {
                "dialogue_session_id": f"id_{i}",
                "created_at": "2026-01-01T00:00:00Z",
                "model": f"model_{i}",
                "base_url": "u",
                "messages": [],
                "requests": [],
                "total_tokens": i * 10,
            }
            p.write_text(json.dumps(data), encoding="utf-8")
            # Принудительно устанавливаем mtime для надёжной сортировки
            os.utime(p, (i, i))

        result = load_last_session()
        assert result is not None
        _, data = result
        assert data["model"] == "model_2"

    def test_returns_none_on_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dialogues = tmp_path / "dialogues"
        dialogues.mkdir()
        (dialogues / "session_bad.json").write_text("NOT JSON", encoding="utf-8")

        result = load_last_session()
        assert result is None
