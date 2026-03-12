"""Тесты для reminders_storage.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from llm_agent.chatbot.reminders_storage import (
    fetch_reminder,
    fetch_reminders,
    get_reminders_path,
    load_reminders_file,
    save_all_reminders,
    update_reminder_in_file,
)


# ---------------------------------------------------------------------------
# Файловые операции
# ---------------------------------------------------------------------------


def test_get_reminders_path():
    path = get_reminders_path("myprofile")
    assert path == os.path.join("dialogues", "myprofile", "reminders.json")


def test_load_reminders_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = load_reminders_file("test")
    assert result == {"last_updated": None, "reminders": []}


def test_load_reminders_file_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile_dir = tmp_path / "dialogues" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "reminders.json").write_text("not json", encoding="utf-8")
    result = load_reminders_file("test")
    assert result == {"last_updated": None, "reminders": []}


def test_load_reminders_file_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile_dir = tmp_path / "dialogues" / "test"
    profile_dir.mkdir(parents=True)
    data = {"last_updated": "2026-01-01T00:00:00Z", "reminders": [{"id": "abc"}]}
    (profile_dir / "reminders.json").write_text(json.dumps(data), encoding="utf-8")
    result = load_reminders_file("test")
    assert result["last_updated"] == "2026-01-01T00:00:00Z"
    assert result["reminders"][0]["id"] == "abc"


def test_load_reminders_file_missing_keys(tmp_path, monkeypatch):
    """Файл без ключей last_updated и reminders должен дополняться дефолтами."""
    monkeypatch.chdir(tmp_path)
    profile_dir = tmp_path / "dialogues" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "reminders.json").write_text("{}", encoding="utf-8")
    result = load_reminders_file("test")
    assert result["reminders"] == []
    assert result["last_updated"] is None


def test_save_all_reminders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reminders = [{"id": "r1", "description": "Test", "status": "pending"}]
    save_all_reminders(reminders, "test")
    path = tmp_path / "dialogues" / "test" / "reminders.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["last_updated"] is not None
    assert len(data["reminders"]) == 1
    assert data["reminders"][0]["id"] == "r1"


def test_save_all_reminders_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_all_reminders([], "newprofile")
    path = tmp_path / "dialogues" / "newprofile" / "reminders.json"
    assert path.exists()


def test_update_reminder_in_file_updates_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reminders = [{"id": "r1", "status": "pending"}, {"id": "r2", "status": "pending"}]
    save_all_reminders(reminders, "test")
    update_reminder_in_file({"id": "r1", "status": "fired"}, "test")
    data = load_reminders_file("test")
    r1 = next(r for r in data["reminders"] if r["id"] == "r1")
    assert r1["status"] == "fired"
    assert len(data["reminders"]) == 2


def test_update_reminder_in_file_adds_new(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_all_reminders([{"id": "r1", "status": "pending"}], "test")
    update_reminder_in_file({"id": "r99", "status": "fired"}, "test")
    data = load_reminders_file("test")
    assert len(data["reminders"]) == 2
    ids = [r["id"] for r in data["reminders"]]
    assert "r99" in ids


def test_update_reminder_in_file_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_reminder_in_file({"id": "new1", "status": "pending"}, "test")
    data = load_reminders_file("test")
    assert len(data["reminders"]) == 1
    assert data["reminders"][0]["id"] == "new1"


def test_update_reminder_sets_last_updated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_reminder_in_file({"id": "x", "status": "pending"}, "test")
    data = load_reminders_file("test")
    assert data["last_updated"] is not None


# ---------------------------------------------------------------------------
# HTTP: fetch_reminders
# ---------------------------------------------------------------------------


def _make_response(status_code: int, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or []
    return resp


def test_fetch_reminders_success():
    mock_resp = _make_response(200, [{"id": "r1"}])
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = fetch_reminders()
    assert result == [{"id": "r1"}]


def test_fetch_reminders_with_status_filter():
    mock_resp = _make_response(200, [{"id": "r2", "status": "pending"}])
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = fetch_reminders(status="pending")
    assert result[0]["status"] == "pending"
    call_kwargs = mock_httpx.get.call_args
    assert call_kwargs[1]["params"]["status"] == "pending"


def test_fetch_reminders_401():
    mock_resp = _make_response(401)
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = fetch_reminders()
    assert result is None


def test_fetch_reminders_connection_error():
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("Connection refused")
        result = fetch_reminders()
    assert result is None


def test_fetch_reminders_httpx_none():
    with patch("llm_agent.chatbot.reminders_storage.httpx", None):
        result = fetch_reminders()
    assert result is None


# ---------------------------------------------------------------------------
# HTTP: fetch_reminder
# ---------------------------------------------------------------------------


def test_fetch_reminder_success():
    mock_resp = _make_response(200, {"id": "r1", "status": "fired"})
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = fetch_reminder("r1")
    assert result["id"] == "r1"
    assert result["status"] == "fired"


def test_fetch_reminder_404():
    mock_resp = _make_response(404)
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = fetch_reminder("missing")
    assert result is None


def test_fetch_reminder_connection_error():
    with patch("llm_agent.chatbot.reminders_storage.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("timeout")
        result = fetch_reminder("r1")
    assert result is None


def test_fetch_reminder_httpx_none():
    with patch("llm_agent.chatbot.reminders_storage.httpx", None):
        result = fetch_reminder("r1")
    assert result is None
