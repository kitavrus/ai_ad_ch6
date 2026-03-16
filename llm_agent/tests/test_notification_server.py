"""Tests for NotificationServer (webhook receiver)."""

import json
import socket
import threading
import time
import urllib.request
from urllib.error import HTTPError

import pytest

from llm_agent.chatbot.notification_server import NotificationServer


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server():
    port = _find_free_port()
    srv = NotificationServer(port=port)
    srv.start()
    time.sleep(0.05)  # let the thread bind
    yield srv
    srv.stop()


def _post(url: str, data: dict) -> int:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status
    except HTTPError as e:
        return e.code


def test_get_url(server: NotificationServer):
    assert server.get_url().startswith("http://localhost:")
    assert server.get_url().endswith("/notify")


def test_check_notifications_empty(server: NotificationServer):
    assert server.check_notifications() == []


def test_webhook_receives_notification(server: NotificationServer):
    status = _post(server.get_url(), {
        "id": "abc",
        "description": "выключить плиту",
        "delay_seconds": 60
    })
    assert status == 200
    time.sleep(0.05)
    notes = server.check_notifications()
    assert len(notes) == 1
    assert "выключить плиту" in notes[0]
    assert "abc" in notes[0]
    assert "прошло 1 мин" in notes[0]
    assert "Описание:" in notes[0]
    assert notes[0].startswith("[REMINDER]")


def test_check_notifications_drains_queue(server: NotificationServer):
    _post(server.get_url(), {"id": "x", "description": "test", "delay_seconds": 5})
    time.sleep(0.05)
    server.check_notifications()  # drain
    assert server.check_notifications() == []


def test_multiple_notifications(server: NotificationServer):
    for i in range(3):
        _post(server.get_url(), {"id": str(i), "description": f"task {i}", "delay_seconds": i * 10})
    time.sleep(0.1)
    notes = server.check_notifications()
    assert len(notes) == 3


def test_wrong_path_returns_404(server: NotificationServer):
    base = server.get_url().rsplit("/notify", 1)[0]
    status = _post(base + "/wrong", {"id": "a", "description": "b"})
    assert status == 404


def test_invalid_json_returns_400(server: NotificationServer):
    body = b"not-json"
    req = urllib.request.Request(
        server.get_url(), data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            code = resp.status
    except HTTPError as e:
        code = e.code
    assert code == 400


def test_watcher_prints_notification_immediately(server, capsys):
    import queue
    from unittest.mock import MagicMock
    from llm_agent.chatbot.main import _start_notification_watcher
    mock_state = MagicMock()
    mock_state.profile_name = "default"
    _start_notification_watcher(server, mock_state, queue.Queue())
    _post(server.get_url(), {"id": "z1", "description": "тест фона", "delay_seconds": 30})
    time.sleep(1.0)  # даём вотчеру проснуться (0.5s poll) и напечатать
    captured = capsys.readouterr().out
    assert "тест фона" in captured
    assert "z1" in captured
    assert "прошло 30 сек" in captured


def test_notification_seconds_format(server: NotificationServer):
    _post(server.get_url(), {"id": "s1", "description": "быстрое", "delay_seconds": 30})
    time.sleep(0.05)
    notes = server.check_notifications()
    assert "прошло 30 сек" in notes[0]


def test_notification_minutes_format(server: NotificationServer):
    _post(server.get_url(), {"id": "m1", "description": "долгое", "delay_seconds": 120})
    time.sleep(0.05)
    notes = server.check_notifications()
    assert "прошло 2 мин" in notes[0]


def test_port_in_use_raises_os_error():
    port = _find_free_port()
    # Occupy the port
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        blocker.bind(("localhost", port))
        blocker.listen(1)
        srv = NotificationServer(port=port)
        with pytest.raises(OSError):
            srv.start()
    finally:
        blocker.close()
