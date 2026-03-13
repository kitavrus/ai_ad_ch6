import sys
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "api_for_mcp", "save_to_file"))

import save_server
from main import app

VALID_KEY = "secret-token"


def _find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{base_url}/", timeout=1.0)
            return
        except (httpx.ConnectError, httpx.ReadError):
            time.sleep(0.1)
    raise RuntimeError("Server did not start")


@pytest.fixture(scope="session")
def live_server_url():
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    _wait_for_server(base_url)
    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def patch_server_url(live_server_url, monkeypatch):
    monkeypatch.setattr(save_server, "BASE_URL", live_server_url)


@pytest.fixture
def valid_auth(monkeypatch):
    monkeypatch.setattr(save_server, "_HEADERS", {"X-API-Key": VALID_KEY})


@pytest.fixture
def invalid_auth(monkeypatch):
    monkeypatch.setattr(save_server, "_HEADERS", {"X-API-Key": "wrong-key"})


@pytest.fixture(autouse=True)
def save_dir_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVE_DIR", str(tmp_path))
    return tmp_path
