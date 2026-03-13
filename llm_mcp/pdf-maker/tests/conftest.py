import sys
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "api_for_mcp", "pdf-maker"))

import importlib.util

import pdf_server
from main import app as pdf_app

_save_main_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "api_for_mcp", "save_to_file", "main.py")
_spec = importlib.util.spec_from_file_location("save_main", _save_main_path)
_save_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_save_module)
save_app = _save_module.app

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


def _start_server(app, env_overrides=None):
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    _wait_for_server(base_url)
    return base_url, server, thread


@pytest.fixture(scope="session")
def live_server_url():
    base_url, server, thread = _start_server(pdf_app)
    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def live_save_url(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("saved_files")
    os.environ["SAVE_DIR"] = str(save_dir)
    base_url, server, thread = _start_server(save_app)
    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def patch_server_urls(live_server_url, live_save_url, monkeypatch):
    monkeypatch.setattr(pdf_server, "BASE_URL", live_server_url)
    monkeypatch.setattr(pdf_server, "SAVE_URL", live_save_url)


@pytest.fixture
def valid_auth(monkeypatch):
    monkeypatch.setattr(pdf_server, "_HEADERS", {"X-API-Key": VALID_KEY})
    monkeypatch.setattr(pdf_server, "_SAVE_HEADERS", {"X-API-Key": VALID_KEY})


@pytest.fixture
def invalid_auth(monkeypatch):
    monkeypatch.setattr(pdf_server, "_HEADERS", {"X-API-Key": "wrong-key"})
