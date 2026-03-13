import base64
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from main import app

VALID_KEY = "secret-token"


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture
def valid_headers():
    return {"X-API-Key": VALID_KEY}


@pytest.fixture
def invalid_headers():
    return {"X-API-Key": "wrong-key"}


@pytest.fixture
def simple_payload():
    return {
        "filename": "test.txt",
        "content_base64": base64.b64encode(b"hello").decode(),
    }


@pytest.fixture(autouse=True)
def save_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVE_DIR", str(tmp_path))
    return tmp_path
