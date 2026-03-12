import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from main import app, tasks

VALID_KEY = "secret-token"


@pytest.fixture(autouse=True)
def clear_tasks():
    tasks.clear()
    yield
    tasks.clear()


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
def sample_payload():
    return {"description": "выключить духовку", "delay_seconds": 60}


@pytest.fixture
def sample_payload_with_webhook():
    return {"description": "позвонить", "delay_seconds": 60, "webhook_url": "http://example.com/hook"}


@pytest.fixture
def anyio_backend():
    return "asyncio"
