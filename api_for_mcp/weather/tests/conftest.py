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
