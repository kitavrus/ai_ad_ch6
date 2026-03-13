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
    return {"title": "Test Document"}


@pytest.fixture
def full_payload():
    return {
        "title": "Full Report",
        "author": "Test Author",
        "sections": [
            {"heading": "Introduction", "content": "This is the intro."},
            {"heading": "List Section", "items": ["Item one", "Item two", "Item three"]},
            {
                "heading": "Table Section",
                "table": {
                    "headers": ["Name", "Value"],
                    "rows": [["Alpha", "1"], ["Beta", "2"]],
                },
            },
        ],
    }
