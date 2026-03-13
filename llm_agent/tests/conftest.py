"""Pytest configuration and fixtures."""

import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires running API servers")
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
