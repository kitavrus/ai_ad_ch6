"""Pytest configuration and fixtures."""

import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_configure():
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
