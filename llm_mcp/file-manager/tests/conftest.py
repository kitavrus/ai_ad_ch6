"""
Pytest fixtures для тестов file-manager MCP сервера.
Изолирует файловую систему через tmp_path и monkeypatch.chdir.
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_project(tmp_path, monkeypatch):
    """
    Меняет CWD на tmp_path для каждого теста.
    Сервер использует os.getcwd() как корень проекта,
    поэтому это гарантирует полную изоляцию файловой системы.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
