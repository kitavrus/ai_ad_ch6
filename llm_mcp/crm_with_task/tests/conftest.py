import pytest
import llm_mcp.crm_with_task.crm_server as crm_module


@pytest.fixture(autouse=True)
def tmp_tasks_file(tmp_path, monkeypatch):
    """Redirect TASKS_FILE to a fresh temp file for every test."""
    tasks_file = tmp_path / "tasks.json"
    monkeypatch.setattr(crm_module, "TASKS_FILE", tasks_file)
    yield tasks_file
