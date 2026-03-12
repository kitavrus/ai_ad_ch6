"""Tests for MCPClient base class, MCPClientManager, and subclasses."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agent.chatbot.mcp_client import (
    MCPClient,
    MCPClientManager,
    MCPSchedulerClient,
    MCPWeatherClient,
    _convert_tools_to_openai,
)


def _make_tool(name, description="desc", schema=None):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {}
    return t


# ---------------------------------------------------------------------------
# MCPClient base class
# ---------------------------------------------------------------------------


def test_mcp_client_connect_success():
    client = MCPClient(server_script_path="/fake/server.py")
    fake_tools = _convert_tools_to_openai([_make_tool("tool1")])
    with patch("asyncio.run", return_value=fake_tools):
        ok = client.connect()
    assert ok is True
    assert client.connected is True
    assert len(client.tools_as_openai_format()) == 1


def test_mcp_client_connect_failure():
    client = MCPClient(server_script_path="/fake/server.py")
    with patch("asyncio.run", side_effect=Exception("refused")):
        ok = client.connect()
    assert ok is False
    assert client.connected is False
    assert client.tools_as_openai_format() == []


def test_mcp_client_call_tool_success():
    client = MCPClient(server_script_path="/fake/server.py")
    client._connected = True
    with patch("asyncio.run", return_value="result text"):
        result = client.call_tool("my_tool", {"arg": "val"})
    assert result == "result text"


def test_mcp_client_call_tool_failure():
    client = MCPClient(server_script_path="/fake/server.py")
    client._connected = True
    with patch("asyncio.run", side_effect=RuntimeError("timeout")):
        result = client.call_tool("my_tool", {})
    assert "[MCP error:" in result
    assert "timeout" in result


# ---------------------------------------------------------------------------
# Subclasses have the right default paths
# ---------------------------------------------------------------------------


def test_mcp_weather_client_is_subclass():
    client = MCPWeatherClient()
    assert isinstance(client, MCPClient)
    assert "weather_server.py" in client.server_script_path


def test_mcp_scheduler_client_is_subclass():
    client = MCPSchedulerClient()
    assert isinstance(client, MCPClient)
    assert "scheduler_server.py" in client.server_script_path


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------


def _connected_client(tools):
    c = MagicMock(spec=MCPClient)
    c.connected = True
    c.server_script_path = f"/fake/{tools[0]}.py"
    c.connect.return_value = True
    c.tools_as_openai_format.return_value = _convert_tools_to_openai(
        [_make_tool(t) for t in tools]
    )
    c.call_tool.side_effect = lambda name, args: f"result-{name}"
    return c


def _disconnected_client(path="/fake/bad.py"):
    c = MagicMock(spec=MCPClient)
    c.connected = False
    c.server_script_path = path
    c.connect.return_value = False
    c.tools_as_openai_format.return_value = []
    return c


def test_manager_connect_all_both_success():
    c1 = _connected_client(["weather"])
    c2 = _connected_client(["create_reminder"])
    manager = MCPClientManager([c1, c2])
    results = manager.connect_all()
    assert all(results.values())
    assert manager.connected is True


def test_manager_connect_all_one_fails():
    c1 = _connected_client(["weather"])
    c2 = _disconnected_client()
    manager = MCPClientManager([c1, c2])
    results = manager.connect_all()
    assert manager.connected is True  # at least one connected
    # one True, one False
    assert True in results.values()
    assert False in results.values()


def test_manager_tools_merged():
    c1 = _connected_client(["weather", "forecast"])
    c2 = _connected_client(["create_reminder", "list_reminders"])
    manager = MCPClientManager([c1, c2])
    manager.connect_all()
    tools = manager.tools_as_openai_format()
    names = [t["function"]["name"] for t in tools]
    assert "weather" in names
    assert "create_reminder" in names
    assert len(names) == 4


def test_manager_call_tool_routes_correctly():
    c1 = _connected_client(["weather"])
    c2 = _connected_client(["create_reminder"])
    manager = MCPClientManager([c1, c2])
    manager.connect_all()
    result = manager.call_tool("create_reminder", {"text": "test"})
    assert result == "result-create_reminder"
    c2.call_tool.assert_called_once_with("create_reminder", {"text": "test"})
    c1.call_tool.assert_not_called()


def test_manager_call_tool_unknown():
    c1 = _connected_client(["weather"])
    manager = MCPClientManager([c1])
    manager.connect_all()
    result = manager.call_tool("unknown_tool", {})
    assert "unknown tool" in result


def test_manager_connected_false_when_all_fail():
    c1 = _disconnected_client("/fake/a.py")
    c2 = _disconnected_client("/fake/b.py")
    manager = MCPClientManager([c1, c2])
    manager.connect_all()
    assert manager.connected is False
