"""Tests for MCPWeatherClient and _convert_tools_to_openai."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agent.chatbot.mcp_client import MCPWeatherClient, _convert_tools_to_openai


# ---------------------------------------------------------------------------
# _convert_tools_to_openai
# ---------------------------------------------------------------------------


def _make_mcp_tool(name, description, input_schema):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = input_schema
    return t


def test_convert_tools_single():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    tool = _make_mcp_tool("get_weather", "Returns weather", schema)
    result = _convert_tools_to_openai([tool])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    fn = result[0]["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Returns weather"
    assert fn["parameters"] == schema


def test_convert_tools_empty():
    assert _convert_tools_to_openai([]) == []


def test_convert_tools_none_description():
    tool = _make_mcp_tool("my_tool", None, {})
    result = _convert_tools_to_openai([tool])
    assert result[0]["function"]["description"] == ""


# ---------------------------------------------------------------------------
# MCPWeatherClient.connect — success
# ---------------------------------------------------------------------------


def test_connect_success():
    client = MCPWeatherClient(server_script_path="/fake/server.py")
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    fake_tools = [_make_mcp_tool("get_weather", "Weather tool", schema)]

    with patch("asyncio.run", return_value=_convert_tools_to_openai(fake_tools)):
        result = client.connect()

    assert result is True
    assert client.connected is True
    tools = client.tools_as_openai_format()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "get_weather"


def test_connect_failure():
    client = MCPWeatherClient(server_script_path="/nonexistent/server.py")
    with patch("asyncio.run", side_effect=Exception("connection refused")):
        result = client.connect()

    assert result is False
    assert client.connected is False
    assert client.tools_as_openai_format() == []


# ---------------------------------------------------------------------------
# MCPWeatherClient.call_tool
# ---------------------------------------------------------------------------


def test_call_tool_success():
    client = MCPWeatherClient(server_script_path="/fake/server.py")
    client._connected = True

    with patch("asyncio.run", return_value="Погода: -3°C"):
        result = client.call_tool("get_weather", {"city": "Москва"})

    assert result == "Погода: -3°C"


def test_call_tool_failure():
    client = MCPWeatherClient(server_script_path="/fake/server.py")
    client._connected = True

    with patch("asyncio.run", side_effect=RuntimeError("timeout")):
        result = client.call_tool("get_weather", {"city": "Москва"})

    assert result.startswith("[MCP error:")
    assert "timeout" in result


# ---------------------------------------------------------------------------
# tools_as_openai_format when not connected
# ---------------------------------------------------------------------------


def test_tools_as_openai_format_not_connected():
    client = MCPWeatherClient(server_script_path="/fake/server.py")
    assert client.connected is False
    assert client.tools_as_openai_format() == []


# ---------------------------------------------------------------------------
# CLI parsing: /mcp commands
# ---------------------------------------------------------------------------


from llm_agent.chatbot.cli import parse_inline_command


def test_cli_mcp_status():
    result = parse_inline_command("/mcp status")
    assert result == {"mcp": {"action": "status", "arg": ""}}


def test_cli_mcp_tools():
    result = parse_inline_command("/mcp tools")
    assert result == {"mcp": {"action": "tools", "arg": ""}}


def test_cli_mcp_reconnect():
    result = parse_inline_command("/mcp reconnect")
    assert result == {"mcp": {"action": "reconnect", "arg": ""}}


def test_cli_mcp_bare():
    result = parse_inline_command("/mcp")
    assert result == {"mcp": {"action": "status", "arg": ""}}


def test_cli_mcp_unknown_action():
    result = parse_inline_command("/mcp foobar")
    assert result == {"mcp": {"action": "foobar", "arg": ""}}


# ---------------------------------------------------------------------------
# _handle_mcp_command
# ---------------------------------------------------------------------------


from llm_agent.chatbot.main import _handle_mcp_command


def test_handle_mcp_command_status_connected(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = True
    schema = {"type": "object", "properties": {}}
    client._tools = _convert_tools_to_openai([_make_mcp_tool("get_weather", "desc", schema)])
    _handle_mcp_command("status", "", client)
    out = capsys.readouterr().out
    assert "подключён" in out
    assert "1" in out


def test_handle_mcp_command_status_not_connected(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = False
    client._tools = []
    _handle_mcp_command("status", "", client)
    out = capsys.readouterr().out
    assert "не подключён" in out


def test_handle_mcp_command_tools_available(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = True
    schema = {"type": "object", "properties": {}}
    client._tools = _convert_tools_to_openai([_make_mcp_tool("get_weather", "Returns weather", schema)])
    _handle_mcp_command("tools", "", client)
    out = capsys.readouterr().out
    assert "get_weather" in out


def test_handle_mcp_command_tools_empty(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = False
    client._tools = []
    _handle_mcp_command("tools", "", client)
    out = capsys.readouterr().out
    assert "недоступны" in out


def test_handle_mcp_command_reconnect_success(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = False
    client._tools = []
    client.server_script_path = "/fake/server.py"
    schema = {"type": "object", "properties": {}}
    with patch("asyncio.run", return_value=_convert_tools_to_openai([_make_mcp_tool("get_weather", "d", schema)])):
        _handle_mcp_command("reconnect", "", client)
    out = capsys.readouterr().out
    assert "успешно" in out


def test_handle_mcp_command_reconnect_failure(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = False
    client._tools = []
    client.server_script_path = "/fake/server.py"
    with patch("asyncio.run", side_effect=Exception("fail")):
        _handle_mcp_command("reconnect", "", client)
    out = capsys.readouterr().out
    assert "не удалось" in out


def test_handle_mcp_command_unknown(capsys):
    client = MCPWeatherClient.__new__(MCPWeatherClient)
    client._connected = False
    client._tools = []
    _handle_mcp_command("unknown_cmd", "", client)
    out = capsys.readouterr().out
    assert "неизвестная подкоманда" in out


# ---------------------------------------------------------------------------
# ChatMessage.to_api_dict with tool_call_id
# ---------------------------------------------------------------------------


from llm_agent.chatbot.models import ChatMessage


def test_chat_message_to_api_dict_no_tool_call_id():
    msg = ChatMessage(role="user", content="hello")
    d = msg.to_api_dict()
    assert d == {"role": "user", "content": "hello"}
    assert "tool_call_id" not in d


def test_chat_message_to_api_dict_with_tool_call_id():
    msg = ChatMessage(role="tool", content="result", tool_call_id="call_abc")
    d = msg.to_api_dict()
    assert d["tool_call_id"] == "call_abc"
    assert d["role"] == "tool"
    assert d["content"] == "result"
