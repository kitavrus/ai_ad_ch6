"""MCP client: sync facade over async MCP stdio transport."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_WEATHER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "llm_mcp", "weather", "weather_server.py")
)

_DEFAULT_SCHEDULER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "llm_mcp", "scheduler", "scheduler_server.py")
)

_DEFAULT_PDF_MAKER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "llm_mcp", "pdf-maker", "pdf_server.py")
)

_DEFAULT_SAVE_TO_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "llm_mcp", "save_to_file", "save_server.py")
)

# Keep for backwards compatibility
_DEFAULT_SERVER_PATH = _DEFAULT_WEATHER_PATH


class MCPClient:
    """Base class: sync facade over async MCP stdio transport."""

    def __init__(self, server_script_path: str) -> None:
        self.server_script_path = os.path.abspath(server_script_path)
        self._tools: List[Dict[str, Any]] = []
        self._connected: bool = False

    def connect(self) -> bool:
        try:
            self._tools = asyncio.run(self._async_discover_tools())
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("MCP connect failed: %s", exc)
            self._connected = False
            return False

    def tools_as_openai_format(self) -> List[Dict[str, Any]]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            return asyncio.run(self._async_call_tool(name, arguments))
        except Exception as exc:
            logger.warning("MCP call_tool(%s) failed: %s", name, exc)
            return f"[MCP error: {exc}]"

    @property
    def connected(self) -> bool:
        return self._connected

    async def _async_discover_tools(self) -> List[Dict[str, Any]]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command="python", args=[self.server_script_path])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                return _convert_tools_to_openai(tools_response.tools)

    async def _async_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command="python", args=[self.server_script_path])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                if result.content:
                    return result.content[0].text
                return ""


class MCPWeatherClient(MCPClient):
    """MCP client for the weather server."""

    def __init__(self, server_script_path: str = _DEFAULT_WEATHER_PATH) -> None:
        super().__init__(server_script_path)


class MCPSchedulerClient(MCPClient):
    """MCP client for the scheduler server."""

    def __init__(self, server_script_path: str = _DEFAULT_SCHEDULER_PATH) -> None:
        super().__init__(server_script_path)


class MCPPdfMakerClient(MCPClient):
    """MCP client for the pdf-maker server."""

    def __init__(self, server_script_path: str = _DEFAULT_PDF_MAKER_PATH) -> None:
        super().__init__(server_script_path)


class MCPSaveToFileClient(MCPClient):
    """MCP client for the save-to-file server."""

    def __init__(self, server_script_path: str = _DEFAULT_SAVE_TO_FILE_PATH) -> None:
        super().__init__(server_script_path)


class MCPClientManager:
    """Manages multiple MCP clients and routes tool calls by tool name."""

    def __init__(self, clients: List[MCPClient]) -> None:
        self._clients = clients
        self._tool_to_client: Dict[str, MCPClient] = {}
        self._all_tools: List[Dict[str, Any]] = []

    def connect_all(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for client in self._clients:
            ok = client.connect()
            results[client.server_script_path] = ok
        self._build_tool_index()
        return results

    def _build_tool_index(self) -> None:
        self._tool_to_client = {}
        self._all_tools = []
        for client in self._clients:
            if client.connected:
                for tool in client.tools_as_openai_format():
                    name = tool.get("function", {}).get("name", "")
                    if name:
                        self._tool_to_client[name] = client
                        self._all_tools.append(tool)

    def tools_as_openai_format(self) -> List[Dict[str, Any]]:
        return list(self._all_tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        client = self._tool_to_client.get(name)
        if client is None:
            return f"[MCP error: unknown tool '{name}']"
        return client.call_tool(name, arguments)

    @property
    def connected(self) -> bool:
        return any(c.connected for c in self._clients)


def _convert_tools_to_openai(mcp_tools) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]
