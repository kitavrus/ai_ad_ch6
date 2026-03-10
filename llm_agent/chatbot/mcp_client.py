"""MCP client: sync facade over async MCP stdio transport."""

import asyncio
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_DEFAULT_SERVER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "llm_mcp", "weather", "weather_server.py")
)


class MCPWeatherClient:
    def __init__(self, server_script_path: str = _DEFAULT_SERVER_PATH) -> None:
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
