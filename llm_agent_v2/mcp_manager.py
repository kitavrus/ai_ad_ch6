"""Менеджер MCP-серверов: запускает серверы, собирает инструменты, маршрутизирует вызовы."""

import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_ROOT = Path(__file__).parent.parent

# (имя, путь к серверному файлу)
_SERVERS: list[tuple[str, Path]] = [
    ("git_commands", _ROOT / "llm_mcp/git-commands/git_server.py"),
]


class MCPManager:
    """Управляет подключениями ко всем MCP-серверам."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}  # server_name → session
        self._tool_to_server: dict[str, str] = {}       # tool_name → server_name
        self.tools_openai: list[dict] = []               # схемы инструментов в формате OpenAI

    async def connect_all(self) -> None:
        """Запускает все MCP-серверы и инициализирует сессии."""
        await self._stack.__aenter__()
        for name, path in _SERVERS:
            try:
                params = StdioServerParameters(command="python3", args=[str(path)])
                devnull = open(os.devnull, "w")
                read, write = await self._stack.enter_async_context(
                    stdio_client(params, errlog=devnull)
                )
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._tool_to_server[tool.name] = name
                    self.tools_openai.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    })
                print(f"  [MCP] {name}: {len(tools_result.tools)} инструмент(ов) подключено")
            except Exception as exc:
                print(f"  [MCP] {name}: не удалось подключить ({exc})")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Вызывает инструмент и возвращает результат строкой."""
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"[Ошибка: инструмент '{tool_name}' не найден]"
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        parts = [item.text if hasattr(item, "text") else str(item) for item in result.content]
        return "\n".join(parts)

    async def close(self) -> None:
        await self._stack.aclose()
