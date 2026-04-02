"""Общие утилиты агента: цикл tool-calls, системный промпт поддержки."""

import json

MAX_TOOL_ROUNDS = 10

SUPPORT_SYSTEM_PROMPT = """Ты — AI-ассистент технической поддержки пользователей.

При получении вопроса от пользователя следуй этому порядку:
1. Вызови инструмент get_tasks() для получения списка тикетов из CRM.
2. Проанализируй тикеты и найди наиболее релевантный к вопросу пользователя.
3. Если найден тикет — сошлись на него в ответе (заголовок, статус, описание).
4. Используй предоставленный FAQ-контекст для технического ответа.
5. Если тикет не найден — отвечай только на основе FAQ и знаний.

Отвечай на русском языке, кратко и по существу."""


async def run_with_tools(client, mcp, messages: list[dict], max_rounds: int = MAX_TOOL_ROUNDS) -> str:
    """Цикл запрос → tool_calls → результаты → повтор до текстового ответа.

    Переиспользуется в script.py (REPL) и support_server.py (FastAPI).
    """
    loop_messages = list(messages)

    for _ in range(max_rounds):
        message = client.chat_raw(loop_messages, tools=mcp.tools_openai or None)

        if not message.tool_calls:
            return message.content or ""

        loop_messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            print(f"  [Tool] {tool_name}({arguments})")
            result = await mcp.call_tool(tool_name, arguments)
            print(f"  [Tool] ← {result[:120]}{'...' if len(result) > 120 else ''}")

            loop_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "[Превышен лимит вызовов инструментов]"
