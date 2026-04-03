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

# Keywords that indicate a "generate file from analysis" request.
# For these requests fm_search_in_files must be called before fm_write_file.
_GENERATE_KEYWORDS = {
    "сгенерируй", "сгенерировать", "генерируй", "генерировать",
    "создай на основе", "создать на основе",
    "changelog", "readme",
    "на основе анализа", "анализ проекта",
    "отчёт", "report",
}


def _is_generate_request(messages: list[dict]) -> bool:
    """Return True if the last user message requests file generation from analysis."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "").lower()
            return any(kw in content for kw in _GENERATE_KEYWORDS)
    return False


async def run_with_tools(client, mcp, messages: list[dict], max_rounds: int = MAX_TOOL_ROUNDS) -> str:
    """Цикл запрос → tool_calls → результаты → повтор до текстового ответа.

    Переиспользуется в script.py (REPL) и support_server.py (FastAPI).
    """
    loop_messages = list(messages)
    search_required = _is_generate_request(messages)
    search_done = False

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

            if tool_name == "fm_search_in_files":
                search_done = True

            # Guard: for generate requests, block fm_write_file until fm_search_in_files is called
            if tool_name == "fm_write_file" and search_required and not search_done:
                result = json.dumps({
                    "error": (
                        "You must call fm_search_in_files at least once before fm_write_file "
                        "when generating a file from project analysis. "
                        "Call fm_search_in_files(query='@mcp.tool()', pattern='**/*.py') first."
                    )
                })
                print(f"  [Guard] blocked fm_write_file — fm_search_in_files not yet called")
            else:
                print(f"  [Tool] {tool_name}({arguments})")
                result = await mcp.call_tool(tool_name, arguments)
                print(f"  [Tool] ← {result[:120]}{'...' if len(result) > 120 else ''}")

            loop_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "[Превышен лимит вызовов инструментов]"
