#!/usr/bin/env python3
"""REPL для работы с LLM API + MCP-инструменты (weather, scheduler, save_to_file, pdf-maker, git-commands)."""

import asyncio
import json
import sys
from pathlib import Path

from llm_agent_v2.llm_client import LLMClient
from llm_agent_v2.config import SessionConfig
from llm_agent_v2.mcp_manager import MCPManager
from llm_agent_v2.rag.retriever import RAGRetriever

MAX_TOOL_ROUNDS = 10  # защита от бесконечного цикла

_PROJECT_ROOT = str(Path(__file__).parent.parent)
_RAG_INDEX_DIR = str(Path(__file__).parent / "rag_index")


async def main() -> None:
    config = SessionConfig()
    client = LLMClient(config)
    mcp = MCPManager()
    retriever = RAGRetriever(index_dir=_RAG_INDEX_DIR)

    print(f"[Модель: {config.model}]")
    print("Подключение MCP-серверов...")
    await mcp.connect_all()
    print(f"Инструменты: {[t['function']['name'] for t in mcp.tools_openai]}")
    print("Введите сообщение (Ctrl+C или 'exit' для выхода)\n")

    messages: list[dict] = []

    try:
        while True:
            try:
                user_input = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nВыход.")
                break

            if user_input.lower() in ("exit", "quit", "выход"):
                break
            if not user_input:
                continue

            if user_input.startswith("/help "):
                question = user_input[len("/help "):].strip()
                if question:
                    await _handle_help(question, client, mcp, retriever)
                else:
                    print("[Укажите вопрос: /help <вопрос>]\n")
                continue

            messages.append({"role": "user", "content": user_input})

            try:
                reply = await _run_with_tools(client, mcp, messages)
            except Exception as exc:
                print(f"[Ошибка: {exc}]")
                messages.pop()
                continue

            print(f"Ассистент: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
    finally:
        await mcp.close()


async def _run_with_tools(client: LLMClient, mcp: MCPManager, messages: list[dict]) -> str:
    """Цикл запрос → tool_calls → результаты → повтор до текстового ответа."""
    loop_messages = list(messages)  # копия, чтобы не загрязнять основную историю промежуточными tool-сообщениями

    for _ in range(MAX_TOOL_ROUNDS):
        message = client.chat_raw(loop_messages, tools=mcp.tools_openai or None)

        if not message.tool_calls:
            return message.content or ""

        # Добавляем ответ ассистента с tool_calls в историю
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

        # Выполняем все вызовы
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


async def _handle_help(
    question: str,
    client: LLMClient,
    mcp: MCPManager,
    retriever: RAGRetriever,
) -> None:
    """One-shot /help: RAG + git-контекст, изолирован от основной истории."""

    # 1. RAG search (sync — приемлемо для REPL)
    try:
        chunks = retriever.search(question, strategy="structure", top_k=3)
        rag_text = "\n\n".join(
            f"[{c.source}]\n{c.text}" for c in chunks
        ) if chunks else "Документация не найдена."
    except Exception as exc:
        rag_text = f"[RAG недоступен: {exc}]"

    # 2. Git-контекст через MCP
    try:
        branch = await mcp.call_tool("get_current_branch", {"repo_path": _PROJECT_ROOT})
    except Exception:
        branch = "неизвестна"
    try:
        files = await mcp.call_tool("list_files", {"repo_path": _PROJECT_ROOT, "pattern": "*.py"})
    except Exception:
        files = ""
    git_context = f"Ветка: {branch}\nФайлы:\n{files}"

    # 3. Системный промпт с ролью ассистента по проекту
    system_prompt = (
        "Ты — ассистент разработчика по проекту. "
        f"Вот документация проекта:\n{rag_text}\n\n"
        f"Вот git-контекст:\n{git_context}\n\n"
        "Отвечай кратко и по существу."
    )

    # 4. Изолированный LLM-вызов — не попадает в основную историю messages
    help_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        message = client.chat_raw(help_messages, tools=None)
        reply = message.content or "[Пустой ответ]"
    except Exception as exc:
        reply = f"[Ошибка LLM: {exc}]"

    print(f"[Help] {reply}\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    asyncio.run(main())
