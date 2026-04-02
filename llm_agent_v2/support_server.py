#!/usr/bin/env python3
"""FastAPI-сервер: AI-ассистент поддержки.

Агент сам вызывает get_tasks() через MCP (crm_server.py), находит
релевантный тикет, дополняет ответ данными из RAG (FAQ).

Запуск:
    python -m llm_agent_v2.support_server
    # → http://localhost:8001
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_agent_v2.config import SessionConfig
from llm_agent_v2.llm_client import LLMClient
from llm_agent_v2.mcp_manager import MCPManager
from llm_agent_v2.rag.retriever import RAGRetriever
from llm_agent_v2.agent_utils import run_with_tools, SUPPORT_SYSTEM_PROMPT

_ROOT = Path(__file__).parent.parent
_CRM_SERVER = _ROOT / "llm_mcp/crm_with_task/crm_server.py"
_RAG_INDEX_DIR = str(Path(__file__).parent / "rag_index")
_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_client: LLMClient
_mcp: MCPManager
_retriever: RAGRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _mcp, _retriever

    config = SessionConfig()
    _client = LLMClient(config)
    # Подключаем только оригинальный crm_server.py (get_tasks, create_task, ...)
    _mcp = MCPManager(servers=[("crm_tasks", _CRM_SERVER)])
    _retriever = RAGRetriever(index_dir=_RAG_INDEX_DIR)

    print(f"[Модель: {config.model}]")
    print("Подключение CRM MCP-сервера...")
    await _mcp.connect_all()
    print(f"Инструменты: {[t['function']['name'] for t in _mcp.tools_openai]}")
    yield
    await _mcp.close()


app = FastAPI(title="Support AI Assistant", lifespan=lifespan)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    html_path = _STATIC_DIR / "support.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"status": "Support AI running. No HTML found in static/"}


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """Основной чат-эндпоинт.

    Flow:
    1. RAG-поиск по вопросу → FAQ-чанки
    2. Собрать messages: system prompt + FAQ context + история + вопрос
    3. run_with_tools() — агент сам вызывает get_tasks() и находит тикет
    4. Вернуть ответ
    """
    # 1. RAG
    rag_context = _build_rag_context(req.message)

    # 2. Build messages
    messages: list[dict] = [{"role": "system", "content": SUPPORT_SYSTEM_PROMPT}]

    if rag_context:
        messages.append({"role": "system", "content": rag_context})

    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": req.message})

    # 3. Run agent (will call get_tasks via MCP autonomously)
    try:
        reply = await run_with_tools(_client, _mcp, messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(reply=reply)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rag_context(query: str) -> str:
    try:
        chunks = _retriever.search(query, strategy="structure", top_k=3)
        if not chunks:
            return ""
        parts = [f"[{c.source} / {c.section or 'FAQ'}]\n{c.text}" for c in chunks]
        return "[FAQ Context — используй при ответе]\n\n" + "\n\n".join(parts)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    sys.path.insert(0, str(_ROOT))
    uvicorn.run("llm_agent_v2.support_server:app", host="0.0.0.0", port=8001, reload=False)
