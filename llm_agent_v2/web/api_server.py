"""
HTTP API server for llm_agent_v2 (OpenAI-compatible API) with RAG, rate limiting and session management.

Usage:
    pip install -r llm_agent_v2/web/requirements.txt
    python -m uvicorn llm_agent_v2.web.api_server:app --port 8081

Environment variables (from llm_agent_v2/.env):
    BASE_URL                - OpenAI-compatible API base URL
    API_KEY                 - API key
    DEFAULT_MODEL           - default model ID
    RATE_LIMIT_PER_MINUTE   - default: 10
    MAX_CONTEXT_MESSAGES    - default: 20
    SESSION_TTL_SECONDS     - default: 3600
    API_PORT                - default: 8081
    API_HOST                - default: 0.0.0.0
"""

import asyncio
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from llm_agent_v2.agent_utils import run_with_tools
from llm_agent_v2.config import BASE_URL, DEFAULT_MODEL, SessionConfig
from llm_agent_v2.llm_client import LLMClient
from llm_agent_v2.mcp_manager import MCPManager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
API_PORT = int(os.getenv("API_PORT", "8081"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
REVIEW_SECRET = os.getenv("REVIEW_SECRET", "")

_BASE_DIR = Path(__file__).parent
_RAG_INDEX_DIR = _BASE_DIR.parent / "rag_index"

# ---------------------------------------------------------------------------
# RAG retriever (optional — loads only if index exists)
# ---------------------------------------------------------------------------

_retriever = None
if (_RAG_INDEX_DIR / "fixed.faiss").exists():
    try:
        from llm_agent_v2.rag.retriever import RAGRetriever
        _retriever = RAGRetriever(index_dir=str(_RAG_INDEX_DIR))
        print(f"[RAG] Loaded index from {_RAG_INDEX_DIR}")
    except Exception as _e:
        print(f"[RAG] Failed to load index: {_e}")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: str = Field(default="")
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    system_prompt: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str
    tokens_used: int


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    messages: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    api_reachable: bool
    api_url: str
    rag_enabled: bool
    mcp_tools: list[str] = []
    active_sessions: int
    rate_limit_per_minute: int
    max_context_messages: int


class ModelsResponse(BaseModel):
    models: list[str]


class PRReviewRequest(BaseModel):
    diff: str = Field(..., description="Full git diff text")
    changed_files: list[str] = Field(default_factory=list)
    pr_number: int = Field(..., ge=1)
    repo: str = Field(..., description="owner/repo")
    github_token: str = Field(..., description="GitHub token for posting comment")


class PRReviewResponse(BaseModel):
    review: str
    comment_url: Optional[str] = None
    has_blocking_issues: bool = False
    blocking_issues: list[str] = []


# ---------------------------------------------------------------------------
# Rate limiter (sliding window, per IP)
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._log: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        self._log[ip] = [t for t in self._log[ip] if now - t < self.window]
        if len(self._log[ip]) >= self.max_requests:
            oldest = self._log[ip][0]
            retry_after = int(self.window - (now - oldest)) + 1
            return False, retry_after
        self._log[ip].append(now)
        return True, 0


# ---------------------------------------------------------------------------
# Session manager (in-memory, TTL cleanup)
# ---------------------------------------------------------------------------


class SessionManager:
    def __init__(self, max_messages: int = 20, ttl: int = 3600):
        self.max_messages = max_messages
        self.ttl = ttl
        self._sessions: dict[str, dict[str, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def _cleanup_expired(self) -> None:
        now = self._now()
        expired = [sid for sid, data in self._sessions.items()
                   if now - data["updated_at"] > self.ttl]
        for sid in expired:
            del self._sessions[sid]

    def get_or_create(self, session_id: str) -> str:
        self._cleanup_expired()
        if not session_id or session_id not in self._sessions:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = {
                "messages": [],
                "created_at": self._now(),
                "updated_at": self._now(),
            }
        return session_id

    def get_messages(self, session_id: str) -> list[dict]:
        data = self._sessions.get(session_id)
        return list(data["messages"]) if data else []

    def add_messages(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        data = self._sessions[session_id]
        data["messages"].append({"role": "user", "content": user_msg})
        data["messages"].append({"role": "assistant", "content": assistant_msg})
        if len(data["messages"]) > self.max_messages:
            data["messages"] = data["messages"][-self.max_messages:]
        data["updated_at"] = self._now()

    def clear(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def count(self) -> int:
        return len(self._sessions)

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_mcp: MCPManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp
    _mcp = MCPManager()
    await _mcp.connect_all()
    print(f"[MCP] Tools loaded: {[t['function']['name'] for t in _mcp.tools_openai]}")
    yield
    await _mcp.close()


app = FastAPI(
    title="LLM Agent Web API",
    description="HTTP API for llm_agent_v2 chat with RAG support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limiter = RateLimiter(max_requests=RATE_LIMIT_PER_MINUTE, window_seconds=60)
session_mgr = SessionManager(max_messages=MAX_CONTEXT_MESSAGES, ttl=SESSION_TTL_SECONDS)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    """Serve the web chat UI."""
    html_path = _BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail=f"UI file not found: {html_path}")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server and API reachability."""
    api_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                BASE_URL.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {os.getenv('API_KEY', '')}"},
            )
            api_ok = r.status_code < 500
    except Exception:
        api_ok = False

    return HealthResponse(
        status="ok",
        api_reachable=api_ok,
        api_url=BASE_URL,
        rag_enabled=_retriever is not None,
        mcp_tools=[t["function"]["name"] for t in (_mcp.tools_openai if _mcp else [])],
        active_sessions=session_mgr.count(),
        rate_limit_per_minute=RATE_LIMIT_PER_MINUTE,
        max_context_messages=MAX_CONTEXT_MESSAGES,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models from the API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                BASE_URL.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {os.getenv('API_KEY', '')}"},
            )
            r.raise_for_status()
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            if models:
                return ModelsResponse(models=models)
    except Exception:
        pass
    return ModelsResponse(models=[DEFAULT_MODEL])


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, client_ip: str = Depends(get_client_ip)):
    """Send a message and get a response from the LLM (with optional RAG context)."""
    # Rate limit
    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Try again in {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )

    # Session
    session_id = session_mgr.get_or_create(req.session_id)
    messages = session_mgr.get_messages(session_id)
    messages.append({"role": "user", "content": req.message})
    context = messages[-MAX_CONTEXT_MESSAGES:]

    # RAG lookup — inject relevant chunks into system prompt
    system_prompt = req.system_prompt or ""
    if _retriever is not None:
        try:
            chunks = _retriever.search(req.message, top_k=3)
            if chunks:
                rag_text = "\n---\n".join(c.text for c in chunks)
                system_prompt += f"\n\nКонтекст из базы знаний:\n{rag_text}"
        except Exception as e:
            print(f"[RAG] retrieval error: {e}")

    # LLM call (synchronous client → offload to thread pool)
    config = SessionConfig(
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        system_prompt=system_prompt.strip() or None,
    )
    llm = LLMClient(config=config)

    try:
        if _mcp and _mcp.tools_openai:
            assistant_msg = await run_with_tools(llm, _mcp, context)
        else:
            assistant_msg = await asyncio.to_thread(llm.chat, context)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    session_mgr.add_messages(session_id, req.message, assistant_msg)

    return ChatResponse(
        response=assistant_msg,
        session_id=session_id,
        model=req.model,
        tokens_used=0,
    )


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get chat history for a session."""
    if not session_mgr.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_mgr.get_messages(session_id)
    return SessionInfo(
        session_id=session_id,
        message_count=len(messages),
        messages=messages,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history."""
    if not session_mgr.clear(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/pr-review", response_model=PRReviewResponse)
async def pr_review(req: PRReviewRequest, request: Request):
    """Run AI code review on a PR diff and post the result as a GitHub comment."""
    # Auth
    token = request.headers.get("X-Review-Token", "")
    if REVIEW_SECRET and token != REVIEW_SECRET:
        raise HTTPException(status_code=403, detail="Invalid review token")

    if not req.diff.strip():
        return PRReviewResponse(review="No changes to review.", comment_url=None)

    # Run review pipeline
    from llm_agent_v2.pr_review import format_comment, run_review as _run_review

    try:
        review_text, blocking = await _run_review(req.diff, req.changed_files)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM review failed: {e}")

    comment_body = format_comment(review_text, req.pr_number, blocking)

    # Post comment to GitHub
    comment_url: Optional[str] = None
    try:
        gh_api_url = f"https://api.github.com/repos/{req.repo}/issues/{req.pr_number}/comments"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                gh_api_url,
                json={"body": comment_body},
                headers={
                    "Authorization": f"Bearer {req.github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if r.status_code == 201:
                comment_url = r.json().get("html_url")
            else:
                print(f"[PR Review] GitHub comment failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[PR Review] GitHub API error: {e}")

    return PRReviewResponse(
        review=review_text,
        comment_url=comment_url,
        has_blocking_issues=bool(blocking),
        blocking_issues=blocking,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"Starting LLM Agent Web API")
    print(f"  API: {BASE_URL}  model: {DEFAULT_MODEL}")
    print(f"  RAG: {'enabled' if _retriever else 'disabled'}")
    print(f"  Rate limit: {RATE_LIMIT_PER_MINUTE} req/min per IP")
    print(f"  Max context: {MAX_CONTEXT_MESSAGES} messages")
    print(f"  Session TTL: {SESSION_TTL_SECONDS}s")
    print(f"  Listening on http://{API_HOST}:{API_PORT}")
    print(f"  Web UI: http://localhost:{API_PORT}/")
    print()

    uvicorn.run(app, host=API_HOST, port=API_PORT)
