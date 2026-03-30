"""
HTTP API server for local LLM (Ollama) with rate limiting and session management.

Usage:
    pip install -r requirements_api.txt
    python api_server.py

Environment variables:
    OLLAMA_BASE_URL         - default: http://localhost:11434
    OLLAMA_DEFAULT_MODEL    - default: qwen3:14b
    RATE_LIMIT_PER_MINUTE   - default: 10
    MAX_CONTEXT_MESSAGES    - default: 20
    SESSION_TTL_SECONDS     - default: 3600
    API_PORT                - default: 8000
    API_HOST                - default: 0.0.0.0
"""

import os
import time
import uuid
from collections import defaultdict
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:14b")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
API_PORT = int(os.getenv("API_PORT", "8000"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: str = Field(default="")
    model: str = Field(default=OLLAMA_DEFAULT_MODEL)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)


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
    ollama_available: bool
    ollama_url: str
    active_sessions: int
    rate_limit_per_minute: int
    max_context_messages: int


class ModelsResponse(BaseModel):
    models: list[str]


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
        timestamps = self._log[ip]
        # Keep only entries within the window
        self._log[ip] = [t for t in timestamps if now - t < self.window]
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
        if data is None:
            return []
        return list(data["messages"])

    def add_messages(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        data = self._sessions[session_id]
        data["messages"].append({"role": "user", "content": user_msg})
        data["messages"].append({"role": "assistant", "content": assistant_msg})
        # Trim to max_messages (keep pairs: last N messages)
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

app = FastAPI(
    title="Local LLM API",
    description="HTTP API for chatting with local Ollama LLM",
    version="1.0.0",
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
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health and Ollama availability."""
    ollama_ok = False
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

    return HealthResponse(
        status="ok",
        ollama_available=ollama_ok,
        ollama_url=OLLAMA_BASE_URL,
        active_sessions=session_mgr.count(),
        rate_limit_per_minute=RATE_LIMIT_PER_MINUTE,
        max_context_messages=MAX_CONTEXT_MESSAGES,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available Ollama models."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
            model_names = [m["name"] for m in data.get("models", [])]
            return ModelsResponse(models=model_names)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama is not available")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch models: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, client_ip: str = Depends(get_client_ip)):
    """Send a message and get a response from the local LLM."""
    # Rate limit check
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

    # Trim context: keep last MAX_CONTEXT_MESSAGES before sending to Ollama
    context = messages[-MAX_CONTEXT_MESSAGES:]

    # Call Ollama
    payload = {
        "model": req.model,
        "messages": context,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama is not available at " + OLLAMA_BASE_URL)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Ollama request timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e.response.text}")

    data = r.json()
    assistant_msg = data["message"]["content"]
    tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

    session_mgr.add_messages(session_id, req.message, assistant_msg)

    return ChatResponse(
        response=assistant_msg,
        session_id=session_id,
        model=req.model,
        tokens_used=tokens_used,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"Starting Local LLM API server")
    print(f"  Ollama: {OLLAMA_BASE_URL}  model: {OLLAMA_DEFAULT_MODEL}")
    print(f"  Rate limit: {RATE_LIMIT_PER_MINUTE} req/min per IP")
    print(f"  Max context: {MAX_CONTEXT_MESSAGES} messages")
    print(f"  Session TTL: {SESSION_TTL_SECONDS}s")
    print(f"  Listening on http://{API_HOST}:{API_PORT}")
    print(f"  Web UI: http://localhost:{API_PORT}/")
    print()

    uvicorn.run(app, host=API_HOST, port=API_PORT)
