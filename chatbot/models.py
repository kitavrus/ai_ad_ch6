"""Pydantic-модели данных: сообщения, метрики, сессия."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Сообщения
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Статистика токенов одного обмена."""

    prompt: int = Field(default=0, ge=0)
    completion: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)


class ChatMessage(BaseModel):
    """Одно сообщение в истории диалога."""

    role: str = Field(description="Роль участника: system | user | assistant")
    content: str = Field(default="")
    tokens: Optional[TokenUsage] = Field(
        default=None,
        description="Метрика токенов (только для assistant-сообщений)",
    )

    def to_api_dict(self) -> Dict[str, str]:
        """Возвращает словарь для отправки в API (без поля tokens)."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Метрики запроса
# ---------------------------------------------------------------------------


class RequestMetric(BaseModel):
    """Метаданные одного API-запроса."""

    model: str
    endpoint: str = "chat"
    temp: float
    ttft: float = Field(description="Time to first token / время вызова API (сек)")
    req_time: float
    total_time: float
    tokens: int
    p_tokens: int
    c_tokens: int
    cost_rub: float


# ---------------------------------------------------------------------------
# Сессия диалога
# ---------------------------------------------------------------------------


class DialogueSession(BaseModel):
    """Полное состояние сессии диалога."""

    dialogue_session_id: str
    created_at: str
    model: str
    base_url: str
    system_prompt: Optional[str] = None
    initial_prompt: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context_summary: str = ""
    turns: int = 0
    last_user_input: Optional[str] = None
    last_assistant_content: Optional[str] = None
    duration_seconds: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    requests: List[RequestMetric] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Состояние текущей сессии (изменяемый объект в ходе диалога)
# ---------------------------------------------------------------------------


class SessionState(BaseModel):
    """Рабочее состояние текущей сессии (не сериализуется напрямую)."""

    model: str
    base_url: str
    max_tokens: Optional[int] = None
    temperature: float
    top_p: float
    top_k: int
    system_prompt: Optional[str] = None
    initial_prompt: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    session_path: Optional[str] = None
    session_id_metrics: str = ""
    dialogue_start_time: float = 0.0
    duration: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    context_summary: str = ""
    request_index: int = 0

    model_config = {"arbitrary_types_allowed": True}
