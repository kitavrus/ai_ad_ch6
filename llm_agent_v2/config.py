"""Конфигурация: константы подключения к LLM API и Pydantic-модель сессии."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Константы API (загружаются из .env)
# ---------------------------------------------------------------------------

BASE_URL: str = os.getenv("BASE_URL", "https://routerai.ru/api/v1")
API_KEY: Optional[str] = os.getenv("API_KEY")

DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "inception/mercury-coder")
_max_tokens_env = os.getenv("DEFAULT_MAX_TOKENS", "")
DEFAULT_MAX_TOKENS: Optional[int] = int(_max_tokens_env) if _max_tokens_env else None
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "50"))


# ---------------------------------------------------------------------------
# Pydantic-модель конфигурации сессии
# ---------------------------------------------------------------------------


class SessionConfig(BaseModel):
    """Конфигурация одной сессии диалога."""

    model: str = Field(default=DEFAULT_MODEL, description="Идентификатор модели")
    base_url: str = Field(default=BASE_URL, description="Базовый URL API")
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Максимальное число токенов в ответе",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Температура генерации",
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus sampling)",
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=0, description="Top-k")
    system_prompt: Optional[str] = Field(
        default=None, description="Системный промпт"
    )
