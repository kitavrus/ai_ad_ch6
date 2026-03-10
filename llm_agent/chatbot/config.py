"""Конфигурация приложения: константы и Pydantic-модель настроек."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Глобальные константы (загружаются из .env)
# ---------------------------------------------------------------------------

BASE_URL: str = os.getenv("BASE_URL", "https://routerai.ru/api/v1")
API_KEY: Optional[str] = os.getenv("API_KEY")

DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "inception/mercury-coder")
_max_tokens_env = os.getenv("DEFAULT_MAX_TOKENS", "")
DEFAULT_MAX_TOKENS: Optional[int] = int(_max_tokens_env) if _max_tokens_env else None
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "50"))

# Стоимость токенов
USD_PER_1K_TOKENS: float = float(os.getenv("USD_PER_1K_TOKENS", "0.0015"))
RUB_PER_USD: float = float(os.getenv("RUB_PER_USD", "100.0"))

# Управление контекстом
CONTEXT_RECENT_MESSAGES: int = int(os.getenv("CONTEXT_RECENT_MESSAGES", "10"))
CONTEXT_SUMMARY_INTERVAL: int = int(os.getenv("CONTEXT_SUMMARY_INTERVAL", "10"))

# Директория для сохранения сессий
DIALOGUES_DIR: str = os.getenv("DIALOGUES_DIR", "dialogues")
DEFAULT_PROFILE: str = os.getenv("DEFAULT_PROFILE", "default")


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
    initial_prompt: Optional[str] = Field(
        default=None, description="Начальное сообщение пользователя (seed)"
    )
