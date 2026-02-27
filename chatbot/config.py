"""Конфигурация приложения: константы и Pydantic-модель настроек."""

import os
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Глобальные константы
# ---------------------------------------------------------------------------

BASE_URL: str = "https://routerai.ru/api/v1"
API_KEY: Optional[str] = os.getenv("API_KEY")

DEFAULT_MODEL: str = "inception/mercury-coder"
DEFAULT_MAX_TOKENS: Optional[int] = None
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 50

# Стоимость токенов
USD_PER_1K_TOKENS: float = 0.0015
RUB_PER_USD: float = 100.0

# Управление контекстом
CONTEXT_RECENT_MESSAGES: int = 10  # сколько последних сообщений держим «как есть»
CONTEXT_SUMMARY_INTERVAL: int = 10  # каждые N сообщений делаем summary старых

# Директория для сохранения сессий
DIALOGUES_DIR: str = "dialogues"
METRICS_DIR: str = "dialogues/metrics"


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
