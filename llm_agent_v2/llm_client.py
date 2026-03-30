"""LLM API client — тонкая обёртка над OpenAI-совместимым API."""

from typing import List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from .config import API_KEY, SessionConfig


class LLMClient:
    """Клиент для OpenAI-совместимого LLM API."""

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self._client = OpenAI(api_key=API_KEY, base_url=self.config.base_url)

    def _build_messages(self, messages: List[dict]) -> list:
        api_messages = []
        if self.config.system_prompt:
            api_messages.append({"role": "system", "content": self.config.system_prompt})
        api_messages.extend(messages)
        return api_messages

    def chat(self, messages: List[dict]) -> str:
        """Отправляет список сообщений и возвращает текст ответа модели."""
        extra = {"top_k": self.config.top_k} if self.config.top_k else None
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=self._build_messages(messages),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body=extra,
        )
        return response.choices[0].message.content or ""

    def chat_raw(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> ChatCompletionMessage:
        """Отправляет сообщения и возвращает полный объект сообщения (с tool_calls)."""
        extra = {"top_k": self.config.top_k} if self.config.top_k else None
        kwargs: dict = dict(
            model=self.config.model,
            messages=self._build_messages(messages),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body=extra,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message
