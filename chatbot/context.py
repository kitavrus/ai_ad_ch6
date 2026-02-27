"""Управление контекстом диалога: суммаризация и скользящее окно."""

import logging
from typing import List, Tuple

from openai import OpenAI

from chatbot.config import CONTEXT_RECENT_MESSAGES, CONTEXT_SUMMARY_INTERVAL
from chatbot.models import ChatMessage

logger = logging.getLogger(__name__)


def summarize_messages(
    client: OpenAI,
    model: str,
    messages: List[ChatMessage],
) -> str:
    """Вызывает LLM для получения краткого резюме списка сообщений.

    Args:
        client: Клиент OpenAI API.
        model: Идентификатор модели для суммаризации.
        messages: Список сообщений для суммаризации.

    Returns:
        Строка-резюме или пустая строка при ошибке.
    """
    if not messages:
        return ""

    dialogue_lines: List[str] = []
    for msg in messages:
        if msg.role in {"user", "assistant"} and msg.content.strip():
            label = "User" if msg.role == "user" else "Assistant"
            dialogue_lines.append(f"{label}: {msg.content.strip()}")

    if not dialogue_lines:
        return ""

    joined = "\n".join(dialogue_lines)
    prompt = (
        "Ниже приведён фрагмент диалога между пользователем и ИИ-ассистентом. "
        "Составь краткое, ёмкое резюме на русском языке, сохранив все ключевые факты, "
        "решения и договорённости. Резюме будет использоваться как контекст "
        f"для продолжения диалога.\n\n{joined}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Не удалось создать summary: %s", exc)
        return ""


def build_context_for_api(
    messages: List[ChatMessage],
    context_summary: str,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> List[dict]:
    """Собирает список сообщений для отправки в API.

    Структура (порядок важен для модели):
      1. Системное сообщение (если есть) — всегда первым.
      2. Сообщение-summary (если есть) — сразу после системного.
      3. Последние recent_n сообщений user/assistant «как есть».

    Args:
        messages: Полная история сообщений.
        context_summary: Накопленное резюме старых сообщений.
        recent_n: Количество последних не-системных сообщений для включения.

    Returns:
        Список словарей {"role": ..., "content": ...} для API.
    """
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    recent = non_system[-recent_n:] if len(non_system) > recent_n else non_system

    result: List[dict] = [m.to_api_dict() for m in system_msgs]

    if context_summary:
        result.append(
            {
                "role": "user",
                "content": f"[Резюме предыдущего диалога для контекста]\n{context_summary}",
            }
        )
        result.append(
            {
                "role": "assistant",
                "content": "Понял, учту контекст из резюме.",
            }
        )

    result.extend(m.to_api_dict() for m in recent)
    return result


def maybe_summarize(
    client: OpenAI,
    model: str,
    messages: List[ChatMessage],
    context_summary: str,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
    interval: int = CONTEXT_SUMMARY_INTERVAL,
) -> Tuple[List[ChatMessage], str, bool]:
    """Проверяет необходимость суммаризации и выполняет её при необходимости.

    Суммаризация происходит каждые `interval` не-системных сообщений.
    Сообщения старше последних `recent_n` сворачиваются в резюме и удаляются
    из messages (остаётся только скользящее окно + системное сообщение).

    Args:
        client: Клиент OpenAI API.
        model: Идентификатор модели.
        messages: Текущая история сообщений.
        context_summary: Текущее накопленное резюме.
        recent_n: Размер скользящего окна (количество сохраняемых сообщений).
        interval: Порог сообщений сверх окна для запуска суммаризации.

    Returns:
        Кортеж (обновлённые messages, обновлённый context_summary, было_ли_обновление).
    """
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]

    excess = len(non_system) - recent_n
    if excess < interval:
        return messages, context_summary, False

    to_summarize = non_system[:-recent_n] if recent_n > 0 else non_system

    summary_context: List[ChatMessage] = []
    if context_summary:
        summary_context = [
            ChatMessage(
                role="user",
                content=f"[Предыдущее резюме]\n{context_summary}",
            ),
            ChatMessage(role="assistant", content="Принято."),
        ]

    new_summary = summarize_messages(client, model, summary_context + to_summarize)

    if new_summary:
        context_summary = new_summary
        logger.info("Summary обновлён: свёрнуто %d сообщений.", len(to_summarize))
        print(f"\n[Контекст: {len(to_summarize)} старых сообщений свёрнуто в summary]\n")

    messages = system_msgs + non_system[-recent_n:]
    return messages, context_summary, True
