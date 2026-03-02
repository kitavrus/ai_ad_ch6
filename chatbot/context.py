"""Управление контекстом диалога: три стратегии + суммаризация."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from chatbot.config import CONTEXT_RECENT_MESSAGES, CONTEXT_SUMMARY_INTERVAL
from chatbot.models import (
    Branch,
    ChatMessage,
    ContextStrategy,
    DialogueCheckpoint,
    StickyFacts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Суммаризация (общая утилита, используется стратегией Sliding Window)
# ---------------------------------------------------------------------------


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


# ===========================================================================
# СТРАТЕГИЯ 1: Sliding Window
# ===========================================================================


def build_context_sliding_window(
    messages: List[ChatMessage],
    context_summary: str = "",
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> List[dict]:
    """Собирает контекст: системное сообщение + опциональный summary + последние N.

    Всё, что выходит за пределы окна recent_n, отбрасывается.
    Если есть накопленный summary — вставляется после системного сообщения.

    Args:
        messages: Полная история сообщений.
        context_summary: Накопленное резюме (опционально).
        recent_n: Размер скользящего окна.

    Returns:
        Список словарей {"role": ..., "content": ...} для API.
    """
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    window = non_system[-recent_n:] if len(non_system) > recent_n else non_system

    result: List[dict] = [m.to_api_dict() for m in system_msgs]

    if context_summary:
        result.append({
            "role": "user",
            "content": f"[Резюме предыдущего диалога для контекста]\n{context_summary}",
        })
        result.append({
            "role": "assistant",
            "content": "Понял, учту контекст из резюме.",
        })

    result.extend(m.to_api_dict() for m in window)
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
    Сообщения старше последних `recent_n` сворачиваются в резюме.

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
            ChatMessage(role="user", content=f"[Предыдущее резюме]\n{context_summary}"),
            ChatMessage(role="assistant", content="Принято."),
        ]

    new_summary = summarize_messages(client, model, summary_context + to_summarize)

    if new_summary:
        context_summary = new_summary
        logger.info("Summary обновлён: свёрнуто %d сообщений.", len(to_summarize))
        print(f"\n[Контекст: {len(to_summarize)} старых сообщений свёрнуто в summary]\n")

    messages = system_msgs + non_system[-recent_n:]
    return messages, context_summary, True


# Псевдоним для обратной совместимости
def build_context_for_api(
    messages: List[ChatMessage],
    context_summary: str,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> List[dict]:
    """Обратносовместимая обёртка над build_context_sliding_window."""
    return build_context_sliding_window(messages, context_summary, recent_n)


# ===========================================================================
# СТРАТЕГИЯ 2: Sticky Facts / Key-Value Memory
# ===========================================================================


def extract_facts_from_llm(
    client: OpenAI,
    model: str,
    user_message: str,
    assistant_message: str,
    existing_facts: Dict[str, str],
) -> Dict[str, str]:
    """Вызывает LLM для извлечения/обновления ключевых фактов из обмена.

    Модель возвращает факты в формате «ключ: значение» по одному на строку.
    Возвращает только новые или изменившиеся факты.

    Args:
        client: Клиент OpenAI.
        model: Идентификатор модели.
        user_message: Последнее сообщение пользователя.
        assistant_message: Последний ответ ассистента.
        existing_facts: Уже накопленные факты.

    Returns:
        Словарь новых/обновлённых фактов (может быть пустым).
    """
    existing_block = ""
    if existing_facts:
        lines = "\n".join(f"{k}: {v}" for k, v in existing_facts.items())
        existing_block = f"Текущие факты:\n{lines}\n\n"

    prompt = (
        f"{existing_block}"
        "Проанализируй следующий обмен и извлеки важные факты: "
        "цели, ограничения, предпочтения, решения, договорённости.\n"
        "Верни ТОЛЬКО новые или изменившиеся факты в формате «ключ: значение», "
        "по одному на строку. Если ничего важного нет — верни пустую строку.\n\n"
        f"Пользователь: {user_message}\n"
        f"Ассистент: {assistant_message}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {}
        new_facts: Dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key and value:
                    new_facts[key] = value
        return new_facts
    except Exception as exc:
        logger.warning("Не удалось извлечь факты: %s", exc)
        return {}


def build_context_sticky_facts(
    messages: List[ChatMessage],
    facts: StickyFacts,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> List[dict]:
    """Собирает контекст: системное + блок фактов + последние N сообщений.

    Структура:
      1. Системное сообщение (если есть).
      2. Блок sticky facts (если есть).
      3. Последние recent_n user/assistant сообщений.

    Args:
        messages: Полная история сообщений.
        facts: Текущий набор sticky facts.
        recent_n: Размер скользящего окна.

    Returns:
        Список словарей для API.
    """
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    window = non_system[-recent_n:] if len(non_system) > recent_n else non_system

    result: List[dict] = [m.to_api_dict() for m in system_msgs]

    if facts.facts:
        facts_lines = "\n".join(f"- {k}: {v}" for k, v in facts.facts.items())
        result.append({
            "role": "user",
            "content": f"[Ключевые факты диалога]\n{facts_lines}",
        })
        result.append({
            "role": "assistant",
            "content": "Принял факты, учту их в ответах.",
        })

    result.extend(m.to_api_dict() for m in window)
    return result


# ===========================================================================
# СТРАТЕГИЯ 3: Branching (ветки диалога)
# ===========================================================================


def create_checkpoint(
    messages: List[ChatMessage],
    facts: Optional[StickyFacts] = None,
) -> DialogueCheckpoint:
    """Создаёт snapshot текущего состояния для ветвления.

    Args:
        messages: Текущая история сообщений.
        facts: Текущие sticky facts (опционально).

    Returns:
        Объект DialogueCheckpoint.
    """
    return DialogueCheckpoint(
        messages_snapshot=[m.model_dump() for m in messages],
        facts_snapshot=dict(facts.facts) if facts else {},
        created_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def create_branch(
    name: str,
    checkpoint: DialogueCheckpoint,
) -> Branch:
    """Создаёт новую независимую ветку диалога от checkpoint.

    Ветка получает копию всех сообщений из snapshot'а и продолжает
    диалог независимо от других веток.

    Args:
        name: Имя ветки (напр. «вариант-А»).
        checkpoint: Точка ветвления.

    Returns:
        Новый объект Branch.
    """
    branch_id = str(uuid.uuid4())[:8]
    messages_copy = [ChatMessage(**msg) for msg in checkpoint.messages_snapshot]
    return Branch(
        branch_id=branch_id,
        name=name,
        checkpoint=checkpoint,
        messages=messages_copy,
    )


def switch_branch(
    branch_id: str,
    branches: List[Branch],
) -> Optional[Branch]:
    """Находит ветку по ID или имени.

    Args:
        branch_id: ID (8 символов) или имя ветки.
        branches: Список всех веток.

    Returns:
        Найденная ветка или None.
    """
    for branch in branches:
        if branch.branch_id == branch_id or branch.name == branch_id:
            return branch
    return None


def build_context_branching(
    branch: Branch,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> List[dict]:
    """Собирает контекст из активной ветки диалога.

    Структура:
      1. Системные сообщения из checkpoint.
      2. Факты из checkpoint (если есть).
      3. Последние recent_n сообщений самой ветки (после точки ветвления).

    Args:
        branch: Активная ветка.
        recent_n: Размер скользящего окна для сообщений ветки.

    Returns:
        Список словарей для API.
    """
    base = [ChatMessage(**m) for m in branch.checkpoint.messages_snapshot]
    system_msgs = [m for m in base if m.role == "system"]

    # Сообщения ветки ПОСЛЕ snapshot'а (новые сообщения ветки)
    checkpoint_len = len(branch.checkpoint.messages_snapshot)
    branch_only = branch.messages[checkpoint_len:]

    non_system = [m for m in branch_only if m.role != "system"]
    window = non_system[-recent_n:] if len(non_system) > recent_n else non_system

    result: List[dict] = [m.to_api_dict() for m in system_msgs]

    if branch.checkpoint.facts_snapshot:
        facts_lines = "\n".join(
            f"- {k}: {v}" for k, v in branch.checkpoint.facts_snapshot.items()
        )
        result.append({
            "role": "user",
            "content": f"[Факты на момент ветвления]\n{facts_lines}",
        })
        result.append({
            "role": "assistant",
            "content": "Принял факты из точки ветвления.",
        })

    result.extend(m.to_api_dict() for m in window)
    return result


# ===========================================================================
# Диспетчер: выбор стратегии
# ===========================================================================


def build_context_by_strategy(
    strategy: ContextStrategy,
    messages: List[ChatMessage],
    context_summary: str = "",
    facts: Optional[StickyFacts] = None,
    active_branch: Optional[Branch] = None,
    recent_n: int = CONTEXT_RECENT_MESSAGES,
) -> List[dict]:
    """Диспетчер стратегий — выбирает нужный способ сборки контекста.

    Args:
        strategy: Активная стратегия (SLIDING_WINDOW / STICKY_FACTS / BRANCHING).
        messages: Полная история (для sliding_window и sticky_facts).
        context_summary: Резюме (для sliding_window).
        facts: Sticky facts (для sticky_facts).
        active_branch: Активная ветка (для branching).
        recent_n: Размер скользящего окна.

    Returns:
        Список словарей для отправки в API.
    """
    if strategy == ContextStrategy.SLIDING_WINDOW:
        return build_context_sliding_window(messages, context_summary, recent_n)

    if strategy == ContextStrategy.STICKY_FACTS:
        return build_context_sticky_facts(messages, facts or StickyFacts(), recent_n)

    if strategy == ContextStrategy.BRANCHING:
        if active_branch is not None:
            return build_context_branching(active_branch, recent_n)
        logger.warning("Branching: нет активной ветки, деградируем до sliding window.")
        return build_context_sliding_window(messages, context_summary, recent_n)

    return build_context_sliding_window(messages, context_summary, recent_n)
