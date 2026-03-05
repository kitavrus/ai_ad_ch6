"""Управление контекстом диалога: три стратегии + суммаризация + agent loop."""

from datetime import datetime
import logging
import re
from typing import Dict, List, Optional, Tuple
import uuid

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


# ===========================================================================
# STATEFUL AI AGENT: системный промпт, валидация, парсинг вывода
# ===========================================================================

_AGENT_SYSTEM_TEMPLATE = """\
# ROLE
You are an advanced Stateful AI Agent operating within a strict architectural loop. \
Your goal is to process user queries while maintaining a persistent state and adhering to rigid invariants.

# CONTEXT VARIABLES
- **Profile**: {profile}
- **Current State**: {state}
- **Invariants**: {invariants}
- **User Query**: [see the last user message in the conversation]

# INSTRUCTIONS
Follow this execution loop strictly for every interaction:

1. **BUILD PROMPT**: Synthesize the User Query with the Profile, Current State, and Invariants.
2. **GENERATE DRAFT**: Create a response based on the synthesized context.
3. **VALIDATE (Self-Correction Loop)**:
   - Check your draft against the Invariants.
   - If any invariant is violated — do NOT output it. Correct and retry internally until valid.
4. **CLARIFICATION** (optional): If you need clarifications to build a better plan or response,
   include a **Questions:** block with numbered questions. The system will collect user answers
   and save them to the active plan before the next iteration.
5. **OUTPUT FORMAT** — you MUST output the following labelled blocks in order:

**Response:**
[Your final, validated answer here]

**Questions:**
[Numbered list of clarifying questions if needed, e.g.:
1. What is the deadline?
2. Which technology stack do you prefer?
Omit this block entirely if no clarifications are needed.]

**State Update:**
[New state variables as "key: value" lines. Write "(none)" if no updates.]
"""


def build_agent_system_prompt(
    profile_text: str,
    state_vars: str,
    invariants: List[str],
) -> str:
    """Строит системный промпт для режима Stateful AI Agent.

    Args:
        profile_text: Текстовое представление профиля пользователя.
        state_vars: Текущие переменные состояния (задача, предпочтения).
        invariants: Список строк-инвариантов.

    Returns:
        Готовый системный промпт.
    """
    if invariants:
        inv_text = "\n".join(f"  {i + 1}. {inv}" for i, inv in enumerate(invariants))
    else:
        inv_text = "  (не заданы)"
    return _AGENT_SYSTEM_TEMPLATE.format(
        profile=profile_text or "(не задан)",
        state=state_vars or "(не задан)",
        invariants=inv_text,
    )


def validate_draft_against_invariants(
    client: OpenAI,
    model: str,
    draft: str,
    invariants: List[str],
) -> Tuple[bool, str]:
    """Вызывает LLM для проверки черновика против списка инвариантов.

    Args:
        client: Клиент OpenAI.
        model: Идентификатор модели.
        draft: Черновик ответа агента.
        invariants: Список строк-инвариантов.

    Returns:
        Кортеж (passed, violation_reason). Если passed=True, reason пустой.
    """
    if not invariants:
        return True, ""
    inv_text = "\n".join(f"{i + 1}. {inv}" for i, inv in enumerate(invariants))
    prompt = (
        f"Invariants:\n{inv_text}\n\n"
        f"Response to validate:\n{draft}\n\n"
        "Does this response violate any of the above invariants?\n"
        'Answer ONLY: "PASS" or "FAIL: <reason>"'
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        result = (response.choices[0].message.content or "").strip()
        if result.upper().startswith("PASS"):
            return True, ""
        if result.upper().startswith("FAIL"):
            reason = result[result.find(":") + 1:].strip() if ":" in result else result
            return False, reason
        return True, ""
    except Exception as exc:
        logger.warning("Agent validation error: %s", exc)
        return True, ""


def parse_agent_output(text: str) -> Tuple[str, Dict[str, str]]:
    """Разбирает вывод агента на блоки Response и State Update.

    Args:
        text: Полный текст ответа агента.

    Returns:
        Кортеж (response_text, state_update_dict).
        Если блоков нет — возвращает (text, {}).
    """
    # Ищем блок Response (до Questions, State Update или до конца)
    r_match = re.search(
        r"\*{0,2}Response[:\s]\*{0,2}\s*\n([\s\S]*?)(?:\n\*{0,2}(?:Questions?|State Update)|\Z)",
        text,
        re.IGNORECASE,
    )
    # Ищем блок State Update (до конца)
    s_match = re.search(
        r"\*{0,2}State Update[:\s]\*{0,2}\s*\n([\s\S]*?)$",
        text,
        re.IGNORECASE,
    )

    if not r_match and not s_match:
        return text, {}

    response_text = r_match.group(1).strip() if r_match else text

    state_update: Dict[str, str] = {}
    if s_match:
        block = s_match.group(1).strip()
        if block.lower() != "(none)":
            for line in block.splitlines():
                line = line.strip().lstrip("-").strip()
                if ":" in line and not line.startswith("#"):
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k and v:
                        state_update[k] = v

    return response_text, state_update


def parse_plan_questions(text: str) -> List[str]:
    """Извлекает уточняющие вопросы из блока **Questions:** в выводе агента.

    Args:
        text: Полный текст ответа агента.

    Returns:
        Список вопросов (пустой список если блока нет).
    """
    q_match = re.search(
        r"\*{0,2}Questions?[:\s]\*{0,2}[^\S\n]*\n([\s\S]*?)(?=\n\*{0,2}State Update|\Z)",
        text,
        re.IGNORECASE,
    )
    if not q_match:
        return []
    block = q_match.group(1).strip()
    questions: List[str] = []
    for line in block.splitlines():
        line = line.strip().lstrip("-").lstrip("0123456789.)").strip()
        if line:
            questions.append(line)
    return questions


_PLAN_DIALOG_SYSTEM_TEMPLATE = """\
# ROLE
You are a Planning Assistant. Your job is to help the user define a task and break it \
into concrete implementation steps.

# CONTEXT
- **Invariants** (must be respected in the plan): {invariants}
- **Conversation so far**: [see the messages above]

# INSTRUCTIONS
1. If this is the start of the dialog — ask the user what task they want to plan.
2. As the user provides information — propose a numbered plan of 3-7 concrete steps.
3. Ask clarifying questions about anything unclear or ambiguous.
4. When you have enough information to produce a solid plan — include the **Draft Plan:**
   block with the final steps AND ask the user to confirm.
5. If the user requests changes — update the plan and ask again.

# OUTPUT FORMAT

**Response:**
[Your message to the user — plan, questions, or confirmation request]

**Draft Plan:**
[JSON array when plan is ready, e.g.:
[
  {{"title": "Set up project", "description": "Initialize FastAPI project with dependencies"}},
  {{"title": "Create models", "description": "Define Pydantic models for request/response"}}
]
Omit this block entirely if you are still gathering information.]
"""


def build_plan_dialog_prompt(invariants: List[str]) -> str:
    """Строит системный промпт для режима диалога планирования."""
    inv_text = (
        "\n".join(f"  {i+1}. {inv}" for i, inv in enumerate(invariants))
        if invariants
        else "  (не заданы)"
    )
    return _PLAN_DIALOG_SYSTEM_TEMPLATE.format(invariants=inv_text)


def parse_draft_plan_block(text: str) -> Optional[str]:
    """Извлекает содержимое блока **Draft Plan:** или None."""
    m = re.search(
        r"\*{0,2}Draft Plan[:\s]\*{0,2}[^\n]*\n([\s\S]*?)(?=\n\*\*[^\n]|\Z)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    block = m.group(1).strip()
    return block if block else None


_BUILDER_SYSTEM_TEMPLATE = """\
# ROLE
You are an advanced Stateful AI Agent executing a plan step-by-step.

# CONTEXT VARIABLES
- **Profile**: {profile}
- **Current State**: {state}
- **Invariants**: {invariants}
- **Current Step**: {step_title}
- **Step Description**: {step_description}
- **Previous Steps**: {previous_steps}

# INSTRUCTIONS
1. **BUILD PROMPT**: Synthesize the Step with Profile, State, Invariants.
2. **GENERATE DRAFT**: Execute the step — produce a concrete result.
3. **VALIDATE**: Ensure the result satisfies all Invariants.
4. **OUTPUT FORMAT** — you MUST output the following labelled blocks:

**Response:**
[Execution result for this step]

**State Update:**
[key: value updates, or (none)]
"""


def build_builder_step_prompt(
    profile_text: str,
    state_vars: str,
    invariants: List[str],
    step_title: str,
    step_description: str,
    previous_steps: str,
) -> str:
    """Строит системный промпт для исполнения одного шага плана в режиме builder.

    Args:
        profile_text: Текстовое представление профиля пользователя.
        state_vars: Текущие переменные состояния.
        invariants: Список строк-инвариантов.
        step_title: Заголовок текущего шага.
        step_description: Описание текущего шага.
        previous_steps: Строка с описанием выполненных шагов.

    Returns:
        Готовый системный промпт.
    """
    if invariants:
        inv_text = "\n".join(f"  {i + 1}. {inv}" for i, inv in enumerate(invariants))
    else:
        inv_text = "  (не заданы)"
    return _BUILDER_SYSTEM_TEMPLATE.format(
        profile=profile_text or "(не задан)",
        state=state_vars or "(не задан)",
        invariants=inv_text,
        step_title=step_title,
        step_description=step_description or "(нет описания)",
        previous_steps=previous_steps or "(нет выполненных шагов)",
    )


def generate_clarification_question(
    client: OpenAI,
    model: str,
    step_title: str,
    step_description: str,
    violation: str,
) -> str:
    """Вызывает LLM для генерации уточняющего вопроса при нарушении инварианта.

    Args:
        client: Клиент OpenAI.
        model: Идентификатор модели.
        step_title: Заголовок шага.
        step_description: Описание шага.
        violation: Текст нарушения инварианта.

    Returns:
        Уточняющий вопрос (строка).
    """
    prompt = (
        f"Given step '{step_title}': '{step_description}'. "
        f"Validation failed: '{violation}'. "
        "Generate ONE short clarifying question that would help resolve this. "
        "Return only the question."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Не удалось сгенерировать уточняющий вопрос: %s", exc)
        return f"Как улучшить шаг '{step_title}' с учётом нарушения: {violation}?"


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
