"""Pydantic-модели данных: сообщения, метрики, сессия."""

from datetime import datetime
import enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Стратегии управления контекстом
# ---------------------------------------------------------------------------


class ContextStrategy(str, enum.Enum):
    """Стратегия управления контекстом диалога."""

    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


# ---------------------------------------------------------------------------
# Режим Stateful AI Agent
# ---------------------------------------------------------------------------


class AgentMode(BaseModel):
    """Конфигурация режима Stateful AI Agent с инвариантами и self-correction."""

    enabled: bool = False
    invariants: List[str] = Field(default_factory=list)
    max_retries: int = Field(default=3, ge=1, le=10)


# ---------------------------------------------------------------------------
# Система управления задачами (Task Planning State Machine)
# ---------------------------------------------------------------------------


class TaskPhase(str, enum.Enum):
    """Фаза задачи в конечном автомате."""

    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"
    PAUSED = "paused"
    FAILED = "failed"


ALLOWED_TRANSITIONS: Dict[TaskPhase, Set[TaskPhase]] = {
    TaskPhase.PLANNING:   {TaskPhase.EXECUTION, TaskPhase.FAILED},
    TaskPhase.EXECUTION:  {TaskPhase.VALIDATION, TaskPhase.PAUSED, TaskPhase.FAILED},
    TaskPhase.VALIDATION: {TaskPhase.DONE, TaskPhase.EXECUTION},
    TaskPhase.PAUSED:     {TaskPhase.EXECUTION, TaskPhase.FAILED},
    TaskPhase.DONE:       set(),
    TaskPhase.FAILED:     set(),
}


def can_transition(from_phase: TaskPhase, to_phase: TaskPhase) -> bool:
    """Возвращает True, если переход между фазами разрешён."""
    return to_phase in ALLOWED_TRANSITIONS.get(from_phase, set())


class StepStatus(str, enum.Enum):
    """Статус отдельного шага задачи."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class TaskStep(BaseModel):
    """Один шаг задачи — хранится в отдельном файле step_NNN.json."""

    step_id: str
    task_id: str
    index: int = Field(ge=1)
    title: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    notes: str = ""
    result: str = ""
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskPlan(BaseModel):
    """Индекс плана задачи — хранится в файле plan.json директории задачи."""

    task_id: str
    profile_name: str = "default"
    name: str
    description: str = ""
    phase: TaskPhase = TaskPhase.PLANNING
    step_ids: List[str] = []
    total_steps: int = 0
    current_step_index: int = 0  # 0-based
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    result: str = ""
    llm_raw_response: Optional[str] = None
    clarifications: List[Dict[str, str]] = Field(default_factory=list)
    """Список уточнений: [{"question": "...", "answer": "..."}]"""
    project_id: Optional[str] = None
    model: Optional[str] = None


class Project(BaseModel):
    """Проект — группирует несколько TaskPlan с единой целью."""

    project_id: str
    name: str
    profile_name: str = "default"
    description: str = ""
    plan_ids: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


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
# Sticky Facts (ключ-значение)
# ---------------------------------------------------------------------------


class StickyFact(BaseModel):
    """Один факт в памяти ключ-значение."""

    key: str = Field(description="Ключ факта (напр: 'цель', 'ограничения', 'предпочтения')")
    value: str = Field(description="Значение факта")
    source_message_index: int = Field(
        default=-1,
        description="Индекс сообщения-источника факта",
    )


class StickyFacts(BaseModel):
    """Коллекция sticky facts."""

    facts: Dict[str, str] = Field(default_factory=dict)

    def get(self, key: str) -> Optional[str]:
        return self.facts.get(key)

    def set(self, key: str, value: str) -> None:
        self.facts[key] = value

    def update_from_message(self, key: str, value: str, msg_index: int) -> None:
        self.facts[key] = value

    def to_list(self) -> List[Dict[str, str]]:
        return [{"key": k, "value": v} for k, v in self.facts.items()]


# ---------------------------------------------------------------------------
# Профиль пользователя
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Профиль пользователя с персональными предпочтениями."""

    name: str = "default"
    style: Dict[str, str] = Field(
        default_factory=dict,
        description="Стиль ответов: тон, краткость, язык. Например: tone=formal, verbosity=concise",
    )
    format: Dict[str, str] = Field(
        default_factory=dict,
        description="Формат вывода: output=markdown, code_blocks=always и т.п.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Ограничения/запреты: 'never use emojis', 'respond in Russian' и т.п.",
    )
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Произвольные дополнительные предпочтения",
    )
    preferred_model: Optional[str] = Field(
        default=None,
        description="Предпочтительная модель для этого профиля",
    )
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_system_prompt(self) -> str:
        """Формирует фрагмент системного промпта из предпочтений профиля."""
        parts = []
        if self.style:
            parts.append("Style: " + "; ".join(f"{k}={v}" for k, v in self.style.items()))
        if self.format:
            parts.append("Format: " + "; ".join(f"{k}={v}" for k, v in self.format.items()))
        if self.constraints:
            parts.append("Constraints: " + "; ".join(self.constraints))
        if self.custom:
            parts.append("Extra: " + "; ".join(f"{k}={v}" for k, v in self.custom.items()))
        return "\n".join(parts)

    def is_empty(self) -> bool:
        """Возвращает True, если в профиле нет ни одного предпочтения."""
        return not any([self.style, self.format, self.constraints, self.custom])


# ---------------------------------------------------------------------------
# Branching (ветки диалога)
# ---------------------------------------------------------------------------


class DialogueCheckpoint(BaseModel):
    """Точка сохранения состояния для ветвления."""

    messages_snapshot: List[Dict[str, Any]] = Field(default_factory=list)
    facts_snapshot: Dict[str, str] = Field(default_factory=dict)
    created_at: str = ""


class Branch(BaseModel):
    """Ветка диалога."""

    branch_id: str = Field(description="Уникальный ID ветки")
    name: str = Field(description="Название ветки")
    checkpoint: DialogueCheckpoint = Field(
        default_factory=DialogueCheckpoint,
        description="Snapshot состояния в момент создания ветки",
    )
    messages: List[ChatMessage] = Field(default_factory=list)


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
    context_strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW
    sticky_facts: Dict[str, str] = Field(default_factory=dict)
    branches: List[Branch] = Field(default_factory=list)
    active_branch_id: Optional[str] = None
    active_task_id: Optional[str] = None
    active_task_ids: List[str] = Field(default_factory=list)
    active_project_id: Optional[str] = None
    agent_mode: Optional[Dict[str, Any]] = None
    plan_dialog_state: Optional[str] = None


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
    context_strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW
    sticky_facts: StickyFacts = Field(default_factory=StickyFacts)
    branches: List[Branch] = Field(default_factory=list)
    active_branch_id: Optional[str] = None
    last_checkpoint: Optional[DialogueCheckpoint] = None
    memory: Any = Field(default=None, description="Объект Memory (три типа памяти)")
    profile_name: str = Field(default="default", description="Активный профиль пользователя")
    active_task_id: Optional[str] = None
    active_task_ids: List[str] = Field(default_factory=list)
    active_project_id: Optional[str] = None
    agent_mode: AgentMode = Field(default_factory=AgentMode)
    plan_dialog_state: Optional[str] = None
    plan_draft_steps: List[dict] = Field(default_factory=list)
    plan_draft_description: str = ""

    model_config = {"arbitrary_types_allowed": True}
