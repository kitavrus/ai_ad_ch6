"""Модель памяти: три типа - краткосрочная, рабочая, долговременная."""

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# ТИПЫ ПАМЯТИ
# ===========================================================================


class ShortTermMemory(BaseModel):
    """Краткосрочная память: текущий диалог (внутри одного сеанса).
    
    Хранит сообщения текущей сессии, удаляются при завершении сеанса.
    """
    messages: List[dict] = Field(default_factory=list)
    session_id: str = ""
    created_at: str = ""
    expires_at: Optional[str] = None
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def get_recent(self, n: int = 10) -> List[dict]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def clear(self) -> None:
        self.messages = []


class WorkingMemory(BaseModel):
    """Рабочая память: данные текущей задачи.
    
    Хранит контекст выполняемой задачи (цели, статус, метаданные).
    Может быть сохранена в долговременную память.
    """
    current_task: Optional[str] = None
    task_status: str = "new"
    task_context: Dict[str, Any] = Field(default_factory=dict)
    recent_actions: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    
    def set_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.current_task = task
        self.task_status = "in_progress"
        if context:
            self.task_context.update(context)
        self.updated_at = datetime.utcnow().isoformat()
    
    def update_status(self, status: str) -> None:
        self.task_status = status
        self.updated_at = datetime.utcnow().isoformat()
    
    def add_action(self, action: str) -> None:
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]
    
    def set_preference(self, key: str, value: Any) -> None:
        self.user_preferences[key] = value
        self.updated_at = datetime.utcnow().isoformat()
    
    def to_short_term_snapshot(self) -> ShortTermMemory:
        """ Преобразует часть рабочей памяти в краткосрочную (для диалога). """
        snapshot = ShortTermMemory(
            messages=[],
            session_id="working_memory_snapshot",
            created_at=datetime.utcnow().isoformat(),
        )
        if self.current_task:
            snapshot.add_message("user", f"Текущая задача: {self.current_task}")
        if self.user_preferences:
            snapshot.add_message(
                "assistant", 
                f"Мои предпочтения: {self.user_preferences}"
            )
        return snapshot


class LongTermMemory(BaseModel):
    """Долговременная память: профиль, решения, знания.
    
    Сохраняется между сеансами. Содержит:
    - Пользовательский профиль
    - История решений/действий
    - Факты и знания
    """
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    decisions_log: List[dict] = Field(default_factory=list)
    knowledge_base: Dict[str, str] = Field(default_factory=dict)
    create_at: str = ""
    last_accessed: str = ""
    
    def add_decision(self, task: str, decision: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.decisions_log.append({
            "task": task,
            "decision": decision,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.last_accessed = datetime.utcnow().isoformat()
    
    def add_knowledge(self, key: str, value: str) -> None:
        self.knowledge_base[key] = value
        self.last_accessed = datetime.utcnow().isoformat()
    
    def get_decision_history(self, task: Optional[str] = None) -> List[dict]:
        if not task:
            return self.decisions_log
        return [d for d in self.decisions_log if d["task"] == task]
    
    def get_knowledge(self, key: str) -> Optional[str]:
        return self.knowledge_base.get(key)
    
    def get_profile(self, key: Optional[str] = None) -> Any:
        if not key:
            return self.user_profile
        return self.user_profile.get(key)
    
    def set_profile(self, key: str, value: Any) -> None:
        self.user_profile[key] = value
        self.last_accessed = datetime.utcnow().isoformat()


# ===========================================================================
# ЦЕНТРАЛЬНЫЙ КЛАСС MEMORY
# ===========================================================================


class Memory(BaseModel):
    """Центральный класс управления памятью.
    
    Содержит три типа памяти и предоставляет единый интерфейс для работы с ними.
    """
    short_term: ShortTermMemory = Field(default_factory=ShortTermMemory)
    working: WorkingMemory = Field(default_factory=WorkingMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)
    
    def add_user_message(self, content: str) -> None:
        """Добавляет сообщение пользователя в краткосрочную память."""
        self.short_term.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Добавляет ответ ассистента в краткосрочную память."""
        self.short_term.add_message("assistant", content)
    
    def add_to_working_memory(
        self,
        task: Optional[str] = None,
        action: Optional[str] = None,
        preference: Optional[str] = None,
        preference_value: Any = None,
    ) -> None:
        """Добавляет данные в рабочую память."""
        if task:
            self.working.set_task(task)
        if action:
            self.working.add_action(action)
        if preference and preference_value is not None:
            self.working.set_preference(preference, preference_value)
    
    def add_to_long_term(
        self,
        decision: Optional[str] = None,
        task: Optional[str] = None,
        knowledge_key: Optional[str] = None,
        knowledge_value: Optional[str] = None,
        profile_key: Optional[str] = None,
        profile_value: Any = None,
    ) -> None:
        """Добавляет данные в долговременную память."""
        if decision and task:
            self.long_term.add_decision(task, decision)
        if knowledge_key and knowledge_value:
            self.long_term.add_knowledge(knowledge_key, knowledge_value)
        if profile_key and profile_value is not None:
            self.long_term.set_profile(profile_key, profile_value)
    
    def get_short_term_context(self, n: int = 10) -> List[dict]:
        """Возвращает последние n сообщений из краткосрочной памяти."""
        return self.short_term.get_recent(n)
    
    def get_working_context(self) -> dict:
        """Возвращает текущую рабочую память для контекста."""
        context = {}
        if self.working.current_task:
            context["current_task"] = self.working.current_task
        context["user_preferences"] = self.working.user_preferences
        return context
    
    def save_working_to_long_term(self, task_name: Optional[str] = None) -> None:
        """Сохраняет текущее состояние рабочей памяти в долговременную."""
        task = task_name or self.working.current_task or "untitled_task"
        self.long_term.add_decision(
            task=task,
            decision=f"Working memory saved: {self.working.model_dump()}",
            context={"timestamp": datetime.utcnow().isoformat()},
        )
    
    def clear_short_term(self) -> None:
        """Очищает краткосрочную память (при завершении сеанса)."""
        self.short_term.clear()
    
    def get_full_state(self) -> dict:
        """Возвращает полное состояние памяти."""
        return {
            "short_term": self.short_term.model_dump(),
            "working": self.working.model_dump(),
            "long_term": self.long_term.model_dump(),
        }
    
    def load_full_state(self, state: dict) -> None:
        """Загружает полное состояние памяти."""
        if "short_term" in state:
            self.short_term = ShortTermMemory(**state["short_term"])
        if "working" in state:
            self.working = WorkingMemory(**state["working"])
        if "long_term" in state:
            self.long_term = LongTermMemory(**state["long_term"])


# ===========================================================================
# ФАКТОРЫ ПАМЯТИ (для извлечения знаний)
# ===========================================================================


class MemoryFactor(str, enum.Enum):
    """Категории информации для извлечения в долговременную память."""
    
    PREFERENCE = "preference"
    DECISION = "decision"
    FACT = "fact"
    SUMMARY = "summary"


def extract_memory_factors(
    user_content: str,
    assistant_content: str,
    working_memory: WorkingMemory,
) -> List[Dict[str, Any]]:
    """Извлекает важные факторы из диалога для сохранения в долговременную память.
    
    Args:
        user_content: Сообщение пользователя.
        assistant_content: Ответ ассистента.
        working_memory: Текущая рабочая память.
    
    Returns:
        Список извлечённых факторов с типом и содержанием.
    """
    factors: List[Dict[str, Any]] = []
    
    # Проверка на упоминание предпочтений
    preference_keywords = [
        "prefer", "лучше", "предпочитаю", "喜欢", "Хочу", "не люблю", 
        "чаще", "регулярно", "всегда", "никогда"
    ]
    
    for keyword in preference_keywords:
        if keyword in user_content.lower() or keyword in assistant_content.lower():
            factors.append({
                "type": MemoryFactor.PREFERENCE,
                "content": f"User preference: {user_content[:100]}",
                "source": "user",
            })
            break
    
    # Проверка на факты
    fact_keywords = [
        "факт", "истина", "действительно", "на самом деле", 
        "важно", "запомни", "запомни это"
    ]
    
    for keyword in fact_keywords:
        if keyword in user_content.lower() or "это факт" in assistant_content.lower():
            factors.append({
                "type": MemoryFactor.FACT,
                "content": user_content,
                "source": "user",
            })
            break
    
    # Проверка на принятые решения
    if working_memory.current_task and "решение" in assistant_content.lower():
        factors.append({
            "type": MemoryFactor.DECISION,
            "content": f"Decision for task '{working_memory.current_task}': {assistant_content[:100]}",
            "source": "assistant",
        })
    
    # Суммаризация (если пользователь просит)
    if "резюме" in user_content.lower() or "копию" in user_content.lower():
        factors.append({
            "type": MemoryFactor.SUMMARY,
            "content": f"Summary request: {user_content}",
            "source": "user",
        })
    
    return factors
