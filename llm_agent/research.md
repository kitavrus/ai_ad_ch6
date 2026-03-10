# Research: Анализ проекта ai_ad_ch6

## 1. Обзор проекта

CLI-чатбот с OpenAI-совместимым API. Основные возможности:
- Трёхуровневая система памяти (краткосрочная / рабочая / долгосрочная)
- Три подключаемые стратегии контекста
- Постоянное управление сессиями (JSON на диске)
- Конечный автомат планирования задач (TaskPlan + TaskStep)
- Режим Plan: stateful AI agent с инвариантами и циклом самокоррекции

**Стек**: Python 3, openai>=2.21.0, pydantic>=2.0.0, python-dotenv>=1.0.0

**Точка входа**: `script.py` → `chatbot.main:main()`

**API**: OpenAI-совместимый, base_url = `https://routerai.ru/api/v1` (по умолчанию)

---

## 2. Модульная структура

| Модуль | Строк | Роль |
|--------|-------|------|
| `script.py` | ~5 | Обёртка над chatbot.main |
| `chatbot/main.py` | 2385 | Диалоговый цикл, все диспетчеры команд, интеграция с LLM |
| `chatbot/cli.py` | ~350 | argparse + parse_inline_command (разбор /команд) |
| `chatbot/config.py` | ~72 | Константы, SessionConfig (Pydantic) |
| `chatbot/models.py` | ~357 | Все Pydantic v2 модели данных |
| `chatbot/memory.py` | ~348 | ShortTerm/Working/LongTerm + фасад Memory |
| `chatbot/memory_storage.py` | ~412 | Сохранение памяти, профилей пользователя |
| `chatbot/storage.py` | ~84 | Сохранение сессий, логирование метрик |
| `chatbot/context.py` | ~841 | 3 стратегии контекста, builder prompts, agent prompts |
| `chatbot/task_storage.py` | ~250 | Сохранение планов задач и шагов |
| `chatbot/project_storage.py` | ~70 | Сохранение проектов (группирование планов) |

---

## 3. Все Pydantic v2 модели

### Перечисления (Enums)

| Enum | Значения |
|------|---------|
| `ContextStrategy` | SLIDING_WINDOW, STICKY_FACTS, BRANCHING |
| `TaskPhase` | PLANNING, EXECUTION, VALIDATION, DONE, PAUSED, FAILED |
| `StepStatus` | PENDING, IN_PROGRESS, DONE, SKIPPED, FAILED |
| `MemoryFactor` | PREFERENCE, DECISION, FACT, SUMMARY |

### Модели сообщений и метрик

**ChatMessage**: `role`, `content`, `tokens: Optional[TokenUsage]`
- Метод `to_api_dict()` → убирает поле tokens для API-запроса

**TokenUsage**: `prompt: int`, `completion: int`, `total: int` (все >= 0)

**RequestMetric**: `model`, `endpoint`, `temp`, `ttft`, `req_time`, `total_time`, `tokens`, `p_tokens`, `c_tokens`, `cost_rub`

### Модели памяти

**ShortTermMemory**: `messages: List[dict]`, `session_id`, `created_at`, `expires_at`

**WorkingMemory**: `current_task`, `task_status`, `task_context: Dict`, `recent_actions: List[str]`, `user_preferences: Dict`

**LongTermMemory**: `profile: UserProfile`, `decisions_log: List[dict]`, `knowledge_base: Dict[str, str]`

**Memory** (фасад): `short_term: ShortTermMemory`, `working: WorkingMemory`, `long_term: LongTermMemory`

### Модели профиля и веток

**UserProfile**: `name`, `style: Dict`, `format: Dict`, `constraints: List[str]`, `custom: Dict`
- Метод `to_system_prompt() -> str`
- Метод `is_empty() -> bool`

**StickyFacts**: `facts: Dict[str, str]`
- Методы: `get(key)`, `set(key, value)`, `update_from_message()`, `to_list()`

**DialogueCheckpoint**: `messages_snapshot`, `facts_snapshot`, `created_at`

**Branch**: `branch_id: str` (UUID[:8]), `name`, `checkpoint: DialogueCheckpoint`, `messages: List[ChatMessage]`

### Модели задач

**TaskStep**: `step_id`, `task_id`, `index` (1-based), `title`, `description`, `status: StepStatus`, `notes`, `result`, `created_at`, `started_at`, `completed_at`

**TaskPlan**: `task_id`, `profile_name`, `name`, `description`, `phase: TaskPhase`, `step_ids: List[str]`, `total_steps`, `current_step_index` (0-based), `result`, `failure_reason`, `clarifications: List[Dict]`, `project_id`, `model`

**Project**: `project_id`, `name`, `profile_name`, `description`, `plan_ids: List[str]`

### Модели сессии

**AgentMode**: `enabled: bool`, `invariants: List[str]`, `max_retries: int` (1-10)

**DialogueSession** (сериализуемое): полное состояние сессии в JSON
- `messages`, `context_summary`, `sticky_facts`, `branches`, `active_branch_id`
- `active_task_id`, `active_task_ids`, `active_project_id`
- `agent_mode`, `plan_dialog_state`

**SessionState** (рабочее): то же + `memory: Memory`, `profile_name`, `plan_draft_steps`, `plan_draft_description`

### Конечный автомат задач

```python
ALLOWED_TRANSITIONS = {
    PLANNING:   {EXECUTION, FAILED},
    EXECUTION:  {VALIDATION, PAUSED, FAILED},
    VALIDATION: {DONE, EXECUTION},
    PAUSED:     {EXECUTION, FAILED},
    DONE:       set(),
    FAILED:     set(),
}
```

---

## 4. Хранилище (структура файловой системы)

```
dialogues/
  {profile_name}/
    profile.json                   ← UserProfile (создаётся при запуске)
    session_<timestamp>_<model>.json ← DialogueSession (после каждого LLM-ответа)
    metrics/
      session_<id>_req_<idx:04d>.log ← RequestMetric (за каждый запрос)
    memory/
      short_term/
        session_<id>_<timestamp>.json
      working/
        task_<name>_<timestamp>.json
      long_term/
        profile_<name>_<timestamp>.json
      state_<timestamp>.json       ← экспорт всей памяти
    tasks/
      {task_id}/
        plan.json                  ← TaskPlan
        step_001.json              ← TaskStep (1-based)
        step_002.json
        ...
        result/                    ← файлы результатов (из builder)
    projects/
      {project_id}/
        project.json               ← Project
```

**Формат**: JSON, UTF-8, indent=2, ensure_ascii=False

---

## 5. Ключевые паттерны

### Построение контекста для LLM

Диспетчер `build_context_by_strategy()` в `context.py` выбирает стратегию:

| Стратегия | Структура контекста | Когда обновляется |
|-----------|--------------------|--------------------|
| `sliding_window` | [системное] → [summary] → [последние N] | Каждые 10 сообщений → LLM-суммаризация |
| `sticky_facts` | [системное] → [факты] → [последние N] | После каждого обмена → LLM извлекает факты |
| `branching` | [системное из checkpoint] → [факты] → [сообщения ветки] | При /checkpoint + /branch |

### Двухуровневое сохранение состояния

1. **После каждого LLM-ответа**: `save_session(DialogueSession, path)`
2. **После ключевых операций** (устойчивость к сбоям):
   - `/task new` → сохранение после `state.active_task_id = task_id`
   - `/plan builder` → сохранение после каждого `save_task_step`

### FSM Plan-диалога (`plan_dialog_state`)

```
None
  └─ /plan on ──────────────────→ awaiting_task
                                      │ пользователь вводит описание
                                      ↓
                               awaiting_invariants
                                      │ да/нет/skip/готово
                                      ↓
                                   active
                                      │ LLM генерирует Draft Plan
                                      ↓
                                 confirming
                                      │ да → создаёт TaskPlan + TaskStep[]
                                      │ нет → None
                                      ↓
                                    None
```

### Builder: пошаговое выполнение (`/plan builder`)

```
Загрузить план → EXECUTION
  ↓
  Для каждого шага:
    LLM-вызов с промптом (профиль + инварианты + шаг)
    ↓
    validate_draft_against_invariants()
    ↓
    passed? → сохранить step.result → advance
    failed? → retry (до max_retries)
               ↓ исчерпаны попытки
               generate_clarification_question() → пользователь уточняет
               ↓ исчерпаны clarifications
               _prompt_invariant_resolution(): edit N / remove N / abort
  ↓
  Все шаги → VALIDATION
  Собрать plan.result из step.result всех шагов
```

### Агентский режим (Agent Mode)

- Системный промпт: `build_agent_system_prompt(profile, state_vars, invariants)`
- Формат вывода: `**Response:**` + `**Questions:**` + `**State Update:**`
- После каждого ответа: State Update → `working.preferences`
- Валидация: отдельный LLM-вызов против каждого инварианта (PASS/FAIL)

### Метрики за запрос

```python
cost_rub = (total_tokens / 1000.0) * 0.0015 * 100.0  # USD_PER_1K * RUB_PER_USD
```

Логируются TTFT (time to first token), время запроса, токены, стоимость в RUB.

---

## 6. Конфигурация по умолчанию

| Константа | Значение |
|-----------|---------|
| `DEFAULT_MODEL` | `inception/mercury-coder` |
| `DEFAULT_TEMPERATURE` | `0.7` |
| `DEFAULT_PROFILE` | `"default"` |
| `CONTEXT_RECENT_MESSAGES` | `10` |
| `CONTEXT_SUMMARY_INTERVAL` | `10` |
| `USD_PER_1K_TOKENS` | `0.0015` |
| `RUB_PER_USD` | `100.0` |

---

## 7. CLI-аргументы запуска

| Флаг | Описание |
|------|---------|
| `-m`, `--model` | Модель API |
| `-u`, `--base-url` | Базовый URL API |
| `--max-tokens` | Максимум токенов в ответе |
| `-T`, `--temperature` | Температура [0.0-2.0] |
| `-p`, `--top-p` | Top-p nucleus sampling |
| `-k`, `--top-k` | Top-k (через extra_body) |
| `--system-prompt` | Системный промпт |
| `--initial-prompt` | Начальное сообщение |
| `--resume` | Загрузить последнюю сессию (только для default профиля) |
| `--strategy` | Стратегия контекста: sliding_window / sticky_facts / branching |
| `--profile` | Имя профиля (автоматически загружает историю) |

---

## 8. Тестовое покрытие

Текущее покрытие: **95%** (865 тестов)

| Тестовый файл | Что покрывает |
|---------------|--------------|
| `test_memory.py` | Трёхуровневая память, Memory facade |
| `test_storage.py` | Сохранение сессий, метрики |
| `test_task_models.py` | Модели TaskPlan/TaskStep |
| `test_task_cli.py` | CLI-команды задач |
| `test_task_storage.py` | Хранение планов/шагов |
| `test_agent.py` | AgentMode, /plan on FSM, инварианты |
| `test_cli_coverage.py` | Ветки cli.py |
| `test_context_coverage.py` | Ветки context.py |
| `test_main_helpers.py` | Хелперы main.py |
| `test_main_coverage2.py` | Глубокие ветки main.py |
| `test_project.py` | Проекты |
