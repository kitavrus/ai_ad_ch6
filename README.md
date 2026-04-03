# CLI Chatbot с системой памяти и управлением задачами

AI-ассистент на Python с трёхуровневой системой памяти (short-term/working/long-term), управлением задачами через State Machine, поддержкой параллельных планов и профилей пользователей.

**Версия:** Day 34+ | **Статус тестов:** ✅ 865 тестов pass | **Модель:** Inception/Mercury-Coder

---

## Архитектура

```
script.py                      ← Entry point
  └─> chatbot/main.py         ← Главный цикл с обработкой команд
       │
       ├─ memory_storage.py    ← Profile-scoped хранилище
       │   ├─ Short-term:  сессионные переменные
       │   ├─ Working:     предпочтения и состояние агента
       │   └─ Long-term:   архивные данные и история
       │
       ├─ task_storage.py      ← Управление планами и задачами
       │   ├─ TaskPlan:     план с шагами (EXECUTION → VALIDATION → DONE)
       │   ├─ Project:      группировка планов
       │   └─ Transitions:  State Machine guards
       │
       ├─ context.py          ← Построение системного промпта
       │   ├─ Agent mode:     инварианты и валидация
       │   └─ RAG retrieval:  поиск по документации
       │
       ├─ llm_client.py        ← OpenAI-совместимый API
       └─ models.py            ← Pydantic v2 модели
```

**Хранилище структурировано по профилям:**
```
dialogues/
  default/
    profile.json              ← UserProfile
    session_*.json
    metrics/session_*_req_*.log
    memory/
      short_term/            ← текущая сессия
      working/               ← preferences, agent state
      long_term/             ← archive
    projects/
      {project_id}/project.json
    tasks/
      {task_id}/plan.json
  Igor1/                      ← профиль через --profile Igor1
    ...
```

---

## Быстрый старт

**Требования:** Python 3.10+

```bash
# 1. Клонировать и установить зависимости
git clone <repo>
cd ai_ad_ch6
pip install -r requirements.txt

# 2. Создать .env (опционально, настройка по умолчанию работает)
cp .env.example .env
# Отредактировать: BASE_URL, API_KEY, DEFAULT_MODEL

# 3. Запустить chatbot
python script.py

# 4. С профилем пользователя (сохраняет историю)
python script.py --profile Igor1

# 5. Возобновить предыдущую сессию
python script.py --resume
```

---

## Основные команды

### Управление профилями

```
/profile show              # текущий профиль
/profile list              # все профили
/profile load <name>       # переключиться на профиль
/profile new <name>        # создать новый профиль
```

### Управление памятью

```
/memory show [short|working|long]    # просмотр уровня памяти
/memory clear <level>                # очистить уровень
/memory search <query>               # поиск в памяти
```

### AI Agent Mode (инварианты и валидация)

```
/agent on                           # включить режим агента
/agent off                          # выключить режим агента
/agent status                       # статус и инварианты
/agent retries <1-10>               # установить макс переходов

/invariant add <текст>              # добавить инвариант
/invariant list                     # показать все инварианты
/invariant edit <N> <новый текст>   # изменить инвариант N
/invariant del <N>                  # удалить инвариант N
/invariant clear                    # очистить все инварианты
```

### Управление задачами и планами

```
/task new <название>               # создать новую задачу
/task list [--plan <name>]          # список активных задач
/task show <id>                     # показать детали задачи
/task start <id> [--plan <name>]    # начать задачу (EXECUTION)
/task step <id>                     # следующий шаг плана
/task pause <id>                    # пауза (только из EXECUTION)
/task resume <id>                   # возобновить (из PAUSED)
/task done <id>                     # завершить (только из VALIDATION)
/task fail <id>                     # отменить (не из DONE/FAILED)

/plan new <название>                # создать новый план
/plan list                          # список планов
/plan builder <--plan <name>>        # интерактивный построитель с инвариантами
/plan show <name>                   # показать план
```

### Управление проектами

```
/project new <name>                 # создать проект
/project list                       # список проектов
/project show <id>                  # показать проект
/project switch <id>                # переключиться на проект
/project add-plan <project_id> <plan_name>  # добавить план в проект
```

### Утилиты

```
/help                              # справка по командам
/clear                             # очистить сессию
/metrics                           # логи запросов текущей сессии
/exit                              # выход
```

---

## Конфигурация

Файл `.env`:

```env
# LLM API
BASE_URL=https://routerai.ru/api/v1    # OpenAI-совместимый API
API_KEY=sk-...                         # API-ключ
DEFAULT_MODEL=inception/mercury-coder  # Модель по умолчанию
DEFAULT_TEMPERATURE=0.7                # Температура (0.0-1.0)

# Профиль и память
DEFAULT_PROFILE=default                # Профиль по умолчанию
MEMORY_RETENTION_DAYS=30               # Дни хранения архива

# Опционально
MAX_RETRIES=3                          # Макс попыток переходов при валидации
DEBUG=false                            # Вывод отладной информации
```

---

## Модели данных (Pydantic v2)

### UserProfile
```python
{
  "name": "Igor1",
  "created_at": "2026-03-05T10:30:00Z",
  "updated_at": "2026-04-03T15:45:00Z",
  "preferences": {...},
  "metrics": {...}
}
```

### TaskPlan
```python
{
  "plan_id": "uuid",
  "name": "Feature: Auth Refactor",
  "project_id": "project-123",     # опционально
  "model": "inception/mercury-coder",
  "steps": [
    {"step": 1, "description": "Design schema", "invariants": [...]},
    {"step": 2, "description": "Implement", "invariants": [...]}
  ],
  "phase": "EXECUTION",  # EXECUTION | VALIDATION | DONE | FAILED
  "created_at": "...",
  "updated_at": "..."
}
```

### AgentMode
```python
{
  "enabled": true,
  "invariants": [
    "Never modify user data without explicit confirmation",
    "Always validate input against schema"
  ],
  "max_retries": 3  # 1-10
}
```

### Project
```python
{
  "project_id": "uuid",
  "name": "Mobile App v2",
  "plan_ids": ["plan-1", "plan-2"],
  "created_at": "...",
  "updated_at": "..."
}
```

---

## State Machine для задач

Граф переходов жёстко определён (guards):

```
┌─ EXECUTION ─────┐
│                 ▼
│            VALIDATION ──── DONE
│                 ▲           │
│                 │           └────(archive)
│                 │
└─────(pause)──PAUSED──(resume)───┘

fail() → FAILED (из любого, кроме DONE/FAILED)
```

**Примеры:**
- `start` задачу → EXECUTION
- `pause` → PAUSED (только из EXECUTION)
- `resume` → EXECUTION (только из PAUSED/VALIDATION)
- `done` → DONE (только из VALIDATION)
- `fail` → FAILED (из любого, кроме DONE)

---

## AI Agent Mode с инвариантами

При `/agent on`:

1. **Вход:** пользователь задаёт инварианты (`/invariant add ...`)
2. **Draft:** LLM генерирует ответ с соответствием инвариантам
3. **Validate:** система проверяет черновик против всех инвариантов
4. **Retry:** при ошибке LLM пересоздаёт черновик (до max_retries)
5. **Parse & Save:** если валидация пройдена → парсируем Response и State Update
6. **Auto-save:** State Update сохраняется в `working.preferences`

**Builder с инвариантами (`/plan builder`):**
- Интерактивно добавляет шаги плана
- Каждый шаг проверяется через инварианты
- При исчерпании retries → предложение редактирования/удаления инварианта
- Анализ влияния: `analyze_invariant_impact()` проверяет противоречия

---

## Структура проекта

```
ai_ad_ch6/
  script.py                      # Entry point
  chatbot/
    main.py                      # Главный цикл
    cli.py                       # Парсинг команд
    models.py                    # Pydantic v2 модели (TaskPlan, Project, AgentMode и т.д.)
    config.py                    # Конфигурация из .env
    llm_client.py                # OpenAI-совместимый клиент
    memory_storage.py            # Profile-scoped хранилище памяти
    task_storage.py              # Управление планами и задачами
    project_storage.py           # Управление проектами
    context.py                   # Построение системного промпта + RAG + Agent mode
    storage.py                   # Логирование метрик и сессий
  tests/
    test_agent.py                # Tests для Agent Mode
    test_invariant_scenarios.py  # Tests для инвариантов
    test_task_state_machine.py   # Tests для State Machine guards
    test_memory_storage.py       # Profile-scoped storage tests
    test_project.py              # Project management tests
    ...                          # 865 total tests

  llm_agent_v2/                  # (Legacy) Web interface + RAG + MCP
  llm_mcp/                       # MCP серверы (CRM, Git, File Manager)

  .env.example                   # Шаблон конфигурации
  requirements.txt               # Зависимости
  README.md                      # Этот файл
```

---

## Примеры использования

### 1. Создание плана с инвариантами

```
> /plan builder --plan "Auth Refactor"
Creating plan...
✓ Added step 1: "Design database schema"
✓ Added step 2: "Implement JWT tokens"
✓ Added step 3: "Write tests"
✓ Plan created with 3 steps
```

### 2. Включение Agent Mode с инвариантами

```
> /agent on
Agent Mode enabled. Add invariants:

> /invariant add "Always validate user input"
> /invariant add "Never store plain-text passwords"
> /invariant list
  1. Always validate user input
  2. Never store plain-text passwords

> User: "Create a login endpoint"
[Draft generated] → [Validated against invariants] → [Response delivered]
✓ Invariant 1: PASS
✓ Invariant 2: PASS
Response: (сгенерированный код)
```

### 3. Управление несколькими профилями

```
> python script.py --profile Igor1
[Профиль: Igor1]
> /memory show short
Short-term memory (session):
  - current_task: "Auth Refactor"
  - branch: "feature/auth"

> /profile list
Profiles:
  ✓ Igor1 (current)
    default

> /profile switch default
[Профиль: default]
```

### 4. Параллельные планы в проекте

```
> /project new "Mobile App v2"
> /project add-plan "Mobile App v2" "Auth"
> /project add-plan "Mobile App v2" "UI"

> /task start plan-auth --plan Auth
> /task start plan-ui --plan UI
[Both tasks run in parallel]

> /task list
  ✓ plan-auth (EXECUTION) [Auth]
  ✓ plan-ui   (EXECUTION) [UI]
```

---

## Тестирование

```bash
# Запустить все тесты
python -m pytest tests/ -v

# Запустить тесты конкретного модуля
pytest tests/test_memory_storage.py -v
pytest tests/test_agent.py -v
pytest tests/test_task_state_machine.py -v

# Тесты с покрытием
pytest tests/ --cov=chatbot --cov-report=html
```

**Статус:** ✅ 865 тестов, все pass

---

## Документация по разделам

- **[USE_CASES.md](USE_CASES.md)** — пошаговые сценарии тестирования
- **[llm_agent_v2/README.md](llm_agent_v2/README.md)** — web interface, RAG, MCP
- **[llm_mcp/](llm_mcp/)** — MCP серверы (CRM, Git, File Manager)

---

## Заметки разработчика

### Profile-scoped storage (день 11+)
- Все данные под `dialogues/{profile_name}/`
- Автоматическое создание профиля при старте
- `--profile Igor1` загружает последнюю сессию Игоря без `--resume`
- `/profile load Bob` переключает историю диалога

### State Machine guards (день 15)
- `ALLOWED_TRANSITIONS` жёстко определяет возможные переходы
- `can_transition(from, to)` проверяет граф
- `pause` только из EXECUTION, `done` только из VALIDATION

### Builder с инвариантами
- `/plan builder` не пропускает шаги без проверки инвариантов
- При исчерпании retries → `_prompt_invariant_resolution`
- `analyze_invariant_impact()` проверяет противоречия перед редактированием

### Параллельные планы (день 15)
- `TaskPlan.project_id: Optional[str]`
- `SessionState.active_project_id: Optional[str]`
- Все task-команды поддерживают `--plan <name>` флаг

---

## Лицензия

MIT
