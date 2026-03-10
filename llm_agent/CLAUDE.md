# CLAUDE.md

Этот файл предоставляет руководство для Claude Code (claude.ai/code) при работе с кодом в этом репозитории.

## Обзор проекта

CLI-чатбот с OpenAI-совместимыми API, включающий трехуровневую систему памяти, подключаемые стратегии контекста, постоянное управление сессиями, конечный автомат планирования задач и режим Plan с циклом самокоррекции (инварианты + авто-ретрай).

## Команды

```bash
# Настройка
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Запуск
python script.py                          # Новая сессия
python script.py --resume                 # Возобновить последнюю сессию
python script.py -m "gpt-4" -T 0.8 --strategy sticky_facts

# Тестирование
python -m pytest tests/ -v
python -m pytest tests/test_memory.py -v  # Отдельный модуль
python -m pytest tests/ -k "memory" -v   # Фильтр по ключевому слову
python -m pytest tests/ --cov=chatbot --cov-report=term-missing  # С покрытием кода
```

Требуется установка `API_KEY` в `.env` (см. `.env.example`).

## Архитектура

### Структура пакетов

```
script.py          # Точка входа (импортирует chatbot.main)
chatbot/
  main.py          # Диалоговый цикл, обработка команд, вызовы API
  cli.py           # Настройка argparse + встроенный парсинг /команд
  config.py        # Константы + SessionConfig (Pydantic)
  models.py        # Все Pydantic-модели (ChatMessage, DialogueSession, AgentMode/TaskPlan и др.)
  memory.py        # Классы трехуровневой памяти + унифицированный фасад Memory
  memory_storage.py # Сохранение всех трех уровней памяти
  storage.py       # Файл сессии + логирование метрик для каждого запроса
  context.py       # Стратегии контекста + функции Plan-режима (build_agent_system_prompt и др.)
  task_storage.py  # Сохранение планов/шагов задач (с привязкой к профилю)
```

### Трехуровневая система памяти

Класс `Memory` объединяет три независимых хранилища:

| Уровень | Класс | Область действия | Путь хранения |
|------|-------|-------|--------------|
| Краткосрочная | `ShortTermMemory` | Сообщения текущей сессии | `dialogues/{profile}/memory/short_term/` |
| Рабочая | `WorkingMemory` | Текущая задача/предпочтения | `dialogues/{profile}/memory/working/` |
| Долгосрочная | `LongTermMemory` | Профиль пользователя, решения | `dialogues/{profile}/memory/long_term/` |

### Стратегии контекста

Три подключаемые стратегии, выбираемые через `--strategy` или встроенную команду `/strategy`:

- **sliding_window** (по умолчанию): Сохраняет последние N сообщений дословно; более старые сообщения суммируются LLM
- **sticky_facts**: Извлекает и добавляет ключевые факты как системное сообщение; управляется через `/setfact`/`/delfact`
- **branching**: Создает контрольные точки (`/checkpoint`) и независимые ветки (`/branch`, `/switch`)

### Хранение с привязкой к профилю

Все данные разделены по пространствам имен `dialogues/{profile_name}/`:

```
dialogues/
  default/
    profile.json          ← UserProfile (автоматически создается при запуске)
    session_*.json        ← полное состояние сессии
    metrics/              ← логи метрик для каждого запроса
    memory/short_term/ | working/ | long_term/
    tasks/{task_id}/      ← планы задач и шаги
      plan.json
      step_001.json ...
  Igor/                   ← именованный профиль через --profile Igor
    ...
```

- `--profile NAME` автоматически возобновляет последнюю сессию для этого профиля
- Профиль по умолчанию требует явного `--resume` для загрузки истории

### Сохранение сессий

Два независимых уровня сохранения:
1. **Файл сессии**: `dialogues/{profile}/session_<timestamp>_<model>.json` — полное состояние (сообщения, стратегия, факты, ветки, active_task_id)
2. **Метрики для каждого запроса**: `dialogues/{profile}/metrics/session_<id>_req_<idx>.log` — TTFT, токены, стоимость в RUB

Дополнительные точки сохранения (устойчивость к сбоям):
- **После `/task new`** — сессия сохраняется сразу после `state.active_task_id = task_id`, до вывода плана пользователю
- **После каждого шага `/plan builder`** — сессия сохраняется сразу после `save_task_step`, до следующего шага

Это гарантирует, что `active_task_id` не теряется при Ctrl+C или обрыве соединения. При `--resume` задача восстанавливается автоматически.

### Встроенные команды (во время диалога)

Формат: `/command value` или `/command=value`

Параметры модели: `/temperature`, `/max-tokens`, `/model`
Память: `/settask`, `/remember`, `/memshow`, `/memstats`, `/memsave`, `/memload`, `/memclear`
Стратегия: `/strategy`, `/checkpoint`, `/branch`, `/switch`, `/showfacts`, `/setfact`, `/delfact`
Сессия: `/resume`, `/showsummary`
Профиль: `/profile show|list|name|load|style|format|constraint`
Задачи: `/task new|show|list|start|step|pause|resume|done|fail|load|delete|result`
Plan: `/plan on|off|status|retries <n>|builder|result`, `/invariant add|del|edit|list|clear`

### Конечный автомат планирования задач

Модели `TaskPlan` / `TaskStep` с фазами: `planning → execution → validation → done` (также `paused`, `failed`).

- Планы хранятся в `dialogues/{profile}/tasks/{task_id}/plan.json`
- Шаги хранятся как `step_001.json`, `step_002.json`, …
- LLM генерирует шаги при `/task new <description>`; трехуровневый парсинг JSON с резервным вариантом нумерованного списка
- Контекст активной задачи добавляется в системный промпт каждого вызова API
- `active_task_id` сохраняется в файле сессии и восстанавливается при возобновлении

#### Агрегация результатов

- `TaskStep.result` — строка с результатом выполнения шага:
  - Вручную: `/task step done <текст>`
  - Автоматически: `_execute_builder_step` сохраняет `step.result = draft` (ответ LLM) перед вызовом `save_task_step`
- `TaskPlan.result` — итог всей задачи:
  - Вручную: `/task done <текст>`
  - Автоматически: `_run_plan_builder` после завершения всех шагов собирает `step.result` в строку формата `"Шаг N. Title:\nresult"` и записывает в `plan.result` → `plan.json`
- `/task result [task_id]` — выводит только результаты (шаги с `[✓]` + итоговый `plan.result`), без лишней информации о статусах
- `/plan result [task_id]` — алиас для `/task result`

**Исправленный баг:** до фикса `_execute_builder_step` выводил `draft` на экран, но не записывал в `step.result`, поэтому `plan.result` всегда оставался пустым после `/plan builder`.

### Режим Plan (Stateful AI Agent)

#### Интерактивный старт `/plan on`

`/plan on` запускает управляемый онбординг из трёх шагов:

1. **`awaiting_task`** — система просит ввести описание задачи
2. **`awaiting_invariants`** — система спрашивает про инварианты:
   - `да` → печатает подсказку, пользователь добавляет через `/invariant add`, затем `готово`
   - `нет` / `skip` / `готово` → сразу запускает LLM-диалог
3. **`active`** — LLM-диалог идёт с контекстом задачи и инвариантами

#### Состояния `plan_dialog_state`

| Состояние | Смысл |
|-----------|-------|
| `"awaiting_task"` | Ждём описание задачи от пользователя |
| `"awaiting_invariants"` | Ждём да/нет/готово про инварианты |
| `"active"` | LLM-диалог планирования идёт |
| `"confirming"` | Подтверждение черновика плана (создать задачи?) |
| `None` | Режим выключен |

#### Агентский цикл (состояние `active`)

1. **Создание промпта** — системный промпт строится из шаблона (Роль + Профиль + Текущее состояние + Инварианты)
2. **Генерация черновика** — обычный вызов LLM
3. **Проверка** — отдельный вызов LLM проверяет черновик против каждого инварианта; если `FAIL` → повтор (до `max_retries`)
4. **Парсинг вывода** — извлекает блоки `**Response:**` и `**State Update:**`; обновления состояния автоматически сохраняются в `working.preferences`

Ключевые функции в `context.py`: `build_agent_system_prompt`, `validate_draft_against_invariants`, `parse_agent_output`, `analyze_invariant_impact`.
Новые функции в `main.py`: `_handle_plan_awaiting_task`, `_handle_plan_awaiting_invariants`, `_prompt_invariant_resolution`.

### Тестовое покрытие

Текущее покрытие: **95%** (865 тестов).

| Модуль | Покрытие |
|--------|---------|
| `cli.py` | 97% |
| `config.py` | 100% |
| `context.py` | 99% |
| `main.py` | 92% |
| `memory.py` | 100% |
| `memory_storage.py` | 100% |
| `models.py` | 100% |
| `storage.py` | 100% |
| `task_storage.py` | 100% |

Тестовые файлы:

| Файл | Что покрывает |
|------|--------------|
| `tests/test_memory.py` | Трёхуровневая память, `Memory` facade |
| `tests/test_storage.py` | Сохранение сессий, метрики |
| `tests/test_task_models.py` | Модели `TaskPlan`/`TaskStep` |
| `tests/test_task_cli.py` | CLI-команды задач |
| `tests/test_task_storage.py` | Хранение планов/шагов |
| `tests/test_agent.py` | Агентский режим, `/plan on` FSM, инварианты |
| `tests/test_cli_coverage.py` | Ветки `cli.py`: алиасы стратегий, setfact, delfact, profile, memory |
| `tests/test_context_coverage.py` | Ветки `context.py`: extract_facts, sticky_facts, branching, builder_prompt |
| `tests/test_memory_storage_coverage.py` | Exception-пути и happy-paths `memory_storage.py` |
| `tests/test_models_coverage.py` | `StickyFacts`, `UserProfile.to_system_prompt` |
| `tests/test_task_storage_coverage.py` | Corrupt JSON, список планов, удаление |
| `tests/test_main_helpers.py` | Хелперы `main.py`: парсинг шагов, plan FSM, команды задач, профили; восстановление веток из сессии; `/memclear all` |
| `tests/test_main_coverage2.py` | Глубокие ветки `main.py`: builder, kick_off_plan, dialog, apply_inline; `/memshow` корректный вывод |

**Изменения (day 15+):**
- `/plan builder` — шаги плана **не могут быть пропущены**; при исчерпании попыток система предлагает только `edit <N> <текст>`, `remove <N>` (с проверкой влияния на другие инварианты) или `abort`
- `/invariant edit <N> <новый текст>` — новая команда для редактирования инварианта по номеру
- `analyze_invariant_impact` (`context.py`) — LLM-анализ последствий изменения/удаления инварианта: не ослабит ли безопасность, не создаст ли противоречие с другими ограничениями
- `_prompt_invariant_resolution` (`main.py`) — интерактивный диалог при неудаче builder; после уточнений предлагает изменить или удалить нарушенный инвариант

**Исправленные баги (day 14):**
- `/memshow` — `mem.long_term.user_profile` → `mem.long_term.profile.name` (AttributeError устранён)
- `/memclear` — теперь уважает параметр: `short`/`short_term`, `working`, `long`/`long_term`, `all`
- `branches` — восстанавливаются при загрузке сессии (`_apply_session_data` + `active_branch_id`)
- `create_at` → `created_at` в `LongTermMemory` (опечатка в поле)
- `/profile load` — сбрасывает `active_task_id`, `agent_mode`, `plan_dialog_state`, токены и ветки перед загрузкой нового профиля

### Ключевые паттерны проектирования

- **Все объекты предметной области используют Pydantic v2** — проверка типов, сериализация, ограничения полей (например, `temperature ∈ [0, 2]`)
- **Интеграция API**: клиент `openai.OpenAI` с настраиваемым `base_url` (по умолчанию: `https://routerai.ru/api/v1`); `top_k` передается через `extra_body`
- **Построение контекста** всегда делегируется функции стратегии в `context.py`; `main.py` только оркестрирует
- **Метрики** вычисляются из временных меток реального времени и логируются независимо от состояния сессии
- **Изоляция профилей**: все функции хранения принимают параметр `profile_name`; `DEFAULT_PROFILE = "default"`

### Конфигурация по умолчанию (из `config.py`)

| Константа | Значение по умолчанию |
|----------|---------|
| `DEFAULT_MODEL` | `inception/mercury-coder` |
| `DEFAULT_TEMPERATURE` | `0.7` |
| `CONTEXT_RECENT_MESSAGES` | `10` |
| `CONTEXT_SUMMARY_INTERVAL` | `10` |
| `USD_PER_1K_TOKENS` | `0.0015` |
| `RUB_PER_USD` | `100.0` |