# AI Ad Ch6 — Интегрированная AI-платформа

Многоуровневая AI-платформа: CLI-чатбот с трёхуровневой памятью, планировщиком задач, RAG-поиском
и агентским режимом; набор MCP-серверов для интеграции с LLM-инструментами; четыре FastAPI-микросервиса
(планировщик, погода, PDF, файлы), используемые чатботом через MCP.

---

## Архитектура

```
llm_agent/          ← CLI-чатбот (Python, основной компонент)
     │
     │  Model Context Protocol (stdio)
     ▼
llm_mcp/            ← MCP-серверы (обёртки над API)
     │
     │  HTTP + API-ключ
     ▼
api_for_mcp/        ← FastAPI-микросервисы

llm_local/          ← Локальный LLM-чат + RAG через Ollama (автономный)
```

---

## Компоненты

### llm_agent/ — CLI-чатбот

Подробная документация: [llm_agent/README.md](llm_agent/README.md) · [llm_agent/CLAUDE.md](llm_agent/CLAUDE.md) · [RAG_QUICKSTART.md](llm_agent/RAG_QUICKSTART.md)

#### Возможности

**Память и контекст**
- Трёхуровневая память: краткосрочная (сообщения сессии), рабочая (задачи/предпочтения), долговременная (профиль/решения)
- Три стратегии контекста: `sliding_window`, `sticky_facts`, `branching`
- Ветки диалога с контрольными точками (`/checkpoint`, `/branch`, `/switch`)

**Планирование задач**
- Конечный автомат: `planning → execution → validation → done` (+ `paused`, `failed`)
- Guards переходов: `pause` — только из `execution`; `resume` — из `paused` или `validation`; `done` — только из `validation`; `fail` — из любого состояния кроме `done`/`failed`
- Команды: `/task new|show|list|start|step|pause|resume|done|fail|load|delete|result`
- Флаг `--plan <name>` поддерживается командами `pause|resume|done|fail|start|step` для адресации конкретного плана
- Проекты (группировка планов): `/project new|list|switch|show|add-plan`
- Builder-режим: LLM автоматически исполняет шаги плана с инвариантной проверкой; при исчерпании retries предлагает разрешение: `edit <N> <текст>`, `remove <N>`, `abort`; LLM-анализ `analyze_invariant_impact` предупреждает о риске ослабления безопасности

**Агентский режим (Plan Mode)**
- `/plan on` — интерактивный онбординг: задача → инварианты → LLM-диалог
- Инварианты: правила, которым должен соответствовать каждый ответ LLM
- Цикл авто-ретраев при нарушении инвариантов (настраивается `/plan retries <n>`)
- `/invariant add|del|edit|list|clear`

**RAG (Retrieval-Augmented Generation)**
- Две стратегии чанкинга: фиксированный размер (`fixed`) и структурная по заголовкам Markdown (`structure`)
- FAISS-индексирование документов с эмбеддингами
- Семантический поиск по индексу
- Сравнение стратегий: `python index_documents.py --compare`
- Индексы хранятся в `llm_agent/rag_index/`

**Профили и хранение**
- Изоляция по профилям: `--profile NAME` (все данные в `dialogues/{profile}/`)
- Автозагрузка последней сессии для именованных профилей
- Уведомления/напоминания: `NotificationServer` + `reminders_storage`
- MCP-интеграция: погода, PDF, планировщик, файлы через `MCPClientManager`

#### Структура пакета

```
llm_agent/
  script.py                   # Точка входа CLI
  chatbot/
    main.py                   # Диалоговый цикл, обработка команд
    cli.py                    # argparse + встроенный парсинг /команд
    config.py                 # Константы (модель, температура, профиль)
    models.py                 # Pydantic-модели: SessionState, TaskPlan, AgentMode, Project…
    memory.py                 # ShortTermMemory / WorkingMemory / LongTermMemory + фасад Memory
    memory_storage.py         # Сохранение всех трёх уровней памяти
    storage.py                # Сохранение сессий + метрики запросов
    context.py                # Стратегии контекста, build_agent_system_prompt, инварианты
    task_storage.py           # Хранение планов/шагов задач
    project_storage.py        # Хранение проектов
    mcp_client.py             # MCP-клиенты: Weather, PDF, Scheduler, SaveToFile
    notification_server.py    # Сервер push-уведомлений (напоминания)
    reminders_storage.py      # Хранение напоминаний
  rag/
    pipeline.py               # IndexingPipeline: файлы → чанки → эмбеддинги → FAISS
    chunking.py               # FixedSizeChunker, StructureChunker
    embeddings.py             # EmbeddingGenerator
    index.py                  # FAISSIndex (build/save/load/search)
    models.py                 # ChunkMetadata, IndexStats
    compare.py                # Сравнение стратегий
  rag_index/                  # Готовые FAISS-индексы
  tests/                      # 865 тестов, покрытие 95%
```

#### Встроенные команды (во время диалога)

| Группа | Команды |
|--------|---------|
| Модель | `/temperature`, `/max-tokens`, `/model` |
| Стратегия контекста | `/strategy`, `/checkpoint`, `/branch`, `/switch`, `/showfacts`, `/setfact`, `/delfact` |
| Память | `/remember`, `/memshow`, `/memstats`, `/memsave`, `/memload`, `/memclear` |
| Профиль | `/profile show\|list\|name\|load\|style\|format\|constraint` |
| Сессия | `/resume` — загрузить последнюю сессию текущего профиля · `/showsummary` · `/settask` |
| Задачи | `/task new\|show\|list\|start\|step\|pause\|resume\|done\|fail\|load\|delete\|result` · флаг `--plan <name>` у `pause\|resume\|done\|fail\|start\|step` |
| Проекты | `/project new\|list\|switch\|show\|add-plan` |
| Агент/Plan | `/plan on\|off\|status\|retries <n>\|builder\|result` |
| Инварианты | `/invariant add\|del\|edit\|list\|clear` |

**Запуск:**
```bash
cd llm_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заполнить API_KEY
python3 script.py                     # новая сессия (профиль default)
python3 script.py --profile Igor      # профиль Igor (авто-возобновление последней сессии)
python3 script.py -m "gpt-4" -T 0.8 --strategy sticky_facts

# Возобновить сессию default-профиля — вводится как команда после запуска:
# /resume
```

**Хранилище данных:**
```
dialogues/
  default/
    profile.json
    session_*.json
    metrics/session_*_req_*.log
    memory/short_term/ | working/ | long_term/
    tasks/{task_id}/plan.json + step_NNN.json
    projects/{project_id}/project.json
```

---

## Примеры сценариев

### Сценарий 1 — Базовый диалог и профили

Демонстрирует запуск, именованный профиль, возобновление сессии.

**Шаг 1 — новая сессия (профиль default):**
```bash
cd llm_agent
python3 script.py
```
```
Сколько уровней памяти в этом чатботе?
exit
```

**Шаг 2 — возобновить сессию default-профиля через /resume:**
```bash
python3 script.py
```
```
/resume
```
Ожидаемый вывод:
```
[Загружена история: 2 сообщений]
>
```
```
Ты упоминал уровни памяти — как хранится рабочая память между сессиями?
```
Ожидаемый результат: ассистент отвечает с учётом предыдущего вопроса.

**Шаг 3 — именованный профиль (авто-возобновление):**
```bash
python3 script.py --profile Igor
```
Ожидаемый вывод:
```
[Профиль: Igor]
[Загружена сессия: session_2026-03-20_...]
>
```

**Шаг 4 — переключить профиль прямо в диалоге:**
```
/profile load Bob
```
Ожидаемый результат: история диалога очищена, загружена последняя сессия Bob
(или создана новая, если сессий нет).

---

### Сценарий 2 — Полный жизненный цикл задачи

Демонстрирует конечный автомат: `planning → execution → validation → done`
с guards переходов и флагом `--plan <name>`.

```
/task new                              # создать план → фаза planning
/task start --plan "рефакторинг"       # planning → execution
/task step --plan "рефакторинг"        # выполнить шаг (остаёмся в execution)
/task pause --plan "рефакторинг"       # execution → paused
/task resume --plan "рефакторинг"      # paused → execution

/task done --plan "рефакторинг"
# ОШИБКА: done допустим только из validation, а не из execution!

/task step --plan "рефакторинг"        # execution → validation (последний шаг)
/task done --plan "рефакторинг"        # validation → done  ✓
```

Ожидаемый вывод при нарушении guard:
```
[Ошибка] Переход done недопустим из фазы execution.
Допустимые переходы из execution: step, pause, fail.
```

---

### Сценарий 3 — Агентский режим с инвариантами (/plan on)

Демонстрирует онбординг, добавление инвариантов, цикл авто-ретраев.

**Шаг 1 — включить агентский режим:**
```
/plan on
```
Ожидаемый результат: LLM спрашивает задачу, формирует начальные инварианты.

**Шаг 2 — просмотреть и настроить инварианты:**
```
/invariant list
/invariant add "Ответ не длиннее 3 абзацев"
/invariant edit 2 "Ответ содержит конкретный пример кода"
```

**Шаг 3 — настроить число ретраев и задать вопрос:**
```
/plan retries 3
```
Ожидаемый результат: при нарушении инварианта LLM автоматически
делает до 3 повторных попыток, затем возвращает итоговый ответ.

---

### Сценарий 4 — Builder-режим и разрешение инвариантов (/plan builder)

Демонстрирует автоматическое исполнение шагов, конфликт инварианта
и диалог разрешения (`edit / remove / abort`).

**Шаг 1 — запустить builder:**
```
/plan builder
```
Ожидаемый результат: онбординг — задача + шаги + инварианты; Builder начинает
исполнять шаги автоматически.

**Шаг 2 — нарушение инварианта + исчерпание retries (на шаге 3):**
```
[Инвариант 2 нарушен. Все попытки исчерпаны.]
Варианты:
  edit 2 <новый текст>  — изменить инвариант
  remove 2              — удалить инвариант
  abort                 — остановить builder
```

**Шаг 3 — изменить инвариант и продолжить:**
```
> edit 2 "Ответ может содержать псевдокод"
[LLM-анализ: изменение не ослабляет безопасность]
Продолжить? (да/нет): да
```
Ожидаемый результат: Builder возобновляет шаг 3 с обновлённым инвариантом.

---

### Сценарий 5 — Проекты с несколькими планами

Демонстрирует группировку планов в проект, параллельную работу над ними.

**Шаг 1 — создать проект и планы:**
```
/project new "Sprint-1"
/task new           # → создаёт план A (planning)
/task new           # → создаёт план B (planning)
```

**Шаг 2 — добавить планы в проект:**
```
/project add-plan "Sprint-1" <plan_id_A>
/project add-plan "Sprint-1" <plan_id_B>
/project show "Sprint-1"    # список планов и их текущие статусы
```

Ожидаемый вывод:
```
Проект: Sprint-1
  [plan_id_A] план A — planning
  [plan_id_B] план B — planning
```

**Шаг 3 — параллельная работа:**
```
/task start --plan "план A"
/task start --plan "план B"
/task pause --plan "план A"   # execution → paused
/task step  --plan "план B"   # план B продолжает независимо
```

---

---

### llm_local/ — Локальный LLM-чат и RAG (Ollama)

Два инструмента для работы с локальными моделями через [Ollama](https://ollama.com/) — без внешних сервисов генерации.

#### `main.py` — интерактивный чат

Минималистичный CLI-чат, работающий с локальными моделями.

**Возможности:**
- Диалог с локальной моделью (по умолчанию `qwen3:14b`)
- Команды во время диалога: `/model <name>`, `/temperature <0-2>`, `/status`, `/help`
- Валидация температуры с понятными сообщениями об ошибках

**Запуск:**
```bash
# Установить и запустить Ollama: https://ollama.com/
ollama serve       # запустить сервер Ollama
ollama pull qwen3:14b

cd llm_local
pip install ollama
python3 main.py
```

```
Модель: qwen3:14b | температура: 0.7
Команды: /model <name>, /temperature <0-2>, /status, /help, exit
> Привет!
> /model llama3.2
> /temperature 0.5
> exit
```

#### `rag_local.py` — RAG-пайплайн с локальной генерацией

`LocalRAGPipeline` — подключает локальную LLM к существующему FAISS-индексу из `llm_agent/rag_index/`.
Retrieval (FAISS-поиск) полностью локальный. Embeddings для запроса используют тот же API, что при построении индекса (`text-embedding-3-small`). Генерация ответа — через Ollama (локально) или OpenAI (для сравнения).

| Метод | Backend | RAG |
|-------|---------|-----|
| `ask_local_norag(q)` | Ollama | Нет |
| `ask_local(q)` | Ollama + FAISS | Да |
| `ask_cloud_norag(q)` | OpenAI | Нет |
| `ask_cloud(q)` | OpenAI + FAISS | Да |

Каждый метод возвращает `{answer, search_time, generate_time, total_time, char_count, rag_used}`.

**Интерактивный тест (3 вопроса):**
```bash
cd /путь/к/проекту
source llm_agent/.venv/bin/activate
python llm_local/rag_local.py
```

#### `compare_local_cloud.py` — сравнение Local vs Cloud

Прогоняет 10 контрольных вопросов в 4 режимах и сохраняет Markdown-отчёт с тайминговой таблицей.

```bash
# Только локальные режимы (API-ключ не нужен)
python llm_local/compare_local_cloud.py --modes LOCAL_NORAG,LOCAL_RAG --output local_report.md

# Полное сравнение (нужен API_KEY в .env)
python llm_local/compare_local_cloud.py --output full_report.md
```

Отчёт включает: ответ каждого режима, время генерации, количество символов и сводную таблицу средних тайминг по режимам.

**Структура файлов:**
```
llm_local/
  main.py              # CLI-чат с Ollama
  rag_local.py         # LocalRAGPipeline (4 режима: local/cloud × rag/norag)
  compare_local_cloud.py  # сравнительный скрипт, генерирует Markdown-отчёт
```

---

### api_for_mcp/ — FastAPI-микросервисы

| Сервис | Порт | Назначение |
|--------|------|-----------|
| `scheduler` | 8881 | Управление напоминаниями |
| `weather` | 8882 | Погода для городов России |
| `pdf-maker` | 8883 | Генерация PDF-документов |
| `save_to_file` | 8884 | Сохранение файлов на сервере |

Все эндпоинты (кроме `GET /`) требуют заголовок `X-API-Key`.
Конфигурация через `.env` (см. `.env.example` в каждом сервисе).

```bash
# Прямой запуск (встроенный uvicorn)
python3 api_for_mcp/scheduler/main.py
python3 api_for_mcp/weather/main.py
python3 api_for_mcp/pdf-maker/main.py
python3 api_for_mcp/save_to_file/main.py

# Через uvicorn CLI
uvicorn api_for_mcp.scheduler.main:app --port 8881
uvicorn api_for_mcp.weather.main:app --port 8882
uvicorn api_for_mcp.pdf-maker.main:app --port 8883
uvicorn api_for_mcp.save_to_file.main:app --port 8884
```

---

### llm_mcp/ — MCP-серверы

Обёртки над FastAPI-микросервисами, реализующие протокол MCP (Model Context Protocol).
LLM-агент подключает их как инструменты через stdio-транспорт.

| Сервер | Директория |
|--------|-----------|
| Планировщик | `llm_mcp/scheduler/` |
| Погода | `llm_mcp/weather/` |
| PDF | `llm_mcp/pdf-maker/` |
| Файлы | `llm_mcp/save_to_file/` |

Зависимости: `mcp>=1.0.0`

---

## Быстрый старт

```bash
# 1. Запустить микросервисы
cd api_for_mcp/weather
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env

python3 api_for_mcp/scheduler/main.py &
python3 api_for_mcp/weather/main.py &
python3 api_for_mcp/pdf-maker/main.py &
python3 api_for_mcp/save_to_file/main.py &

# 2. Запустить чатбот
cd llm_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполнить API_KEY
python3 script.py

# 3. (Опционально) Проиндексировать документы для RAG
cd llm_agent
python index_documents.py          # обе стратегии
python index_documents.py --compare  # сравнение стратегий
```

---

## Тесты

```bash
# Чатбот (865 тестов, покрытие 95%)
cd llm_agent
python -m pytest tests/ -v
python -m pytest tests/ --cov=chatbot --cov-report=term-missing

# Только RAG-тесты (44 теста, ключ API не нужен)
python -m pytest tests/test_rag_chunking.py tests/test_rag_index.py tests/test_rag_pipeline.py -v

# FastAPI-сервисы
cd api_for_mcp/scheduler    && python -m pytest tests/ -v  # 51 тест
cd api_for_mcp/weather      && python -m pytest tests/ -v  # 42 теста
cd api_for_mcp/pdf-maker    && python -m pytest tests/ -v  # 23 теста
cd api_for_mcp/save_to_file && python -m pytest tests/ -v  # 26 тестов

# MCP-серверы
cd llm_mcp/pdf-maker     && python -m pytest tests/ -v
cd llm_mcp/save_to_file  && python -m pytest tests/ -v
cd llm_mcp/scheduler     && python -m pytest tests/ -v
cd llm_mcp/weather       && python -m pytest tests/ -v
```

---

## Требования

- Python 3.10+
- `faiss-cpu`, `numpy` — для RAG-индексирования
- API-ключ для OpenAI-совместимого сервиса (чатбот, по умолчанию `inception/mercury-coder`)
- API-ключ для микросервисов (задаётся в `.env` каждого сервиса)
