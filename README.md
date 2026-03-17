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
- Команды: `/task new|show|list|start|step|pause|resume|done|fail|load|delete|result`
- Проекты (группировка планов): `/project new|list|switch|show|add-plan`
- Builder-режим: LLM автоматически исполняет шаги плана с инвариантной проверкой

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
| Сессия | `/resume`, `/showsummary`, `/settask` |
| Задачи | `/task new\|show\|list\|start\|step\|pause\|resume\|done\|fail\|load\|delete\|result` |
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
python3 script.py --resume            # возобновить последнюю сессию
python3 script.py --profile Igor      # профиль Igor (авто-возобновление)
python3 script.py -m "gpt-4" -T 0.8 --strategy sticky_facts
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
