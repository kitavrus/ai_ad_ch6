# AI Ad Ch6 — Интегрированная AI-платформа

Многоуровневая AI-платформа: CLI-чатбот с трёхуровневой памятью, планировщиком задач, RAG-поиском
и агентским режимом; набор MCP-серверов для интеграции с LLM-инструментами; четыре FastAPI-микросервиса
(планировщик, погода, PDF, файлы), используемые чатботом через MCP.

---

## Архитектура

```
llm_agent/          ← CLI-чатбот v1 (Python, основной компонент)
llm_agent_v2/       ← CLI-агент v2 (async, встроенный RAG, MCPManager, PR Review)
     │
     │  Model Context Protocol (stdio)
     ▼
llm_mcp/            ← MCP-серверы (обёртки над API + git-команды)
     │
     │  HTTP + API-ключ
     ▼
api_for_mcp/        ← FastAPI-микросервисы

llm_local/          ← Локальный LLM-чат + RAG через Ollama (автономный)

.github/workflows/  ← CI/CD: AI PR Review (pr_review.yml)
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

### llm_agent_v2/ — Async CLI-агент v2

Переработанный агент с async-архитектурой, встроенным RAG-пайплайном и MCP-менеджером.

#### Ключевые отличия от v1

- **Async-first** — весь цикл диалога на `asyncio`
- **Встроенный RAG** — FAISS-индекс + две стратегии чанкинга (`FixedSize`, `Structure`) без дополнительных настроек
- **MCPManager** — централизованное async-подключение к MCP-серверам с автодискавери инструментов
- **`/help <вопрос>`** — изолированный запрос, не загрязняющий основную историю диалога:
  - RAG-поиск по документации (top-3 чанка)
  - Git-контекст через MCP (`get_current_branch` + `list_files`)
  - Системный промпт роли «project helper»
- **`_run_with_tools()`** — цикл вызова инструментов, до 10 раундов

#### AI PR Review Pipeline (автоматическое ревью кода)

Автоматическое AI-ревью при открытии/обновлении Pull Request.
GitHub Action отправляет diff на удалённый сервер, сервер запускает RAG + LLM-анализ
и постит структурированный комментарий в PR.

**Архитектура:**
```
PR opened/updated
       │
       ▼
GitHub Action (.github/workflows/pr_review.yml)
  ├── git diff → diff.patch
  ├── git diff --name-only → changed_files
  └── POST http://<server>:8080/pr-review
              │
              ▼
       FastAPI /pr-review endpoint
         ├── Validate X-Review-Token
         ├── RAG: запрос к rag_index/fixed.faiss
         ├── LLM: structured review
         ├── TODO/FIXME/HACK detection
         ├── POST GitHub API → comment in PR
         └── Return { review, has_blocking_issues }
```

**Что делает ревью:**
- **Потенциальные баги** — null-pointer, off-by-one, необработанные исключения, гонки
- **Архитектурные проблемы** — coupling, нарушения SOLID, несогласованные паттерны
- **Рекомендации** — именование, тесты, edge-cases, производительность
- **Блокирующие проблемы** — `TODO` / `FIXME` / `HACK` в добавленных строках → Action завершается с ошибкой

**Endpoint:** `POST /pr-review`

| Поле запроса | Тип | Описание |
|-------------|-----|----------|
| `diff` | `str` | Полный git diff |
| `changed_files` | `list[str]` | Список изменённых файлов |
| `pr_number` | `int` | Номер PR |
| `repo` | `str` | `owner/repo` |
| `github_token` | `str` | GitHub токен для постинга комментария |

| Поле ответа | Тип | Описание |
|-------------|-----|----------|
| `review` | `str` | Текст ревью (Markdown) |
| `comment_url` | `str\|null` | URL комментария в GitHub |
| `has_blocking_issues` | `bool` | Есть ли TODO/FIXME/HACK |
| `blocking_issues` | `list[str]` | Список блокирующих строк |

**Защита:** заголовок `X-Review-Token` проверяется против переменной `REVIEW_SECRET`.

**GitHub Secrets:**

| Name | Описание |
|------|----------|
| `REVIEW_SECRET` | Shared secret для аутентификации endpoint |

Подробное руководство по настройке: [PR_REVIEW_SETUP.md](llm_agent_v2/PR_REVIEW_SETUP.md)

#### Структура пакета

```
llm_agent_v2/
  script.py            # Точка входа (async main)
  config.py            # Константы
  llm_client.py        # OpenAI-совместимый клиент
  mcp_manager.py       # Async MCP-менеджер
  pr_review.py         # AI PR Review pipeline (RAG + LLM → structured review)
  index_documents.py   # Индексация документов для RAG
  rag/                 # RAG-пайплайн: pipeline, chunking, embeddings,
                       #   index, retriever, reranker, query_rewrite, compare
  rag_index/           # Готовые FAISS-индексы
  web/
    api_server.py      # FastAPI: /chat, /pr-review, /health, /models, /session
    index.html         # Веб-интерфейс чата
  HELP_COMMAND.md      # Документация команды /help
  PR_REVIEW_SETUP.md   # Руководство по настройке PR Review
```

#### Запуск

```bash
cd llm_agent_v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполнить API_KEY
python3 script.py
```

```
# Внутри диалога:
/help как работает RAG-пайплайн?   # изолированный запрос с RAG + git-контекстом
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

Инструменты для работы с локальными моделями через [Ollama](https://ollama.com/).
Подробная документация с примерами: [llm_local/GUIDE.md](llm_local/GUIDE.md)

#### Web Chat Interface (FastAPI + Ollama)

**HTTP API + веб-интерфейс для удалённых серверов**

Пакет `api_server.py` развёртывает FastAPI-сервер на порту 8080 с веб-чатом и HTTP API.

**Развёртывание на удалённом сервере (день 31):**
```bash
# На сервере: установить Ollama и Python-зависимости
ollama serve
ollama pull qwen2.5:3b   # ~2 GB, оптимально для 4-8 GB RAM

cd /path/to/llm_local
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_api.txt

# Создать systemd-сервис для API
sudo cat > /etc/systemd/system/llm-api.service << 'EOF'
[Unit]
Description=LLM Chat API Server
After=network.target ollama.service
Requires=ollama.service

[Service]
User=debian
WorkingDirectory=/path/to/llm_local
EnvironmentFile=/path/to/llm_local/.env
ExecStart=/path/to/venv/bin/python api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llm-api
```

**API эндпоинты:**
- `GET /health` — статус сервера + доступность API + RAG
- `GET /models` — список доступных моделей
- `POST /chat` — отправить сообщение (JSON: `message`, `session_id`, `model`, `temperature`, `max_tokens`)
- `POST /pr-review` — AI-ревью PR (JSON: `diff`, `changed_files`, `pr_number`, `repo`, `github_token`)
- `GET /session/{session_id}` — история сессии
- `DELETE /session/{session_id}` — очистить сессию

**Пример запроса:**
```bash
curl -X POST http://server:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","session_id":"user1","model":"qwen2.5:3b","temperature":0.7}'
```

**Веб-интерфейс:** `http://server:8080/` (HTML + JS, подключается к API автоматически)

#### `main.py` — интерактивный чат с пресетами и бенчмарком

Полнофункциональный CLI-чат с ролевыми пресетами, настройкой параметров генерации,
логированием и сравнительным тестированием.

**Запуск:**
```bash
ollama serve
ollama pull qwen3:14b

cd llm_local
pip install ollama
python3 main.py
```

**Команды:**

| Команда | Описание |
|---------|----------|
| `/model <name>` | Сменить модель (в т.ч. квантованные) |
| `/settings k=v …` | Настроить `temperature`, `max_tokens`, `ctx` |
| `/preset <name>` | Выбрать ролевой системный промпт |
| `/models` | Список установленных моделей + советы по квантованию |
| `/benchmark` | Тест текущего пресета (3 вопроса, замер времени/скорости) |
| `/benchmark-all` | Сравнение всех пресетов на одних вопросах → MD-отчёт |
| `/status` | Текущие параметры |
| `/help` | Справка |
| `exit` | Выйти |

Любая неизвестная команда (начинается с `/`) — выводит подсказку, в LLM не отправляется.
Команды без параметров (`/preset`, `/model`, `/settings`) выводят список доступных значений.

**Пресеты (`/preset <name>`):**

| Пресет | Роль | Формат ответов |
|--------|------|----------------|
| `architect` | Архитектор ПО | C4 (Context/Container/Component), DFD, Sequence Diagrams (Mermaid/PlantUML) |
| `analyst` | Бизнес-аналитик | User stories, acceptance criteria, BPMN-процессы |
| `tester` | QA-инженер | Тест-планы, test cases (ID/шаги/ожидаемый результат), edge cases |
| `devops` | DevOps-инженер | CI/CD пайплайны, Docker/K8s YAML, мониторинг, rollback |
| `pm` | Проджект-менеджер | Roadmap, риски (матрица P×I), KPI, коммуникационный план |
| `developer` | Senior-разработчик | SOLID, паттерны, code review, рефакторинг before/after |
| `default` | Без роли | Прямой диалог без системного промпта |

**Настройка параметров:**
```
/settings temperature=0.3 max_tokens=4096 ctx=8192
```

| Параметр | Ollama-опция | Дефолт | Рекомендация для архитектуры |
|----------|-------------|--------|------------------------------|
| `temperature` | `temperature` | 0.7 | 0.2–0.3 |
| `max_tokens` | `num_predict` | 2048 | 4096 (диаграммы длинные) |
| `ctx` | `num_ctx` | 4096 | 8192 |

**Квантование (через имя модели):**
```bash
ollama pull qwen3:14b-q4_K_M   # Q4, ~8 GB, скорость +40%, качество ~95%
ollama pull qwen3:14b-q8_0     # Q8, ~15 GB, качество ≈ fp16
```
```
/model qwen3:14b-q4_K_M
```

**Бенчмарк:**
```
/benchmark          # текущий пресет — сохраняет benchmark_results/{ts}_{model}_{preset}.md
/benchmark-all      # все пресеты   — сохраняет benchmark_results/{ts}_compare_{model}.md
```

Итоговая таблица в сравнительном отчёте:
```
| Пресет    | Вопрос 1 (сим) | Вопрос 2 (сим) | Вопрос 3 (сим) | Всего (s) |
|-----------|---------------|---------------|---------------|-----------|
| default   | 823           | 412           | 198           | 42.3s     |
| architect | 1204          | 380           | 210           | 61.7s     |
| ...
```

Во время ожидания ответа отображается анимированный спиннер с таймером:
```
  ⠹ [architect] думает... 12.4s
```

При исчерпании `max_tokens` выводится предупреждение:
```
⚠️  ОТВЕТ ОБРЕЗАН — закончились токены (max_tokens=2048). Увеличьте: /settings max_tokens=4096
```

**Сохраняемые файлы:**
```
llm_local/
  chat_logs/{ts}_chat.md            # лог диалога в реальном времени (каждый вопрос/ответ)
  benchmark_results/{ts}_{model}_{preset}.md   # одиночный бенчмарк
  benchmark_results/{ts}_compare_{model}.md    # сравнение всех пресетов
```

---

#### `rag_local.py` — RAG-пайплайн с локальной генерацией

`LocalRAGPipeline` — подключает локальную LLM к существующему FAISS-индексу из `llm_agent/rag_index/`.
Retrieval (FAISS-поиск) полностью локальный. Embeddings используют `text-embedding-3-small`.
Генерация — через Ollama (локально) или OpenAI (для сравнения).

| Метод | Backend | RAG |
|-------|---------|-----|
| `ask_local_norag(q)` | Ollama | Нет |
| `ask_local(q)` | Ollama + FAISS | Да |
| `ask_cloud_norag(q)` | OpenAI | Нет |
| `ask_cloud(q)` | OpenAI + FAISS | Да |

```bash
source llm_agent/.venv/bin/activate
python llm_local/rag_local.py
```

#### `compare_local_cloud.py` — сравнение Local vs Cloud

Прогоняет 10 вопросов в 4 режимах, сохраняет Markdown-отчёт.

```bash
python llm_local/compare_local_cloud.py --modes LOCAL_NORAG,LOCAL_RAG --output local_report.md
python llm_local/compare_local_cloud.py --output full_report.md   # нужен API_KEY
```

**Структура файлов:**
```
llm_local/
  main.py                  # CLI-чат: пресеты, настройки, бенчмарк
  rag_local.py             # LocalRAGPipeline (4 режима)
  compare_local_cloud.py   # сравнение local vs cloud
  GUIDE.md                 # подробная документация с примерами
  chat_logs/               # логи диалогов (MD)
  benchmark_results/       # отчёты бенчмарков (MD)
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
| Git-команды | `llm_mcp/git-commands/` |

**Git MCP-сервер** (`git-commands/git_server.py`) предоставляет три инструмента:

| Инструмент | Параметры | Назначение |
|-----------|-----------|-----------|
| `get_current_branch` | `repo_path` | Текущая ветка Git |
| `list_files` | `repo_path`, `pattern` (glob) | Список tracked-файлов (`git ls-files`) |
| `get_diff` | `repo_path`, `staged`, `file_path` | Git diff (staged/unstaged, файл или весь репо) |

Используется агентом v2 для команды `/help` (git-контекст).

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

# 4. (Опционально) Запустить агент v2
cd llm_agent_v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 script.py
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
cd llm_mcp/git-commands  && python -m pytest tests/ -v
```

---

## Развёртывание на удалённом сервере

Для запуска веб-чата с локальной LLM на удалённом сервере требуется:
- **CPU:** 2+ ядра
- **RAM:** 4 GB минимум (для qwen2.5:3b ~2 GB + OS + API), 8+ GB рекомендуется
- **Диск:** 5+ GB для модели + логов
- **Порт:** 8080 для API (по умолчанию)

**Пример:** 185.146.1.116:8080 — боевой сервер, развёрнут через systemd (`llm-api.service`).

На сервере также работает endpoint `/pr-review` для автоматического AI-ревью PR
(см. [PR_REVIEW_SETUP.md](llm_agent_v2/PR_REVIEW_SETUP.md)).

Детали: см. раздел "Web Chat Interface (FastAPI + Ollama)" и "AI PR Review Pipeline" выше.

---

## Требования

- Python 3.10+
- `faiss-cpu`, `numpy` — для RAG-индексирования
- API-ключ для OpenAI-совместимого сервиса (чатбот, по умолчанию `inception/mercury-coder`)
- API-ключ для микросервисов (задаётся в `.env` каждого сервиса)
- **Для llm_local/api_server.py:** `fastapi>=0.110.0`, `uvicorn[standard]>=0.27.0`, `httpx>=0.27.0`, `pydantic>=2.0.0`
