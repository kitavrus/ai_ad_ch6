# LLM Agent v2

AI-агент с веб-интерфейсом, RAG-поиском по документации и MCP-инструментами для работы с CRM и Git.

---

## Архитектура

```
llm_agent_v2/
  web/api_server.py   ← FastAPI HTTP-сервер (порт 8081)
       │
       │  run_with_tools()
       ▼
  mcp_manager.py      ← подключение MCP-серверов по stdio
       │
       ├── llm_mcp/crm_with_task/crm_server.py   (CRM-задачи)
       └── llm_mcp/git-commands/git_server.py     (Git-операции)
       │
       │  RAG
       ▼
  rag_index/          ← FAISS-индекс (structure + fixed стратегии)
```

---

## Быстрый старт

**Требования:** Python 3.10+

```bash
# 1. Установить зависимости
pip install -r llm_agent_v2/web/requirements.txt
pip install openai mcp

# 2. Создать .env (см. раздел Конфигурация)
cp llm_agent_v2/.env.example llm_agent_v2/.env  # или создать вручную

# 3. Запустить сервер
python -m uvicorn llm_agent_v2.web.api_server:app --port 8081

# 4. Открыть веб-интерфейс
open http://localhost:8081
```

---

## Конфигурация

Файл `llm_agent_v2/.env`:

```env
BASE_URL=https://routerai.ru/api/v1   # OpenAI-совместимый API
API_KEY=sk-...                         # API-ключ
DEFAULT_MODEL=inception/mercury-coder  # Модель по умолчанию

# RAG (эмбеддинги)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=...          # опционально, кастомный endpoint
EMBEDDING_MODEL=text-embedding-3-small

# Опционально
RATE_LIMIT_PER_MINUTE=10     # лимит запросов на IP
MAX_CONTEXT_MESSAGES=20      # глубина истории сессии
SESSION_TTL_SECONDS=3600     # TTL сессии
REVIEW_SECRET=...            # токен для /pr-review
```

---

## API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/` | Веб-чат UI |
| `GET` | `/health` | Статус сервера, RAG, MCP-инструменты |
| `GET` | `/models` | Список доступных моделей |
| `POST` | `/chat` | Отправить сообщение (с MCP-инструментами и RAG) |
| `GET` | `/session/{id}` | История сессии |
| `DELETE` | `/session/{id}` | Удалить сессию |
| `POST` | `/pr-review` | AI-ревью Pull Request (публикует комментарий в GitHub) |

### POST /chat

```json
{
  "message": "Покажи задачи в статусе NEW",
  "session_id": "",          // оставить пустым для новой сессии
  "model": "inception/mercury-coder",
  "temperature": 0.7,
  "max_tokens": 1024,
  "system_prompt": null      // опциональный системный промпт
}
```

---

## MCP-инструменты

При старте сервер автоматически подключает два MCP-сервера:

### CRM-задачи (`llm_mcp/crm_with_task/crm_server.py`)

| Инструмент | Описание |
|-----------|----------|
| `get_tasks(status?)` | Список задач, фильтр по NEW / IN_PROCESS / DONE |
| `create_task(title, description?, assignee?, priority?)` | Создать задачу (priority: LOW/MEDIUM/HIGH) |
| `update_status(task_id, status)` | Сменить статус задачи |
| `delete_task(task_id)` | Удалить задачу |

Данные хранятся в `llm_mcp/crm_with_task/tasks.json`.

### Git-операции (`llm_mcp/git-commands/git_server.py`)

| Инструмент | Описание |
|-----------|----------|
| `get_current_branch(repo_path)` | Текущая ветка |
| `list_files(repo_path, pattern?)` | Список файлов в репозитории |
| `get_diff(repo_path, staged?, file_path?)` | Git diff |

---

## RAG

Векторный поиск по документации — результаты автоматически добавляются в системный промпт при каждом запросе.

**Проиндексированные документы:**
- `llm_agent_v2/support_docs/faq.md` — FAQ по продукту
- `llm_agent_v2/support_docs/crm_guide.md` — инструкции по CRM-процессам
- `llm_agent_v2/HELP_COMMAND.md` — справка по командам
- `llm_agent_v2/PR_REVIEW_SETUP.md` — настройка PR-ревью
- `README.md` — описание проекта

**Переиндексировать:**
```bash
python -m llm_agent_v2.index_documents
```

**Тест поиска:**
```bash
python -m llm_agent_v2.index_documents --query "связаться с клиентом"
```

---

## Структура проекта

```
llm_agent_v2/
  web/
    api_server.py         # FastAPI-приложение, все эндпоинты
    index.html            # Веб-чат UI
    requirements.txt      # Зависимости
  rag/                    # RAG-пайплайн
    retriever.py          # Поиск по индексу
    pipeline.py           # Индексирование документов
    chunking.py           # Fixed / Structure стратегии чанкинга
    embeddings.py         # Генерация эмбеддингов
    index.py              # FAISS-индекс
  support_docs/
    faq.md
    crm_guide.md
  rag_index/              # Файлы FAISS-индексов (не коммитить)
  config.py               # Конфигурация из .env
  llm_client.py           # OpenAI-совместимый клиент
  mcp_manager.py          # Управление MCP-серверами
  agent_utils.py          # Цикл tool-calls (run_with_tools)
  pr_review.py            # Логика AI PR-ревью
  index_documents.py      # CLI для переиндексации RAG

llm_mcp/
  crm_with_task/
    crm_server.py         # MCP-сервер: управление CRM-задачами
    support_crm_server.py # MCP-сервер: тикеты поддержки
    tasks.json            # Хранилище задач
    tests/
  git-commands/
    git_server.py         # MCP-сервер: git-операции
    tests/

USE_CASES.md              # Пошаговые сценарии для ручного тестирования
```

---

## Примеры использования

Полные пошаговые сценарии с ожидаемыми результатами — в файле **[USE_CASES.md](USE_CASES.md)**.

Краткий пример сессии через веб-интерфейс:

> **«Покажи все задачи из CRM»**
> → агент вызывает `get_tasks()`, возвращает таблицу задач

> **«Найди документацию по задаче "Связаться с клиентом Иванов"»**
> → RAG находит релевантные чанки из `crm_guide.md`

> **«Смени статус задачи "Связаться с клиентом Иванов" на IN_PROCESS»**
> → агент вызывает `update_status()`, подтверждает изменение

---

## Тесты

```bash
# CRM-сервер
pytest llm_mcp/crm_with_task/tests/

# Git-сервер
pytest llm_mcp/git-commands/tests/
```
