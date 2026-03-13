# AI Ad Ch6 — Интегрированная AI-платформа

Многоуровневая AI-платформа: CLI-чатбот с трёхуровневой памятью и планировщиком задач,
набор MCP-серверов для интеграции с LLM-инструментами и четыре FastAPI-микросервиса
(планировщик, погода, PDF, файлы), используемые чатботом через MCP.

---

## Архитектура

```
llm_agent/          ← CLI-чатбот (Python)
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

Подробная документация: [llm_agent/README.md](llm_agent/README.md)

Возможности:
- Трёхуровневая память: краткосрочная, рабочая, долговременная
- Три стратегии контекста: sliding window, sticky facts, branching
- Конечный автомат планирования задач (`/task`, `/plan`)
- Агентский режим с инвариантами и авто-ретраями
- Хранение данных с изоляцией по профилям (`--profile`)

**Запуск:**
```bash
cd llm_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заполнить API_KEY
python3 script.py
```

---

### api_for_mcp/ — FastAPI-микросервисы

| Сервис | Порт | Назначение |
|--------|------|-----------|
| `scheduler` | 8881 | Управление напоминаниями |
| `weather` | 8882 | Погода для городов России |
| `pdf-maker` | 8883 | Генерация PDF-документов |
| `save_to_file` | 8884 | Сохранение файлов на сервере |

Каждый сервис запускается двумя способами:

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

Все эндпоинты (кроме `GET /`) требуют заголовок `X-API-Key`.
Конфигурация через `.env` (см. `.env.example` в каждом сервисе).

---

### llm_mcp/ — MCP-серверы

Обёртки над FastAPI-микросервисами, реализующие протокол MCP (Model Context Protocol).
LLM-агент подключает их как инструменты через stdio-транспорт.

| Сервер | Файл |
|--------|------|
| Планировщик | `llm_mcp/scheduler/` |
| Погода | `llm_mcp/weather/` |
| PDF | `llm_mcp/pdf-maker/pdf_server.py` |
| Файлы | `llm_mcp/save_to_file/save_server.py` |

Зависимости: `mcp>=1.0.0`

---

## Быстрый старт

```bash
# 1. Установить зависимости нужного компонента
cd api_for_mcp/weather
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Настроить окружение
cp .env.example .env
# Заполнить API_KEY в .env

# 3. Запустить микросервисы
python3 api_for_mcp/scheduler/main.py &
python3 api_for_mcp/weather/main.py &
python3 api_for_mcp/pdf-maker/main.py &
python3 api_for_mcp/save_to_file/main.py &

# 4. Запустить чатбот
cd llm_agent
python3 script.py
```

---

## Тесты

```bash
# Чатбот (865 тестов)
cd llm_agent
python3 -m pytest tests/ -v

# FastAPI-сервисы
cd api_for_mcp/scheduler   && python3 -m pytest tests/ -v  # 51 тест
cd api_for_mcp/weather     && python3 -m pytest tests/ -v  # 42 теста
cd api_for_mcp/pdf-maker   && python3 -m pytest tests/ -v  # 23 теста
cd api_for_mcp/save_to_file && python3 -m pytest tests/ -v # 26 тестов

# MCP-серверы
cd llm_mcp/pdf-maker       && python3 -m pytest tests/ -v
cd llm_mcp/save_to_file    && python3 -m pytest tests/ -v
```

---

## Требования

- Python 3.10+
- API-ключ для OpenAI-совместимого сервиса (чатбот)
- API-ключ для микросервисов (задаётся в `.env` каждого сервиса)
