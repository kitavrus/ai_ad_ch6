# Weather MCP Server

MCP-сервер на базе [FastMCP](https://github.com/jlowin/fastmcp), предоставляющий LLM-инструменты для получения погодных данных. Работает как прокси к [Weather API](../../api_for_mcp/weather/README.md) (`localhost:8000`).

## Архитектура

```
LLM / MCP Client
      │
      ▼
weather_server.py  (FastMCP, этот сервер)
      │  X-API-Key
      ▼
FastAPI backend    (api_for_mcp/weather/, порт 8000)
```

## Настройка

Создайте файл `.env` в директории `llm_mcp/weather/`:

```
WEATHER_API_KEY=secret-token
```

Значение должно совпадать с `WEATHER_API_KEY` в `.env` API-бэкенда.

## Установка зависимостей

```bash
cd llm_mcp/weather
pip install -r requirements.txt
```

## Запуск

Перед запуском MCP-сервера убедитесь, что API-бэкенд запущен:

```bash
# 1. Запустить API backend (в отдельном терминале)
cd api_for_mcp/weather
uvicorn main:app --port 8000

# 2. Запустить MCP сервер в режиме разработки
cd llm_mcp/weather
mcp dev weather_server.py
```

Либо запустить сервер напрямую:

```bash
python weather_server.py
```

## Инструменты (Tools)

### `get_weather(city: str) -> str`
Возвращает текущую погоду для указанного города на русском языке.

**Параметры:**
- `city` — название города (строчными буквами, например `москва`, `санкт-петербург`)

**Примеры ответов:**

Успешный запрос:
```
Погода в Москва:
  Температура: -3°C
  Состояние: Облачно, небольшой снег
  Влажность: 80%
  Ветер: 5 м/с, С
```

Город не найден:
```
Город 'лондон' не найден. Доступные города: Москва, Санкт-Петербург, ...
```

Ошибка авторизации:
```
Ошибка авторизации: неверный или отсутствующий API ключ (X-API-Key).
```

API недоступен:
```
Не удалось подключиться к сервису погоды. Убедитесь, что API запущен на localhost:8000.
```

---

### `list_cities() -> str`
Возвращает список всех доступных городов.

**Пример ответа:**
```
Доступные города:
  - Москва
  - Санкт-Петербург
  - Новосибирск
  - ...
```

## Интеграция с Claude Desktop

Добавьте в конфигурацию Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/llm_mcp/weather/weather_server.py"]
    }
  }
}
```

## Зависимости

| Пакет          | Версия    | Назначение                        |
|----------------|-----------|-----------------------------------|
| mcp            | >=1.0.0   | FastMCP фреймворк для MCP-серверов |
| httpx          | >=0.27.0  | HTTP-клиент для запросов к API    |
| python-dotenv  | >=1.0.0   | Загрузка `WEATHER_API_KEY` из `.env` |
