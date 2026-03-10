# AI Chatbot CLI — Документация

Интерактивный CLI-чатбот с поддержкой OpenAI-совместимых API, тремя типами памяти и стратегиями управления контекстом.

---

## Содержание

1. [Обзор](#обзор)
2. [Быстрый старт](#быстрый-старт)
3. [Архитектура](#архитектура)
4. [Модель памяти](#модель-памяти)
5. [Стратегии контекста](#стратегии-контекста)
6. [CLI интерфейс](#cli-интерфейс)
7. [Inline-команды](#inline-команды)
8. [Структура файлов](#структура-файлов)
9. [API модулей](#api-модулей)
10. [Тестирование](#тестирование)

---

## Обзор

### Возможности

- Интерактивный диалог с AI-моделями через OpenAI-совместимый API
- **Три типа памяти**: краткосрочная, рабочая, долговременная
- **Три стратегии контекста**: sliding window, sticky facts, branching
- Сохранение сессий и метрик на диск
- Возобновление прерванных диалогов
- Настройка параметров генерации в runtime

### Требования

- Python 3.10+
- API-ключ для OpenAI-совместимого сервиса

---

## Быстрый старт

### Установка

```bash
# Клонировать репозиторий
git clone <repo-url>
cd ai_ad_ch6

# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Конфигурация

Создайте файл `.env` в корне проекта:

```env
API_KEY=your-api-key-here
```

### Запуск

```bash
# Начать новый диалог
python script.py

# Возобновить последнюю сессию
python script.py --resume

# Указать модель
python script.py -m "deepseek/deepseek-v3.2"

# Задать системный промпт
python script.py --system-prompt "Ты опытный программист"
```

---

## Архитектура

```
ai_ad_ch6/
├── script.py              # Точка входа
├── chatbot/               # Основной пакет
│   ├── __init__.py
│   ├── main.py            # Главный модуль, цикл диалога
│   ├── cli.py             # Разбор аргументов и inline-команд
│   ├── config.py          # Константы и конфигурация
│   ├── models.py          # Pydantic-модели данных
│   ├── context.py         # Стратегии управления контекстом
│   ├── memory.py          # Три типа памяти
│   ├── memory_storage.py  # Сохранение/загрузка памяти
│   └── storage.py         # Сохранение сессий
├── tests/                 # Тесты pytest
├── dialogues/             # Сохранённые данные
│   ├── session_*.json     # Файлы сессий
│   ├── metrics/           # Логи запросов
│   └── memory/            # Файлы памяти
│       ├── short_term/
│       ├── working/
│       └── long_term/
└── .env                   # Переменные окружения
```

### Модули

| Модуль | Назначение |
|--------|------------|
| `main.py` | Точка входа, интерактивный цикл, обработка команд |
| `cli.py` | Парсинг аргументов CLI и inline-команд |
| `config.py` | Константы, конфиг по умолчанию |
| `models.py` | Pydantic-модели: SessionState, ChatMessage, Branch и др. |
| `context.py` | Стратегии: sliding window, sticky facts, branching |
| `memory.py` | Классы ShortTermMemory, WorkingMemory, LongTermMemory, Memory |
| `memory_storage.py` | Функции save/load для каждого типа памяти |
| `storage.py` | Сохранение сессий и метрик |

---

## Модель памяти

Система реализует **три независимых типа памяти**, каждый со своим storage и жизненным циклом.

### Краткосрочная память (ShortTermMemory)

**Хранит**: Сообщения текущего диалога (user/assistant)

**Жизненный цикл**: Только в рамках текущей сессии. Очищается при выходе.

**Storage**: `dialogues/memory/short_term/session_*.json`

**API**:
```python
memory.short_term.add_message(role, content)
memory.short_term.get_recent(n=10)  # последние N сообщений
memory.short_term.clear()
```

### Рабочая память (WorkingMemory)

**Хранит**:
- Текущую задачу (`current_task`)
- Статус задачи (`task_status`)
- Предпочтения пользователя (`user_preferences`)
- Последние действия (`recent_actions`)

**Жизненный цикл**: Может сохраняться между сессиями.

**Storage**: `dialogues/memory/working/task_<name>_*.json`

**API**:
```python
memory.working.set_task("Написать тесты")
memory.working.set_preference("lang", "python")
memory.working.add_action("Создан файл test_x.py")
memory.working.update_status("in_progress")
```

### Долговременная память (LongTermMemory)

**Хранит**:
- Профиль пользователя (`user_profile`)
- Историю решений (`decisions_log`)
- Базу знаний (`knowledge_base`)

**Жизненный цикл**: Персистентная, сохраняется между сессиями.

**Storage**: `dialogues/memory/long_term/profile_<name>_*.json`

**API**:
```python
memory.long_term.add_decision("setup_db", "Использовать PostgreSQL")
memory.long_term.add_knowledge("python_tip", "List comprehensions are fast")
memory.long_term.set_profile("name", "Игорь")
memory.long_term.get_knowledge("python_tip")
```

### Центральный класс Memory

```python
from llm_agent.chatbot import Memory

memory = Memory()

# Автоматическое добавление в краткосрочную
memory.add_user_message("Привет")
memory.add_assistant_message("Привет! Чем помочь?")

# Добавление в рабочую память
memory.add_to_working_memory(task="Write code", action="Started")

# Добавление в долговременную память
memory.add_to_long_term(
    decision="Use FastAPI",
    task="backend",
    knowledge_key="framework",
    knowledge_value="FastAPI",
    profile_key="role",
    profile_value="backend_dev"
)

# Сохранение/загрузка полного состояния
state = memory.get_full_state()
memory.load_full_state(state)
```

### Команды управления памятью

| Команда | Описание |
|---------|----------|
| `/memshow` | Показать состояние всех типов памяти |
| `/memstats` | Статистика файлов на диске |
| `/memsave` | Явно сохранить все типы памяти на диск |
| `/memload` | Загрузить рабочую и долговременную память |
| `/memclear` | Очистить краткосрочную память |
| `/settask <задача>` | Установить текущую задачу (working) |
| `/setpref <к>=<з>` | Установить предпочтение (working) |
| `/remember <к>=<з>` | Сохранить знание (long_term) |

---

## Стратегии контекста

### 1. Sliding Window (по умолчанию)

Последние N сообщений + опциональное резюме старых.

```bash
/strategy sliding_window
```

**Параметры**:
- `CONTEXT_RECENT_MESSAGES = 10` — размер окна
- `CONTEXT_SUMMARY_INTERVAL = 10` — интервал суммаризации

**Команды**:
- `/showsummary` — показать текущее резюме

### 2. Sticky Facts

Извлекает и хранит ключевые факты из диалога.

```bash
/strategy sticky_facts
```

**Команды**:
- `/showfacts` — показать накопленные факты
- `/setfact <ключ>: <значение>` — добавить факт вручную
- `/delfact <ключ>` — удалить факт

### 3. Branching

Позволяет создавать независимые ветки диалога.

```bash
/strategy branching
```

**Команды**:
- `/checkpoint` — создать точку сохранения
- `/branch <имя>` — создать новую ветку от checkpoint
- `/switch <id|имя>` — переключиться на ветку
- `/branches` — список всех веток

---

## CLI интерфейс

### Аргументы командной строки

| Аргумент | Короткий | По умолчанию | Описание |
|----------|----------|--------------|----------|
| `--model` | `-m` | `inception/mercury-coder` | Модель |
| `--base-url` | `-u` | `https://routerai.ru/api/v1` | URL API |
| `--max-tokens` | | `None` | Макс. токенов в ответе |
| `--temperature` | `-T` | `0.7` | Температура |
| `--top-p` | `-p` | `0.9` | Top-p sampling |
| `--top-k` | `-k` | `50` | Top-k sampling |
| `--system-prompt` | | | Системный промпт |
| `--initial-prompt` | | | Начальное сообщение |
| `--resume` | | `False` | Возобновить сессию |
| `--strategy` | | `sliding_window` | Стратегия контекста |

### Примеры

```bash
# Базовый запуск
python script.py

# С настройками
python script.py -m "gpt-4" -T 0.5 --max-tokens 1000

# Со стратегией sticky_facts
python script.py --strategy sticky_facts

# Возобновить с возобновлением памяти
python script.py --resume
```

---

## Inline-команды

Команды вводятся прямо в диалоге, начинаются с `/`.

### Параметры модели

```
/model gpt-4
/temperature 0.8
/top-p 0.95
/max-tokens 2000
/system-prompt Ты программист
```

### Управление сессией

```
/resume true
/showsummary
```

### Стратегии контекста

```
/strategy sliding_window
/strategy sticky_facts
/strategy branching
/showfacts
/setfact цель: написать тесты
/delfact цель
```

### Ветвление (branching)

```
/checkpoint
/branch вариант-А
/switch вариант-Б
/branches
```

### Память

```
/memshow
/memstats
/memsave
/memload
/memclear
/settask написать документацию
/setpref lang=python
/remember framework=FastAPI
```

---

## Структура файлов

### Сессия (`dialogues/session_*.json`)

```json
{
  "dialogue_session_id": "path/to/file",
  "created_at": "2026-03-03T12:00:00Z",
  "model": "deepseek/deepseek-v3.2",
  "base_url": "https://routerai.ru/api/v1",
  "system_prompt": "Ты помощник",
  "initial_prompt": "Привет",
  "messages": [
    {"role": "user", "content": "Привет"},
    {"role": "assistant", "content": "Привет!", "tokens": {...}}
  ],
  "context_summary": "Пользователь представился...",
  "turns": 5,
  "total_tokens": 1500,
  "context_strategy": "sliding_window",
  "sticky_facts": {"goal": "написать код"},
  "branches": []
}
```

### Метрики (`dialogues/metrics/session_*_req_*.log`)

```json
{
  "model": "deepseek/deepseek-v3.2",
  "endpoint": "chat",
  "temp": 0.7,
  "ttft": 1.23,
  "req_time": 1.23,
  "total_time": 45.6,
  "tokens": 512,
  "p_tokens": 256,
  "c_tokens": 256,
  "cost_rub": 0.077
}
```

### Память

**Рабочая** (`dialogues/memory/working/task_*.json`):
```json
{
  "current_task": "Написать документацию",
  "task_status": "in_progress",
  "task_context": {},
  "recent_actions": ["Создан README.md"],
  "user_preferences": {"lang": "ru"},
  "created_at": "...",
  "updated_at": "..."
}
```

**Долговременная** (`dialogues/memory/long_term/profile_*.json`):
```json
{
  "user_profile": {"name": "Игорь", "role": "developer"},
  "decisions_log": [
    {"task": "backend", "decision": "Use FastAPI", "timestamp": "..."}
  ],
  "knowledge_base": {
    "python_tip": "Use list comprehensions",
    "framework": "FastAPI"
  },
  "last_accessed": "..."
}
```

---

## API модулей

### chatbot.memory

```python
from llm_agent.chatbot import (
    Memory,
    ShortTermMemory,
    WorkingMemory,
    LongTermMemory,
    MemoryFactor,
    extract_memory_factors,
)

# Создание
memory = Memory()

# Краткосрочная
memory.add_user_message("Привет")
memory.add_assistant_message("Привет!")
memory.short_term.get_recent(n=5)
memory.clear_short_term()

# Рабочая
memory.working.set_task("Задача")
memory.working.set_preference("key", "value")
memory.working.add_action("Действие")

# Долговременная
memory.add_to_long_term(decision="Решение", task="Задача")
memory.add_to_long_term(knowledge_key="key", knowledge_value="value")
memory.long_term.get_knowledge("key")
memory.long_term.get_decision_history("Задача")

# Состояние
state = memory.get_full_state()
memory.load_full_state(state)
```

### chatbot.memory_storage

```python
from llm_agent.chatbot import (
    save_short_term,
    load_short_term_last,
    save_working_memory,
    load_working_memory,
    save_long_term,
    load_long_term,
    export_memory_state,
    import_memory_state,
    get_memory_stats,
)

# Сохранение
save_short_term(data, "session_id")
save_working_memory(data, "task_name")
save_long_term(data, "profile_name")

# Загрузка
data = load_short_term_last("session_id")
data = load_working_memory("task_name")
data = load_long_term("profile_name")

# Экспорт/импорт
path = export_memory_state(st, wm, lt)
st, wm, lt = import_memory_state(path)

# Статистика
stats = get_memory_stats()
```

### chatbot.context

```python
from llm_agent.chatbot import (
    build_context_by_strategy,
    build_context_sliding_window,
    build_context_sticky_facts,
    build_context_branching,
    maybe_summarize,
    summarize_messages,
    extract_facts_from_llm,
    create_checkpoint,
    create_branch,
    switch_branch,
)
```

### chatbot.models

```python
from llm_agent.chatbot.models import (
    ChatMessage,
    TokenUsage,
    RequestMetric,
    DialogueSession,
    SessionState,
    ContextStrategy,
    StickyFacts,
    Branch,
    DialogueCheckpoint,
)
```

---

## Тестирование

```bash
# Запустить все тесты
python -m pytest tests/ -v

# Запустить с покрытием
python -m pytest tests/ --cov=chatbot --cov-report=html

# Конкретный файл
python -m pytest tests/test_memory.py -v

# С фильтром
python -m pytest tests/ -k "memory" -v
```

### Структура тестов

```
tests/
├── conftest.py          # Фикстуры pytest
├── test_main.py         # Тесты main.py
├── test_cli.py          # Тесты парсинга команд
├── test_config.py       # Тесты конфигурации
├── test_models.py       # Тесты Pydantic-моделей
├── test_context.py      # Тесты стратегий контекста
├── test_storage.py      # Тесты storage
└── test_memory.py       # Тесты модели памяти
```

---

## Константы конфигурации

Файл `chatbot/config.py`:

| Константа | Значение | Описание |
|-----------|----------|----------|
| `BASE_URL` | `https://routerai.ru/api/v1` | URL API |
| `DEFAULT_MODEL` | `inception/mercury-coder` | Модель по умолчанию |
| `DEFAULT_TEMPERATURE` | `0.7` | Температура |
| `DEFAULT_TOP_P` | `0.9` | Top-p |
| `DEFAULT_TOP_K` | `50` | Top-k |
| `USD_PER_1K_TOKENS` | `0.0015` | Цена за 1K токенов |
| `RUB_PER_USD` | `100.0` | Курс |
| `CONTEXT_RECENT_MESSAGES` | `10` | Размер окна контекста |
| `CONTEXT_SUMMARY_INTERVAL` | `10` | Интервал суммаризации |
| `DIALOGUES_DIR` | `dialogues` | Директория сессий |

---

## Лицензия

MIT
