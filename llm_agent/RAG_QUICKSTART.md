# Руководство по быстрому старту RAG

## 1. Предварительные требования

- Активированное виртуальное окружение Python (`.venv`)
- `OPENAI_API_KEY` задан в `.env` (нужен для индексирования/поиска; тесты не требуют)
- Опционально: `OPENAI_BASE_URL` в `.env` для собственных endpoint'ов

## 2. Установка

```bash
# Активировать виртуальное окружение
source .venv/bin/activate           # macOS/Linux
# .venv\Scripts\activate            # Windows

# Установить зависимости RAG
pip install faiss-cpu numpy

# Или через requirements.txt (включает все зависимости проекта)
pip install -r llm_agent/requirements.txt
```

Файл `.env` (в директории `llm_agent/` или корне проекта):
```
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://your-endpoint/v1  # optional
```

## 3. Индексирование документов

Запускать из директории `llm_agent/`:

```bash
cd llm_agent

# Индексировать обеими стратегиями (создаёт rag_index/fixed.faiss + rag_index/structure.faiss)
python index_documents.py

# Только стратегия фиксированного размера
python index_documents.py --strategy fixed

# Только структурная стратегия (по заголовкам markdown)
python index_documents.py --strategy structure
```

## 4. Сравнение стратегий

```bash
python index_documents.py --compare
# Выводит таблицу сравнения и сохраняет rag_index/compare_report.json
```

Ожидаемый вывод:
```
Strategy       Chunks   Avg chars    Min    Max      Std  Sources
fixed             147       512       12    512       89        8
structure          63      1204       87   3201      741        8
```

| Аспект | Фиксированный размер | Структурная |
|---|---|---|
| Количество чанков | Больше (мелкая нарезка) | Меньше (крупная нарезка) |
| Связность контекста | Ниже (может разрезать на середине предложения) | Выше (учитывает разделы) |
| Богатство метаданных | Нет информации о разделах | Полный путь раздела |
| Лучше подходит для | Запросов с высоким recall | Тематических запросов |

## 5. Семантический поиск

```bash
# Поиск по индексу фиксированного размера (по умолчанию)
python index_documents.py --query "three-tier memory"

# Поиск по структурному индексу
python index_documents.py --query "task state machine" --strategy structure

# Вернуть топ-10 результатов (по умолчанию: 5)
python index_documents.py --query "profile" --top-k 10
```

Каждый результат содержит: заголовок, раздел, исходный файл и превью текста.

## 6. Запуск тестов (ключ API не нужен)

```bash
cd llm_agent

# Только RAG-тесты (44 теста, ~0.5с, ключ API не требуется)
python -m pytest tests/test_rag_chunking.py tests/test_rag_index.py tests/test_rag_pipeline.py -v

# Все тесты проекта
python -m pytest tests/ -v
```

## 7. Структура выходных файлов

```
llm_agent/rag_index/
  fixed.faiss               # FAISS-индекс (стратегия фиксированного размера)
  fixed_metadata.json       # массив метаданных чанков (JSON)
  structure.faiss           # FAISS-индекс (структурная стратегия)
  structure_metadata.json   # метаданные чанков с информацией о разделах
  compare_report.json       # отчёт со статистикой сравнения
```

## 8. Устранение проблем

| Ошибка | Решение |
|---|---|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `OPENAI_API_KEY not set` | Создайте `.env` с вашим API-ключом |
| `No markdown files found` | Запускайте из директории `llm_agent/`, а не из корня проекта |
| Индекс не найден при `--query` | Сначала запустите индексирование (без `--query`) |
| `ModuleNotFoundError: numpy` | `pip install numpy` |
