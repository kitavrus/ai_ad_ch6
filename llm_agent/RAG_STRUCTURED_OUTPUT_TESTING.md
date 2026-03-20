# RAG Structured Output — Руководство по проверке

## Что реализовано

- Обязательный формат ответа: **Ответ** / **Источники** / **Цитаты**
- IDK-режим: если релевантных чанков нет и `threshold > 0` — ассистент отвечает "Не знаю"
- Заголовок каждого чанка содержит `score=` и `chunk_id=`

### Готовый индекс

`rag_index/` уже содержит проиндексированные документы (336 чанков):

| Файл | Содержимое |
|------|-----------|
| `CLAUDE.md` | Архитектура проекта, команды, стратегии контекста |
| `EXAMPLES.md` | Примеры `/plan on` и `/plan builder` |
| `RAG_QUICKSTART.md` | Руководство по быстрому старту RAG |
| `README.md` | Обзор проекта |
| `SCENARIOS.md` | Сценарии инвариантной системы |
| `research.md` | Анализ проекта, стек, архитектура |
| `schema-graph.md` | Граф схемы данных |
| `USE_CASE_NOTIFICATION_WATCHER.md` | Use case: наблюдатель уведомлений |
| `VERIFICATION_CHECKLIST.md` | Чеклист верификации |

---

## Шаг 1 — Запуск чатбота

```bash
cd ai_ad_ch6
source .venv/bin/activate
cd llm_agent
python script.py
```

---

## Шаг 2 — Включение RAG

Индекс `rag_index/` загружается **автоматически** — индексировать ничего не нужно.

```
/rag on
```

Ожидаемый вывод:
```
[RAG включён. Стратегия: fixed, top_k: 5]
```

Проверить статус:
```
/rag status
```

---

## Шаг 3 — 10 вопросов по реальным документам

Задайте следующие вопросы по одному и заполняйте чеклист.

### Вопросы

```
1. Какие три уровня памяти есть в чатботе?
2. Как запустить чатбот с именованным профилем?
3. Что делает команда /plan builder?
4. Какие стратегии контекста поддерживает чатбот?
5. Как добавить инвариант и что он делает?
6. Где хранятся файлы сессий для профиля Igor?
7. Что такое sticky_facts стратегия?
8. Какие зависимости нужны для работы RAG?
9. Как работает валидация черновика против инвариантов?
10. Что происходит при исчерпании попыток в plan builder?
```

### Чек-лист результатов

| # | Вопрос (кратко) | **Ответ:** | **Источники:** | **Цитаты:** | Смысл совпадает |
|---|-----------------|:----------:|:--------------:|:-----------:|:---------------:|
| 1 | Три уровня памяти | ☐ | ☐ | ☐ | ☐ |
| 2 | Запуск с профилем | ☐ | ☐ | ☐ | ☐ |
| 3 | /plan builder | ☐ | ☐ | ☐ | ☐ |
| 4 | Стратегии контекста | ☐ | ☐ | ☐ | ☐ |
| 5 | Инварианты | ☐ | ☐ | ☐ | ☐ |
| 6 | Хранение сессий | ☐ | ☐ | ☐ | ☐ |
| 7 | sticky_facts | ☐ | ☐ | ☐ | ☐ |
| 8 | Зависимости RAG | ☐ | ☐ | ☐ | ☐ |
| 9 | Валидация черновика | ☐ | ☐ | ☐ | ☐ |
| 10 | Исчерпание попыток | ☐ | ☐ | ☐ | ☐ |

### Ожидаемый формат каждого ответа

```
**Ответ:** Развёрнутый ответ на вопрос...

**Источники:**
- CLAUDE.md §Трехуровневая система памяти (chunk_id=claude_5, score=0.87)

**Цитаты:**
> "ShortTermMemory / WorkingMemory / LongTermMemory — три независимых хранилища"
```

Консоль при каждом запросе:
```
[RAG] Найдено 3 чанк(ов) из: CLAUDE.md, research.md
```

---

## Шаг 4 — Проверка IDK-режима

### 4.1 Установить порог релевантности

```
/rag filter 0.5
```

### 4.2 Задать вопросы НЕ по теме документов

```
11. Как приготовить борщ?
12. Расскажи про фотосинтез у растений.
13. Какой курс доллара сегодня?
```

**Ожидаемый вывод в консоли:**
```
[RAG] Релевантных чанков не найдено (threshold=0.5)
```

**Ожидаемый ответ ассистента:**
```
Не знаю — в доступных документах нет информации по этому вопросу.
Уточните, пожалуйста, ваш запрос или добавьте нужные документы в индекс.
```

### 4.3 Отключить IDK-режим

```
/rag filter 0
```

Задайте тот же вопрос — ассистент попытается ответить своими словами без IDK.

---

## Шаг 5 — Тонкая настройка (опционально)

| Команда | Эффект |
|---------|--------|
| `/rag strategy structure` | Переключиться на структурную стратегию чанкинга |
| `/rag topk 10` | Увеличить количество извлекаемых чанков |
| `/rag rewrite on` | Включить переформулировку запроса перед поиском |
| `/rag preset quality` | threshold=0.5, top_k_before=10, rewrite=on |
| `/rag preset fast` | threshold=0, top_k_before=0, rewrite=off |

---

## Шаг 6 — Автотесты

```bash
python -m pytest tests/test_rag_structured_output.py -v
```

**Ожидаемый результат:**
```
tests/test_rag_structured_output.py::test_contains_answer_header                          PASSED
tests/test_rag_structured_output.py::test_contains_sources_header                         PASSED
tests/test_rag_structured_output.py::test_contains_citations_header                       PASSED
tests/test_rag_structured_output.py::test_chunk_header_contains_score_and_chunk_id        PASSED
tests/test_rag_structured_output.py::test_build_rag_idk_system_contains_ne_znayu          PASSED
tests/test_rag_structured_output.py::test_empty_results_returns_empty_string              PASSED
tests/test_rag_structured_output.py::test_idk_mode_activated_when_no_results_and_threshold_positive PASSED
tests/test_rag_structured_output.py::test_idk_mode_not_activated_when_threshold_zero      PASSED
tests/test_rag_structured_output.py::test_idk_not_activated_when_results_present          PASSED
tests/test_rag_structured_output.py::test_source_line_contains_section_and_chunk_id       PASSED

10 passed
```

---

## Итоговый чеклист

| Критерий | Способ проверки | Статус |
|----------|----------------|:------:|
| `**Ответ:**` присутствует в каждом из 10 ответов | Визуально | ☐ |
| `**Источники:**` с `chunk_id=` и `score=` | Визуально | ☐ |
| `**Цитаты:**` с фрагментом из документа | Визуально | ☐ |
| Смысл ответа совпадает с цитатой | Ручная проверка | ☐ |
| IDK при `threshold=0.5` и нерелевантном вопросе | Шаг 4.2 | ☐ |
| IDK **не** срабатывает при `threshold=0` | Шаг 4.3 | ☐ |
| Консоль показывает `[RAG] Найдено N чанк(ов) из: ...` | Консоль | ☐ |
| 10 автотестов зелёные | `pytest` | ☐ |
