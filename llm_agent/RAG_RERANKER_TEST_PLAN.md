# RAG: Reranker + Query Rewrite — пошаговый чеклист

## 1. Что реализовано

| Компонент | Файл | Описание |
|-----------|------|----------|
| `RetrievalResult` | `rag/models.py` | Модель с полями `score`, `distance`, `query` |
| `FAISSIndex.search_with_scores()` | `rag/index.py` | Поиск с возвратом расстояний |
| `RelevanceFilter` | `rag/reranker.py` | Отсечение по порогу + top_k_after |
| `QueryRewriter` | `rag/query_rewrite.py` | LLM-переформулировка (`rewrite` / `rewrite_multi`) |
| `RAGRetriever` | `rag/retriever.py` | Объединяет фильтрацию и rewrite; `search()` и `search_with_scores()` |
| `compare_rag.py` | `compare_rag.py` | CLI-сравнение режимов A/B/C/D с LLM-ответами |
| `RagMode` (расширен) | `chatbot/models.py` | `threshold`, `top_k_before`, `top_k_after`, `rewrite_query` |
| `/rag` команды в чатботе | `chatbot/main.py` | Управление RAG прямо из диалога |

---

## 2. Режимы RAG

| Режим | Конфиг | Что проверяет |
|-------|--------|---------------|
| A | Без RAG | Baseline — знания только из модели |
| B | `top_k=3`, `threshold=0` | Базовый RAG без фильтрации |
| C | `top_k_before=10`, `top_k_after=3`, `threshold=0.5` | Фильтрация нерелевантных чанков |
| D | как C + `rewrite_query=True` | Фильтрация + переформулировка запроса |

---

## 3. Подготовка

### 3.1 Проверить `.env`

```bash
cat .env | grep -E "API_KEY|BASE_URL|MODEL"
```

- [ ] `API_KEY` или `OPENAI_API_KEY` задан
- [ ] `OPENAI_BASE_URL` задан (если нужен кастомный эндпоинт)

### 3.2 Проверить RAG-индекс

```bash
ls rag_index/
```

Ожидаемые файлы:
- [ ] `rag_index/fixed.faiss` + `rag_index/fixed_metadata.json`
- [ ] `rag_index/structure.faiss` + `rag_index/structure_metadata.json`

Если индекс отсутствует — создать:

```bash
python index_documents.py
```

---

## 4. Сценарий A: сравнение через чатбот (`/rag compare`)

### 4.1 Запустить чатбот

```bash
cd llm_agent
python script.py
```

### 4.2 Активировать RAG и проверить статус

```
/rag on
/rag status
```

- [ ] Статус показывает: `enabled=True`, `threshold`, `top_k_before`, `top_k_after`, `rewrite`

### 4.3 Прогнать 4 контрольных вопроса через `/rag compare`

Для каждого вопроса выполнить `/rag compare <вопрос>` и отметить результат.

---

**Вопрос 1:**
```
/rag compare Как устроена трёхуровневая система памяти и где физически хранятся данные каждого уровня?
```

Ожидать в выводе:
- [ ] Режим B: чанки найдены, упоминается `short_term/`, `working/`, `long_term/`
- [ ] Режим C: чанков ≤ чанков режима B (фильтрация сработала)
- [ ] Режим D: поле `query_used` отличается от исходного вопроса

---

**Вопрос 2:**
```
/rag compare Как работает агентский цикл /plan on?
```

Ожидать:
- [ ] Режим B: чанки из `CLAUDE.md` или `EXAMPLES.md`
- [ ] Режим C: нерелевантные чанки отсечены
- [ ] Режим D: `query_used` показывает переформулированный запрос

---

**Вопрос 3:**
```
/rag compare Что происходит при нарушении инварианта в plan builder?
```

Ожидать:
- [ ] Режим B: упоминаются `retry`, `max_retries`
- [ ] Режим C/D: более точные чанки, меньше "шума"

---

**Вопрос 4:**
```
/rag compare Как проиндексировать документы для RAG?
```

Ожидать:
- [ ] Режим B: чанки из `RAG_QUICKSTART.md`
- [ ] Режим C: `top_k_after ≤ top_k_before` (фильтрация)
- [ ] Режим D: переформулировка улучшает поиск

---

### 4.4 Дополнительно: `/rag search` для отладки

```
/rag search трёхуровневая память
```

- [ ] Вывод показывает чанки с полями `score` и `distance`
- [ ] Источники файлов (`source`) читаемы

---

## 5. Сценарий B: полный отчёт через `compare_rag.py`

### 5.1 Запустить CLI — все 4 режима

```bash
cd llm_agent
python compare_rag.py --output rag_comparison_report.md
```

Прогресс: в терминале отображаются `[1/10]`, `[2/10]`, … `[10/10]`.

- [ ] Скрипт завершился без ошибок
- [ ] Файл `rag_comparison_report.md` создан

### 5.2 Запустить только B/C/D (без baseline)

```bash
python compare_rag.py --modes B,C,D --output rag_comparison_BCD.md
```

- [ ] Файл `rag_comparison_BCD.md` создан

### 5.3 Проверить содержимое отчёта

Открыть `rag_comparison_report.md` и проверить:

- [ ] Раздел `## Вопрос 1` содержит ответы всех 4 режимов
- [ ] Режим C/D: количество источников (`sources=`) ≤ режима B
- [ ] Режим D: `query_used=` отличается от исходного вопроса хотя бы для одного из 10 вопросов
- [ ] Режимы B/C/D упоминают конкретные детали из документов (не просто общие слова)
- [ ] Раздел `## Сводная таблица` присутствует в конце файла

### 5.4 Пример проверки метаданных режима D

В отчёте найти строку вида:
```
*Метаданные:* sources=CLAUDE.md; query_used=<переформулированный запрос>
```

- [ ] `query_used` ≠ исходный вопрос (rewrite сработал)
- [ ] `sources` содержит имя файла (не пустая строка)

---

## 6. Эксперименты с порогом (`/rag filter`)

Тонкая настройка прямо в чатботе без перезапуска:

```
/rag mode B
/rag search Как устроена трёхуровневая память
```
Запомнить количество чанков (baseline).

```
/rag filter 0.3
/rag search Как устроена трёхуровневая память
```
- [ ] Чанков стало меньше или столько же

```
/rag filter 0.7
/rag search Как устроена трёхуровневая память
```
- [ ] Чанков ещё меньше (более строгий порог)

```
/rag filter 0.0
/rag search Как устроена трёхуровневая память
```
- [ ] Чанков столько же, сколько в baseline (фильтр отключён)

Поиграть с `top_k_before` / `top_k_after`:

```
/rag topk_before 15
/rag topk_after 5
/rag search Как работает агентский цикл
```
- [ ] Вывод показывает до 5 чанков после фильтрации

---

## 7. Тесты

```bash
# Новые /rag команды чатбота
python -m pytest tests/test_rag_commands.py -v    # 22 теста

# Reranker + retriever
python -m pytest tests/test_rag_reranker.py -v    # 26 тестов
python -m pytest tests/test_rag_retriever.py -v

# Все RAG-тесты одной командой
python -m pytest tests/test_rag_commands.py tests/test_rag_reranker.py tests/test_rag_retriever.py -v
```

- [ ] `test_rag_commands.py` — все 22 теста прошли
- [ ] `test_rag_reranker.py` — все 26 тестов прошли
- [ ] `test_rag_retriever.py` — все тесты прошли
