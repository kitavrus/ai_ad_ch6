# PLAN: Сценарии работы MCP File Manager

## Цель
Запустить и проверить 4 сценария, где AI-ассистент самостоятельно
инициирует работу с файлами проекта через MCP-сервер `file-manager`.

## Сценарии

### Сценарий 1 — Поиск использования FastMCP
**Задача (уровень цели):** «Найди все места в проекте, где используется FastMCP»
- Инструменты: `fm_search_in_files`
- Файлов затрагивает: ≥ 3 (все серверы используют FastMCP)
- Результат: список файл:строка с контекстом

### Сценарий 2 — Обновление документации
**Задача:** «Обнови README.md для модуля file-manager на основе кода»
- Инструменты: `fm_list_files` → `fm_search_in_files` → `fm_patch_file` / `fm_write_file`
- Файлов затрагивает: ≥ 2 (.py + README.md)
- Результат: README.md с актуальной секцией инструментов + unified diff

### Сценарий 3 — Генерация файла
**Задача:** «Сгенерируй CHANGELOG.md на основе анализа проекта»
- Инструменты: `fm_list_files` + `fm_search_in_files` → `fm_write_file`
- Файлов затрагивает: ≥ 3 (анализ всего проекта)
- Результат: CHANGELOG.md создан / обновлён с diff при повторном запуске

### Сценарий 4 — Проверка инвариантов
**Задача:** «Проверь все файлы на соответствие правилам rules.json»
- Инструменты: `fm_check_invariants`
- Файлов затрагивает: все .py файлы в FILE_MANAGER_ROOT
- Результат: список нарушений по severity + файл:строка

## Архитектура запуска

```
scenarios.py (клиент)
    │
    ├─ StdioServerParameters → subprocess: file_manager_server.py
    │       env: FILE_MANAGER_ROOT=/path/to/ai_ad_ch6
    │
    └─ ClientSession
            ├─ fm_search_in_files(...)
            ├─ fm_list_files(...)
            ├─ fm_read_file(...)
            ├─ fm_write_file(...)
            ├─ fm_patch_file(...)
            └─ fm_check_invariants(...)
```

## Файлы

| Файл | Роль |
|---|---|
| `llm_mcp/file-manager/scenarios.py` | Основной запускаемый скрипт сценариев |
| `llm_mcp/file-manager/file_manager_server.py` | MCP-сервер (уже готов) |
| `rules.json` (корень проекта) | Правила для проверки инвариантов |
| `llm_mcp/file-manager/README.md` | Генерируется/обновляется в сценарии 2 |
| `llm_mcp/file-manager/CHANGELOG.md` | Генерируется в сценарии 3 |

## Воспроизведение

```bash
cd /path/to/ai_ad_ch6
.venv/bin/python llm_mcp/file-manager/scenarios.py
```

Повторный запуск — для сценария 3 покажет diff изменений.
