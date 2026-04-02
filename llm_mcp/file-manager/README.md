# File Manager MCP Server

MCP-сервер для работы с файлами проекта.
Корень проекта задаётся через `FILE_MANAGER_ROOT`.

## Инструменты MCP

| Инструмент | Описание |
|---|---|
| `fm_read_file(path)` | Читает содержимое файла проекта. |
| `fm_list_files(path?, pattern?)` | Перечисляет файлы проекта с поддержкой glob-паттерна. |
| `fm_search_in_files(query, path?, pattern?, is_regex?)` | Ищет строку или regex по файлам проекта. |
| `fm_write_file(path, content)` | Создаёт или перезаписывает файл в проекте. |
| `fm_patch_file(path, old_string, new_string)` | Заменяет первое вхождение old_string на new_string в файле проекта. |
| `fm_check_invariants(path?)` | Проверяет файлы на соответствие правилам из rules.json. |