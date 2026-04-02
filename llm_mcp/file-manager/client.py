"""
Демо-клиент для MCP File Manager Server.

Демонстрирует 4 сценария:
  1. Поиск использования FastMCP по всем .py файлам
  2. Обновление/создание README.md с секцией инструментов
  3. Генерация CHANGELOG.md (показывает diff при повторном запуске)
  4. Проверка инвариантов из rules.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_SCRIPT = Path(__file__).parent / "file_manager_server.py"


async def call_tool(session: ClientSession, name: str, **kwargs) -> dict:
    """Вызывает MCP-инструмент и десериализует JSON-ответ."""
    result = await session.call_tool(name, arguments=kwargs)
    text = result.content[0].text if result.content else "{}"
    return json.loads(text)


async def scenario_1_search_fastmcp(session: ClientSession) -> None:
    """Сценарий 1: найти все места использования FastMCP в .py файлах."""
    print("\n" + "=" * 60)
    print("Сценарий 1: Поиск использования FastMCP")
    print("=" * 60)

    result = await call_tool(
        session,
        "search_in_files",
        query="FastMCP",
        pattern="**/*.py",
    )

    if "error" in result:
        print(f"Ошибка: {result['error']}")
        return

    print(f"Найдено совпадений: {result['total_matches']} в {result['files_searched']} файлах\n")

    # Группируем по файлам
    by_file: dict[str, list] = {}
    for match in result["matches"]:
        by_file.setdefault(match["file"], []).append(match)

    for filepath, matches in by_file.items():
        print(f"  {filepath}:")
        for m in matches:
            print(f"    Строка {m['line_number']:4d}: {m['line_content'].strip()}")
    print()


async def scenario_2_update_readme(session: ClientSession) -> None:
    """Сценарий 2: обновить README.md — добавить/обновить секцию с инструментами сервера."""
    print("=" * 60)
    print("Сценарий 2: Обновление документации (README.md)")
    print("=" * 60)

    # Собираем список .py файлов чтобы показать какие инструменты есть
    files_result = await call_tool(session, "list_files", pattern="**/*.py")
    if "error" in files_result:
        print(f"Ошибка list_files: {files_result['error']}")
        return

    py_files = [f["path"] for f in files_result["files"]]
    print(f"Найдено .py файлов: {len(py_files)}")

    # Ищем функции-инструменты (декоратор @mcp.tool())
    tools_result = await call_tool(
        session,
        "search_in_files",
        query="@mcp.tool()",
        pattern="**/*.py",
    )

    tool_names = []
    if "matches" in tools_result:
        for m in tools_result["matches"]:
            # Следующая строка после @mcp.tool() — def tool_name(
            for ctx_line in m["context_after"]:
                stripped = ctx_line.strip()
                if stripped.startswith("def "):
                    name = stripped[4:].split("(")[0]
                    tool_names.append(f"- `{name}`")
                    break

    tools_section = "\n".join(tool_names) if tool_names else "- (инструменты не найдены)"

    new_section = f"""## Инструменты MCP

Сервер `file-manager` предоставляет следующие инструменты:

{tools_section}
"""

    # Читаем существующий README или создаём с нуля
    readme_result = await call_tool(session, "read_file", path="README.md")

    if "error" in readme_result:
        # README не существует — создаём
        content = f"# File Manager MCP Server\n\n{new_section}"
        write_result = await call_tool(session, "write_file", path="README.md", content=content)
        print("README.md создан.")
        if write_result.get("diff"):
            print("Diff:\n" + write_result["diff"])
    else:
        old_content = readme_result["content"]

        if "## Инструменты MCP" in old_content:
            # Обновляем существующую секцию
            # Находим начало секции и следующий ## заголовок
            start = old_content.index("## Инструменты MCP")
            end = old_content.find("\n## ", start + 1)
            if end == -1:
                end = len(old_content)
            old_section = old_content[start:end]

            patch_result = await call_tool(
                session,
                "patch_file",
                path="README.md",
                old_string=old_section,
                new_string=new_section.rstrip(),
            )
            print("README.md обновлён.")
            if "diff" in patch_result:
                print("Diff:\n" + patch_result["diff"])
        else:
            # Добавляем секцию в конец
            new_content = old_content.rstrip() + "\n\n" + new_section
            write_result = await call_tool(
                session, "write_file", path="README.md", content=new_content
            )
            print("Секция добавлена в README.md.")
            if write_result.get("diff"):
                print("Diff:\n" + write_result["diff"])
    print()


async def scenario_3_generate_changelog(session: ClientSession) -> None:
    """Сценарий 3: сгенерировать CHANGELOG.md. При повторном запуске показывает diff."""
    print("=" * 60)
    print("Сценарий 3: Генерация CHANGELOG.md")
    print("=" * 60)

    # Собираем информацию о файлах проекта
    files_result = await call_tool(session, "list_files", pattern="**/*")
    if "error" in files_result:
        print(f"Ошибка: {files_result['error']}")
        return

    files = files_result["files"]
    py_count = sum(1 for f in files if f["path"].endswith(".py"))
    json_count = sum(1 for f in files if f["path"].endswith(".json"))
    total = files_result["total"]

    # Поиск инструментов для changelog
    tools_result = await call_tool(session, "search_in_files", query="@mcp.tool()", pattern="**/*.py")
    tool_count = tools_result.get("total_matches", 0)

    from datetime import date
    today = date.today().isoformat()

    changelog_content = f"""# CHANGELOG

## [Unreleased] — {today}

### Added
- MCP File Manager Server (`file_manager_server.py`)
- {tool_count} MCP-инструментов: read_file, list_files, search_in_files, write_file, patch_file, check_invariants
- Поддержка .gitignore-исключений через паттерны
- Проверка инвариантов через rules.json
- Демо-клиент с 4 сценариями (client.py)
- Тесты с изоляцией файловой системы (tests/)

### Project Stats (auto-generated)
- Python файлов: {py_count}
- JSON файлов: {json_count}
- Всего файлов: {total}

### Notes
- Транспорт: stdio (стандарт проекта)
- Корень проекта: CWD сервера
- Без LLM-бэкенда — логика на стороне клиента
"""

    result = await call_tool(session, "write_file", path="CHANGELOG.md", content=changelog_content)

    if "error" in result:
        print(f"Ошибка: {result['error']}")
        return

    if result["created"]:
        print("CHANGELOG.md создан.")
    else:
        print("CHANGELOG.md обновлён.")

    if result.get("diff"):
        print("Diff:\n" + result["diff"])
    else:
        print("(изменений нет)")
    print()


async def scenario_4_check_invariants(session: ClientSession) -> None:
    """Сценарий 4: проверить все файлы на соответствие правилам из rules.json."""
    print("=" * 60)
    print("Сценарий 4: Проверка инвариантов (rules.json)")
    print("=" * 60)

    result = await call_tool(session, "check_invariants")

    if "error" in result:
        print(f"Ошибка: {result['error']}")
        return

    violations = result["violations"]
    print(f"Правил проверено: {result['rules_checked']}")
    print(f"Файлов проверено: {result['files_checked']}")
    print(f"Нарушений найдено: {len(violations)}\n")

    if not violations:
        print("Нарушений нет.")
        return

    # Группируем по severity
    errors = [v for v in violations if v["severity"] == "error"]
    warnings = [v for v in violations if v["severity"] == "warning"]

    if errors:
        print(f"ОШИБКИ ({len(errors)}):")
        for v in errors:
            print(f"  [{v['rule_id']}] {v['file']}:{v['line_number']}")
            print(f"         {v['line_content'].strip()}")
            print(f"         → {v['message']}")

    if warnings:
        print(f"\nПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
        for v in warnings:
            print(f"  [{v['rule_id']}] {v['file']}:{v['line_number']}")
            print(f"         {v['line_content'].strip()}")
            print(f"         → {v['message']}")
    print()


async def main() -> None:
    # Запускаем сервер относительно директории file-manager
    server_dir = str(SERVER_SCRIPT.parent)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env={**os.environ, "PWD": server_dir},
    )

    print(f"Запуск file-manager MCP сервера из: {server_dir}")
    print(f"Python: {sys.executable}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Перечисляем доступные инструменты
            tools = await session.list_tools()
            print(f"Доступных инструментов: {len(tools.tools)}")
            for t in tools.tools:
                print(f"  - {t.name}")

            # Запускаем сценарии
            # Меняем CWD клиента чтобы относительные пути работали корректно
            original_cwd = os.getcwd()
            os.chdir(server_dir)
            try:
                await scenario_1_search_fastmcp(session)
                await scenario_2_update_readme(session)
                await scenario_3_generate_changelog(session)
                await scenario_4_check_invariants(session)
            finally:
                os.chdir(original_cwd)

    print("Готово.")


if __name__ == "__main__":
    asyncio.run(main())
