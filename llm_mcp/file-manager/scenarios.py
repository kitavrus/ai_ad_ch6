"""
Сценарии работы MCP File Manager Assistant.

Ассистент сам инициирует работу с файлами — задача задаётся на уровне цели.

Запуск:
    python llm_mcp/file-manager/scenarios.py
    # или из корня проекта:
    .venv/bin/python llm_mcp/file-manager/scenarios.py

Сценарии:
  1. Поиск использования FastMCP  — fm_search_in_files
  2. Обновление документации       — fm_list_files + fm_search_in_files + fm_write_file/fm_patch_file
  3. Генерация CHANGELOG.md        — fm_list_files + fm_write_file (diff при повторном запуске)
  4. Проверка инвариантов          — fm_check_invariants
"""

import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent          # ai_ad_ch6/
_SERVER_SCRIPT = _HERE / "file_manager_server.py"

# Директория, о которой пишем документацию (сам file-manager модуль)
_FM_DIR = "llm_mcp/file-manager"


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _hr(title: str) -> None:
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ! {msg}")


async def tool(session: ClientSession, name: str, **kwargs) -> dict:
    """Вызывает MCP-инструмент и возвращает dict."""
    result = await session.call_tool(name, arguments=kwargs)
    raw = result.content[0].text if result.content else "{}"
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Сценарий 1: Поиск использования FastMCP
# ---------------------------------------------------------------------------

async def scenario_search(session: ClientSession) -> None:
    """
    Задача (уровень цели): «Найди все места в проекте, где используется FastMCP»

    Ассистент сам решает:
      - искать по всему проекту (**/*.py)
      - использовать fm_search_in_files
      - сгруппировать результат по файлам
    """
    _hr("Сценарий 1 — Поиск использования FastMCP")
    print("  Задача: «Найди все места в проекте, где используется FastMCP»\n")

    result = await tool(session, "fm_search_in_files",
                        query="FastMCP", pattern="**/*.py", path="llm_mcp")

    if "error" in result:
        _warn(f"Ошибка: {result['error']}")
        return

    total = result["total_matches"]
    searched = result["files_searched"]
    _ok(f"Просмотрено файлов: {searched}")
    _ok(f"Совпадений найдено: {total}\n")

    # Группируем по файлам
    by_file: dict[str, list] = {}
    for m in result["matches"]:
        by_file.setdefault(m["file"], []).append(m)

    for fpath, matches in sorted(by_file.items()):
        print(f"  📄 {fpath}  ({len(matches)} вхождений)")
        for m in matches[:5]:  # не более 5 строк на файл
            line = m["line_content"].strip()
            print(f"       L{m['line_number']:3d}: {line}")
        if len(matches) > 5:
            print(f"       ... и ещё {len(matches) - 5}")


# ---------------------------------------------------------------------------
# Сценарий 2: Обновление документации README.md
# ---------------------------------------------------------------------------

async def scenario_update_readme(session: ClientSession) -> None:
    """
    Задача: «Обнови README.md для модуля file-manager на основе текущего кода»

    Ассистент сам:
      1. Перечисляет .py файлы в модуле
      2. Ищет все fm_* инструменты по паттерну @mcp.tool()
      3. Читает или создаёт README.md
      4. Обновляет/дополняет секцию инструментов
    """
    _hr("Сценарий 2 — Обновление документации (README.md)")
    print(f"  Задача: «Обнови README.md для {_FM_DIR} на основе кода»\n")

    # Шаг 1: перечислить .py файлы модуля
    files_result = await tool(session, "fm_list_files",
                               path=_FM_DIR, pattern="**/*.py")
    if "error" in files_result:
        _warn(f"list_files: {files_result['error']}")
        return
    py_files = [f["path"] for f in files_result["files"]]
    _ok(f"Найдено .py файлов в модуле: {len(py_files)}")
    for p in py_files:
        print(f"       {p}")

    # Шаг 2: найти все @mcp.tool() → имена функций
    tools_search = await tool(session, "fm_search_in_files",
                               query="@mcp.tool()", pattern="**/*.py",
                               path=_FM_DIR)
    tool_entries = []
    for m in tools_search.get("matches", []):
        for ctx in m["context_after"]:
            stripped = ctx.strip()
            if stripped.startswith("def fm_"):
                fn_name = stripped[4:].split("(")[0]
                # Читаем первую строку докстринга из context_after[1]
                idx = m["context_after"].index(ctx)
                docline = ""
                if idx + 1 < len(m["context_after"]):
                    dl = m["context_after"][idx + 1].strip().strip('"""').strip("'")
                    if dl and not dl.startswith("def ") and not dl.startswith("@"):
                        docline = dl
                tool_entries.append((fn_name, docline))
                break

    _ok(f"Инструментов обнаружено: {len(tool_entries)}")

    # Строим секцию
    rows = ""
    for fn, doc in tool_entries:
        rows += f"| `{fn}` | {doc} |\n"

    new_section = (
        "## Инструменты MCP\n\n"
        "| Инструмент | Описание |\n"
        "|---|---|\n"
        f"{rows}"
    )

    # Шаг 3: читаем или создаём README.md
    readme_path = f"{_FM_DIR}/README.md"
    readme_result = await tool(session, "fm_read_file", path=readme_path)

    if "error" in readme_result:
        # Создаём с нуля
        content = (
            "# File Manager MCP Server\n\n"
            "MCP-сервер для работы с файлами проекта.\n"
            "Корень проекта задаётся через `FILE_MANAGER_ROOT`.\n\n"
            + new_section
        )
        write_result = await tool(session, "fm_write_file",
                                   path=readme_path, content=content)
        _ok("README.md создан.")
        if write_result.get("diff"):
            print("\nDiff:\n" + write_result["diff"])
    else:
        old_content = readme_result["content"]
        if "## Инструменты MCP" in old_content:
            # Обновляем существующую секцию
            start = old_content.index("## Инструменты MCP")
            end = old_content.find("\n## ", start + 1)
            old_section = old_content[start:] if end == -1 else old_content[start:end]
            patch_result = await tool(session, "fm_patch_file",
                                       path=readme_path,
                                       old_string=old_section,
                                       new_string=new_section.rstrip())
            if "error" in patch_result:
                _warn(patch_result["error"])
            else:
                _ok("README.md обновлён.")
                print("\nDiff:\n" + patch_result["diff"])
        else:
            new_content = old_content.rstrip() + "\n\n" + new_section
            write_result = await tool(session, "fm_write_file",
                                       path=readme_path, content=new_content)
            _ok("Секция добавлена в README.md.")
            if write_result.get("diff"):
                print("\nDiff:\n" + write_result["diff"])


# ---------------------------------------------------------------------------
# Сценарий 3: Генерация CHANGELOG.md
# ---------------------------------------------------------------------------

async def scenario_generate_changelog(session: ClientSession) -> None:
    """
    Задача: «Сгенерируй CHANGELOG.md для file-manager на основе анализа проекта»

    Ассистент сам:
      1. Считает статистику файлов
      2. Находит все инструменты
      3. Создаёт/обновляет CHANGELOG.md
      4. При повторном запуске показывает diff изменений
    """
    _hr("Сценарий 3 — Генерация CHANGELOG.md")
    print(f"  Задача: «Сгенерируй CHANGELOG.md для {_FM_DIR}»\n")

    # Статистика файлов проекта
    all_files = await tool(session, "fm_list_files", path=_FM_DIR, pattern="**/*")
    if "error" in all_files:
        _warn(all_files["error"])
        return

    files = all_files["files"]
    py_count = sum(1 for f in files if f["path"].endswith(".py"))
    test_count = sum(1 for f in files if "test_" in f["path"])
    total = all_files["total"]
    _ok(f"Файлов в модуле: {total} (python: {py_count}, тесты: {test_count})")

    # Считаем инструменты
    tools_res = await tool(session, "fm_search_in_files",
                            query="@mcp.tool()", pattern="**/*.py", path=_FM_DIR)
    tool_names = []
    for m in tools_res.get("matches", []):
        for ctx in m["context_after"]:
            if ctx.strip().startswith("def fm_"):
                tool_names.append(ctx.strip()[4:].split("(")[0])
                break

    _ok(f"Инструментов: {len(tool_names)}: {', '.join(tool_names)}")

    today = date.today().isoformat()
    changelog_path = f"{_FM_DIR}/CHANGELOG.md"

    new_entry = (
        f"## [Unreleased] — {today}\n\n"
        "### Инструменты\n"
        + "".join(f"- `{t}()`\n" for t in tool_names)
        + "\n"
        "### Статистика\n"
        f"- Python файлов: {py_count}\n"
        f"- Тестов: {test_count}\n"
        f"- Всего файлов: {total}\n"
        "\n"
        "### Конфигурация\n"
        "- Транспорт: stdio\n"
        "- Корень: `FILE_MANAGER_ROOT` из env\n"
        "- Правила: `FILE_MANAGER_RULES_PATH` из env\n"
    )

    # Читаем существующий CHANGELOG или создаём
    existing = await tool(session, "fm_read_file", path=changelog_path)

    header_line = f"## [Unreleased] — {today}"

    if "error" in existing:
        content = f"# CHANGELOG — file-manager\n\n{new_entry}"
        result = await tool(session, "fm_write_file",
                             path=changelog_path, content=content)
        _ok("CHANGELOG.md создан.")
    else:
        old = existing["content"]
        if header_line in old:
            # Запись за сегодня уже есть — заменяем её
            start = old.index(header_line)
            end = old.find("\n## ", start + 1)
            old_entry = old[start:] if end == -1 else old[start:end + 1]
            new_content = old.replace(old_entry, new_entry, 1)
        else:
            # Вставляем новую запись после заголовка файла
            insert_after = "# CHANGELOG — file-manager\n\n"
            if insert_after in old:
                new_content = old.replace(insert_after, insert_after + new_entry, 1)
            else:
                new_content = f"# CHANGELOG — file-manager\n\n{new_entry}" + old
        result = await tool(session, "fm_write_file",
                             path=changelog_path, content=new_content)
        _ok("CHANGELOG.md обновлён.")

    if result.get("diff"):
        print("\nDiff:\n" + result["diff"])
    else:
        _ok("(содержимое не изменилось)")


# ---------------------------------------------------------------------------
# Сценарий 4: Проверка инвариантов
# ---------------------------------------------------------------------------

async def scenario_check_invariants(session: ClientSession) -> None:
    """
    Задача: «Проверь все файлы проекта на соответствие правилам»

    Ассистент сам:
      1. Загружает rules.json
      2. Проверяет все файлы через regex-паттерны
      3. Группирует нарушения по severity
    """
    _hr("Сценарий 4 — Проверка инвариантов (rules.json)")
    print("  Задача: «Проверь все файлы проекта на соответствие правилам»\n")

    result = await tool(session, "fm_check_invariants", path="llm_mcp")

    if "error" in result:
        _warn(f"Ошибка: {result['error']}")
        return

    violations = result["violations"]
    _ok(f"Правил проверено:  {result['rules_checked']}")
    _ok(f"Файлов проверено:  {result['files_checked']}")
    _ok(f"Нарушений найдено: {len(violations)}\n")

    if not violations:
        print("  Нарушений не найдено.")
        return

    errors = [v for v in violations if v["severity"] == "error"]
    warnings = [v for v in violations if v["severity"] == "warning"]

    if errors:
        print(f"  ❌ ОШИБКИ ({len(errors)}):")
        # топ-10
        for v in errors[:10]:
            print(f"     [{v['rule_id']}] {v['file']}:{v['line_number']}")
            print(f"            {v['line_content'].strip()[:80]}")
            print(f"            → {v['message']}")
        if len(errors) > 10:
            print(f"     ... и ещё {len(errors) - 10} ошибок")

    if warnings:
        print(f"\n  ⚠️  ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
        for v in warnings[:10]:
            print(f"     [{v['rule_id']}] {v['file']}:{v['line_number']}")
            print(f"            {v['line_content'].strip()[:80]}")
            print(f"            → {v['message']}")
        if len(warnings) > 10:
            print(f"     ... и ещё {len(warnings) - 10} предупреждений")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

async def main() -> None:
    project_root = str(_PROJECT_ROOT.resolve())

    server_env = {
        **os.environ,
        "FILE_MANAGER_ROOT": project_root,
        "FILE_MANAGER_RULES_PATH": str(_PROJECT_ROOT / "rules.json"),
    }

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(_SERVER_SCRIPT)],
        env=server_env,
    )

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       MCP File Manager — Сценарии ассистента            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Сервер:       {_SERVER_SCRIPT.name}")
    print(f"  Корень:       {project_root}")
    print(f"  Python:       {sys.executable}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"  Инструментов подключено: {len(tools.tools)}")
            for t in tools.tools:
                print(f"    • {t.name}")

            await scenario_search(session)
            await scenario_update_readme(session)
            await scenario_generate_changelog(session)
            await scenario_check_invariants(session)

    print("\n" + "═" * 62)
    print("  Все сценарии выполнены.")
    print("  Повторный запуск покажет diff изменений (сц. 3).")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
