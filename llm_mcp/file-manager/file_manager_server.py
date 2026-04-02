"""
MCP File Manager Server

Предоставляет инструменты для работы с файлами проекта:
- fm_read_file       — чтение файла
- fm_list_files      — перечисление файлов с glob-фильтрацией
- fm_search_in_files — поиск текста/regex с контекстом ±3 строки
- fm_write_file      — создание/перезапись файла (возвращает unified diff)
- fm_patch_file      — замена фрагмента текста в файле (возвращает unified diff)
- fm_check_invariants — проверка файлов на соответствие rules.json

Конфигурация через .env:
  FILE_MANAGER_ROOT       — корень проекта (по умолчанию CWD)
  FILE_MANAGER_RULES_PATH — путь к rules.json (по умолчанию ROOT/rules.json)
"""

import difflib
import fnmatch
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("file-manager")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".DS_Store",
    ".env",
    ".claude",
    ".idea",
    "rag_index",
}

DEFAULT_EXCLUDE_PATTERNS = {"*.pyc", "*.pyo", "*.pyd"}


def _root() -> Path:
    """Корень проекта: FILE_MANAGER_ROOT из env или CWD."""
    root_env = os.environ.get("FILE_MANAGER_ROOT")
    return Path(root_env).resolve() if root_env else Path(os.getcwd()).resolve()


def _rules_path(root: Path) -> Path:
    """Путь к rules.json: FILE_MANAGER_RULES_PATH из env или ROOT/rules.json."""
    rules_env = os.environ.get("FILE_MANAGER_RULES_PATH")
    return Path(rules_env).resolve() if rules_env else root / "rules.json"


def _load_gitignore(root: Path) -> list[str]:
    """Читает .gitignore из корня и возвращает список паттернов."""
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    lines = []
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _is_ignored(path: Path, root: Path, gitignore_patterns: list[str]) -> bool:
    """Проверяет, должен ли путь быть исключён из обхода."""
    rel = path.relative_to(root)
    parts = rel.parts

    # Дефолтные исключения по имени папки/файла
    for part in parts:
        if part in DEFAULT_EXCLUDES:
            return True
        for pat in DEFAULT_EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(part, pat):
                return True

    # .gitignore паттерны
    rel_str = str(rel).replace("\\", "/")
    name = path.name
    for pattern in gitignore_patterns:
        if pattern.endswith("/"):
            # Директорный паттерн — проверяем компоненты пути
            dir_pat = pattern.rstrip("/")
            for part in parts:
                if fnmatch.fnmatch(part, dir_pat):
                    return True
        elif "/" not in pattern:
            # Простой паттерн по имени файла/директории
            if fnmatch.fnmatch(name, pattern):
                return True
        else:
            # Паттерн с путём — проверяем по относительному пути
            clean = pattern.lstrip("/")
            if fnmatch.fnmatch(rel_str, clean):
                return True
    return False


def _iter_files(search_root: Path, glob_pattern: str, root: Path, gitignore: list[str]):
    """Генератор файлов, удовлетворяющих паттерну и не попадающих в исключения."""
    for path in sorted(search_root.glob(glob_pattern)):
        if path.is_file() and not _is_ignored(path, root, gitignore):
            yield path


def _resolve_path(rel_path: str, root: Path) -> Path | None:
    """
    Разрешает путь относительно root.
    Возвращает None если путь выходит за пределы root (path traversal).
    """
    try:
        resolved = (root / rel_path).resolve()
        resolved.relative_to(root)  # raises ValueError if outside
        return resolved
    except (ValueError, OSError):
        return None


def _unified_diff(old_text: str, new_text: str, filename: str) -> str:
    """Возвращает unified diff двух строк текста."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------


@mcp.tool()
def fm_read_file(path: str) -> dict:
    """
    Читает содержимое файла проекта.

    Args:
        path: Путь к файлу относительно корня проекта (FILE_MANAGER_ROOT).

    Returns:
        {"path": str, "content": str, "lines": int} или {"error": str}
    """
    root = _root()
    resolved = _resolve_path(path, root)
    if resolved is None:
        return {"error": f"Путь '{path}' выходит за пределы корня проекта"}
    if not resolved.exists():
        return {"error": f"Файл не найден: {path}"}
    if not resolved.is_file():
        return {"error": f"'{path}' не является файлом"}
    try:
        content = resolved.read_text(encoding="utf-8")
        return {
            "path": str(resolved.relative_to(root)),
            "content": content,
            "lines": len(content.splitlines()),
        }
    except UnicodeDecodeError:
        return {"error": f"Файл '{path}' не является текстовым (binary)"}


@mcp.tool()
def fm_list_files(path: str = ".", pattern: str = "**/*") -> dict:
    """
    Перечисляет файлы проекта с поддержкой glob-паттерна.
    Исключает .git, __pycache__, node_modules, *.pyc, .DS_Store, .env и .gitignore-паттерны.

    Args:
        path: Директория поиска относительно корня (по умолчанию ".").
        pattern: Glob-паттерн для фильтрации (по умолчанию "**/*").

    Returns:
        {"files": [{"path": str, "size": int}], "total": int}
    """
    root = _root()
    resolved = _resolve_path(path, root)
    if resolved is None:
        return {"error": f"Путь '{path}' выходит за пределы корня проекта"}
    if not resolved.is_dir():
        return {"error": f"'{path}' не является директорией"}

    gitignore = _load_gitignore(root)
    files = []
    for f in _iter_files(resolved, pattern, root, gitignore):
        files.append({
            "path": str(f.relative_to(root)),
            "size": f.stat().st_size,
        })
    return {"files": files, "total": len(files)}


@mcp.tool()
def fm_search_in_files(
    query: str,
    path: str = ".",
    pattern: str = "**/*",
    is_regex: bool = False,
) -> dict:
    """
    Ищет строку или regex по файлам проекта.
    Для каждого совпадения возвращает контекст ±3 строки.

    Args:
        query: Искомая строка или regex-паттерн.
        path: Директория поиска относительно корня (по умолчанию ".").
        pattern: Glob-паттерн файлов (по умолчанию "**/*").
        is_regex: Если True — query трактуется как regex (re.search).

    Returns:
        {"matches": [{file, line_number, line_content, context_before, context_after}],
         "total_matches": int, "files_searched": int}
    """
    root = _root()
    resolved = _resolve_path(path, root)
    if resolved is None:
        return {"error": f"Путь '{path}' выходит за пределы корня проекта"}

    if is_regex:
        try:
            compiled = re.compile(query)
        except re.error as e:
            return {"error": f"Невалидный regex: {e}"}
    else:
        compiled = None

    gitignore = _load_gitignore(root)
    matches = []
    files_searched = 0

    for filepath in _iter_files(resolved, pattern, root, gitignore):
        try:
            content = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        lines = content.splitlines()
        files_searched += 1

        for i, line in enumerate(lines):
            found = compiled.search(line) if is_regex else query in line
            if found:
                context_before = lines[max(0, i - 3) : i]
                context_after = lines[i + 1 : i + 4]
                matches.append({
                    "file": str(filepath.relative_to(root)),
                    "line_number": i + 1,
                    "line_content": line,
                    "context_before": context_before,
                    "context_after": context_after,
                })

    return {
        "matches": matches,
        "total_matches": len(matches),
        "files_searched": files_searched,
    }


@mcp.tool()
def fm_write_file(path: str, content: str) -> dict:
    """
    Создаёт или перезаписывает файл в проекте.
    Если файл уже существует — вычисляет и возвращает unified diff.

    Args:
        path: Путь к файлу относительно корня проекта.
        content: Новое содержимое файла.

    Returns:
        {"path": str, "created": bool, "diff": str | null}
    """
    root = _root()
    resolved = _resolve_path(path, root)
    if resolved is None:
        return {"error": f"Путь '{path}' выходит за пределы корня проекта"}

    created = not resolved.exists()
    diff_text = None

    if not created:
        try:
            old_content = resolved.read_text(encoding="utf-8")
            diff_text = _unified_diff(old_content, content, path)
            if not diff_text:
                diff_text = None  # файл не изменился
        except (UnicodeDecodeError, OSError):
            diff_text = None

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")

    return {
        "path": str(resolved.relative_to(root)),
        "created": created,
        "diff": diff_text,
    }


@mcp.tool()
def fm_patch_file(path: str, old_string: str, new_string: str) -> dict:
    """
    Заменяет первое вхождение old_string на new_string в файле проекта.

    Args:
        path: Путь к файлу относительно корня проекта.
        old_string: Строка, которую нужно заменить.
        new_string: Строка для замены.

    Returns:
        {"path": str, "diff": str} или {"error": str}
    """
    root = _root()
    resolved = _resolve_path(path, root)
    if resolved is None:
        return {"error": f"Путь '{path}' выходит за пределы корня проекта"}
    if not resolved.exists():
        return {"error": f"Файл не найден: {path}"}

    try:
        old_content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": f"Файл '{path}' не является текстовым (binary)"}

    if old_string not in old_content:
        return {"error": f"Строка не найдена в файле '{path}'"}

    new_content = old_content.replace(old_string, new_string, 1)
    diff_text = _unified_diff(old_content, new_content, path)
    resolved.write_text(new_content, encoding="utf-8")

    return {
        "path": str(resolved.relative_to(root)),
        "diff": diff_text,
    }


@mcp.tool()
def fm_check_invariants(path: str = ".") -> dict:
    """
    Проверяет файлы на соответствие правилам из rules.json.
    Путь к rules.json задаётся через FILE_MANAGER_RULES_PATH (по умолчанию ROOT/rules.json).

    Формат rules.json:
    [{"id": "R001", "pattern": "TODO", "message": "...", "severity": "warning",
      "file_pattern": "*.py"}]

    Args:
        path: Директория для проверки относительно корня (по умолчанию ".").

    Returns:
        {"violations": [{rule_id, file, line_number, line_content, message, severity}],
         "rules_checked": int, "files_checked": int}
    """
    root = _root()
    rules_file_path = _rules_path(root)
    if not rules_file_path.exists():
        return {"error": f"rules.json не найден: {rules_file_path}"}

    try:
        rules = json.loads(rules_file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"Ошибка чтения rules.json: {e}"}

    resolved = _resolve_path(path, root)
    if resolved is None:
        return {"error": f"Путь '{path}' выходит за пределы корня проекта"}

    gitignore = _load_gitignore(root)
    violations = []
    files_checked = set()
    rules_resolved = rules_file_path.resolve()

    for rule in rules:
        rule_id = rule.get("id", "?")
        rule_pattern = rule.get("pattern", "")
        rule_message = rule.get("message", "")
        rule_severity = rule.get("severity", "warning")
        file_pattern = rule.get("file_pattern", "**/*")

        try:
            compiled = re.compile(rule_pattern)
        except re.error:
            continue

        for filepath in _iter_files(resolved, "**/*", root, gitignore):
            # Пропускаем сам rules.json
            if filepath.resolve() == rules_resolved:
                continue
            # Фильтрация по file_pattern правила
            if file_pattern != "**/*" and not fnmatch.fnmatch(filepath.name, file_pattern):
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            rel_path = str(filepath.relative_to(root))
            files_checked.add(rel_path)

            for i, line in enumerate(content.splitlines()):
                if compiled.search(line):
                    violations.append({
                        "rule_id": rule_id,
                        "file": rel_path,
                        "line_number": i + 1,
                        "line_content": line,
                        "message": rule_message,
                        "severity": rule_severity,
                    })

    return {
        "violations": violations,
        "rules_checked": len(rules),
        "files_checked": len(files_checked),
    }


if __name__ == "__main__":
    mcp.run()
