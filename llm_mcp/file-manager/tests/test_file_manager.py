"""
Тесты для file_manager_server.py.

Каждый тест запускается в изолированной tmp_path директории
(через autouse-фикстуру isolated_project из conftest.py).
"""

import json
from pathlib import Path

import pytest

from file_manager_server import (
    fm_check_invariants as check_invariants,
    fm_list_files       as list_files,
    fm_patch_file       as patch_file,
    fm_read_file        as read_file,
    fm_search_in_files  as search_in_files,
    fm_write_file       as write_file,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def make_file(tmp_path: Path, rel_path: str, content: str) -> Path:
    """Создаёт файл в tmp_path с заданным содержимым."""
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TestReadFile
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing_file(self, isolated_project):
        make_file(isolated_project, "hello.txt", "Hello\nWorld\n")
        result = read_file("hello.txt")
        assert result["path"] == "hello.txt"
        assert result["content"] == "Hello\nWorld\n"
        assert result["lines"] == 2

    def test_read_nonexistent_file(self):
        result = read_file("nonexistent.txt")
        assert "error" in result
        assert "не найден" in result["error"]

    def test_read_outside_root(self):
        result = read_file("../../etc/passwd")
        assert "error" in result
        assert "за пределы" in result["error"]

    def test_read_directory_returns_error(self, isolated_project):
        (isolated_project / "subdir").mkdir()
        result = read_file("subdir")
        assert "error" in result

    def test_read_nested_file(self, isolated_project):
        make_file(isolated_project, "a/b/c.txt", "nested content")
        result = read_file("a/b/c.txt")
        assert result["content"] == "nested content"
        assert result["lines"] == 1


# ---------------------------------------------------------------------------
# TestListFiles
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_list_all_files(self, isolated_project):
        make_file(isolated_project, "a.py", "")
        make_file(isolated_project, "b.txt", "")
        result = list_files()
        paths = [f["path"] for f in result["files"]]
        assert "a.py" in paths
        assert "b.txt" in paths
        assert result["total"] == 2

    def test_list_with_glob_pattern(self, isolated_project):
        make_file(isolated_project, "a.py", "")
        make_file(isolated_project, "b.txt", "")
        make_file(isolated_project, "c.py", "")
        result = list_files(pattern="**/*.py")
        paths = [f["path"] for f in result["files"]]
        assert "a.py" in paths
        assert "c.py" in paths
        assert "b.txt" not in paths

    def test_excludes_default_dirs(self, isolated_project):
        make_file(isolated_project, "__pycache__/module.pyc", "")
        make_file(isolated_project, ".git/config", "")
        make_file(isolated_project, "node_modules/pkg/index.js", "")
        make_file(isolated_project, "real.py", "code")
        result = list_files()
        paths = [f["path"] for f in result["files"]]
        assert "real.py" in paths
        assert not any("__pycache__" in p for p in paths)
        assert not any(".git" in p for p in paths)
        assert not any("node_modules" in p for p in paths)

    def test_excludes_pyc_files(self, isolated_project):
        make_file(isolated_project, "main.py", "code")
        make_file(isolated_project, "main.pyc", "binary")
        result = list_files()
        paths = [f["path"] for f in result["files"]]
        assert "main.py" in paths
        assert "main.pyc" not in paths

    def test_respects_gitignore(self, isolated_project):
        (isolated_project / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
        make_file(isolated_project, "app.py", "")
        make_file(isolated_project, "debug.log", "")
        make_file(isolated_project, "build/output.js", "")
        result = list_files()
        paths = [f["path"] for f in result["files"]]
        assert "app.py" in paths
        assert "debug.log" not in paths
        assert not any("build" in p for p in paths)

    def test_list_empty_directory(self, isolated_project):
        result = list_files()
        assert result["files"] == []
        assert result["total"] == 0

    def test_invalid_path_outside_root(self):
        result = list_files(path="../../")
        assert "error" in result

    def test_returns_file_size(self, isolated_project):
        make_file(isolated_project, "data.txt", "hello")
        result = list_files()
        assert result["files"][0]["size"] == 5


# ---------------------------------------------------------------------------
# TestSearchInFiles
# ---------------------------------------------------------------------------


class TestSearchInFiles:
    def test_simple_search(self, isolated_project):
        make_file(isolated_project, "a.py", "def foo():\n    pass\n")
        make_file(isolated_project, "b.py", "def bar():\n    foo()\n")
        result = search_in_files("foo")
        assert result["total_matches"] == 2
        files = {m["file"] for m in result["matches"]}
        assert "a.py" in files
        assert "b.py" in files

    def test_no_matches(self, isolated_project):
        make_file(isolated_project, "a.py", "def foo(): pass")
        result = search_in_files("xyz_not_found")
        assert result["total_matches"] == 0
        assert result["matches"] == []

    def test_context_lines(self, isolated_project):
        content = "\n".join([f"line{i}" for i in range(10)])
        make_file(isolated_project, "file.txt", content)
        result = search_in_files("line5")
        assert len(result["matches"]) == 1
        m = result["matches"][0]
        assert m["line_number"] == 6
        assert len(m["context_before"]) == 3
        assert len(m["context_after"]) == 3
        assert "line4" in m["context_before"]
        assert "line6" in m["context_after"]

    def test_context_at_file_start(self, isolated_project):
        make_file(isolated_project, "f.txt", "MATCH\nline2\nline3\n")
        result = search_in_files("MATCH")
        m = result["matches"][0]
        assert m["context_before"] == []
        assert m["line_number"] == 1

    def test_regex_search(self, isolated_project):
        make_file(isolated_project, "code.py", "x = 123\ny = 'hello'\nz = 456\n")
        result = search_in_files(r"\d+", is_regex=True)
        assert result["total_matches"] == 2

    def test_invalid_regex(self, isolated_project):
        result = search_in_files("[invalid", is_regex=True)
        assert "error" in result

    def test_glob_pattern_filter(self, isolated_project):
        make_file(isolated_project, "a.py", "TODO fix this")
        make_file(isolated_project, "b.md", "TODO update docs")
        result = search_in_files("TODO", pattern="**/*.py")
        assert result["total_matches"] == 1
        assert result["matches"][0]["file"] == "a.py"

    def test_files_searched_count(self, isolated_project):
        make_file(isolated_project, "a.py", "content")
        make_file(isolated_project, "b.py", "content")
        result = search_in_files("xyz")
        assert result["files_searched"] == 2


# ---------------------------------------------------------------------------
# TestWriteFile
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_create_new_file(self, isolated_project):
        result = write_file("new.txt", "hello world")
        assert result["created"] is True
        assert result["diff"] is None
        assert (isolated_project / "new.txt").read_text() == "hello world"

    def test_overwrite_existing_file_returns_diff(self, isolated_project):
        make_file(isolated_project, "existing.txt", "old content\n")
        result = write_file("existing.txt", "new content\n")
        assert result["created"] is False
        assert result["diff"] is not None
        assert "-old content" in result["diff"]
        assert "+new content" in result["diff"]

    def test_overwrite_with_same_content_no_diff(self, isolated_project):
        make_file(isolated_project, "same.txt", "content\n")
        result = write_file("same.txt", "content\n")
        assert result["created"] is False
        assert result["diff"] is None

    def test_creates_parent_directories(self, isolated_project):
        result = write_file("deep/nested/dir/file.txt", "content")
        assert result["created"] is True
        assert (isolated_project / "deep/nested/dir/file.txt").exists()

    def test_path_outside_root_rejected(self):
        result = write_file("../../evil.txt", "pwned")
        assert "error" in result

    def test_diff_format(self, isolated_project):
        make_file(isolated_project, "f.txt", "line1\nline2\n")
        result = write_file("f.txt", "line1\nchanged\n")
        assert "--- a/f.txt" in result["diff"]
        assert "+++ b/f.txt" in result["diff"]


# ---------------------------------------------------------------------------
# TestPatchFile
# ---------------------------------------------------------------------------


class TestPatchFile:
    def test_patch_replaces_first_occurrence(self, isolated_project):
        make_file(isolated_project, "code.py", "foo = 1\nfoo = 2\n")
        result = patch_file("code.py", "foo = 1", "bar = 1")
        assert "error" not in result
        content = (isolated_project / "code.py").read_text()
        assert "bar = 1" in content
        assert "foo = 2" in content  # второе вхождение не тронуто

    def test_patch_returns_unified_diff(self, isolated_project):
        make_file(isolated_project, "doc.md", "# Old Title\n\nContent\n")
        result = patch_file("doc.md", "# Old Title", "# New Title")
        assert "diff" in result
        assert "-# Old Title" in result["diff"]
        assert "+# New Title" in result["diff"]

    def test_patch_old_string_not_found(self, isolated_project):
        make_file(isolated_project, "f.txt", "hello world")
        result = patch_file("f.txt", "not_there", "replacement")
        assert "error" in result
        assert "не найдена" in result["error"]

    def test_patch_nonexistent_file(self):
        result = patch_file("ghost.txt", "old", "new")
        assert "error" in result
        assert "не найден" in result["error"]

    def test_patch_multiline_string(self, isolated_project):
        make_file(isolated_project, "readme.md", "## Section\n\nOld text\nmore old\n")
        result = patch_file("readme.md", "Old text\nmore old", "New text\nmore new")
        assert "error" not in result
        content = (isolated_project / "readme.md").read_text()
        assert "New text" in content
        assert "more new" in content

    def test_patch_path_outside_root(self):
        result = patch_file("../../file.txt", "old", "new")
        assert "error" in result


# ---------------------------------------------------------------------------
# TestCheckInvariants
# ---------------------------------------------------------------------------


def make_rules(tmp_path: Path, rules: list) -> None:
    (tmp_path / "rules.json").write_text(json.dumps(rules), encoding="utf-8")


class TestCheckInvariants:
    def test_finds_violations(self, isolated_project):
        make_rules(isolated_project, [
            {"id": "R001", "pattern": "TODO", "message": "Есть TODO", "severity": "warning"}
        ])
        make_file(isolated_project, "code.py", "# TODO fix this\nx = 1\n")
        result = check_invariants()
        assert result["rules_checked"] == 1
        assert len(result["violations"]) == 1
        v = result["violations"][0]
        assert v["rule_id"] == "R001"
        assert v["file"] == "code.py"
        assert v["line_number"] == 1
        assert v["severity"] == "warning"

    def test_no_violations(self, isolated_project):
        make_rules(isolated_project, [
            {"id": "R001", "pattern": "TODO", "message": "Есть TODO", "severity": "warning"}
        ])
        make_file(isolated_project, "clean.py", "x = 1\ny = 2\n")
        result = check_invariants()
        assert result["violations"] == []

    def test_file_pattern_filter(self, isolated_project):
        make_rules(isolated_project, [
            {
                "id": "R001",
                "pattern": "print\\(",
                "message": "Нет print",
                "severity": "warning",
                "file_pattern": "*.py",
            }
        ])
        make_file(isolated_project, "code.py", "print('hello')\n")
        make_file(isolated_project, "notes.txt", "print(this)\n")
        result = check_invariants()
        files = {v["file"] for v in result["violations"]}
        assert "code.py" in files
        assert "notes.txt" not in files

    def test_missing_rules_json(self, isolated_project):
        result = check_invariants()
        assert "error" in result
        assert "rules.json" in result["error"]

    def test_invalid_rules_json(self, isolated_project):
        (isolated_project / "rules.json").write_text("not valid json", encoding="utf-8")
        result = check_invariants()
        assert "error" in result

    def test_multiple_rules(self, isolated_project):
        make_rules(isolated_project, [
            {"id": "R001", "pattern": "TODO", "message": "TODO", "severity": "warning"},
            {"id": "R002", "pattern": "FIXME", "message": "FIXME", "severity": "error"},
        ])
        make_file(isolated_project, "code.py", "# TODO\n# FIXME\nx = 1\n")
        result = check_invariants()
        assert result["rules_checked"] == 2
        assert len(result["violations"]) == 2
        severities = {v["severity"] for v in result["violations"]}
        assert "warning" in severities
        assert "error" in severities

    def test_files_checked_count(self, isolated_project):
        make_rules(isolated_project, [
            {"id": "R001", "pattern": "TODO", "message": "TODO", "severity": "warning"}
        ])
        make_file(isolated_project, "a.py", "x = 1")
        make_file(isolated_project, "b.py", "y = 2")
        make_file(isolated_project, "c.txt", "z = 3")
        result = check_invariants()
        assert result["files_checked"] == 3

    def test_violation_includes_line_content(self, isolated_project):
        make_rules(isolated_project, [
            {"id": "R001", "pattern": "secret", "message": "Секрет", "severity": "error"}
        ])
        make_file(isolated_project, "config.py", "password = 'secret123'\n")
        result = check_invariants()
        assert len(result["violations"]) == 1
        assert "secret123" in result["violations"][0]["line_content"]

    def test_path_argument(self, isolated_project):
        make_rules(isolated_project, [
            {"id": "R001", "pattern": "TODO", "message": "TODO", "severity": "warning"}
        ])
        make_file(isolated_project, "src/code.py", "# TODO\n")
        make_file(isolated_project, "other/code.py", "# TODO\n")
        result = check_invariants(path="src")
        files = {v["file"] for v in result["violations"]}
        assert any("src" in f for f in files)
        assert not any("other" in f for f in files)
