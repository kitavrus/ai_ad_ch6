import subprocess
import pytest
from pathlib import Path

from git_server import get_current_branch, list_files, get_diff


@pytest.fixture()
def git_repo(tmp_path):
    """Create a minimal git repo in tmp_path."""
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True, capture_output=True)
    return tmp_path


def _commit(repo: Path, message: str = "init"):
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True, capture_output=True)


class TestGetCurrentBranch:
    def test_returns_main_branch(self, git_repo):
        (git_repo / "file.txt").write_text("hello")
        _commit(git_repo)
        result = get_current_branch(str(git_repo))
        assert result == "Current branch: main"

    def test_invalid_repo(self, tmp_path):
        result = get_current_branch(str(tmp_path / "nonexistent"))
        assert "Git error" in result or "Git command failed" in result or "error" in result.lower()

    def test_not_a_git_repo(self, tmp_path):
        result = get_current_branch(str(tmp_path))
        assert "Git error" in result or "not a git" in result.lower()


class TestListFiles:
    def test_lists_tracked_files(self, git_repo):
        (git_repo / "a.py").write_text("x = 1")
        (git_repo / "b.txt").write_text("hello")
        _commit(git_repo)
        result = list_files(str(git_repo))
        assert "a.py" in result
        assert "b.txt" in result
        assert "2 file(s)" in result

    def test_filter_by_pattern(self, git_repo):
        (git_repo / "a.py").write_text("x = 1")
        (git_repo / "b.txt").write_text("hello")
        _commit(git_repo)
        result = list_files(str(git_repo), pattern="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_empty_repo(self, git_repo):
        # No commits yet — ls-files returns nothing
        result = list_files(str(git_repo))
        assert "No tracked files found" in result

    def test_untracked_file_not_shown(self, git_repo):
        (git_repo / "tracked.txt").write_text("a")
        _commit(git_repo)
        (git_repo / "untracked.txt").write_text("b")
        result = list_files(str(git_repo))
        assert "tracked.txt" in result
        assert "untracked.txt" not in result


class TestGetDiff:
    def test_no_changes(self, git_repo):
        (git_repo / "file.txt").write_text("hello")
        _commit(git_repo)
        result = get_diff(str(git_repo))
        assert result == "No changes"

    def test_unstaged_changes(self, git_repo):
        (git_repo / "file.txt").write_text("hello")
        _commit(git_repo)
        (git_repo / "file.txt").write_text("hello world")
        result = get_diff(str(git_repo))
        assert "hello world" in result or "+" in result

    def test_staged_changes(self, git_repo):
        (git_repo / "file.txt").write_text("hello")
        _commit(git_repo)
        (git_repo / "file.txt").write_text("hello world")
        subprocess.run(["git", "-C", str(git_repo), "add", "file.txt"], check=True, capture_output=True)
        result = get_diff(str(git_repo), staged=True)
        assert "hello world" in result or "+" in result

    def test_staged_flag_false_shows_no_staged(self, git_repo):
        (git_repo / "file.txt").write_text("hello")
        _commit(git_repo)
        (git_repo / "file.txt").write_text("hello world")
        subprocess.run(["git", "-C", str(git_repo), "add", "file.txt"], check=True, capture_output=True)
        # After staging, unstaged diff should be empty
        result = get_diff(str(git_repo), staged=False)
        assert result == "No changes"

    def test_diff_specific_file(self, git_repo):
        (git_repo / "a.txt").write_text("aaa")
        (git_repo / "b.txt").write_text("bbb")
        _commit(git_repo)
        (git_repo / "a.txt").write_text("aaa modified")
        (git_repo / "b.txt").write_text("bbb modified")
        result = get_diff(str(git_repo), file_path="a.txt")
        assert "a.txt" in result or "aaa modified" in result
        assert "b.txt" not in result
