import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("git-commands")


def _run_git(args: list[str], repo_path: str) -> tuple[bool, str]:
    """Run a git command in the given repo path. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip()
        return False, f"Git error: {stderr}" if stderr else f"Git command failed (exit code {e.returncode})"
    except FileNotFoundError:
        return False, "Git is not installed or not found in PATH"


@mcp.tool()
def get_current_branch(repo_path: str = ".") -> str:
    """Возвращает имя текущей git-ветки репозитория."""
    ok, output = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    if not ok:
        return output
    return f"Current branch: {output}"


@mcp.tool()
def list_files(repo_path: str = ".", pattern: str | None = None) -> str:
    """Возвращает список файлов, отслеживаемых git (git ls-files).

    Args:
        repo_path: Путь к репозиторию (по умолчанию текущая директория).
        pattern: Опциональный glob-паттерн для фильтрации файлов.
    """
    args = ["ls-files"]
    if pattern:
        args += ["--", pattern]
    ok, output = _run_git(args, repo_path)
    if not ok:
        return output
    if not output:
        return "No tracked files found"
    lines = output.splitlines()
    return f"{len(lines)} file(s):\n" + output


@mcp.tool()
def get_diff(repo_path: str = ".", staged: bool = False, file_path: str | None = None) -> str:
    """Возвращает git diff для репозитория или конкретного файла.

    Args:
        repo_path: Путь к репозиторию (по умолчанию текущая директория).
        staged: Если True — показывает staged-изменения (git diff --staged).
        file_path: Опциональный путь к файлу для просмотра diff только по нему.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if file_path:
        args += ["--", file_path]
    ok, output = _run_git(args, repo_path)
    if not ok:
        return output
    if not output:
        return "No changes"
    return output


if __name__ == "__main__":
    mcp.run()
