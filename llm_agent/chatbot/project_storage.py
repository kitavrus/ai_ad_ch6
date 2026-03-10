"""Сохранение и загрузка проектов (Project → TaskPlan → TaskStep)."""

import json
import logging
from pathlib import Path
from typing import List, Optional

from llm_agent.chatbot.config import DEFAULT_PROFILE
from llm_agent.chatbot.models import Project

logger = logging.getLogger(__name__)


def _projects_dir(profile_name: str = DEFAULT_PROFILE) -> Path:
    return Path("dialogues") / profile_name / "projects"


def _project_path(project_id: str, profile_name: str = DEFAULT_PROFILE) -> Path:
    return _projects_dir(profile_name) / project_id / "project.json"


def save_project(project: Project, profile_name: str = DEFAULT_PROFILE) -> None:
    path = _project_path(project.project_id, profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(project.model_dump_json(indent=2), encoding="utf-8")


def load_project(project_id: str, profile_name: str = DEFAULT_PROFILE) -> Optional[Project]:
    path = _project_path(project_id, profile_name)
    if not path.exists():
        return None
    try:
        return Project.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ошибка загрузки проекта %s: %s", project_id, exc)
        return None


def list_projects(profile_name: str = DEFAULT_PROFILE) -> List[dict]:
    """Возвращает краткое описание всех проектов профиля."""
    base = _projects_dir(profile_name)
    if not base.exists():
        return []
    result = []
    for project_dir in sorted(base.iterdir()):
        pfile = project_dir / "project.json"
        if not pfile.exists():
            continue
        try:
            data = json.loads(pfile.read_text(encoding="utf-8"))
            result.append({
                "project_id": data.get("project_id", ""),
                "name": data.get("name", ""),
                "plan_count": len(data.get("plan_ids", [])),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception as exc:
            logger.warning("Ошибка чтения проекта %s: %s", project_dir.name, exc)
    return result


def delete_project(project_id: str, profile_name: str = DEFAULT_PROFILE) -> bool:
    import shutil
    path = _project_path(project_id, profile_name).parent
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True
