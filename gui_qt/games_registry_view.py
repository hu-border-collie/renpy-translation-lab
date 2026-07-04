"""GUI helpers for displaying the workspace games registry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from games_registry import (
    REGISTRY_FILENAME,
    default_registry_path,
    default_workspace_root,
    load_registry,
)
from translator_runtime import canonical_abs_path, resolve_effective_game_root

REGISTRY_TABLE_COLUMNS = ("项目", "路径", "版本", "游玩", "翻译")


@dataclass(frozen=True)
class RegistryRow:
    project_id: str
    name: str
    path: str
    version: str
    play_status: str
    translation_status: str
    notes: str
    engine: str
    in_renpy_pipeline: bool
    work_dir: str

    @property
    def tooltip(self) -> str:
        parts = [self.notes] if self.notes else []
        if self.engine and self.engine != "renpy":
            parts.append(f"引擎：{self.engine}")
        if not self.in_renpy_pipeline:
            parts.append("不纳入 Ren'Py 汉化流程")
        return "\n".join(part for part in parts if part.strip())


def resolve_workspace_root(tool_root: Path | None = None) -> Path:
    if tool_root is not None:
        return tool_root.parent
    return default_workspace_root()


def resolve_registry_path(workspace_root: Path | None = None) -> Path:
    return default_registry_path(workspace_root)


def resolve_project_work_dir(workspace_root: Path, project_path: str) -> Path:
    project_root = workspace_root / Path(project_path.replace("\\", "/"))
    return Path(canonical_abs_path(resolve_effective_game_root(str(project_root))))


def registry_row_from_project(workspace_root: Path, project: dict[str, Any]) -> RegistryRow:
    path = str(project.get("path") or "").strip()
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    engine = str(project.get("engine") or auto.get("engine") or "renpy")
    return RegistryRow(
        project_id=str(project.get("id") or ""),
        name=str(project.get("name") or ""),
        path=path,
        version=str(project.get("version") or "待确认"),
        play_status=str(project.get("play_status") or "待确认"),
        translation_status=str(project.get("translation_status") or "待确认"),
        notes=str(project.get("notes") or ""),
        engine=engine,
        in_renpy_pipeline=bool(project.get("in_renpy_pipeline", True)),
        work_dir=resolve_project_work_dir(workspace_root, path).as_posix() if path else "",
    )


def load_registry_rows(
    *,
    workspace_root: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[list[RegistryRow], str]:
    workspace = workspace_root or default_workspace_root()
    registry_file = registry_path or (workspace / REGISTRY_FILENAME)
    if not registry_file.is_file():
        return [], f"未找到 {registry_file.name}，请在工作区根目录运行 import-md。"

    registry = load_registry(registry_file)
    projects = registry.get("projects")
    if not isinstance(projects, list) or not projects:
        return [], f"{registry_file.name} 中没有项目记录。"

    rows = [
        registry_row_from_project(workspace, project)
        for project in projects
        if isinstance(project, dict) and project.get("path")
    ]
    rows.sort(key=lambda row: row.name.lower())
    summary = str(registry.get("update_summary") or "").strip()
    return rows, summary


def row_matches_game_root(row: RegistryRow, game_root: Path | str | None) -> bool:
    if game_root is None or not row.work_dir:
        return False
    current = canonical_abs_path(str(game_root))
    target = canonical_abs_path(row.work_dir)
    return current == target


def format_registry_status_message(row_count: int, summary: str, *, missing_message: str = "") -> str:
    if missing_message:
        return missing_message
    if row_count <= 0:
        return "工作区项目总览：暂无记录。"
    message = f"工作区项目总览：共 {row_count} 个项目。"
    if summary:
        message = f"{message} {summary}"
    return message