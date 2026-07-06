"""GUI helpers for displaying the workspace games registry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from games_registry import (
    ENGINES,
    GAMES_MD_FILENAME,
    REGISTRY_FILENAME,
    TRANSLATION_STATUSES,
    default_registry_path,
    default_workspace_root,
    discover_new_project_paths,
    get_registry_preferences,
    load_registry,
    resolve_doctor_mode,
    resolve_layout_status,
)
from translator_runtime import canonical_abs_path, resolve_effective_game_root

REGISTRY_TABLE_COLUMNS = ("项目", "路径", "版本", "目录状态", "游玩", "翻译")

REGISTRY_SORT_NAME_ASC = "name_asc"
REGISTRY_SORT_NAME_DESC = "name_desc"
REGISTRY_SORT_PATH_ASC = "path_asc"
REGISTRY_SORT_TRANSLATION = "translation"
REGISTRY_SORT_LAST_REFRESH = "last_refresh"

REGISTRY_SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    (REGISTRY_SORT_NAME_ASC, "名称 A→Z"),
    (REGISTRY_SORT_NAME_DESC, "名称 Z→A"),
    (REGISTRY_SORT_PATH_ASC, "路径"),
    (REGISTRY_SORT_TRANSLATION, "翻译状态"),
    (REGISTRY_SORT_LAST_REFRESH, "最近刷新"),
)

REGISTRY_PREF_AUTO_DISCOVER = "auto_discover_on_open"


@dataclass(frozen=True)
class RegistryRow:
    project_id: str
    name: str
    path: str
    version: str
    layout_status: str
    play_status: str
    translation_status: str
    notes: str
    engine: str
    in_renpy_pipeline: bool
    work_dir: str

    auto_summary: str = ""
    translation_status_source: str = ""
    doctor_mode: str = ""
    last_refresh_at: str = ""

    @property
    def tooltip(self) -> str:
        parts: list[str] = []
        source_hint = _translation_status_source_label(self.translation_status_source)
        if source_hint:
            parts.append(source_hint)
        if self.auto_summary:
            parts.append(self.auto_summary)
        if self.notes:
            parts.append(self.notes)
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


def format_auto_summary(auto: dict[str, Any]) -> str:
    if not auto:
        return ""
    parts: list[str] = []
    tl_files = auto.get("tl_rpy_files")
    if isinstance(tl_files, int) and tl_files > 0:
        parts.append(f"翻译文件 {tl_files} 个")
    pending = auto.get("pending_tasks")
    if isinstance(pending, int):
        parts.append(f"待译 {pending} 条")
    pct = auto.get("dialogue_translated_pct")
    if isinstance(pct, (int, float)):
        parts.append(f"对话约 {pct}% 已译")
    refresh_mode = auto.get("refresh_mode")
    if refresh_mode == "deep":
        parts.append("深度扫描（含 doctor）")
    elif refresh_mode == "lite":
        parts.append("快速扫描")
    if auto.get("glossary"):
        parts.append(f"术语：{auto['glossary']}")
    batch_summary = auto.get("last_batch_summary")
    if isinstance(batch_summary, str) and batch_summary.strip():
        parts.append(batch_summary.strip())
    return "；".join(parts)


def resolve_project_work_dir(workspace_root: Path, project_path: str) -> Path:
    project_root = workspace_root / Path(project_path.replace("\\", "/"))
    return Path(canonical_abs_path(resolve_effective_game_root(str(project_root))))


def _translation_status_source_label(source: str) -> str:
    if source == "manual":
        return "翻译状态：人工维护（快速/深度刷新均不会改写）"
    if source == "doctor":
        return "翻译状态：由深度刷新推断"
    if source == "scan":
        return "翻译状态：由快速刷新推断"
    if source == "batch":
        return "翻译状态：由最近批次更新"
    return ""


def registry_row_from_project(workspace_root: Path, project: dict[str, Any]) -> RegistryRow:
    path = str(project.get("path") or "").strip()
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    engine = str(project.get("engine") or auto.get("engine") or "renpy")
    status_source = str(project.get("translation_status_source") or "")
    return RegistryRow(
        project_id=str(project.get("id") or ""),
        name=str(project.get("name") or ""),
        path=path,
        version=str(project.get("version") or "待确认"),
        layout_status=resolve_layout_status(project) or "待确认",
        play_status=str(project.get("play_status") or "待确认"),
        translation_status=str(project.get("translation_status") or "待确认"),
        notes=str(project.get("notes") or ""),
        engine=engine,
        in_renpy_pipeline=bool(project.get("in_renpy_pipeline", True)),
        work_dir=resolve_project_work_dir(workspace_root, path).as_posix() if path else "",
        auto_summary=format_auto_summary(auto),
        translation_status_source=status_source,
        doctor_mode=resolve_doctor_mode(project),
        last_refresh_at=str(auto.get("last_refresh_at") or ""),
    )


def find_project_id_for_game_root(
    *,
    workspace_root: Path,
    game_root: Path | str | None,
    registry_path: Path | None = None,
) -> str | None:
    if game_root is None:
        return None
    registry_file = registry_path or resolve_registry_path(workspace_root)
    if not registry_file.is_file():
        return None
    registry_data = load_registry(registry_file)
    projects = registry_data.get("projects")
    if not isinstance(projects, list):
        return None
    for project in projects:
        if not isinstance(project, dict) or not project.get("path"):
            continue
        row = registry_row_from_project(workspace_root, project)
        if row_matches_game_root(row, game_root):
            return row.project_id or None
    return None


def load_registry_rows(
    *,
    workspace_root: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[list[RegistryRow], str]:
    workspace = workspace_root or default_workspace_root()
    registry_file = registry_path or (workspace / REGISTRY_FILENAME)
    if not registry_file.is_file():
        games_md = workspace / GAMES_MD_FILENAME
        if games_md.is_file():
            return [], (
                f"未找到 {registry_file.name}。"
                f"可点击「从 GAMES.md 导入」，或「扫描新项目」从 Game_* 目录创建总表。"
            )
        return [], (
            f"未找到 {registry_file.name}。"
            f"可点击「扫描新项目」发现 Game_* 目录，或先准备 {GAMES_MD_FILENAME} 后导入。"
        )

    registry = load_registry(registry_file)
    projects = registry.get("projects")
    if not isinstance(projects, list) or not projects:
        return [], f"{registry_file.name} 中没有项目记录。可点击「扫描新项目」或「从 GAMES.md 导入」。"

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


def count_undiscovered_projects(
    *,
    workspace_root: Path,
    registry_path: Path | None = None,
) -> int:
    registry_file = registry_path or resolve_registry_path(workspace_root)
    if not registry_file.is_file():
        registry_data: dict[str, Any] = {"projects": []}
    else:
        registry_data = load_registry(registry_file)
    return len(discover_new_project_paths(workspace_root, registry_data))


def load_registry_preferences(
    *,
    workspace_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace_root or default_workspace_root()
    registry_file = registry_path or (workspace / REGISTRY_FILENAME)
    if not registry_file.is_file():
        return {}
    return get_registry_preferences(load_registry(registry_file))


def _row_search_blob(row: RegistryRow) -> str:
    return " ".join(
        (
            row.name,
            row.path,
            row.version,
            row.layout_status,
            row.play_status,
            row.translation_status,
            row.notes,
            row.engine,
            row.doctor_mode,
            row.auto_summary,
        )
    ).lower()


def filter_and_sort_registry_rows(
    rows: list[RegistryRow],
    *,
    search_text: str = "",
    engine_filter: str = "",
    translation_filter: str = "",
    sort_key: str = REGISTRY_SORT_NAME_ASC,
) -> list[RegistryRow]:
    filtered = list(rows)
    query = search_text.strip().lower()
    if query:
        filtered = [row for row in filtered if query in _row_search_blob(row)]

    engine_value = engine_filter.strip().lower()
    if engine_value and engine_value != "all":
        filtered = [row for row in filtered if row.engine.lower() == engine_value]

    translation_value = translation_filter.strip()
    if translation_value and translation_value != "全部":
        filtered = [
            row for row in filtered if row.translation_status == translation_value
        ]

    if sort_key == REGISTRY_SORT_NAME_DESC:
        filtered.sort(key=lambda row: row.name.lower(), reverse=True)
    elif sort_key == REGISTRY_SORT_PATH_ASC:
        filtered.sort(key=lambda row: row.path.lower())
    elif sort_key == REGISTRY_SORT_TRANSLATION:
        filtered.sort(key=lambda row: row.translation_status)
    elif sort_key == REGISTRY_SORT_LAST_REFRESH:
        filtered.sort(key=lambda row: row.last_refresh_at, reverse=True)
    else:
        filtered.sort(key=lambda row: row.name.lower())
    return filtered


def registry_engine_filter_options() -> list[tuple[str, str]]:
    return [("all", "全部引擎"), *[(engine, engine) for engine in sorted(ENGINES)]]


def registry_translation_filter_options() -> list[str]:
    return ["全部", *sorted(TRANSLATION_STATUSES)]


def format_registry_status_message(row_count: int, summary: str, *, missing_message: str = "") -> str:
    if missing_message:
        return missing_message
    if row_count <= 0:
        return "工作区项目总览：暂无记录。"
    message = f"工作区项目总览：共 {row_count} 个项目。"
    if summary:
        message = f"{message} {summary}"
    return message