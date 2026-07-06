"""GUI-facing actions for workspace games registry maintenance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

from games_registry import (
    REFRESH_MODE_DEEP,
    REFRESH_MODE_LITE,
    CancelCheck,
    default_games_md_path,
    discover_new_project_paths,
    empty_registry,
    import_from_games_md,
    load_registry,
    merge_discovered_projects,
    record_batch,
    refresh_all,
    refresh_project,
    save_registry,
    update_project_manual_fields,
    write_games_md,
)

from .games_registry_view import find_project_id_for_game_root, resolve_registry_path

RegistryProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class RegistryActionResult:
    ok: bool
    message: str
    rendered_games_md: bool = False
    cancelled: bool = False


def _registry_file(workspace_root: Path) -> Path:
    return resolve_registry_path(workspace_root)


def _mode_label(mode: str) -> str:
    return "深度" if mode == REFRESH_MODE_DEEP else "快速"


def refresh_registry_projects(
    workspace_root: Path,
    *,
    project_id: str | None = None,
    refresh_everything: bool = False,
    mode: str = REFRESH_MODE_LITE,
    on_progress: RegistryProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> RegistryActionResult:
    registry_path = _registry_file(workspace_root)
    if not registry_path.is_file():
        return RegistryActionResult(False, f"未找到 {registry_path.name}。")

    if should_cancel and should_cancel():
        return RegistryActionResult(False, "刷新已取消。", cancelled=True)

    data = load_registry(registry_path)
    label = _mode_label(mode)

    if refresh_everything:
        count, cancelled = refresh_all(
            data,
            workspace_root=workspace_root,
            mode=mode,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        save_registry(registry_path, data)
        if cancelled:
            return RegistryActionResult(
                False,
                f"已停止{label}刷新，已完成 {count} 个项目。",
                cancelled=True,
            )
        return RegistryActionResult(True, f"已{label}刷新全部 {count} 个项目。")

    if not project_id:
        return RegistryActionResult(False, "未指定要刷新的项目。")

    if on_progress is not None:
        project = next(
            (item for item in data.get("projects", []) if item.get("id") == project_id),
            None,
        )
        name = str((project or {}).get("name") or project_id)
        on_progress(1, 1, name)

    if should_cancel and should_cancel():
        return RegistryActionResult(False, "刷新已取消。", cancelled=True)

    project = refresh_project(
        data,
        project_id,
        workspace_root=workspace_root,
        mode=mode,
    )
    if project is None:
        return RegistryActionResult(False, f"未找到项目：{project_id}")
    save_registry(registry_path, data)

    if should_cancel and should_cancel():
        name = str(project.get("name") or project_id)
        return RegistryActionResult(
            False,
            f"已停止{label}刷新；项目 {name} 的结果已保存。",
            cancelled=True,
        )

    name = str(project.get("name") or project_id)
    return RegistryActionResult(True, f"已{label}刷新项目 {name}。")


def import_registry_from_games_md(
    workspace_root: Path,
    *,
    merge: bool = False,
) -> RegistryActionResult:
    md_path = default_games_md_path(workspace_root)
    if not md_path.is_file():
        return RegistryActionResult(False, f"未找到 {md_path.name}。")

    registry_path = _registry_file(workspace_root)
    try:
        registry = import_from_games_md(
            md_path=md_path,
            registry_path=registry_path,
            workspace_root=workspace_root,
            merge=merge and registry_path.is_file(),
        )
    except (OSError, ValueError) as exc:
        return RegistryActionResult(False, f"从 GAMES.md 导入失败：{exc}")

    summary = str(registry.get("update_summary") or "").strip()
    count = len(registry.get("projects") or [])
    message = f"已从 GAMES.md 导入 {count} 个项目。"
    if summary:
        message = f"{message} {summary}"
    return RegistryActionResult(True, message)


def discover_registry_projects(
    workspace_root: Path,
    *,
    refresh_new: bool = True,
    mode: str = REFRESH_MODE_LITE,
) -> RegistryActionResult:
    registry_path = _registry_file(workspace_root)
    if registry_path.is_file():
        data = load_registry(registry_path)
    else:
        data = empty_registry(workspace_root)
        data["workspace_root"] = workspace_root.as_posix()

    pending_before = len(discover_new_project_paths(workspace_root, data))
    if pending_before <= 0:
        return RegistryActionResult(True, "未发现新的 Game_* 项目。")

    added_count, added_paths = merge_discovered_projects(
        data,
        workspace_root=workspace_root,
        refresh_new=refresh_new,
        mode=mode,
    )
    save_registry(registry_path, data)
    names = "、".join(added_paths[:5])
    if len(added_paths) > 5:
        names = f"{names} 等"
    label = _mode_label(mode)
    message = f"已扫描并登记 {added_count} 个新项目"
    if refresh_new:
        message = f"{message}（已{label}刷新）"
    message = f"{message}：{names}。"
    return RegistryActionResult(True, message)


def save_registry_project_fields(
    workspace_root: Path,
    *,
    project_id: str,
    play_status: str,
    translation_status: str,
    notes: str,
) -> RegistryActionResult:
    registry_path = _registry_file(workspace_root)
    if not registry_path.is_file():
        return RegistryActionResult(False, f"未找到 {registry_path.name}。")

    data = load_registry(registry_path)
    project = update_project_manual_fields(
        data,
        project_id,
        play_status=play_status,
        translation_status=translation_status,
        notes=notes,
    )
    if project is None:
        return RegistryActionResult(False, f"未找到项目：{project_id}")

    data["update_summary"] = f"已更新项目 {project.get('name') or project_id} 的人工字段"
    save_registry(registry_path, data)
    return RegistryActionResult(True, f"已保存项目 {project.get('name') or project_id} 的修改。")


def render_registry_games_md(workspace_root: Path) -> RegistryActionResult:
    registry_path = _registry_file(workspace_root)
    if not registry_path.is_file():
        return RegistryActionResult(False, f"未找到 {registry_path.name}。")

    data = load_registry(registry_path)
    md_path = default_games_md_path(workspace_root)
    data["update_summary"] = data.get("update_summary") or "由图形界面同步生成"
    save_registry(registry_path, data)
    write_games_md(data, md_path)
    return RegistryActionResult(True, f"已更新 {md_path.name}。", rendered_games_md=True)


def record_apply_batch_for_game_root(
    workspace_root: Path,
    *,
    game_root: Path | str | None,
    manifest_path: str | Path,
) -> RegistryActionResult:
    registry_path = _registry_file(workspace_root)
    if not registry_path.is_file():
        return RegistryActionResult(False, f"跳过 registry 记录：未找到 {registry_path.name}。")

    project_id = find_project_id_for_game_root(
        workspace_root=workspace_root,
        game_root=game_root,
        registry_path=registry_path,
    )
    if not project_id:
        return RegistryActionResult(False, "跳过 registry 记录：当前项目不在工作区总表中。")

    data = load_registry(registry_path)
    try:
        project = record_batch(
            data,
            project_id=project_id,
            manifest_path=Path(manifest_path),
        )
    except (ValueError, OSError) as exc:
        return RegistryActionResult(False, f"registry 批次记录失败：{exc}")

    if project is None:
        return RegistryActionResult(False, f"registry 批次记录失败：未找到项目 {project_id}。")

    refresh_project(
        data,
        project_id,
        workspace_root=workspace_root,
        mode=REFRESH_MODE_LITE,
    )
    save_registry(registry_path, data)
    summary = (project.get("auto") or {}).get("last_batch_summary") or ""
    message = "已记录写回批次到 games_registry.json。"
    if summary:
        message = f"{message} {summary}"
    return RegistryActionResult(True, message)


def prompt_render_games_md(parent: "QWidget | None", workspace_root: Path) -> RegistryActionResult:
    from PySide6.QtWidgets import QMessageBox

    reply = QMessageBox.question(
        parent,
        "同步 GAMES.md",
        "已将写回结果记录到 games_registry.json。\n是否据此更新 GAMES.md？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return RegistryActionResult(True, "已跳过 GAMES.md 同步。")
    return render_registry_games_md(workspace_root)


def handle_post_apply_registry_update(
    parent: "QWidget | None",
    *,
    workspace_root: Path,
    game_root: Path | str | None,
    manifest_path: str | Path,
) -> RegistryActionResult:
    record_result = record_apply_batch_for_game_root(
        workspace_root,
        game_root=game_root,
        manifest_path=manifest_path,
    )
    if not record_result.ok:
        return record_result

    render_result = prompt_render_games_md(parent, workspace_root)
    if render_result.rendered_games_md:
        return RegistryActionResult(
            True,
            f"{record_result.message} {render_result.message}".strip(),
            rendered_games_md=True,
        )
    return RegistryActionResult(True, f"{record_result.message} {render_result.message}".strip())