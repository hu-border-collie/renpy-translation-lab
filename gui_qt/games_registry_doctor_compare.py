"""Compare workbench doctor results with games registry snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from games_registry import (
    find_project,
    load_registry,
    resolve_doctor_mode,
    resolve_layout_status,
)

from .games_registry_view import find_project_id_for_game_root, resolve_registry_path


@dataclass(frozen=True)
class RegistryDoctorCompareResult:
    matched: bool | None
    registry_layout: str
    registry_mode: str
    doctor_layout: str
    doctor_mode: str
    last_refresh_at: str
    project_name: str
    message: str
    log_line: str


def _normalize_field(value: object) -> str:
    return str(value or "").strip()


def compare_registry_with_doctor_report(
    workspace_root: Path,
    *,
    game_root: Path | str | None,
    report: dict[str, Any] | None,
) -> RegistryDoctorCompareResult | None:
    if report is None:
        return None

    registry_path = resolve_registry_path(workspace_root)
    if not registry_path.is_file():
        return None

    project_id = find_project_id_for_game_root(
        workspace_root=workspace_root,
        game_root=game_root,
        registry_path=registry_path,
    )
    if not project_id:
        return RegistryDoctorCompareResult(
            matched=None,
            registry_layout="",
            registry_mode="",
            doctor_layout=_normalize_field(report.get("layout_status")),
            doctor_mode=_normalize_field(report.get("mode")),
            last_refresh_at="",
            project_name="",
            message="当前项目不在工作区总表中，无法与 registry 对比 layout。",
            log_line="[总表对比] 当前项目未登记在 games_registry.json。",
        )

    registry = load_registry(registry_path)
    project = find_project(registry, project_id)
    if project is None:
        return None

    registry_layout = resolve_layout_status(project)
    registry_mode = resolve_doctor_mode(project)
    doctor_layout = _normalize_field(report.get("layout_status"))
    doctor_mode = _normalize_field(report.get("mode"))
    auto = project.get("auto") if isinstance(project.get("auto"), dict) else {}
    last_refresh_at = _normalize_field(auto.get("last_refresh_at"))
    project_name = _normalize_field(project.get("name")) or project_id

    layout_match = registry_layout == doctor_layout
    mode_match = registry_mode == doctor_mode or (not registry_mode and not doctor_mode)
    matched = layout_match and mode_match

    if matched:
        message = (
            f"与工作区总表一致：layout={registry_layout or '（空）'}"
            f"，mode={registry_mode or '（空）'}。"
        )
        log_line = (
            f"[总表对比] 与 games_registry 记录一致"
            f"（layout={registry_layout or '-'}，mode={registry_mode or '-'}）。"
        )
    else:
        message = (
            f"与工作区总表不一致：总表 layout={registry_layout or '（空）'}"
            f" / mode={registry_mode or '（空）'}，"
            f"当前检查 layout={doctor_layout or '（空）'}"
            f" / mode={doctor_mode or '（空）'}。"
        )
        if last_refresh_at:
            message = f"{message} 总表最近刷新：{last_refresh_at}。"
        message = f"{message} 建议在「工作区项目…」中对当前项执行刷新。"
        log_line = (
            f"[总表对比] 不一致 — registry: layout={registry_layout or '-'}, "
            f"mode={registry_mode or '-'}；doctor: layout={doctor_layout or '-'}, "
            f"mode={doctor_mode or '-'}。"
        )

    return RegistryDoctorCompareResult(
        matched=matched,
        registry_layout=registry_layout,
        registry_mode=registry_mode,
        doctor_layout=doctor_layout,
        doctor_mode=doctor_mode,
        last_refresh_at=last_refresh_at,
        project_name=project_name,
        message=message,
        log_line=log_line,
    )


def format_registry_compare_hint(
    compare: RegistryDoctorCompareResult | None,
    *,
    for_registry_dialog: bool = False,
) -> str:
    if compare is None:
        if for_registry_dialog:
            return "请先在工作台运行「环境检查」，或选中已检查过的当前项目。"
        return ""

    if compare.matched is None:
        return compare.message

    if compare.matched:
        prefix = "与环境检查一致"
    else:
        prefix = "与环境检查不一致"

    if for_registry_dialog:
        return (
            f"{prefix}：总表 layout={compare.registry_layout or '—'}"
            f" / mode={compare.registry_mode or '—'}；"
            f"检查 layout={compare.doctor_layout or '—'}"
            f" / mode={compare.doctor_mode or '—'}。"
        )
    return compare.message