"""Create and resume GUI workflows for each workbench work mode."""
from __future__ import annotations

from typing import Any, Protocol

from .sync_translation_workflow import SyncTranslationWorkflow
from .translation_workflow import TranslationWorkflow, WorkflowStep, WorkflowUpdate
from .work_modes import WorkMode, manifest_mode_for_work_mode, work_mode_spec


class GuiWorkflow(Protocol):
    manifest_path: str

    def current_step(self) -> WorkflowStep | None: ...

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate: ...


def create_workflow(mode: WorkMode | str) -> GuiWorkflow | None:
    spec = work_mode_spec(mode)
    if not spec.implemented:
        return None
    if spec.mode == WorkMode.BATCH_TRANSLATION:
        return TranslationWorkflow.start_new()
    if spec.mode == WorkMode.SYNC_TRANSLATION:
        return SyncTranslationWorkflow.start_new()
    return None


def resume_workflow(
    mode: WorkMode | str,
    manifest_path: str,
    manifest: dict[str, Any],
) -> GuiWorkflow | None:
    spec = work_mode_spec(mode)
    if not spec.implemented or not spec.supports_resume:
        return None
    if spec.mode == WorkMode.BATCH_TRANSLATION:
        return TranslationWorkflow.resume_manifest(manifest_path, manifest)
    return None


def validate_resume_manifest(
    mode: WorkMode | str,
    manifest: dict[str, Any],
    *,
    game_root: str | None,
    normalized_path_text,
) -> None:
    spec = work_mode_spec(mode)
    expected_mode = manifest_mode_for_work_mode(spec.mode)
    actual_mode = manifest.get("mode", "translation")
    actual_text = actual_mode.strip() if isinstance(actual_mode, str) else "translation"
    if expected_mode == "translation":
        if actual_text not in {"", "translation"}:
            raise ValueError(f"最新任务不是{spec.label}任务，不能在这里继续。")
    elif actual_text != expected_mode:
        raise ValueError(f"最新任务不是{spec.label}任务，不能在这里继续。")

    base_dir = manifest.get("base_dir")
    if not isinstance(base_dir, str) or not base_dir.strip():
        raise ValueError("最新任务缺少项目目录信息，不能安全继续。")
    if game_root is not None and normalized_path_text(base_dir) != normalized_path_text(game_root):
        raise ValueError("最新任务属于其他项目，请先选择对应 work 目录。")