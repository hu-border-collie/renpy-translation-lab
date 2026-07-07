"""GUI workflow helpers for retry package follow-up after build-retry."""
from __future__ import annotations

from .check_report import WritebackSummary
from .diagnostics_context import existing_retry_manifest_path, manifest_check_safety_level
from .translation_workflow import TranslationWorkflow, WorkflowStep

_RETRY_STEP_BUTTON_LABELS = {
    "submit": "提交补译任务",
    "status": "继续补译",
    "download": "继续补译",
    "check": "继续补译",
    "merge-retry": "合并补译结果",
    "check-parent": "继续补译",
}


def create_retry_followup_workflow(
    retry_manifest_path: str,
    retry_manifest: dict[str, object],
    parent_manifest_path: str,
    *,
    submit_max_cost: float | None = None,
) -> TranslationWorkflow:
    workflow = TranslationWorkflow.resume_retry_manifest(
        retry_manifest_path,
        retry_manifest,
        parent_manifest_path,
        submit_max_cost=submit_max_cost,
    )
    workflow.restore_latest_manifest_path = parent_manifest_path
    return workflow


def describe_retry_followup_button(
    retry_manifest_path: str,
    retry_manifest: dict[str, object],
    parent_manifest_path: str,
) -> tuple[str, str]:
    workflow = create_retry_followup_workflow(
        retry_manifest_path,
        retry_manifest,
        parent_manifest_path,
    )
    step = workflow.current_step()
    if step is None:
        return "继续补译", "补译流程暂无后续步骤。"
    label = _RETRY_STEP_BUTTON_LABELS.get(step.key, "继续补译")
    return label, step.message


def retry_followup_workflow_ready(
    summary: WritebackSummary,
    *,
    parent_manifest: dict[str, object] | None,
    retry_manifest: dict[str, object] | None,
    retry_manifest_path: str,
    parent_manifest_path: str,
    confirmed_parent_paths: set[str] | frozenset[str],
    supports_translation_writeback: bool,
) -> bool:
    if not supports_translation_writeback:
        return False
    if summary.status != "warn" or not parent_manifest_path:
        return False
    if parent_manifest is None or retry_manifest is None:
        return False
    if manifest_check_safety_level(parent_manifest) != "warn":
        return False
    if not retry_manifest_path:
        return False
    if parent_manifest_path not in confirmed_parent_paths:
        return False
    workflow = create_retry_followup_workflow(
        retry_manifest_path,
        retry_manifest,
        parent_manifest_path,
    )
    return workflow.current_step() is not None


def retry_followup_next_step(
    retry_manifest_path: str,
    retry_manifest: dict[str, object],
    parent_manifest_path: str,
) -> WorkflowStep | None:
    workflow = create_retry_followup_workflow(
        retry_manifest_path,
        retry_manifest,
        parent_manifest_path,
    )
    return workflow.current_step()