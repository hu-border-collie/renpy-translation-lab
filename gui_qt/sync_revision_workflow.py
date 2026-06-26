"""Sync revision workflow state for the GUI."""
from __future__ import annotations

from .revision_report import summarize_sync_revision_output
from .translation_workflow import WorkflowStep, WorkflowUpdate


class SyncRevisionWorkflow:
    def __init__(self, pending_steps: list[str] | None = None):
        self.manifest_path = ""
        self._pending_steps = ["sync-revisions"] if pending_steps is None else list(pending_steps)

    @classmethod
    def start_new(cls) -> "SyncRevisionWorkflow":
        return cls()

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        return WorkflowStep(
            key="sync-revisions",
            args=["sync-revisions"],
            heading="正在同步订正",
            message="正在扫描已有译文并调用同步 API 生成订正预览报告。",
            script_basename="gemini_translate_batch.py",
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._pending_steps:
            return WorkflowUpdate(
                status="failed",
                heading="同步订正状态异常",
                message="没有正在等待完成的步骤。",
                facts=[],
            )

        self._pending_steps.clear()
        return summarize_sync_revision_output(output, exit_code)