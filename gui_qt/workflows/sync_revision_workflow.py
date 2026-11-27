"""Sync revision workflow state for the GUI."""
from __future__ import annotations

from gui_qt.reports.revision_report import summarize_sync_revision_output
from gui_qt.user_copy import SYNC_REVISION_COPY
from gui_qt.workflows.translation_workflow import WorkflowStep, WorkflowUpdate


class SyncRevisionWorkflow:
    def __init__(self, pending_steps: list[str] | None = None, *,
                 manifest_path: str = "", retry_unknown: bool = False):
        self.manifest_path = manifest_path
        self._retry_unknown = retry_unknown
        self._pending_steps = ["sync-revisions"] if pending_steps is None else list(pending_steps)

    @classmethod
    def start_new(cls) -> "SyncRevisionWorkflow":
        return cls()

    @classmethod
    def resume_manifest(cls, manifest_path: str, *, retry_unknown: bool = False) -> "SyncRevisionWorkflow":
        return cls(manifest_path=manifest_path, retry_unknown=retry_unknown)

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        args = ["sync-revisions"]
        if self.manifest_path:
            args.extend(["--resume", self.manifest_path])
            if self._retry_unknown:
                args.append("--retry-unknown")
        return WorkflowStep(
            key="sync-revisions",
            args=args,
            heading=SYNC_REVISION_COPY['resume_heading'] if self.manifest_path else "正在同步订正",
            message=(SYNC_REVISION_COPY['resume_message'] if self.manifest_path
                     else "正在扫描已有译文并调用同步 API 生成订正预览报告。"),
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
