"""Sync keyword extraction workflow state for the GUI."""
from __future__ import annotations

from gui_qt.reports.keyword_report import summarize_sync_keyword_output
from gui_qt.user_copy import SYNC_KEYWORD_SCAN_MESSAGE
from gui_qt.workflows.translation_workflow import WorkflowStep, WorkflowUpdate


class SyncKeywordWorkflow:
    def __init__(self, pending_steps: list[str] | None = None, *, manifest_path="", retry_unknown=False):
        self.manifest_path = manifest_path
        self.retry_unknown = retry_unknown
        self._pending_steps = ["sync-keywords"] if pending_steps is None else list(pending_steps)

    @classmethod
    def start_new(cls) -> "SyncKeywordWorkflow":
        return cls()

    @classmethod
    def resume_manifest(cls, manifest_path: str, *, retry_unknown=False) -> "SyncKeywordWorkflow":
        """Recover the same frozen keyword package through the CLI."""
        return cls(manifest_path=manifest_path, retry_unknown=retry_unknown)

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        args = ["sync-keywords"]
        if self.manifest_path:
            args.extend(["--resume", self.manifest_path])
            if self.retry_unknown:
                args.append("--retry-unknown")
        return WorkflowStep(
            key="sync-keywords",
            args=args,
            heading="正在同步提取关键词",
            message=SYNC_KEYWORD_SCAN_MESSAGE,
            script_basename="gemini_translate_batch.py",
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._pending_steps:
            return WorkflowUpdate(
                status="failed",
                heading="同步关键词提取状态异常",
                message="没有正在等待完成的步骤。",
                facts=[],
            )

        self._pending_steps.clear()
        return summarize_sync_keyword_output(output, exit_code)
