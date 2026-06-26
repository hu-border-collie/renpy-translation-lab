"""Sync translation workflow state for the GUI."""
from __future__ import annotations

from .sync_translation_report import summarize_sync_translation_output
from .translation_workflow import WorkflowStep, WorkflowUpdate


class SyncTranslationWorkflow:
    def __init__(self, pending_steps: list[str] | None = None):
        self.manifest_path = ""
        self._pending_steps = list(pending_steps or ["run"])

    @classmethod
    def start_new(cls) -> "SyncTranslationWorkflow":
        return cls()

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        return WorkflowStep(
            key="run",
            args=[],
            heading="正在同步翻译",
            message="正在扫描待翻译文本并调用同步 API 写回项目文件。",
            script_basename="gemini_translate.py",
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._pending_steps:
            return WorkflowUpdate(
                status="failed",
                heading="同步翻译状态异常",
                message="没有正在等待完成的步骤。",
                facts=[],
            )

        self._pending_steps.clear()
        return summarize_sync_translation_output(output, exit_code)