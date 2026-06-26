"""Sync keyword extraction workflow state for the GUI."""
from __future__ import annotations

from .keyword_report import summarize_sync_keyword_output
from .translation_workflow import WorkflowStep, WorkflowUpdate


class SyncKeywordWorkflow:
    def __init__(self, pending_steps: list[str] | None = None):
        self.manifest_path = ""
        self._pending_steps = ["sync-keywords"] if pending_steps is None else list(pending_steps)

    @classmethod
    def start_new(cls) -> "SyncKeywordWorkflow":
        return cls()

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        return WorkflowStep(
            key="sync-keywords",
            args=["sync-keywords"],
            heading="正在同步提取关键词",
            message="正在扫描翻译文本并生成术语与剧情报告。",
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