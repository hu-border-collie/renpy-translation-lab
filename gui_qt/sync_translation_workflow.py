"""Sync translation workflow state for the GUI."""
from __future__ import annotations

import re

from .sync_translation_report import summarize_sync_translation_output
from .translation_workflow import WorkflowStep, WorkflowUpdate


class SyncTranslationWorkflow:
    def __init__(self, pending_steps: list[str] | None = None):
        self.manifest_path = ""
        self._pending_steps = ["preview"] if pending_steps is None else list(pending_steps)

    @classmethod
    def start_new(cls) -> "SyncTranslationWorkflow":
        return cls()

    @classmethod
    def apply_existing(cls, manifest_path: str) -> "SyncTranslationWorkflow":
        workflow = cls(["apply"])
        workflow.manifest_path = manifest_path
        return workflow

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        step = self._pending_steps[0]
        if step == "apply":
            return WorkflowStep(
                key="apply",
                args=["--apply", self.manifest_path],
                heading="正在写回同步翻译预览",
                message="正在重新校验项目与源文件，然后原子写回已确认的预览。",
                script_basename="gemini_translate.py",
            )
        return WorkflowStep(
            key="preview",
            args=[],
            heading="正在生成同步翻译预览",
            message="正在扫描并翻译待处理文本；项目脚本不会在此步骤中修改。",
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

        step = self._pending_steps.pop(0)
        if step == "preview" and exit_code == 0:
            match = re.search(r"^Sync preview manifest:\s*(.+?)\s*$", output, re.MULTILINE)
            if match:
                self.manifest_path = match.group(1).strip()
        return summarize_sync_translation_output(output, exit_code, operation=step)
