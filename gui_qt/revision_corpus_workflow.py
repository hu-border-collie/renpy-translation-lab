"""GUI workflow wrapper for the read-only revision-corpus export."""
from __future__ import annotations

from .batch_workflow_support import machine_output_args
from .revision_corpus_report import (
    RevisionCorpusExportResult,
    summarize_revision_corpus_output,
)
from .translation_workflow import WorkflowStep, WorkflowUpdate


class RevisionCorpusExportWorkflow:
    """Run one machine-contract corpus export through the shared CLI runner."""

    def __init__(self, *, operation_identity: str = "") -> None:
        self.operation_identity = str(operation_identity or "")
        self.manifest_path = ""
        self.result: RevisionCorpusExportResult | None = None
        self._pending = True

    def current_step(self) -> WorkflowStep | None:
        if not self._pending:
            return None
        return WorkflowStep(
            key="export-revision-corpus",
            args=machine_output_args(
                ["export-revision-corpus", "--strict-exit-codes"]
            ),
            heading="正在导出润色语料",
            message=(
                "正在复用订正扫描导出 JSONL、Markdown 和 manifest；"
                "不会修改游戏脚本。"
            ),
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._pending:
            return WorkflowUpdate(
                status="failed",
                heading="润色语料导出状态异常",
                message="没有正在等待完成的导出步骤。",
                facts=[],
            )
        self._pending = False
        update, result = summarize_revision_corpus_output(output, exit_code)
        self.result = result
        return update

    def stale_update(self) -> WorkflowUpdate:
        """Return a non-success update when the project identity changed."""

        self._pending = False
        self.result = None
        self.manifest_path = ""
        return WorkflowUpdate(
            status="stale",
            heading="导出结果已过期",
            message="项目已切换，刚才的导出结果已丢弃；请在当前项目重新导出。",
            facts=[],
        )
