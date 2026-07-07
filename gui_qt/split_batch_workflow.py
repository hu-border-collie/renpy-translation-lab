"""GUI workflows for split batch package operations."""
from __future__ import annotations

from .batch_workflow_support import build_submit_cli_args
from .split_batch import SplitManifestEntry
from .translation_workflow import WorkflowStep, WorkflowUpdate
from .user_copy import format_manifest_path_fact


class SplitBatchQueueWorkflow:
    def __init__(
        self,
        *,
        action: str,
        manifest_paths: list[str],
        anchor_manifest_path: str,
        total_count: int,
        submit_max_cost: float | None = None,
    ):
        self.action = action
        self.manifest_path = anchor_manifest_path
        self.restore_latest_manifest_path = anchor_manifest_path
        self.submit_max_cost = submit_max_cost
        self._pending_paths = list(manifest_paths)
        self._current_path = ""
        self._done_count = 0
        self._total_count = total_count

    @classmethod
    def submit_remaining(
        cls,
        entries: list[SplitManifestEntry],
        *,
        anchor_manifest_path: str,
        submit_max_cost: float | None = None,
    ) -> "SplitBatchQueueWorkflow":
        return cls(
            action="submit",
            manifest_paths=[entry.manifest_path for entry in entries if entry.needs_submit],
            anchor_manifest_path=anchor_manifest_path,
            total_count=len([entry for entry in entries if entry.needs_submit]),
            submit_max_cost=submit_max_cost,
        )

    @classmethod
    def refresh_status(
        cls,
        entries: list[SplitManifestEntry],
        *,
        anchor_manifest_path: str,
    ) -> "SplitBatchQueueWorkflow":
        return cls(
            action="status",
            manifest_paths=[
                entry.manifest_path
                for entry in entries
                if entry.job_name and not entry.applied
            ],
            anchor_manifest_path=anchor_manifest_path,
            total_count=len([
                entry
                for entry in entries
                if entry.job_name and not entry.applied
            ]),
        )

    def current_step(self) -> WorkflowStep | None:
        if not self._current_path:
            if not self._pending_paths:
                return None
            self._current_path = self._pending_paths.pop(0)

        heading, message = self._step_text()
        if self.action == "submit":
            args = build_submit_cli_args(self._current_path, self.submit_max_cost)
        else:
            args = [self.action, self._current_path]
        return WorkflowStep(
            key=self.action,
            args=args,
            heading=heading,
            message=message,
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        del output
        current_path = self._current_path
        self._current_path = ""
        if exit_code != 0:
            self._pending_paths.clear()
            return WorkflowUpdate(
                status="failed",
                heading=self._failed_heading(),
                message=self._failed_message(),
                facts=self._facts(current_path),
            )

        self._done_count += 1
        next_step = self.current_step()
        if next_step is not None:
            return WorkflowUpdate(
                status="running",
                heading=next_step.heading,
                message=next_step.message,
                facts=self._facts(current_path),
                should_continue=True,
            )

        return WorkflowUpdate(
            status=self._done_status(),
            heading=self._done_heading(),
            message=self._done_message(),
            facts=self._facts(current_path),
            timeline_step_key=self._done_timeline_step_key(),
        )

    def _step_text(self) -> tuple[str, str]:
        progress = self._progress_text()
        if self.action == "submit":
            return (
                "正在提交拆分包",
                f"正在按顺序提交剩余拆分包。{progress}",
            )
        return (
            "正在刷新拆分包状态",
            f"正在按顺序查询已提交拆分包的云端状态。{progress}",
        )

    def _done_heading(self) -> str:
        if self.action == "submit":
            return "剩余拆分包已提交"
        return "拆分包状态已刷新"

    def _done_message(self) -> str:
        if self.action == "submit":
            return "已提交所有尚未提交的拆分包；请点击「查询云端状态」刷新全部云端状态。"
        return "已刷新全部已提交拆分包的状态；云端已完成的包可以在列表中选择。"

    def _done_status(self) -> str:
        if self.action == "submit":
            return "waiting"
        return "ready"

    def _done_timeline_step_key(self) -> str:
        if self.action == "submit":
            return "status"
        return self.action

    def _failed_heading(self) -> str:
        if self.action == "submit":
            return "提交拆分包中断"
        return "刷新拆分包状态中断"

    def _failed_message(self) -> str:
        if self.action == "submit":
            return "某个拆分包提交失败；已停止继续提交，请查看原始输出。"
        return "某个拆分包状态查询失败；已停止继续查询，请查看原始输出。"

    def _progress_text(self) -> str:
        total = max(self._total_count, 1)
        current = min(self._done_count + 1, total)
        return f"进度：{current}/{total}"

    def _facts(self, current_path: str = "") -> list[str]:
        facts = [
            f"拆分包操作进度：{self._done_count}/{self._total_count}",
            format_manifest_path_fact(self.manifest_path),
        ]
        if current_path:
            facts.append(f"刚处理：{current_path}")
        return facts

