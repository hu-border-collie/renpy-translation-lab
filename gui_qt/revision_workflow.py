"""Batch revision workflow state for the GUI."""
from __future__ import annotations

import re

from .revision_report import summarize_revision_preview_output
from .batch_workflow_support import build_submit_cli_args, output_blocked_by_max_cost
from .translation_workflow import (
    TERMINAL_FAILURE_STATES,
    WorkflowStep,
    WorkflowUpdate,
    extract_job_state,
    extract_manifest_path,
    manifest_path_for_package,
)
from .user_copy import format_job_state_fact, format_manifest_path_fact, job_state_label


def extract_created_revision_package_path(output: str) -> str:
    return _extract_line_value(output, "Created revision package:")


def _extract_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


STEP_TEXT = {
    "build-revisions": ("正在准备订正内容", "正在扫描已有译文并准备待提交内容。"),
    "submit": ("正在提交订正任务", "正在上传请求文件并创建云端批量任务。"),
    "status": ("正在刷新订正任务状态", "正在查询云端任务处理状态。"),
    "download": ("正在获取订正结果", "任务已完成，正在下载结果文件。"),
    "preview-revisions": ("正在预览订正结果", "正在校验结果并生成订正预览报告。"),
}


class RevisionBatchWorkflow:
    def __init__(
        self,
        pending_steps: list[str],
        manifest_path: str = "",
        *,
        submit_max_cost: float | None = None,
    ):
        self._pending_steps = list(pending_steps)
        self.manifest_path = manifest_path
        self.submit_max_cost = submit_max_cost

    @classmethod
    def start_new(cls, *, submit_max_cost: float | None = None) -> "RevisionBatchWorkflow":
        return cls(["build-revisions", "submit", "status"], submit_max_cost=submit_max_cost)

    @classmethod
    def resume_latest(
        cls,
        manifest_path: str,
        *,
        submit_max_cost: float | None = None,
    ) -> "RevisionBatchWorkflow":
        return cls(["status"], manifest_path=manifest_path, submit_max_cost=submit_max_cost)

    @classmethod
    def resume_manifest(
        cls,
        manifest_path: str,
        manifest: dict[str, object],
        *,
        submit_max_cost: float | None = None,
    ) -> "RevisionBatchWorkflow":
        if manifest.get("last_revision_preview"):
            return cls([], manifest_path=manifest_path, submit_max_cost=submit_max_cost)
        if manifest.get("job_state") == "JOB_STATE_SUCCEEDED":
            return cls(
                ["download", "preview-revisions"],
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
            )
        if not manifest.get("job_name"):
            return cls(
                ["submit", "status"],
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
            )
        return cls.resume_latest(manifest_path, submit_max_cost=submit_max_cost)

    def current_step(self) -> WorkflowStep | None:
        if not self._pending_steps:
            return None
        key = self._pending_steps[0]
        heading, message = STEP_TEXT[key]
        return WorkflowStep(
            key=key,
            args=self._args_for_step(key),
            heading=heading,
            message=message,
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        if not self._pending_steps:
            return WorkflowUpdate(
                status="failed",
                heading="订正流程状态异常",
                message="没有正在等待完成的步骤。",
                facts=[],
            )

        key = self._pending_steps.pop(0)
        if exit_code != 0:
            self._pending_steps.clear()
            message = f"{STEP_TEXT[key][0]}没有正常完成，请查看下方原始输出。"
            if key == "submit" and output_blocked_by_max_cost(output):
                message = (
                    "提交被成本上限拦截。请在高级设置中提高「提交成本上限」，"
                    "或先拆分任务包以降低单次提交成本。"
                )
            return WorkflowUpdate(
                status="failed",
                heading="订正流程中断",
                message=message,
                facts=self._facts(),
            )

        if key == "build-revisions":
            return self._finish_build(output)
        if key == "submit":
            manifest_path = extract_manifest_path(output)
            if manifest_path:
                self.manifest_path = manifest_path
        if key == "status":
            status_update = self._finish_status(output)
            if status_update is not None:
                if getattr(self, "only_query", False) and status_update.status == "running":
                    return WorkflowUpdate(
                        status="ready",
                        heading="云端任务已完成",
                        message="查询结果：云端批量任务已成功完成！请点击「继续订正」下载并预览订正内容。",
                        facts=status_update.facts,
                        should_continue=False,
                    )
                return status_update
        if key == "preview-revisions":
            return self._finish_preview(output)

        return self._continue_or_finish()

    def _finish_build(self, output: str) -> WorkflowUpdate:
        package_path = extract_created_revision_package_path(output)
        if not package_path:
            if (
                "No revision source lines found." in output
                or "No revision chunks built." in output
            ):
                self._pending_steps.clear()
                return WorkflowUpdate(
                    status="done",
                    heading="没有可订正的源行",
                    message="当前项目没有可用于订正的已有译文行。",
                    facts=[],
                )
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="无法完成订正任务准备",
                message="订正任务准备未完成，请查看诊断日志。",
                facts=[],
            )

        self.manifest_path = manifest_path_for_package(package_path)
        return self._continue_or_finish()

    def _finish_status(self, output: str) -> WorkflowUpdate | None:
        state = extract_job_state(output)
        if state == "JOB_STATE_SUCCEEDED":
            if "download" not in self._pending_steps:
                self._pending_steps[:0] = ["download", "preview-revisions"]
            return self._continue_or_finish(
                extra_facts=[format_job_state_fact(state)],
            )
        if state in TERMINAL_FAILURE_STATES:
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="订正批量任务没有成功完成",
                message=f"当前状态为 {job_state_label(state)}，请查看原始输出后重试或重新生成任务。",
                facts=self._facts([format_job_state_fact(state)]),
            )

        self._pending_steps.clear()
        state_text = state or "未知"
        return WorkflowUpdate(
            status="waiting",
            heading="订正批量任务仍在处理",
            message="稍后可以继续刷新最新任务状态；任务成功后再下载并预览结果。",
            facts=self._facts([format_job_state_fact(state_text)]),
        )

    def _finish_preview(self, output: str) -> WorkflowUpdate:
        update = summarize_revision_preview_output(output, 0)
        self._pending_steps.clear()
        facts = list(update.facts)
        if self.manifest_path and not any("manifest" in fact.lower() for fact in facts):
            facts.insert(0, format_manifest_path_fact(self.manifest_path))
        return WorkflowUpdate(
            status=update.status,
            heading=update.heading,
            message=update.message,
            facts=facts,
        )

    def _continue_or_finish(self, extra_facts: list[str] | None = None) -> WorkflowUpdate:
        next_step = self.current_step()
        if next_step is None:
            return WorkflowUpdate(
                status="done",
                heading="订正流程完成",
                message="当前订正任务流程已完成。",
                facts=self._facts(extra_facts),
            )
        return WorkflowUpdate(
            status="running",
            heading=next_step.heading,
            message=next_step.message,
            facts=self._facts(extra_facts),
            should_continue=True,
        )

    def _args_for_step(self, key: str) -> list[str]:
        if key == "build-revisions" or not self.manifest_path:
            return [key]
        if key == "submit":
            return build_submit_cli_args(self.manifest_path, self.submit_max_cost)
        return [key, self.manifest_path]

    def _facts(self, extra_facts: list[str] | None = None) -> list[str]:
        facts: list[str] = []
        if self.manifest_path:
            facts.append(format_manifest_path_fact(self.manifest_path))
        if extra_facts:
            facts.extend(extra_facts)
        return facts