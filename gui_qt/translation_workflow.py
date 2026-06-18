"""Batch translation workflow state for the GUI.

This module only plans and interprets CLI steps. The GUI still executes the
existing command-line tool through ``CliRunner``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath


TERMINAL_FAILURE_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@dataclass(frozen=True)
class WorkflowStep:
    key: str
    args: list[str]
    heading: str
    message: str


@dataclass(frozen=True)
class WorkflowUpdate:
    status: str
    heading: str
    message: str
    facts: list[str]
    should_continue: bool = False


STEP_TEXT = {
    "build": ("正在准备翻译内容", "正在扫描待翻译文本并生成 Batch 请求包。"),
    "submit": ("正在提交翻译任务", "正在上传请求文件并创建 Batch 任务。"),
    "status": ("正在刷新任务状态", "正在向 Batch API 查询当前处理状态。"),
    "download": ("正在获取翻译结果", "任务已完成，正在下载结果文件。"),
    "check": ("正在检查翻译结果", "正在校验结果是否可以进入写回前预览。"),
}


def extract_created_package_path(output: str) -> str:
    return _extract_line_value(output, "Created batch package:")


def extract_manifest_path(output: str) -> str:
    return _extract_line_value(output, "Manifest:")


def extract_job_state(output: str) -> str:
    return _extract_line_value(output, "State:")


def extract_safety_status(output: str) -> str:
    return _extract_line_value(output, "Safety status:")


def manifest_path_for_package(package_path: str) -> str:
    if re.match(r"^[A-Za-z]:", package_path) or package_path.startswith("\\\\") or "\\" in package_path:
        return str(PureWindowsPath(package_path) / "manifest.json")
    return str(PurePosixPath(package_path) / "manifest.json")


def _extract_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


class TranslationWorkflow:
    def __init__(self, pending_steps: list[str], manifest_path: str = ""):
        self._pending_steps = list(pending_steps)
        self.manifest_path = manifest_path

    @classmethod
    def start_new(cls) -> "TranslationWorkflow":
        return cls(["build", "submit", "status"])

    @classmethod
    def resume_latest(cls, manifest_path: str) -> "TranslationWorkflow":
        return cls(["status"], manifest_path=manifest_path)

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
                heading="翻译流程状态异常",
                message="没有正在等待完成的步骤。",
                facts=[],
            )

        key = self._pending_steps.pop(0)
        if exit_code != 0:
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="翻译流程中断",
                message=f"{STEP_TEXT[key][0]}没有正常完成，请查看下方原始输出。",
                facts=self._facts(),
            )

        if key == "build":
            return self._finish_build(output)
        if key == "submit":
            manifest_path = extract_manifest_path(output)
            if manifest_path:
                self.manifest_path = manifest_path
        if key == "status":
            status_update = self._finish_status(output)
            if status_update is not None:
                return status_update
        if key == "check":
            return self._finish_check(output)

        return self._continue_or_finish()

    def _finish_build(self, output: str) -> WorkflowUpdate:
        package_path = extract_created_package_path(output)
        if not package_path:
            if "No pending lines to translate." in output:
                self._pending_steps.clear()
                return WorkflowUpdate(
                    status="done",
                    heading="没有待翻译内容",
                    message="当前项目没有需要提交到 Batch 的待翻译行。",
                    facts=[],
                )
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="无法定位 Batch 请求包",
                message="build 已结束，但输出中没有请求包路径；请查看原始输出。",
                facts=[],
            )

        self.manifest_path = manifest_path_for_package(package_path)
        return self._continue_or_finish()

    def _finish_status(self, output: str) -> WorkflowUpdate | None:
        state = extract_job_state(output)
        if state == "JOB_STATE_SUCCEEDED":
            if "download" not in self._pending_steps:
                self._pending_steps[:0] = ["download", "check"]
            return self._continue_or_finish(
                extra_facts=[f"Batch 状态：{state}"],
            )
        if state in TERMINAL_FAILURE_STATES:
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="Batch 任务没有成功完成",
                message=f"当前状态为 {state}，请查看原始输出后重试或重新生成任务。",
                facts=self._facts([f"Batch 状态：{state}"]),
            )

        self._pending_steps.clear()
        state_text = state or "未知"
        return WorkflowUpdate(
            status="waiting",
            heading="Batch 任务仍在处理",
            message="稍后可以继续刷新最新任务状态；任务成功后再下载并检查结果。",
            facts=self._facts([f"Batch 状态：{state_text}"]),
        )

    def _finish_check(self, output: str) -> WorkflowUpdate:
        safety = extract_safety_status(output)
        if safety == "safe":
            heading = "翻译结果检查通过"
            message = "结果为 safe，可以进入后续写回前预览。"
        elif safety in {"warn", "block"}:
            heading = "翻译结果需要处理"
            message = f"结果为 {safety}，普通流程不应写回；请先查看问题并重试或修复。"
        else:
            heading = "翻译结果检查完成"
            message = "未能识别安全状态，请查看原始输出。"

        self._pending_steps.clear()
        facts = self._facts([f"安全状态：{safety or '未知'}"])
        return WorkflowUpdate(status="done", heading=heading, message=message, facts=facts)

    def _continue_or_finish(self, extra_facts: list[str] | None = None) -> WorkflowUpdate:
        next_step = self.current_step()
        if next_step is None:
            return WorkflowUpdate(
                status="done",
                heading="翻译流程完成",
                message="当前翻译任务流程已完成。",
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
        if key == "build" or not self.manifest_path:
            return [key]
        return [key, self.manifest_path]

    def _facts(self, extra_facts: list[str] | None = None) -> list[str]:
        facts: list[str] = []
        if self.manifest_path:
            facts.append(f"Manifest：{self.manifest_path}")
        if extra_facts:
            facts.extend(extra_facts)
        return facts
