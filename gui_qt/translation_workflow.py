"""Batch translation workflow state for the GUI.

This module only plans and interprets CLI steps. The GUI still executes the
existing command-line tool through ``CliRunner``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .batch_workflow_support import (
    build_recover_submit_cli_args,
    build_submit_cli_args,
    load_cost_estimate_facts_from_manifest,
    load_non_chinese_rules_facts_from_manifest,
    load_target_language_facts_from_manifest,
    output_blocked_by_max_cost,
    output_blocked_by_uncertain_submit,
    plan_unsubmitted_workflow_steps,
    uncertain_submit_failure_message,
)
from .user_copy import (
    format_job_state_fact,
    format_manifest_path_fact,
    format_safety_fact,
    job_state_label,
    safety_level_label,
)
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
    script_basename: str = "gemini_translate_batch.py"


@dataclass(frozen=True)
class WorkflowUpdate:
    status: str
    heading: str
    message: str
    facts: list[str]
    should_continue: bool = False
    timeline_step_key: str | None = None


STEP_TEXT = {
    "build": ("正在准备翻译内容", "正在扫描待翻译文本并准备待提交内容。"),
    "submit": ("正在提交翻译任务", "正在上传请求文件并创建云端批量任务。"),
    "recover-submit": ("正在恢复提交状态", "正在从提交日志恢复远端批量任务信息。"),
    "status": ("正在刷新任务状态", "正在查询云端任务处理状态。"),
    "download": ("正在获取翻译结果", "任务已完成，正在下载结果文件。"),
    "check": ("正在检查翻译结果", "正在校验结果是否可以进入写回前预览。"),
    "merge-retry": ("正在合并补译结果", "正在把补译结果合并回父翻译包。"),
    "check-parent": ("正在重新检查父任务", "补译结果已合并，正在重新校验父翻译包。"),
}


def extract_created_package_path(output: str) -> str:
    return _extract_line_value(output, "Created batch package:")


def extract_manifest_path(output: str) -> str:
    return _extract_line_value(output, "Manifest:")


def extract_job_state(output: str) -> str:
    return _extract_line_value(output, "State:")


def extract_safety_status(output: str) -> str:
    return _extract_line_value(output, "Safety status:")


def output_has_quota_error(output: str) -> bool:
    return any(
        marker in output
        for marker in ("RESOURCE_EXHAUSTED", "429", "Quota/resource limit hit")
    )


def extract_suggested_split_command(output: str) -> str:
    return _extract_line_value(output, "Suggested split command:")


def manifest_path_for_package(package_path: str) -> str:
    if re.match(r"^[A-Za-z]:", package_path) or package_path.startswith("\\\\") or "\\" in package_path:
        return str(PureWindowsPath(package_path) / "manifest.json")
    return str(PurePosixPath(package_path) / "manifest.json")


def _extract_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


class TranslationWorkflow:
    def __init__(
        self,
        pending_steps: list[str],
        manifest_path: str = "",
        *,
        submit_max_cost: float | None = None,
        retry_parent_manifest_path: str = "",
    ):
        self._pending_steps = list(pending_steps)
        self.manifest_path = manifest_path
        self.submit_max_cost = submit_max_cost
        self.retry_parent_manifest_path = retry_parent_manifest_path

    @classmethod
    def start_new(cls, *, submit_max_cost: float | None = None) -> "TranslationWorkflow":
        return cls(["build", "submit", "status"], submit_max_cost=submit_max_cost)

    @classmethod
    def resume_latest(
        cls,
        manifest_path: str,
        *,
        submit_max_cost: float | None = None,
    ) -> "TranslationWorkflow":
        return cls(["status"], manifest_path=manifest_path, submit_max_cost=submit_max_cost)

    @classmethod
    def resume_manifest(
        cls,
        manifest_path: str,
        manifest: dict[str, object],
        *,
        submit_max_cost: float | None = None,
    ) -> "TranslationWorkflow":
        retry_parent = _stripped(manifest.get("retry_of_manifest"))
        if retry_parent:
            return cls.resume_retry_manifest(
                manifest_path,
                manifest,
                retry_parent,
                submit_max_cost=submit_max_cost,
            )
        if manifest.get("last_check") or manifest.get("last_check_summary"):
            return cls([], manifest_path=manifest_path, submit_max_cost=submit_max_cost)
        if manifest.get("job_state") == "JOB_STATE_SUCCEEDED":
            return cls(
                ["download", "check"],
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
            )
        if not manifest.get("job_name"):
            return cls(
                plan_unsubmitted_workflow_steps(manifest_path),
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
            )
        return cls.resume_latest(manifest_path, submit_max_cost=submit_max_cost)

    @classmethod
    def resume_retry_manifest(
        cls,
        manifest_path: str,
        manifest: dict[str, object],
        retry_parent_manifest_path: str,
        *,
        submit_max_cost: float | None = None,
    ) -> "TranslationWorkflow":
        check_summary = manifest.get("last_check_summary")
        safety = ""
        if isinstance(check_summary, dict):
            safety = _stripped(check_summary.get("safety_level"))
        if safety == "safe":
            return cls(
                ["merge-retry", "check-parent"],
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
                retry_parent_manifest_path=retry_parent_manifest_path,
            )
        if check_summary:
            return cls(
                [],
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
                retry_parent_manifest_path=retry_parent_manifest_path,
            )
        if manifest.get("job_state") == "JOB_STATE_SUCCEEDED":
            return cls(
                ["download", "check"],
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
                retry_parent_manifest_path=retry_parent_manifest_path,
            )
        if not manifest.get("job_name"):
            return cls(
                plan_unsubmitted_workflow_steps(manifest_path),
                manifest_path=manifest_path,
                submit_max_cost=submit_max_cost,
                retry_parent_manifest_path=retry_parent_manifest_path,
            )
        return cls(
            ["status"],
            manifest_path=manifest_path,
            submit_max_cost=submit_max_cost,
            retry_parent_manifest_path=retry_parent_manifest_path,
        )

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
            return self._finish_failed_step(key, output)

        if key == "build":
            return self._finish_build(output)
        if key in {"submit", "recover-submit"}:
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
                        message="查询结果：云端批量任务已成功完成！请点击「继续翻译」下载并校验结果。",
                        facts=status_update.facts,
                        should_continue=False,
                    )
                return status_update
        if key == "check":
            return self._finish_check(output)
        if key == "merge-retry":
            if self.retry_parent_manifest_path:
                self.manifest_path = self.retry_parent_manifest_path
            return self._continue_or_finish()
        if key == "check-parent":
            if self.retry_parent_manifest_path:
                self.manifest_path = self.retry_parent_manifest_path
            self.retry_parent_manifest_path = ""
            return self._finish_check(output)

        return self._continue_or_finish()

    def _finish_failed_step(self, key: str, output: str) -> WorkflowUpdate:
        message = f"{STEP_TEXT[key][0]}没有正常完成，请查看下方原始输出。"
        extra_facts: list[str] = []
        if key in {"submit", "recover-submit"} and output_blocked_by_max_cost(output):
            message = (
                "提交被成本上限拦截。请在高级设置中提高「提交成本上限」，"
                "或先拆分任务包以降低单次提交成本。"
            )
            extra_facts.append("建议：检查高级设置中的提交成本上限，或先运行 split 拆包。")
        elif key in {"submit", "recover-submit"} and output_blocked_by_uncertain_submit(output):
            message = uncertain_submit_failure_message(output)
            extra_facts.append("建议：在诊断与工具运行 recover-submit，或按提示使用 --resume / --force。")
        elif key == "submit" and output_has_quota_error(output):
            message = (
                "提交云端任务时命中配额或资源限制。当前批量包可能过大，"
                "建议先运行拆包，把当前任务拆成小包后再继续提交。"
            )
            extra_facts.append("建议：先运行 split 拆包，再继续最新任务。")
            command = extract_suggested_split_command(output)
            if command:
                extra_facts.append(f"拆包命令：{command}")

        return WorkflowUpdate(
            status="failed",
            heading="翻译流程中断",
            message=message,
            facts=self._facts(extra_facts),
        )

    def _finish_build(self, output: str) -> WorkflowUpdate:
        package_path = extract_created_package_path(output)
        if not package_path:
            if "No pending lines to translate." in output:
                self._pending_steps.clear()
                return WorkflowUpdate(
                    status="done",
                    heading="没有待翻译内容",
                    message="当前项目没有需要翻译的待译文本。",
                    facts=[],
                )
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="无法完成翻译任务准备",
                message="翻译任务准备未完成，请查看诊断日志。",
                facts=[],
            )

        self.manifest_path = manifest_path_for_package(package_path)
        extra_facts = load_target_language_facts_from_manifest(self.manifest_path)
        extra_facts.extend(load_non_chinese_rules_facts_from_manifest(self.manifest_path))
        extra_facts.extend(load_cost_estimate_facts_from_manifest(self.manifest_path))
        return self._continue_or_finish(extra_facts=extra_facts)

    def _finish_status(self, output: str) -> WorkflowUpdate | None:
        state = extract_job_state(output)
        if state == "JOB_STATE_SUCCEEDED":
            if "download" not in self._pending_steps:
                self._pending_steps[:0] = ["download", "check"]
            return self._continue_or_finish(
                extra_facts=[format_job_state_fact(state)],
            )
        if state in TERMINAL_FAILURE_STATES:
            self._pending_steps.clear()
            return WorkflowUpdate(
                status="failed",
                heading="批量任务没有成功完成",
                message=f"当前状态为 {job_state_label(state)}，请查看原始输出后重试或重新生成任务。",
                facts=self._facts([format_job_state_fact(state)]),
            )

        self._pending_steps.clear()
        state_text = state or "未知"
        return WorkflowUpdate(
            status="waiting",
            heading="批量任务仍在处理",
            message="稍后可以继续刷新最新任务状态；任务成功后再下载并检查结果。",
            facts=self._facts([format_job_state_fact(state_text)]),
        )

    def _finish_check(self, output: str) -> WorkflowUpdate:
        safety = extract_safety_status(output)
        if safety == "safe":
            if self.retry_parent_manifest_path:
                self._pending_steps[:] = ["merge-retry", "check-parent"]
                return self._continue_or_finish(
                    extra_facts=[format_safety_fact(safety, prefix="补译检查结果")],
                )
            heading = "翻译结果检查通过"
            message = "检查结果为可写回，可以进入写回前确认。"
        elif safety in {"warn", "block"}:
            heading = "补译结果仍需处理" if self.retry_parent_manifest_path else "翻译结果需要处理"
            message = (
                f"检查结果为 {safety_level_label(safety)}，普通流程不应写回；"
                "请先查看问题并重试或修复。"
            )
            if self.retry_parent_manifest_path:
                message = (
                    f"补译包检查结果为 {safety_level_label(safety)}，暂不能合并回父任务；"
                    "请先查看问题清单，必要时继续补译或人工处理。"
                )
        else:
            heading = "翻译结果检查完成"
            message = "未能识别安全状态，请查看原始输出。"

        self._pending_steps.clear()
        facts = self._facts([format_safety_fact(safety or "", prefix="检查结果")])
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
        if key == "merge-retry":
            return ["merge-retry", self.retry_parent_manifest_path, self.manifest_path]
        if key == "check-parent":
            return ["check", self.retry_parent_manifest_path]
        if key == "build" or not self.manifest_path:
            return [key]
        if key == "submit":
            return build_submit_cli_args(self.manifest_path, self.submit_max_cost)
        if key == "recover-submit":
            return build_recover_submit_cli_args(self.manifest_path)
        return [key, self.manifest_path]

    def _facts(self, extra_facts: list[str] | None = None) -> list[str]:
        facts: list[str] = []
        if self.manifest_path:
            facts.append(format_manifest_path_fact(self.manifest_path))
        if extra_facts:
            facts.extend(extra_facts)
        return facts


def _stripped(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
