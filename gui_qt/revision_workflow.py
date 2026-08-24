"""Batch revision workflow state for the GUI."""
from __future__ import annotations

import re

import cli_contract

from .revision_report import summarize_revision_preview_output
from .batch_workflow_support import (
    build_recover_submit_cli_args,
    build_submit_cli_args,
    machine_output_args,
    output_blocked_by_max_cost,
    output_blocked_by_uncertain_submit,
    plan_unsubmitted_workflow_steps,
    uncertain_submit_failure_message,
)
from .translation_workflow import (
    TERMINAL_FAILURE_STATES,
    WorkflowStep,
    WorkflowUpdate,
    extract_job_state,
    extract_manifest_path,
    manifest_path_for_package,
)
from .user_copy import (
    REVISION_PROPOSAL_COPY,
    format_job_state_fact,
    format_manifest_path_fact,
    job_state_label,
)


def extract_created_revision_package_path(output: str) -> str:
    return _extract_line_value(output, "Created revision package:")


def _extract_line_value(output: str, prefix: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


STEP_TEXT = {
    "build-revisions": ("正在准备订正内容", "正在扫描已有译文并准备待提交内容。"),
    "submit": ("正在提交订正任务", "正在上传请求文件并创建云端批量任务。"),
    "recover-submit": ("正在恢复提交状态", "正在从提交日志恢复远端批量任务信息。"),
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
                plan_unsubmitted_workflow_steps(manifest_path),
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
            if key in {"submit", "recover-submit"} and output_blocked_by_max_cost(output):
                message = (
                    "提交被成本上限拦截。请在高级设置中提高「提交成本上限」，"
                    "或先拆分任务包以降低单次提交成本。"
                )
            elif key in {"submit", "recover-submit"} and output_blocked_by_uncertain_submit(output):
                message = uncertain_submit_failure_message(output)
            return WorkflowUpdate(
                status="failed",
                heading="订正流程中断",
                message=message,
                facts=self._facts(),
            )

        if key == "build-revisions":
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
        if key == "recover-submit":
            return build_recover_submit_cli_args(self.manifest_path)
        if key in {"status", "download"}:
            return machine_output_args([key, self.manifest_path])
        return [key, self.manifest_path]

    def _facts(self, extra_facts: list[str] | None = None) -> list[str]:
        facts: list[str] = []
        if self.manifest_path:
            facts.append(format_manifest_path_fact(self.manifest_path))
        if extra_facts:
            facts.extend(extra_facts)
        return facts


class RevisionProposalImportWorkflow:
    """Import proposals, optionally stopping at a shared staged-selection artifact."""

    def __init__(
        self,
        proposal_path: str,
        corpus_manifest_path: str = "",
        *,
        stage: bool = False,
        operation_identity: str = "",
    ):
        self.proposal_path = proposal_path
        self.corpus_manifest_path = corpus_manifest_path
        self.stage = bool(stage)
        self.operation_identity = str(operation_identity or "")
        self.manifest_path = ""
        self.stage_result: dict[str, object] | None = None
        self._pending = True

    def can_open_selection(self) -> bool:
        """Return whether a staged import is ready for the selection dialog."""

        result = self.stage_result
        return bool(
            self.stage
            and isinstance(result, dict)
            and str(result.get("session_status") or "") == "ready"
            and int(result.get("selectable_count") or 0) > 0
        )

    def current_step(self) -> WorkflowStep | None:
        if not self._pending:
            return None
        args = ["import-revision-proposals", self.proposal_path]
        if self.corpus_manifest_path:
            args.extend(["--corpus-manifest", self.corpus_manifest_path])
        if self.stage:
            args.append("--stage")
            if self.operation_identity:
                args.extend(["--operation-identity", self.operation_identity])
            args = [*args, "--strict-exit-codes", "--output", "json", "--non-interactive"]
        return WorkflowStep(
            key="import-revision-proposals",
            args=args,
            heading="正在导入润色提案",
            message=(
                "正在校验身份、快照和格式，并准备候选会话。"
                if self.stage
                else "正在校验身份、快照和格式，并生成安全订正预览。"
            ),
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        self._pending = False
        if self.stage:
            return self._complete_staged_import(exit_code, output)
        self.manifest_path = _extract_line_value(output, "Manifest:")
        status = _extract_line_value(output, "Revision proposal import status:")
        facts = self._facts()
        if exit_code != 0 or status in {"blocked", "stale", "partial"}:
            return WorkflowUpdate(
                status="failed",
                heading="润色提案未通过安全校验",
                message="没有修改任何 .rpy；请根据导入报告修正提案后重试。",
                facts=facts,
            )
        if status == "no_op":
            return WorkflowUpdate(
                status="done",
                heading="润色提案无需写回",
                message="所选提案没有产生译文变化。",
                facts=facts,
            )
        return WorkflowUpdate(
            status="done",
            heading="润色提案预览已生成",
            message="请检查订正预览；确认后可使用“写回订正”。",
            facts=facts,
        )

    def _complete_staged_import(self, exit_code: int, output: str) -> WorkflowUpdate:
        try:
            envelope = cli_contract.parse_result_envelope(output)
        except ValueError:
            return WorkflowUpdate(
                status="failed",
                heading="润色提案导入结果不可识别",
                message="导入没有返回可识别的机器结果；请查看诊断日志后重试。",
                facts=[],
            )
        result = envelope.get("result")
        result_map = result if isinstance(result, dict) else {}
        artifacts = envelope.get("artifacts")
        artifact_map = artifacts if isinstance(artifacts, dict) else {}
        self.stage_result = {
            **result_map,
            "status": str(envelope.get("status") or ""),
            "ok": envelope.get("ok") is True,
            "paths": dict(artifact_map),
            "candidates": list(result_map.get("candidates") or []),
        }
        if exit_code != 0 or envelope.get("ok") is not True:
            error = envelope.get("error")
            error_map = error if isinstance(error, dict) else {}
            code = str(error_map.get("code") or "UNKNOWN_ERROR")
            return WorkflowUpdate(
                status="failed",
                heading="润色提案导入失败",
                message="导入没有完成；请根据诊断报告修正提案后重试。",
                facts=[f"错误码：{code}"],
            )
        facts = [
            f"候选总数：{int(result_map.get('candidate_count') or 0)}",
            f"有效候选：{int(result_map.get('selectable_count') or 0)}",
            f"初始选择：{int(result_map.get('selected_count') or 0)}",
            f"未选择：{int(result_map.get('unselected_count') or 0)}",
            f"无需修改：{int(result_map.get('no_op_count') or 0)}",
            f"无效：{int(result_map.get('invalid_count') or 0)}",
            f"过期：{int(result_map.get('stale_count') or 0)}",
            f"冲突：{int(result_map.get('conflict_count') or 0)}",
        ]
        if artifact_map.get("staged_selection"):
            facts.append(f"候选会话：{artifact_map['staged_selection']}")
        session_status = str(result_map.get("session_status") or "")
        if session_status == "stale":
            return WorkflowUpdate(
                status="warning",
                heading="润色提案已导入，但会话已过期",
                message=REVISION_PROPOSAL_COPY["selection_stale"],
                facts=facts,
            )
        if int(result_map.get("selectable_count") or 0) <= 0:
            return WorkflowUpdate(
                status="done",
                heading="润色提案已导入，但没有有效候选",
                message="无效、过期或冲突候选不能进入订正预览。",
                facts=facts,
            )
        return WorkflowUpdate(
            status="done",
            heading="润色提案已导入",
            message="请筛选并明确勾选有效候选；确认后才会生成订正预览。",
            facts=facts,
        )

    def _facts(self) -> list[str]:
        if self.manifest_path:
            return [format_manifest_path_fact(self.manifest_path)]
        return []

    def stale_update(self) -> WorkflowUpdate:
        """Discard a late import result after the project identity changed."""
        self._pending = False
        self.stage_result = None
        self.manifest_path = ""
        return WorkflowUpdate(
            status="stale",
            heading="润色提案导入会话已过期",
            message="项目或提案文件已变化，迟到结果已丢弃；请重新导入当前项目。",
            facts=[],
        )


class RevisionProposalConfirmWorkflow:
    """Confirm a serialized staged selection and run the existing preview gate."""

    def __init__(
        self,
        staged_selection_path: str,
        selection_path: str,
        *,
        operation_identity: str = "",
    ):
        self.staged_selection_path = staged_selection_path
        self.selection_path = selection_path
        self.operation_identity = str(operation_identity or "")
        self.manifest_path = ""
        self.result: dict[str, object] | None = None
        self._pending = True

    def current_step(self) -> WorkflowStep | None:
        if not self._pending:
            return None
        return WorkflowStep(
            key="confirm-revision-proposals",
            args=[
                "confirm-revision-proposals",
                self.staged_selection_path,
                "--selection-file",
                self.selection_path,
                "--strict-exit-codes",
                "--output",
                "json",
                "--non-interactive",
            ],
            heading="正在确认润色候选",
            message=REVISION_PROPOSAL_COPY["selection_running"],
        )

    def complete_current_step(self, exit_code: int, output: str) -> WorkflowUpdate:
        self._pending = False
        try:
            envelope = cli_contract.parse_result_envelope(output)
        except ValueError:
            return WorkflowUpdate(
                status="failed",
                heading="订正预览结果不可识别",
                message="确认没有返回可识别的机器结果；请查看诊断日志后重试。",
                facts=[],
            )
        result = envelope.get("result")
        result_map = result if isinstance(result, dict) else {}
        artifacts = envelope.get("artifacts")
        artifact_map = artifacts if isinstance(artifacts, dict) else {}
        self.result = {**result_map, "status": envelope.get("status") or "", "paths": dict(artifact_map)}
        self.manifest_path = str(
            artifact_map.get("manifest") or result_map.get("manifest_path") or ""
        )
        facts = [
            f"选中候选：{int(result_map.get('selected_count') or 0)}",
        ]
        if self.manifest_path:
            facts.insert(0, format_manifest_path_fact(self.manifest_path))
        if exit_code != 0 or envelope.get("ok") is not True:
            error = envelope.get("error")
            error_map = error if isinstance(error, dict) else {}
            facts.append(f"错误码：{error_map.get('code') or 'UNKNOWN_ERROR'}")
            return WorkflowUpdate(
                status="failed",
                heading="所选润色候选未通过安全校验",
                message="没有修改任何 .rpy；请重新导入或查看诊断报告。",
                facts=facts,
            )
        status = str(envelope.get("status") or "")
        if status in {"stale", "blocked", "partial"}:
            return WorkflowUpdate(
                status="failed",
                heading="所选润色候选已过期或被阻止",
                message="没有修改任何 .rpy；请重新导入当前语料并重新确认。",
                facts=facts,
            )
        if status == "no_op":
            return WorkflowUpdate(
                status="done",
                heading="所选润色候选无需写回",
                message="确认的候选没有产生译文变化。",
                facts=facts,
            )
        return WorkflowUpdate(
            status="done",
            heading="所选润色候选预览已生成",
            message="请检查订正预览；确认后可使用“写回订正”。",
            facts=facts,
        )

    def stale_update(self) -> WorkflowUpdate:
        self._pending = False
        self.result = None
        self.manifest_path = ""
        return WorkflowUpdate(
            status="stale",
            heading="润色候选会话已过期",
            message="项目或提案文件已变化，迟到结果已丢弃；请重新导入当前项目。",
            facts=[],
        )
