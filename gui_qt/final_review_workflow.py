"""Final-review Batch workflow and selected-finding hand-off for the GUI."""
from __future__ import annotations

import re
from typing import Sequence

from .batch_workflow_support import build_submit_cli_args
from .revision_report import summarize_revision_preview_output
from .translation_workflow import (
    TERMINAL_FAILURE_STATES, WorkflowStep, WorkflowUpdate, extract_job_state,
    extract_manifest_path, manifest_path_for_package,
)
from .user_copy import format_job_state_fact, format_manifest_path_fact, job_state_label


STEP_TEXT = {
    "final-review-build": ("正在准备最终审校", "正在冻结项目上下文并生成审查任务。"),
    "final-review-resume": ("正在刷新审查范围", "正在重新采集上下文并重建未完成任务。"),
    "submit": ("正在提交最终审校", "正在上传审查请求并创建云端批量任务。"),
    "status": ("正在刷新审查状态", "正在查询云端任务处理状态。"),
    "download": ("正在获取审查结果", "任务已完成，正在下载问题报告。"),
    "final-review-ingest-results": ("正在整理问题报告", "正在校验结果并写入最终审校 findings。"),
    "final-review-create-revisions": ("正在生成订正预览", "正在把人工选择的问题转换为安全订正候选。"),
}


def _created_campaign(output: str) -> str:
    match = re.search(r"^Created final-review campaign:\s*(.+?)\s*$", output, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _created_revision(output: str) -> str:
    match = re.search(r"^Created final-review revision package:\s*(.+?)\s*$", output, re.MULTILINE)
    return match.group(1).strip() if match else ""


class FinalReviewWorkflow:
    def __init__(self, steps: list[str], manifest_path: str = "", *, finding_ids: Sequence[str] = (), submit_max_cost=None):
        self._steps = list(steps)
        self.manifest_path = manifest_path
        self.finding_ids = tuple(finding_ids)
        self.submit_max_cost = submit_max_cost

    @classmethod
    def start_new(cls, *, submit_max_cost=None):
        return cls(["final-review-build", "submit", "status"], submit_max_cost=submit_max_cost)

    @classmethod
    def resume_manifest(cls, manifest_path, manifest, *, submit_max_cost=None):
        summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
        counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), dict) else {}
        done_count = counts.get("done")
        unit_count = summary.get("unit_count")
        if manifest.get("status") == "done" or (
            done_count is not None
            and unit_count is not None
            and done_count == unit_count
        ):
            return cls([], manifest_path, submit_max_cost=submit_max_cost)
        if manifest.get("job_state") == "JOB_STATE_SUCCEEDED":
            steps = ["download", "final-review-ingest-results"]
        elif not manifest.get("job_name"):
            steps = ["final-review-resume", "submit", "status"]
        else:
            steps = ["status"]
        return cls(steps, manifest_path, submit_max_cost=submit_max_cost)

    @classmethod
    def create_revisions(cls, manifest_path: str, finding_ids: Sequence[str]):
        return cls(["final-review-create-revisions"], manifest_path, finding_ids=finding_ids)

    def current_step(self):
        if not self._steps:
            return None
        key = self._steps[0]
        heading, message = STEP_TEXT[key]
        return WorkflowStep(key=key, args=self._args(key), heading=heading, message=message)

    def complete_current_step(self, exit_code: int, output: str):
        key = self._steps.pop(0)
        if exit_code != 0:
            self._steps.clear()
            return WorkflowUpdate(status="failed", heading="最终审校流程中断",
                                  message=f"{STEP_TEXT[key][0]}没有正常完成，请查看原始输出。", facts=self._facts())
        if key == "final-review-build":
            package = _created_campaign(output)
            if not package:
                self._steps.clear()
                return WorkflowUpdate(status="failed", heading="无法准备最终审校",
                                      message="未生成最终审校任务包，请查看诊断日志。", facts=[])
            self.manifest_path = manifest_path_for_package(package)
        elif key == "submit":
            path = extract_manifest_path(output)
            if path:
                self.manifest_path = path
        elif key == "status":
            state = extract_job_state(output)
            if state == "JOB_STATE_SUCCEEDED":
                if getattr(self, "only_query", False):
                    self._steps.clear()
                    return WorkflowUpdate(
                        status="ready",
                        heading="最终审校云端任务已完成",
                        message="查询完成；请点击「继续审查」下载并整理问题报告。",
                        facts=self._facts([format_job_state_fact(state)]),
                    )
                self._steps[:0] = ["download", "final-review-ingest-results"]
            elif state in TERMINAL_FAILURE_STATES:
                self._steps.clear()
                return WorkflowUpdate(status="failed", heading="最终审校任务失败",
                                      message=f"云端状态为 {job_state_label(state)}。", facts=self._facts([format_job_state_fact(state)]))
            else:
                self._steps.clear()
                return WorkflowUpdate(status="waiting", heading="最终审校仍在处理",
                                      message="稍后点击「查询云端状态」继续。", facts=self._facts([format_job_state_fact(state or '未知')]))
        elif key == "final-review-create-revisions":
            package = _created_revision(output)
            if package:
                self.manifest_path = manifest_path_for_package(package)
            update = summarize_revision_preview_output(output, 0)
            self._steps.clear()
            return WorkflowUpdate(status=update.status, heading="订正预览已生成",
                                  message="所选问题已进入标准订正预览；确认后才可写回。",
                                  facts=self._facts(list(update.facts)))
        return self._continue()

    def _continue(self):
        step = self.current_step()
        if step is None:
            return WorkflowUpdate(status="done", heading="最终审校报告已就绪",
                                  message="请审核问题并选择需要进入订正预览的项目。", facts=self._facts())
        return WorkflowUpdate(status="running", heading=step.heading, message=step.message,
                              facts=self._facts(), should_continue=True)

    def _args(self, key: str) -> list[str]:
        if key == "final-review-build":
            return [key]
        if key == "submit":
            return build_submit_cli_args(self.manifest_path, self.submit_max_cost)
        if key == "final-review-create-revisions":
            args = [key, self.manifest_path]
            for finding_id in self.finding_ids:
                args.extend(["--finding-id", finding_id])
            return args
        return [key, self.manifest_path]

    def _facts(self, extra=None):
        facts = [format_manifest_path_fact(self.manifest_path)] if self.manifest_path else []
        if extra:
            facts.extend(extra)
        return facts
