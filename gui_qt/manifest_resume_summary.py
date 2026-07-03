"""Summarize resumable batch manifest state for the GUI workbench."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .translation_workflow import TERMINAL_FAILURE_STATES
from .user_copy import (
    format_job_fact,
    format_job_state_fact,
    format_manifest_path_fact,
    job_state_label,
    safety_level_label,
)
from .workflow_factory import resume_workflow
from .work_modes import WorkMode, WorkModeSpec


@dataclass(frozen=True)
class ManifestWorkflowDisplay:
    status: str
    heading: str
    message: str
    facts: tuple[str, ...]
    timeline_step_key: str | None
    workflow: Any | None
    selected_manifest_path: str
    archive_when_idle: bool


@dataclass(frozen=True)
class _ResumeDisplayCopy:
    safety_heading: str
    safety_message: str
    done_heading: str
    done_message: str
    ready_heading: str
    ready_message: str


def _display_for_resumed_workflow(
    workflow: Any | None,
    *,
    latest_safety: str,
    copy: _ResumeDisplayCopy,
) -> tuple[str, str, str, str | None]:
    if workflow and workflow.current_step() is None and latest_safety in {"warn", "block"}:
        status = "stale" if latest_safety == "warn" else "failed"
        message = copy.safety_message.format(safety=safety_level_label(latest_safety))
        return status, copy.safety_heading, message, "check"
    if workflow and workflow.current_step() is None:
        return "done", copy.done_heading, copy.done_message, None
    return "ready", copy.ready_heading, copy.ready_message, None


def _resume_display_copy(spec: WorkModeSpec, *, is_retry: bool) -> _ResumeDisplayCopy:
    if is_retry:
        return _ResumeDisplayCopy(
            safety_heading="补译结果仍需处理",
            safety_message=(
                "补译包检查结果为「{safety}」，暂不能合并回父任务。"
                "请先查看问题清单，必要时继续补译或人工处理。"
            ),
            done_heading="补译任务已完成",
            done_message="补译包流程已完成。",
            ready_heading="可继续补译后续处理",
            ready_message="检测到补译任务还有后续步骤，点击「继续翻译」继续执行。",
        )
    return _ResumeDisplayCopy(
        safety_heading="需要先处理问题",
        safety_message=(
            "最近一次检查结果为「{safety}」，暂不能写回。"
            "可先查看问题清单，必要时生成「补译包」并预览；"
            "处理完重新检查后，显示「可写回」才能写入项目。"
        ),
        done_heading=f"最新{spec.label}任务已完成",
        done_message="该任务流程的所有步骤已全部执行完毕。",
        ready_heading=f"可继续最新{spec.label}任务",
        ready_message=(
            f"检测到未完成的最新任务，"
            f"点击「{spec.resume_button_label or '继续任务'}」继续执行。"
        ),
    )


def _manifest_item_count_label(spec: WorkModeSpec, item_count: object) -> str | None:
    if item_count is None:
        return None
    if spec.mode == WorkMode.KEYWORD_EXTRACTION:
        return f"待处理行：{item_count} 行"
    if spec.mode == WorkMode.REVISION:
        return f"待修订项：{item_count} 项"
    return f"待翻译项：{item_count} 项"


def build_manifest_workflow_display(
    spec: WorkModeSpec,
    manifest_path: str,
    manifest: dict[str, object],
    *,
    extra_facts: list[str] | None = None,
) -> ManifestWorkflowDisplay:
    job_state = manifest.get("job_state")
    job_state_text = job_state if isinstance(job_state, str) else ""
    job_name = manifest.get("job_name")
    job_error = manifest.get("job_error")
    retry_parent = manifest.get("retry_of_manifest")
    retry_parent_text = retry_parent.strip() if isinstance(retry_parent, str) else ""

    latest_safety = ""
    check_summary = manifest.get("last_check_summary")
    if isinstance(check_summary, dict):
        safety_value = check_summary.get("safety_level")
        latest_safety = safety_value.strip().lower() if isinstance(safety_value, str) else ""

    facts: list[str] = [format_manifest_path_fact(manifest_path)]
    if job_name:
        facts.append(format_job_fact(str(job_name)))
    if job_state_text:
        facts.append(format_job_state_fact(job_state_text))

    summary_info = manifest.get("summary", {})
    if isinstance(summary_info, dict):
        file_count = summary_info.get("file_count")
        chunk_count = summary_info.get("chunk_count")
        item_count = summary_info.get("item_count")
        if file_count is not None:
            facts.append(f"扫描文件：{file_count} 个")
        if chunk_count is not None:
            facts.append(f"分块数量：{chunk_count} 个")
        item_label = _manifest_item_count_label(spec, item_count)
        if item_label:
            facts.append(item_label)

    if retry_parent_text:
        facts.append(f"父任务：{retry_parent_text}")
    if extra_facts:
        facts.extend(extra_facts)

    is_waiting = job_state_text in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING")
    timeline_step_key: str | None = None
    workflow = None

    if is_waiting:
        status = "waiting"
        timeline_step_key = "status"
        heading = f"最新{spec.label}任务进行中"
        message = "云端批量任务处理中。可以点击下方「查询云端状态」按钮进行刷新。"
    elif job_state_text in TERMINAL_FAILURE_STATES:
        status = "failed"
        timeline_step_key = "status"
        if job_state_text == "JOB_STATE_FAILED":
            heading = f"最新{spec.label}任务已失败"
            message = (
                f"云端任务执行失败：{job_error or '未知错误'}。"
                "可以重新开始或继续任务以重试。"
            )
        else:
            heading = f"最新{spec.label}任务无法继续"
            detail = f"：{job_error}" if job_error else ""
            message = (
                f"云端任务状态为「{job_state_label(job_state_text)}」{detail}，"
                "无法继续该任务；可以重新开始。"
            )
    else:
        workflow = resume_workflow(spec.mode, manifest_path, manifest)
        copy = _resume_display_copy(spec, is_retry=bool(retry_parent_text))
        status, heading, message, timeline_step_key = _display_for_resumed_workflow(
            workflow,
            latest_safety=latest_safety,
            copy=copy,
        )

    selected_manifest_path = retry_parent_text or manifest_path
    return ManifestWorkflowDisplay(
        status=status,
        heading=heading,
        message=message,
        facts=tuple(facts),
        timeline_step_key=timeline_step_key,
        workflow=workflow,
        selected_manifest_path=selected_manifest_path,
        archive_when_idle=status == "done",
    )


def completed_manifest_entry_fact(spec: WorkModeSpec, manifest_path: str) -> str:
    label = Path(manifest_path).parent.name or manifest_path
    return f"上次{spec.label}任务：{label}（已完成，可查看详情）"