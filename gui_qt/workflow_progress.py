"""Parse CLI progress lines into compact GUI progress-bar state."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace

from .duration_format import format_remaining_duration_zh


@dataclass(frozen=True)
class WorkflowProgressState:
    kind: str
    current: int = 0
    total: int = 0
    label: str = ""
    facts: tuple[str, ...] = ()
    visible: bool = False
    indeterminate: bool = False
    current_file: str = ""
    file_done: int = 0
    file_total: int = 0
    current_file_done: int = 0
    current_file_total: int = 0
    started_at: float | None = None
    last_sample_at: float | None = None
    last_sample_current: int = 0
    seconds_per_unit: float | None = None
    remaining_seconds: int | None = None


_SOURCE_INDEX_BUILD_TOTAL_RE = re.compile(
    r"Source index retrieval for build:\s*(\d+)\s*chunks to query\."
)
_SOURCE_INDEX_BUILD_PROGRESS_RE = re.compile(
    r"Source index retrieval progress:\s*(\d+)/(\d+)\s*chunks,\s*file=(.*?),\s*chunk=(.+)\."
)
_SOURCE_INDEX_BUILD_COMPLETE_RE = re.compile(
    r"Source index retrieval complete:\s*(\d+)/(\d+)\s*chunks queried\."
)
_SYNC_REQUEST_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s*$")
_SYNC_FILES_TOTAL_RE = re.compile(r"Found\s+(\d+)\s+files\.")
_SYNC_PROCESSING_FILE_RE = re.compile(r"Processing:\s*(.+?)\s*$")
_SYNC_FOUND_LINES_RE = re.compile(r"Found\s+(\d+)\s+lines to translate\.")
_SYNC_TRANSLATED_ITEMS_RE = re.compile(r"Translated\s+(\d+)/(\d+)\s+items\.")
_SYNC_DONE_FILE_RE = re.compile(r"Done with\s+(.+)\.")
_RAG_SCAN_RE = re.compile(
    r"RAG scan progress:\s*(\d+)\s*records scanned from\s*(\d+)\s*files,\s*(\d+)\s*pending\."
)
_RAG_EMBED_RE = re.compile(
    r"RAG update progress:\s*(\d+)/(\d+)\s*records\."
)
_WORK_COPY_TOTAL_RE = re.compile(r"Work bootstrap copy progress:\s*0/(\d+)\s*files\.")
_WORK_COPY_PROGRESS_RE = re.compile(
    r"Work bootstrap copy progress:\s*(\d+)/(\d+)\s*files,\s*file=(.+)\."
)
_WORKFLOW_ETA_MIN_ELAPSED_SECONDS = 5.0
_WORKFLOW_ETA_MIN_PROGRESS_UNITS = 2


def create_workflow_progress_state(kind: str) -> WorkflowProgressState | None:
    if kind == "work_bootstrap":
        return WorkflowProgressState(
            kind=kind,
            label="正在统计文件…",
            visible=True,
            indeterminate=True,
        )
    if kind in {
        "source_index_build",
        "sync_translation",
        "sync_requests",
        "rag_bootstrap",
    }:
        return WorkflowProgressState(kind=kind)
    return None


def update_workflow_progress_from_line(
    line: str,
    state: WorkflowProgressState | None,
    *,
    now: float | None = None,
) -> WorkflowProgressState | None:
    text = line.strip()
    if state is None or not text:
        return state
    if state.kind == "source_index_build":
        return _with_progress_timing(state, _update_source_index_build_progress(text, state), now=now)
    if state.kind == "sync_translation":
        return _with_progress_timing(state, _update_sync_translation_progress(text, state), now=now)
    if state.kind == "sync_requests":
        return _with_progress_timing(state, _update_sync_request_progress(text, state), now=now)
    if state.kind == "rag_bootstrap":
        return _with_progress_timing(state, _update_rag_bootstrap_progress(text, state), now=now)
    if state.kind == "work_bootstrap":
        return _with_progress_timing(state, _update_work_bootstrap_progress(text, state), now=now)
    return state


def _update_source_index_build_progress(
    text: str,
    state: WorkflowProgressState,
) -> WorkflowProgressState:
    match = _SOURCE_INDEX_BUILD_TOTAL_RE.search(text)
    if match:
        total = int(match.group(1))
        return replace(
            state,
            current=0,
            total=total,
            label=f"检索原文索引 0/{total}",
            facts=("正在为批量包检索相关原文…",),
            visible=True,
        )

    match = _SOURCE_INDEX_BUILD_PROGRESS_RE.search(text)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        file_name = match.group(3).strip()
        chunk = match.group(4).strip()
        return replace(
            state,
            current=current,
            total=total,
            label=f"检索原文索引 {current}/{total}",
            facts=(f"当前文件：{file_name}", f"当前分块：{chunk}"),
            visible=True,
        )

    match = _SOURCE_INDEX_BUILD_COMPLETE_RE.search(text)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        return replace(
            state,
            current=current,
            total=total,
            label=f"检索原文索引 {current}/{total}",
            facts=("原文索引检索完成。",),
            visible=True,
        )
    return state


def _update_sync_request_progress(
    text: str,
    state: WorkflowProgressState,
) -> WorkflowProgressState:
    match = _SYNC_REQUEST_PROGRESS_RE.search(text)
    if not match:
        return state
    current = int(match.group(1))
    total = int(match.group(2))
    key = match.group(3).strip()
    return replace(
        state,
        current=current,
        total=total,
        label=f"请求 {current}/{total}",
        facts=(f"当前请求：{key}",),
        visible=True,
    )


def _update_sync_translation_progress(
    text: str,
    state: WorkflowProgressState,
) -> WorkflowProgressState:
    match = _SYNC_FILES_TOTAL_RE.search(text)
    if match:
        file_total = int(match.group(1))
        if file_total <= 0:
            return replace(state, visible=False)
        return replace(
            state,
            current=0,
            total=file_total,
            label=f"文件 0/{file_total}",
            facts=(),
            visible=True,
            file_done=0,
            file_total=file_total,
            current_file="",
            current_file_done=0,
            current_file_total=0,
        )

    match = _SYNC_PROCESSING_FILE_RE.search(text)
    if match:
        current_file = match.group(1).strip()
        return replace(
            state,
            current=state.file_done,
            total=max(state.file_total, 1),
            label=f"正在处理 {current_file}",
            facts=(f"文件：{_current_file_index(state)}/{state.file_total}",)
            if state.file_total
            else (),
            visible=True,
            current_file=current_file,
            current_file_done=0,
            current_file_total=0,
        )

    match = _SYNC_FOUND_LINES_RE.search(text)
    if match:
        total = int(match.group(1))
        return replace(
            state,
            current=0,
            total=max(total, 1),
            label=f"{state.current_file or '当前文件'} 0/{total}",
            facts=(f"文件：{_current_file_index(state)}/{state.file_total}",)
            if state.file_total
            else (),
            visible=total > 0,
            current_file_done=0,
            current_file_total=total,
        )

    match = _SYNC_TRANSLATED_ITEMS_RE.search(text)
    if match:
        translated = int(match.group(1))
        total = state.current_file_total or int(match.group(2))
        current_done = min(total, state.current_file_done + max(translated, 0))
        return replace(
            state,
            current=current_done,
            total=max(total, 1),
            label=f"{state.current_file or '当前文件'} {current_done}/{total}",
            facts=(f"文件：{_current_file_index(state)}/{state.file_total}",)
            if state.file_total
            else (),
            visible=True,
            current_file_done=current_done,
            current_file_total=total,
        )

    match = _SYNC_DONE_FILE_RE.search(text)
    if match:
        file_done = min(
            state.file_total or state.file_done + 1,
            state.file_done + 1,
        )
        label = (
            f"文件 {file_done}/{state.file_total}"
            if state.file_total
            else f"已完成 {file_done} 个文件"
        )
        return replace(
            state,
            current=file_done,
            total=max(state.file_total, 1),
            label=label,
            facts=(),
            visible=True,
            file_done=file_done,
            current_file="",
            current_file_done=0,
            current_file_total=0,
        )

    if "No new lines to translate." in text and state.current_file:
        file_done = min(
            state.file_total or state.file_done + 1,
            state.file_done + 1,
        )
        label = (
            f"文件 {file_done}/{state.file_total}"
            if state.file_total
            else f"已完成 {file_done} 个文件"
        )
        return replace(
            state,
            current=file_done,
            total=max(state.file_total, 1),
            label=label,
            facts=(f"无新增内容：{state.current_file}",),
            visible=True,
            file_done=file_done,
            current_file="",
            current_file_done=0,
            current_file_total=0,
        )

    return state


def _update_rag_bootstrap_progress(
    text: str,
    state: WorkflowProgressState,
) -> WorkflowProgressState:
    match = _RAG_SCAN_RE.search(text)
    if match:
        scanned = int(match.group(1))
        files = int(match.group(2))
        pending = int(match.group(3))
        if pending <= 0:
            return replace(
                state,
                current=scanned,
                total=max(scanned, 1),
                label="记忆库无需更新",
                facts=(f"扫描记录：{scanned}", f"扫描文件：{files}"),
                visible=True,
            )
        return replace(
            state,
            current=0,
            total=pending,
            label=f"记忆库待写入 0/{pending}",
            facts=(f"扫描记录：{scanned}", f"扫描文件：{files}"),
            visible=True,
        )

    match = _RAG_EMBED_RE.search(text)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        return replace(
            state,
            current=current,
            total=max(total, 1),
            label=f"更新记忆库 {current}/{total}",
            facts=state.facts,
            visible=True,
        )
    return state


def _update_work_bootstrap_progress(
    text: str,
    state: WorkflowProgressState,
) -> WorkflowProgressState:
    match = _WORK_COPY_TOTAL_RE.search(text)
    if match:
        total = int(match.group(1))
        return replace(
            state,
            current=0,
            total=max(total, 1),
            label=f"复制文件 0/{total}",
            visible=total > 0,
            indeterminate=False,
        )

    match = _WORK_COPY_PROGRESS_RE.search(text)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        file_name = match.group(3).strip()
        return replace(
            state,
            current=current,
            total=max(total, 1),
            label=f"复制文件 {current}/{total}",
            facts=(f"当前文件：{file_name}",),
            visible=True,
            indeterminate=False,
        )
    return state


def _with_progress_timing(
    previous: WorkflowProgressState,
    updated: WorkflowProgressState,
    *,
    now: float | None = None,
) -> WorkflowProgressState:
    if updated is previous:
        return updated
    if not updated.visible or updated.indeterminate or updated.total <= 0:
        return replace(
            updated,
            started_at=None,
            last_sample_at=None,
            last_sample_current=updated.current,
            seconds_per_unit=None,
            remaining_seconds=None,
        )

    if now is None:
        now = time.monotonic()

    current = min(max(updated.current, 0), max(updated.total, 1))
    total = max(updated.total, 1)
    same_series = (
        previous.visible
        and not previous.indeterminate
        and previous.total == updated.total
        and previous.current <= current
    )
    started_at = previous.started_at if same_series and previous.started_at is not None else now
    last_sample_at = previous.last_sample_at if same_series else None
    last_sample_current = previous.last_sample_current if same_series else 0
    seconds_per_unit = previous.seconds_per_unit if same_series else None

    if current <= 0:
        return replace(
            updated,
            started_at=started_at,
            last_sample_at=None,
            last_sample_current=0,
            seconds_per_unit=None,
            remaining_seconds=None,
        )

    if current >= total:
        return replace(
            updated,
            started_at=started_at,
            last_sample_at=now,
            last_sample_current=current,
            remaining_seconds=0,
        )

    delta_units = current - last_sample_current
    if last_sample_at is not None and delta_units > 0:
        delta_seconds = now - last_sample_at
        if delta_seconds > 0:
            sample_rate = delta_seconds / delta_units
            seconds_per_unit = (
                sample_rate
                if seconds_per_unit is None
                else seconds_per_unit * 0.6 + sample_rate * 0.4
            )
        last_sample_at = now
        last_sample_current = current
    elif last_sample_at is None:
        last_sample_at = now
        last_sample_current = current

    elapsed = now - started_at if started_at is not None else 0
    if (
        seconds_per_unit is None
        and current >= _WORKFLOW_ETA_MIN_PROGRESS_UNITS
        and elapsed >= _WORKFLOW_ETA_MIN_ELAPSED_SECONDS
    ):
        seconds_per_unit = elapsed / current

    remaining_seconds = None
    label = updated.label
    if seconds_per_unit is not None:
        remaining_seconds = max(0, int((total - current) * seconds_per_unit + 0.5))
        if remaining_seconds > 0:
            label = f"{label} · {format_remaining_duration_zh(remaining_seconds)}"

    return replace(
        updated,
        label=label,
        started_at=started_at,
        last_sample_at=last_sample_at,
        last_sample_current=last_sample_current,
        seconds_per_unit=seconds_per_unit,
        remaining_seconds=remaining_seconds,
    )


def _current_file_index(state: WorkflowProgressState) -> int:
    if state.file_total <= 0:
        return 0
    return min(state.file_total, state.file_done + 1)
