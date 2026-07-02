"""Helpers for displaying and selecting split batch manifests in the GUI."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .user_copy import job_state_label, safety_level_label


RUNNING_STATES = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
@dataclass(frozen=True)
class SplitManifestEntry:
    index: int
    total: int
    manifest_path: str
    display_name: str
    job_name: str
    job_state: str
    safety_level: str
    item_count: int | None
    chunk_count: int | None
    applied: bool
    has_result: bool
    selectable: bool
    needs_submit: bool
    needs_status: bool
    status_label: str
    status_kind: str

    @property
    def part_label(self) -> str:
        if self.index > 0 and self.total > 0:
            return f"part{self.index:02d}/{self.total:02d}"
        return self.display_name or Path(self.manifest_path).parent.name


def read_json_object(path: str | Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
        data = json.loads(raw or "{}")
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def split_group_paths(manifest_path: str | Path, manifest: dict[str, Any] | None = None) -> list[Path]:
    path = Path(manifest_path)
    data = manifest if manifest is not None else read_json_object(path)
    if not isinstance(data, dict):
        return []

    parent_path = data.get("split_from_manifest")
    if isinstance(parent_path, str) and parent_path.strip():
        parent = read_json_object(parent_path)
        children = parent.get("split_children")
        if isinstance(children, list):
            paths = [
                Path(child)
                for child in children
                if isinstance(child, str) and child.strip()
            ]
            if paths:
                return _sort_split_paths(paths)

    children = data.get("split_children")
    if isinstance(children, list):
        paths = [
            Path(child)
            for child in children
            if isinstance(child, str) and child.strip()
        ]
        if paths:
            return _sort_split_paths(paths)

    split_index = _positive_int(data.get("split_index"))
    split_total = _positive_int(data.get("split_total"))
    if not split_index or not split_total:
        return []

    split_root = path.parent.parent
    if not split_root.exists():
        return [path] if path.exists() else []

    paths = list(split_root.glob(f"part*_of_{split_total:02d}/manifest.json"))
    if path.exists() and path not in paths:
        paths.append(path)
    return _sort_split_paths(paths)


def load_split_manifest_entries(
    manifest_path: str | Path,
    manifest: dict[str, Any] | None = None,
) -> list[SplitManifestEntry]:
    paths = split_group_paths(manifest_path, manifest)
    entries: list[SplitManifestEntry] = []
    for path in paths:
        data = read_json_object(path)
        if not data:
            continue
        entries.append(entry_from_manifest(path, data))
    return sorted(entries, key=lambda entry: (entry.total or 0, entry.index or 999999, entry.manifest_path))


def entry_from_manifest(path: str | Path, manifest: dict[str, Any]) -> SplitManifestEntry:
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    check_summary = manifest.get("last_check_summary")
    if not isinstance(check_summary, dict):
        check_summary = {}

    job_name = _stripped(manifest.get("job_name"))
    job_state = _stripped(manifest.get("job_state"))
    safety_level = _stripped(check_summary.get("safety_level"))
    applied = bool(manifest.get("applied_at"))
    result_path = _stripped(manifest.get("result_jsonl_path"))
    has_result = bool(result_path)
    needs_submit = not job_name and not applied
    needs_status = bool(job_name) and job_state in RUNNING_STATES
    selectable = _is_selectable(job_name, job_state, safety_level, applied, has_result)

    return SplitManifestEntry(
        index=_positive_int(manifest.get("split_index")) or _split_index_from_path(path),
        total=_positive_int(manifest.get("split_total")) or _split_total_from_path(path),
        manifest_path=str(path),
        display_name=_stripped(manifest.get("display_name")),
        job_name=job_name,
        job_state=job_state,
        safety_level=safety_level,
        item_count=_optional_int(summary.get("item_count")),
        chunk_count=_optional_int(summary.get("chunk_count")),
        applied=applied,
        has_result=has_result,
        selectable=selectable,
        needs_submit=needs_submit,
        needs_status=needs_status,
        status_label=_status_label(job_name, job_state, safety_level, applied, has_result),
        status_kind=_status_kind(job_name, job_state, safety_level, applied, has_result),
    )


def summarize_split_entries(entries: list[SplitManifestEntry]) -> list[str]:
    if not entries:
        return []
    total = len(entries)
    submitted = sum(1 for entry in entries if entry.job_name)
    running = sum(1 for entry in entries if entry.job_state in RUNNING_STATES)
    succeeded = sum(1 for entry in entries if entry.job_state == "JOB_STATE_SUCCEEDED")
    checked = sum(1 for entry in entries if entry.safety_level)
    applied = sum(1 for entry in entries if entry.applied)
    unsubmitted = sum(1 for entry in entries if entry.needs_submit)
    facts = [f"拆分包：{total} 个；已提交 {submitted} 个；未提交 {unsubmitted} 个"]
    if running:
        facts.append(f"云端处理中：{running} 个")
    if succeeded:
        facts.append(f"云端已完成：{succeeded} 个")
    if checked:
        facts.append(f"已检查：{checked} 个")
    if applied:
        facts.append(f"已写回：{applied} 个")
    return facts


def _sort_split_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (_split_index_from_path(path), str(path)))


def _is_selectable(
    job_name: str,
    job_state: str,
    safety_level: str,
    applied: bool,
    has_result: bool,
) -> bool:
    if applied:
        return False
    if safety_level:
        return True
    if job_state == "JOB_STATE_SUCCEEDED":
        return True
    if has_result and job_name:
        return True
    return False


def _status_label(
    job_name: str,
    job_state: str,
    safety_level: str,
    applied: bool,
    has_result: bool,
) -> str:
    if applied:
        return "已写回"
    if safety_level:
        return f"检查：{safety_level_label(safety_level)}"
    if not job_name:
        return "未提交"
    if job_state == "JOB_STATE_SUCCEEDED" and has_result:
        return "已下载，待检查"
    if job_state:
        return job_state_label(job_state)
    return "已提交，待查询"

def _status_kind(
    job_name: str,
    job_state: str,
    safety_level: str,
    applied: bool,
    has_result: bool,
) -> str:
    if applied:
        return "applied"
    if safety_level == "safe":
        return "checked_safe"
    if safety_level == "warn":
        return "checked_warn"
    if safety_level == "block":
        return "checked_block"
    if job_state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
        return "failed"
    if job_state in RUNNING_STATES:
        return "running"
    if job_state == "JOB_STATE_SUCCEEDED" and has_result:
        return "downloaded"
    if job_state == "JOB_STATE_SUCCEEDED":
        return "succeeded"
    if job_name:
        return "submitted"
    return "unsubmitted"

def _stripped(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _split_index_from_path(path: str | Path) -> int:
    match = re.search(r"part(\d+)_of_(\d+)", str(path))
    return int(match.group(1)) if match else 999999


def _split_total_from_path(path: str | Path) -> int:
    match = re.search(r"part(\d+)_of_(\d+)", str(path))
    return int(match.group(2)) if match else 0
