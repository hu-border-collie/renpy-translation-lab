"""Retry package eligibility, CLI parsing, and preview summaries for the GUI."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .check_failures_report import (
    REASON_CATEGORY_REPAIR,
    REASON_CATEGORY_RETRY,
    build_check_issues_report,
    reason_code_label,
)
from .diagnostics_context import existing_retry_manifest_path, manifest_check_safety_level
from .user_copy import format_manifest_path_fact

_MANIFEST_LINE_RE = re.compile(r"^\s*Manifest:\s*(.+?)\s*$", re.MULTILINE)
_RETRY_PACKAGE_LINE_RE = re.compile(r"^\s*Created retry package:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RetryEligibility:
    eligible: bool
    heading: str
    message: str


@dataclass(frozen=True)
class BuildRetryResult:
    status: str
    heading: str
    message: str
    retry_manifest_path: str = ""
    package_dir: str = ""


@dataclass(frozen=True)
class RetryPreviewReport:
    status: str
    heading: str
    message: str
    facts: list[str]
    detail_lines: list[str]
    retry_manifest_path: str
    parent_manifest_path: str
    package_dir: str
    chunk_count: int
    item_count: int
    file_count: int


def assess_retry_eligibility(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> RetryEligibility:
    """Return whether the current warn state is a good candidate for build-retry."""
    if manifest_check_safety_level(manifest) != "warn":
        return RetryEligibility(
            eligible=False,
            heading="暂不适合 retry",
            message="只有最近一次 check 为需处理时，才可能生成 retry 包。",
        )

    report = build_check_issues_report(manifest, manifest_path=manifest_path)
    retry_count = report.category_counts.get(REASON_CATEGORY_RETRY, 0)
    repair_count = report.category_counts.get(REASON_CATEGORY_REPAIR, 0)

    if retry_count > 0:
        return RetryEligibility(
            eligible=True,
            heading="可生成 retry 包",
            message=(
                f"检测到 {retry_count} 个适合 retry 的问题项。"
                "生成后请先预览范围，确认后再到补救命令里手动提交任务。"
            ),
        )

    if repair_count > 0:
        return RetryEligibility(
            eligible=False,
            heading="不建议直接 retry",
            message="当前问题更适合 repair 或人工处理，不建议直接生成 retry 包。",
        )

    return RetryEligibility(
        eligible=False,
        heading="不适合 retry",
        message="当前问题属于源文件漂移、重定位失败或需重新 build/check 的类型，不应默认走 retry。",
    )


def parse_build_retry_output(output: str, exit_code: int) -> BuildRetryResult:
    """Parse build-retry stdout into a structured result."""
    if exit_code != 0:
        return BuildRetryResult(
            status="failed",
            heading="生成 retry 包失败",
            message="build-retry 没有正常完成，请查看诊断日志。",
        )

    if "No retry chunks needed." in output:
        return BuildRetryResult(
            status="empty",
            heading="无需 retry",
            message="当前没有需要重新翻译的 chunk，请回到问题清单确认原因。",
        )

    manifest_match = _MANIFEST_LINE_RE.search(output)
    package_match = _RETRY_PACKAGE_LINE_RE.search(output)
    retry_manifest_path = manifest_match.group(1).strip() if manifest_match else ""
    package_dir = package_match.group(1).strip() if package_match else ""

    if not retry_manifest_path:
        return BuildRetryResult(
            status="unknown",
            heading="retry 包状态不明确",
            message="命令已结束，但未能从输出中识别 retry manifest 路径，请查看诊断日志。",
        )

    return BuildRetryResult(
        status="ok",
        heading="retry 包已生成",
        message="请预览 retry 范围并确认后，再到补救命令里手动执行后续步骤。",
        retry_manifest_path=retry_manifest_path,
        package_dir=package_dir,
    )


def _sorted_file_paths(files: object) -> list[str]:
    if isinstance(files, dict):
        return sorted(str(path) for path in files.keys() if str(path).strip())
    return []


def _format_reason_count_lines(reason_counts: object) -> list[str]:
    if not isinstance(reason_counts, dict) or not reason_counts:
        return ["- 未记录 retry 原因统计"]

    lines: list[str] = []
    for reason_code, count in sorted(reason_counts.items()):
        if not isinstance(reason_code, str) or not reason_code.strip():
            continue
        if not isinstance(count, int) or count <= 0:
            continue
        lines.append(f"- {reason_code_label(reason_code)} ({reason_code})：{count}")
    return lines or ["- 未记录 retry 原因统计"]


def summarize_retry_manifest(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> RetryPreviewReport:
    """Build a user-facing preview report from a retry child manifest."""
    if not manifest_path:
        raw_path = manifest.get("_manifest_path")
        manifest_path = raw_path.strip() if isinstance(raw_path, str) else ""

    parent_manifest_path = manifest.get("retry_of_manifest")
    parent_text = parent_manifest_path.strip() if isinstance(parent_manifest_path, str) else ""

    summary = manifest.get("summary")
    chunk_count = 0
    item_count = 0
    file_count = 0
    if isinstance(summary, dict):
        if isinstance(summary.get("chunk_count"), int):
            chunk_count = summary["chunk_count"]
        if isinstance(summary.get("item_count"), int):
            item_count = summary["item_count"]
        if isinstance(summary.get("file_count"), int):
            file_count = summary["file_count"]

    package_dir = manifest.get("_package_dir")
    if not isinstance(package_dir, str) or not package_dir.strip():
        if manifest_path:
            from .diagnostics_context import resolve_package_dir

            package_dir = resolve_package_dir(manifest_path, manifest)
        else:
            package_dir = ""

    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if parent_text:
        facts.append(f"父任务清单：{parent_text}")
    if package_dir:
        facts.append(f"retry 包目录：{package_dir}")
    facts.append(f"影响范围：{file_count} 个文件，{chunk_count} 个 chunk，{item_count} 个条目")

    reason_lines = _format_reason_count_lines(manifest.get("retry_reason_counts"))
    file_paths = _sorted_file_paths(manifest.get("files"))
    detail_lines = ["【retry 原因统计】", *reason_lines]
    if file_paths:
        detail_lines.append("")
        detail_lines.append("【涉及文件】")
        detail_lines.extend(f"- {path}" for path in file_paths)

    return RetryPreviewReport(
        status="ok",
        heading="retry 包预览",
        message=(
            "请确认影响范围后再继续。"
            "GUI 不会自动提交 retry 任务；确认后可到「补救命令」手动执行 submit / download / merge-retry。"
        ),
        facts=facts,
        detail_lines=detail_lines,
        retry_manifest_path=manifest_path,
        parent_manifest_path=parent_text,
        package_dir=package_dir if isinstance(package_dir, str) else "",
        chunk_count=chunk_count,
        item_count=item_count,
        file_count=file_count,
    )


def retry_followup_allowed(
    manifest: dict[str, object],
    *,
    parent_manifest_path: str,
    confirmed_parent_paths: set[str] | frozenset[str],
) -> bool:
    """Return whether remediation commands for submit/merge may be shown as next steps."""
    if not parent_manifest_path:
        return False
    if not existing_retry_manifest_path(manifest):
        return True
    return parent_manifest_path in confirmed_parent_paths


def build_retry_cli_args(parent_manifest_path: str) -> list[str]:
    """Return CLI args for build-retry; kept separate to make non-submit behavior testable."""
    return ["build-retry", parent_manifest_path]