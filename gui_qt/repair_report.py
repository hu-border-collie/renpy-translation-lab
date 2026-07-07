"""Repair workflow summaries, eligibility, and CLI helpers for the GUI."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .check_failures_report import (
    REASON_CATEGORY_REPAIR,
    REASON_CATEGORY_RETRY,
    build_check_issues_report,
    classify_reason_category,
    infer_reason_code_from_entry,
    normalize_failure_entry,
    parse_check_failures_jsonl,
    resolve_check_report_path,
)
from .diagnostics_context import DiagnosticsContext, DiagnosticsPathEntry, resolve_package_dir
from .diagnostics_context import manifest_check_safety_level
from .user_copy import format_manifest_path_fact

_DERIVED_REPAIR_REPORT_NAME = "repair_from_check_failures.jsonl"
_REPAIR_RUN_DIR_RE = re.compile(r"^\s*Repair run dir:\s*(.+?)\s*$", re.MULTILINE)
_REPAIR_SUMMARY_INT_RE = re.compile(
    r"^\s*(Requested items|Repair jobs|Applied items|Applied files|Failure items|"
    r"Request errors|Parse errors|Validation failures|Missing item ids|Unresolved items):"
    r"\s*(-?\d+)\s*$",
    re.MULTILINE,
)
_STORY_MEMORY_REPAIR_RE = re.compile(
    r"^\s*Story Memory repair hits:\s*(.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RepairEligibility:
    eligible: bool
    heading: str
    message: str
    repair_count: int = 0
    retry_count: int = 0


@dataclass(frozen=True)
class RepairReportCandidate:
    label: str
    path: str
    source: str


@dataclass(frozen=True)
class RepairSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    report_path: str = ""
    run_dir: str = ""
    applied_items: int | None = None
    applied_files: int | None = None
    failure_items: int | None = None


def assess_repair_eligibility(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
) -> RepairEligibility:
    if manifest_check_safety_level(manifest) != "warn":
        return RepairEligibility(
            eligible=False,
            heading="暂不适合同步修补",
            message="只有最近一次检查为「需处理」时，才可能使用同步修补。",
        )

    report = build_check_issues_report(manifest, manifest_path=manifest_path)
    retry_count = report.category_counts.get(REASON_CATEGORY_RETRY, 0)
    repair_count = report.category_counts.get(REASON_CATEGORY_REPAIR, 0)

    if repair_count <= 0:
        return RepairEligibility(
            eligible=False,
            heading="暂无适合修补的问题",
            message="当前问题清单中没有「可尝试修复」类条目。",
            repair_count=repair_count,
            retry_count=retry_count,
        )

    if retry_count > repair_count:
        return RepairEligibility(
            eligible=False,
            heading="建议优先补译",
            message=(
                f"检测到 {retry_count} 个适合补译的问题项，"
                f"仅 {repair_count} 个适合同步修补。"
                "请先考虑「生成补译包」或人工处理。"
            ),
            repair_count=repair_count,
            retry_count=retry_count,
        )

    return RepairEligibility(
        eligible=True,
        heading="可同步修补",
        message=(
            f"检测到 {repair_count} 个适合同步修补的问题项。"
            "该操作会直接修改翻译文件，请先确认已在副本或备份上验证。"
        ),
        repair_count=repair_count,
        retry_count=retry_count,
    )


def repair_action_ready(
    manifest: dict[str, object] | None,
    *,
    manifest_path: str = "",
) -> bool:
    if manifest is None or not manifest_path:
        return False
    return assess_repair_eligibility(manifest, manifest_path=manifest_path).eligible


def _glob_paths(pattern: str) -> list[str]:
    path = Path(pattern)
    return sorted(str(match) for match in path.parent.glob(path.name))


def discover_repair_report_candidates(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
    search_roots: list[str] | None = None,
    path_exists: Callable[[str], bool] | None = None,
    glob_paths: Callable[[str], list[str]] | None = None,
) -> list[RepairReportCandidate]:
    exists = path_exists or (lambda path: Path(path).exists())
    globber = glob_paths or _glob_paths

    package_dir = resolve_package_dir(manifest_path, manifest)
    roots: list[str] = []
    if package_dir:
        roots.append(package_dir)
    if search_roots:
        for root in search_roots:
            text = str(root).strip()
            if text and text not in roots:
                roots.append(text)

    candidates: list[RepairReportCandidate] = []
    seen: set[str] = set()

    def add(label: str, path: str, source: str) -> None:
        normalized = str(Path(path))
        key = normalized.casefold()
        if key in seen or not exists(normalized):
            return
        seen.add(key)
        candidates.append(RepairReportCandidate(label=label, path=normalized, source=source))

    for root in roots:
        for path in globber(str(Path(root) / "remaining_need_translate_*.jsonl")):
            add("剩余未译报告", path, "remaining_need_translate")
        failures_path = str(Path(root) / "failures.jsonl")
        add("翻译包失败明细", failures_path, "failures")

    derived_path = build_derived_repair_report_path(manifest, manifest_path=manifest_path)
    if derived_path:
        add("从检查报告提取（repair 类）", derived_path, "derived_check_failures")

    return candidates


def build_derived_repair_report_path(
    manifest: dict[str, object],
    *,
    manifest_path: str = "",
    path_exists: Callable[[str], bool] | None = None,
    read_file: Callable[[str], str] | None = None,
) -> str:
    exists = path_exists or (lambda path: Path(path).exists())
    reader = read_file or (lambda path: Path(path).read_text(encoding="utf-8"))

    package_dir = resolve_package_dir(manifest_path, manifest)
    if not package_dir:
        return ""

    check_report_path = resolve_check_report_path(manifest, manifest_path=manifest_path)
    if not check_report_path or not exists(check_report_path):
        return ""

    try:
        raw_entries = parse_check_failures_jsonl(reader(check_report_path))
    except (OSError, UnicodeError, ValueError):
        return ""

    repair_rows: list[dict[str, object]] = []
    for entry in raw_entries:
        reason_code = infer_reason_code_from_entry(entry)
        error = str(entry.get("error") or "")
        if classify_reason_category(reason_code, error) != REASON_CATEGORY_REPAIR:
            continue
        item = normalize_failure_entry(entry)
        if not item.file_rel_path or item.line is None:
            continue
        source_text = entry.get("text")
        if not isinstance(source_text, str) or not source_text.strip():
            continue
        line_value = item.line - 1 if item.line > 0 else 0
        repair_rows.append(
            {
                "file_rel_path": item.file_rel_path,
                "line": line_value,
                "text": source_text.strip(),
            }
        )

    if not repair_rows:
        return ""

    output_path = str(Path(package_dir) / _DERIVED_REPAIR_REPORT_NAME)
    Path(output_path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in repair_rows) + "\n",
        encoding="utf-8",
    )
    return output_path


def select_repair_report_path(
    candidates: list[RepairReportCandidate],
) -> str:
    if not candidates:
        return ""
    preferred = (
        "derived_check_failures",
        "remaining_need_translate",
        "failures",
    )
    for source in preferred:
        for candidate in candidates:
            if candidate.source == source:
                return candidate.path
    return candidates[0].path


def build_repair_cli_args(
    report_path: str,
    *,
    limit: int = 0,
    offset: int = 0,
    batch_size: int = 2,
    context_before: int = 2,
    context_after: int = 2,
    api_key_index: int | None = None,
) -> list[str]:
    args = [
        "repair",
        report_path,
        "--batch-size",
        str(batch_size),
        "--context-before",
        str(context_before),
        "--context-after",
        str(context_after),
    ]
    if limit > 0:
        args.extend(["--limit", str(limit)])
    if offset > 0:
        args.extend(["--offset", str(offset)])
    if api_key_index is not None:
        args.extend(["--api-key-index", str(api_key_index)])
    return args


def parse_repair_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    field_map = {
        "requested items": "requested_items",
        "repair jobs": "job_count",
        "applied items": "applied_items",
        "applied files": "applied_files",
        "failure items": "failure_items",
        "request errors": "request_errors",
        "parse errors": "parse_errors",
        "validation failures": "validation_failures",
        "missing item ids": "missing_item_ids",
        "unresolved items": "unresolved_items",
    }
    for match in _REPAIR_SUMMARY_INT_RE.finditer(output):
        key = field_map.get(match.group(1).lower(), match.group(1).lower().replace(" ", "_"))
        parsed[key] = int(match.group(2))
    run_dir_match = _REPAIR_RUN_DIR_RE.search(output)
    if run_dir_match:
        parsed["run_dir"] = run_dir_match.group(1).strip()
    story_match = _STORY_MEMORY_REPAIR_RE.search(output)
    if story_match:
        parsed["story_memory_hits"] = story_match.group(1).strip()
    report_match = re.search(r"^\s*Repair report:\s*(.+?)\s*$", output, re.MULTILINE)
    if report_match:
        parsed["report_path"] = report_match.group(1).strip()
    return parsed


def summarize_repair_output(
    output: str,
    exit_code: int,
    *,
    report_path: str = "",
    manifest_path: str = "",
) -> RepairSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if report_path:
        facts.append(f"修补报告：{report_path}")

    if exit_code != 0:
        return RepairSummary(
            status="failed",
            heading="同步修补失败",
            message="同步修补没有正常完成，请查看诊断日志。",
            facts=facts,
            findings=["修补会直接修改翻译文件；失败时请先在副本上排查。"],
            report_path=report_path,
        )

    parsed = parse_repair_output(output)
    applied_items = parsed.get("applied_items")
    applied_files = parsed.get("applied_files")
    failure_items = parsed.get("failure_items")
    run_dir = parsed.get("run_dir")

    if isinstance(applied_items, int):
        facts.append(f"已修补条目：{applied_items}")
    if isinstance(applied_files, int):
        facts.append(f"已修补文件：{applied_files}")
    if isinstance(failure_items, int) and failure_items > 0:
        facts.append(f"失败条目：{failure_items}")
    if isinstance(run_dir, str) and run_dir:
        facts.append(f"运行目录：{run_dir}")
    if isinstance(parsed.get("story_memory_hits"), str):
        facts.append(f"Story Memory：{parsed['story_memory_hits']}")

    findings = [
        "同步修补已直接写入翻译文件；建议点击「重新检查」更新写回摘要。",
    ]
    if isinstance(failure_items, int) and failure_items > 0:
        findings.append("部分条目修补失败，请查看运行目录中的 repair_failures.jsonl。")

    if isinstance(applied_items, int) and applied_items > 0:
        status = "ok" if not failure_items else "warn"
        heading = "同步修补完成" if status == "ok" else "同步修补部分完成"
        message = (
            "同步修补已完成。请重新检查翻译结果，确认显示「可写回」后再写入项目。"
            if status == "ok"
            else "部分条目已修补，但仍需查看失败明细并重新检查。"
        )
        return RepairSummary(
            status=status,
            heading=heading,
            message=message,
            facts=facts,
            findings=findings,
            report_path=report_path,
            run_dir=run_dir if isinstance(run_dir, str) else "",
            applied_items=applied_items,
            applied_files=applied_files if isinstance(applied_files, int) else None,
            failure_items=failure_items if isinstance(failure_items, int) else None,
        )

    return RepairSummary(
        status="unknown",
        heading="同步修补结果不明确",
        message="命令已结束，但未能识别修补摘要，请查看诊断日志。",
        facts=facts,
        findings=findings,
        report_path=report_path,
        run_dir=run_dir if isinstance(run_dir, str) else "",
    )


def running_repair_summary(
    *,
    report_path: str = "",
    manifest_path: str = "",
) -> RepairSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if report_path:
        facts.append(f"修补报告：{report_path}")
    return RepairSummary(
        status="running",
        heading="正在同步修补",
        message="正在按报告同步修补剩余条目；完成后这里会显示摘要。",
        facts=facts,
        findings=[],
        report_path=report_path,
    )


def repair_summary_to_diagnostics_context(
    summary: RepairSummary,
    base: DiagnosticsContext,
) -> DiagnosticsContext:
    paths = list(base.paths)
    if summary.run_dir:
        paths.append(DiagnosticsPathEntry(label="Repair 运行目录", path=summary.run_dir))
    if summary.report_path:
        paths.append(DiagnosticsPathEntry(label="Repair 报告", path=summary.report_path))

    facts = [*summary.facts, *base.facts]
    if summary.findings:
        facts.extend(summary.findings)

    status = summary.status
    if status == "ok":
        status = "ready"
    elif status in {"warn", "unknown"}:
        status = "warning"
    elif status == "running":
        status = "running"

    message = summary.message
    if base.message and summary.status not in {"running", "failed"}:
        message = f"{summary.message}\n\n{base.message}"

    return DiagnosticsContext(
        status=status,
        heading=summary.heading,
        message=message,
        facts=facts,
        paths=paths,
        commands=base.commands,
        manifest_json_preview=base.manifest_json_preview,
    )