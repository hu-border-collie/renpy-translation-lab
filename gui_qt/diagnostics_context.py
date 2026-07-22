"""Diagnostics context for the advanced diagnostics tab."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

from keyword_glossary_merge import build_merge_keywords_cli_command

from .batch_workflow_support import (
    build_recover_submit_cli_args,
    build_submit_cli_args,
    format_cost_estimate_facts,
    format_non_chinese_rules_facts,
    get_uncertain_submit_kind,
    load_uncertain_submit_facts_from_manifest,
)
from .user_copy import (
    format_job_fact,
    format_job_state_fact,
    format_manifest_path_fact,
    format_package_dir_fact,
    format_safety_fact,
    manifest_mode_label,
)


@dataclass(frozen=True)
class DiagnosticsPathEntry:
    label: str
    path: str


@dataclass(frozen=True)
class DiagnosticsCommand:
    label: str
    command: str


@dataclass(frozen=True)
class DiagnosticsContext:
    status: str
    heading: str
    message: str
    facts: list[str]
    paths: list[DiagnosticsPathEntry]
    commands: list[DiagnosticsCommand]
    manifest_json_preview: str


STANDARD_REPORT_FILES = (
    ("check_failures.jsonl", "检查失败明细"),
    ("failures.jsonl", "翻译失败明细"),
    ("apply_failure_report.json", "写回失败报告"),
    ("requests.jsonl", "批量请求"),
    ("results.jsonl", "批量结果"),
    ("last_status_snapshot.json", "最近状态快照"),
)


def is_windows_style_path(path: str) -> bool:
    return bool(
        re.match(r"^[A-Za-z]:", path)
        or path.startswith("\\\\")
        or "\\" in path
    )


def parent_directory(path: str) -> str:
    if not path:
        return ""
    if is_windows_style_path(path):
        parent = str(PureWindowsPath(path).parent)
    else:
        parent = str(PurePosixPath(path).parent)
    return parent if parent and parent != "." else ""


def join_directory_file(directory: str, filename: str) -> str:
    if is_windows_style_path(directory):
        return str(PureWindowsPath(directory) / filename)
    return str(PurePosixPath(directory) / filename)


def _canonical_compare_path(path: str) -> str:
    if not path:
        return ""
    if is_windows_style_path(path):
        return str(PureWindowsPath(path)).replace("/", "\\").lower()
    return os.path.normcase(os.path.abspath(path))


def _default_path_exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def quote_cli_arg(value: str) -> str:
    if value == "" or any(char in value for char in ' \t"&'):
        return f'"{value}"'
    return value


def format_cli_command(python_exe: str, script_path: str, args: list[str]) -> str:
    parts = [quote_cli_arg(python_exe), quote_cli_arg(script_path)]
    parts.extend(quote_cli_arg(arg) for arg in args)
    return " ".join(parts)


def resolve_package_dir(manifest_path: str, manifest: dict[str, object] | None = None) -> str:
    parent = parent_directory(manifest_path)
    if parent:
        return parent
    if manifest:
        package_dir = manifest.get("_package_dir")
        if isinstance(package_dir, str) and package_dir.strip():
            return package_dir
    return ""


def manifest_for_preview(manifest: dict[str, object]) -> dict[str, object]:
    preview: dict[str, object] = {}
    for key, value in manifest.items():
        if key in {"chunks", "files"}:
            continue
        preview[key] = value

    files = manifest.get("files")
    chunks = manifest.get("chunks")
    summary = manifest.get("summary")
    file_count = len(files) if isinstance(files, dict) else 0
    chunk_count = len(chunks) if isinstance(chunks, list) else 0
    if isinstance(summary, dict):
        if not file_count and isinstance(summary.get("file_count"), int):
            file_count = summary["file_count"]
        if not chunk_count and isinstance(summary.get("chunk_count"), int):
            chunk_count = summary["chunk_count"]
    if file_count or chunk_count:
        preview["_preview_note"] = (
            f"预览已省略条目明细（{chunk_count} 块 / {file_count} 个文件）"
        )
    return preview


def format_manifest_json_preview(
    manifest: dict[str, object],
    *,
    max_chars: int = 48_000,
) -> str:
    if not manifest:
        return ""
    preview = manifest_for_preview(manifest)
    try:
        text = json.dumps(preview, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(preview)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def collect_existing_report_paths(
    package_dir: str,
    manifest: dict[str, object],
    *,
    path_exists: Callable[[str], bool] | None = None,
) -> list[DiagnosticsPathEntry]:
    exists = path_exists or _default_path_exists
    entries: list[DiagnosticsPathEntry] = []
    seen: set[str] = set()

    def add(label: str, path: str) -> None:
        if not path:
            return
        normalized = _canonical_compare_path(path)
        if normalized in seen or not exists(path):
            return
        seen.add(normalized)
        entries.append(DiagnosticsPathEntry(label=label, path=path))

    if package_dir:
        for filename, label in STANDARD_REPORT_FILES:
            add(label, join_directory_file(package_dir, filename))

    report_path = manifest.get("last_check_report_path")
    if isinstance(report_path, str) and report_path.strip():
        add("最近检查报告", report_path.strip())

    return entries


def build_cli_commands(
    *,
    python_exe: str,
    batch_script_path: str,
    manifest_path: str,
    manifest: dict[str, object],
    submit_max_cost: float | None = None,
) -> list[DiagnosticsCommand]:
    if not manifest_path:
        return []

    commands: list[DiagnosticsCommand] = [
        DiagnosticsCommand(
            label="项目检查",
            command=format_cli_command(python_exe, batch_script_path, ["doctor"]),
        ),
        DiagnosticsCommand(
            label="项目分析状态",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["project-analysis-status"],
            ),
        ),
        DiagnosticsCommand(
            label="项目分析·导入关键词摘要",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                [
                    "project-analysis-ingest-keywords",
                    "--summary-jsonl",
                    "path/to/keyword_chunk_summaries.jsonl",
                ],
            ),
        ),
        DiagnosticsCommand(
            label="项目分析·构建结构草稿",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["project-analysis-build-structure"],
            ),
        ),
        DiagnosticsCommand(
            label="项目分析·LLM 生成摘要",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["project-analysis-generate"],
            ),
        ),
        DiagnosticsCommand(
            label="项目分析·发布 brief",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["project-analysis-publish"],
            ),
        ),
        DiagnosticsCommand(
            label="项目分析·撤销发布",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["project-analysis-unpublish"],
            ),
        ),
    ]

    mode = manifest.get("mode")
    mode_text = mode.strip() if isinstance(mode, str) else ""
    if mode_text == "revision":
        commands.extend(
            build_cloud_job_commands(
                python_exe=python_exe,
                batch_script_path=batch_script_path,
                manifest_path=manifest_path,
                manifest=manifest,
                submit_max_cost=submit_max_cost,
                submit_label="提交订正任务",
                status_label="查询订正状态",
                download_label="下载订正结果",
            )
        )
        commands.append(
            DiagnosticsCommand(
                label="预览订正结果",
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    ["preview-revisions", manifest_path],
                ),
            )
        )
        if not manifest.get("revision_applied_at"):
            commands.append(
                DiagnosticsCommand(
                    label="写回订正（预览确认后）",
                    command=format_cli_command(
                        python_exe,
                        batch_script_path,
                        ["apply-revisions", manifest_path],
                    ),
                )
            )
        return commands

    if mode_text == "keyword_extraction":
        commands.extend(
            build_cloud_job_commands(
                python_exe=python_exe,
                batch_script_path=batch_script_path,
                manifest_path=manifest_path,
                manifest=manifest,
                submit_max_cost=submit_max_cost,
                submit_label="提交关键词任务",
                status_label="查询关键词状态",
                download_label="下载关键词结果",
            )
        )
        commands.append(
            DiagnosticsCommand(
                label="导出关键词报告",
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    ["export-keywords", manifest_path],
                ),
            )
        )
        keyword_export = manifest.get("keyword_export")
        if isinstance(keyword_export, dict):
            jsonl_path = keyword_export.get("jsonl_path")
            if isinstance(jsonl_path, str) and jsonl_path.strip():
                commands.append(
                    DiagnosticsCommand(
                        label="合并候选到 glossary（预览）",
                        command=format_cli_command(
                            python_exe,
                            batch_script_path,
                            build_merge_keywords_cli_command(
                                manifest_path,
                                dry_run=True,
                            ),
                        ),
                    )
                )
                commands.append(
                    DiagnosticsCommand(
                        label="合并候选到 glossary",
                        command=format_cli_command(
                            python_exe,
                            batch_script_path,
                            build_merge_keywords_cli_command(manifest_path),
                        ),
                    )
                )
        return commands

    retry_parent = manifest.get("retry_of_manifest")
    if isinstance(retry_parent, str) and retry_parent.strip():
        parent_manifest_path = retry_parent.strip()
        commands.extend(
            build_cloud_job_commands(
                python_exe=python_exe,
                batch_script_path=batch_script_path,
                manifest_path=manifest_path,
                manifest=manifest,
                submit_max_cost=submit_max_cost,
                submit_label="提交补译任务",
                status_label="查询补译状态",
                download_label="下载补译结果",
            )
        )
        commands.extend(
            [
                DiagnosticsCommand(
                    label="合并补译结果",
                    command=format_cli_command(
                        python_exe,
                        batch_script_path,
                        ["merge-retry", parent_manifest_path, manifest_path],
                    ),
                ),
                DiagnosticsCommand(
                    label="重新检查父任务",
                    command=format_cli_command(
                        python_exe,
                        batch_script_path,
                        ["check", parent_manifest_path],
                    ),
                ),
            ]
        )
        return commands

    commands.extend(
        build_cloud_job_commands(
            python_exe=python_exe,
            batch_script_path=batch_script_path,
            manifest_path=manifest_path,
            manifest=manifest,
            submit_max_cost=submit_max_cost,
        )
    )
    commands.append(
        DiagnosticsCommand(
            label="估算提交成本",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["estimate-cost", manifest_path],
            ),
        )
    )
    commands.append(
        DiagnosticsCommand(
            label="检查翻译结果",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["check", manifest_path],
            ),
        )
    )
    from .ab_experiment_report import build_compare_variants_cli_args

    commands.append(
        DiagnosticsCommand(
            label="翻译 A/B 对比（试跑）",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                build_compare_variants_cli_args(
                    manifest_path,
                    "<variants.json>",
                    dry_run=True,
                ),
            ),
        )
    )

    safety_level = manifest_check_safety_level(manifest)
    if not manifest.get("applied_at") and safety_level not in {"warn", "block"}:
        commands.append(
            DiagnosticsCommand(
                label="写回翻译（仅可写回）",
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    ["apply", manifest_path],
                ),
            )
        )

    if safety_level == "warn":
        commands.extend(
            build_warn_remediation_commands(
                python_exe=python_exe,
                batch_script_path=batch_script_path,
                manifest_path=manifest_path,
                manifest=manifest,
                submit_max_cost=submit_max_cost,
            )
        )

    commands.extend(
        build_split_child_submit_commands(
            python_exe=python_exe,
            batch_script_path=batch_script_path,
            manifest=manifest,
            submit_max_cost=submit_max_cost,
        )
    )

    return commands


def build_split_child_submit_commands(
    *,
    python_exe: str,
    batch_script_path: str,
    manifest: dict[str, object],
    submit_max_cost: float | None = None,
) -> list[DiagnosticsCommand]:
    children = manifest.get("split_children")
    if not isinstance(children, list):
        return []

    commands: list[DiagnosticsCommand] = []
    for index, child_path in enumerate(children, start=1):
        if not isinstance(child_path, str) or not child_path.strip():
            continue
        commands.append(
            DiagnosticsCommand(
                label=f"提交拆分包 {index:02d}",
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    build_submit_cli_args(child_path.strip(), submit_max_cost),
                ),
            )
        )
    return commands


def build_cloud_job_commands(
    *,
    python_exe: str,
    batch_script_path: str,
    manifest_path: str,
    manifest: dict[str, object],
    submit_max_cost: float | None = None,
    submit_label: str = "提交批量任务",
    status_label: str = "查询任务状态",
    download_label: str = "下载翻译结果",
) -> list[DiagnosticsCommand]:
    commands: list[DiagnosticsCommand] = []
    if not manifest.get("job_name"):
        uncertain_kind = get_uncertain_submit_kind(manifest_path)
        if uncertain_kind == "job_created_uncommitted":
            commands.append(
                DiagnosticsCommand(
                    label="恢复提交状态",
                    command=format_cli_command(
                        python_exe,
                        batch_script_path,
                        build_recover_submit_cli_args(manifest_path),
                    ),
                )
            )
        commands.append(
            DiagnosticsCommand(
                label=submit_label,
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    build_submit_cli_args(manifest_path, submit_max_cost),
                ),
            )
        )

    commands.extend(
        [
            DiagnosticsCommand(
                label=status_label,
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    ["status", manifest_path],
                ),
            ),
            DiagnosticsCommand(
                label=download_label,
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    ["download", manifest_path],
                ),
            ),
        ]
    )
    return commands


def manifest_check_safety_level(manifest: dict[str, object]) -> str:
    summary = manifest.get("last_check_summary")
    if not isinstance(summary, dict):
        return ""
    safety = summary.get("safety_level")
    return safety.strip().lower() if isinstance(safety, str) else ""


def existing_retry_manifest_path(manifest: dict[str, object]) -> str:
    retry_path = manifest.get("last_retry_manifest_path")
    if isinstance(retry_path, str) and retry_path.strip():
        return retry_path.strip()
    return ""


def build_warn_remediation_commands(
    *,
    python_exe: str,
    batch_script_path: str,
    manifest_path: str,
    manifest: dict[str, object],
    submit_max_cost: float | None = None,
) -> list[DiagnosticsCommand]:
    existing_retry_path = existing_retry_manifest_path(manifest)
    retry_manifest_path = existing_retry_path or "RETRY_MANIFEST_PATH"

    commands: list[DiagnosticsCommand] = []
    if not existing_retry_path:
        commands.append(
            DiagnosticsCommand(
                label="生成补译包",
                command=format_cli_command(
                    python_exe,
                    batch_script_path,
                    ["build-retry", manifest_path],
                ),
            )
        )

    commands.extend(
        [
        DiagnosticsCommand(
            label="提交补译任务",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                build_submit_cli_args(retry_manifest_path, submit_max_cost),
            ),
        ),
        DiagnosticsCommand(
            label="查询补译状态",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["status", retry_manifest_path],
            ),
        ),
        DiagnosticsCommand(
            label="下载补译结果",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["download", retry_manifest_path],
            ),
        ),
        DiagnosticsCommand(
            label="合并补译结果",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["merge-retry", manifest_path, retry_manifest_path],
            ),
        ),
        DiagnosticsCommand(
            label="重新检查翻译结果",
            command=format_cli_command(
                python_exe,
                batch_script_path,
                ["check", manifest_path],
            ),
        ),
        ]
    )
    return commands


def build_manifest_facts(manifest: dict[str, object], manifest_path: str) -> list[str]:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    package_dir = resolve_package_dir(manifest_path, manifest)
    if package_dir:
        facts.append(format_package_dir_fact(package_dir))

    job_name = manifest.get("job_name")
    if isinstance(job_name, str) and job_name.strip():
        facts.append(format_job_fact(job_name.strip()))
    else:
        facts.append("云端任务：尚未提交")

    job_state = manifest.get("job_state")
    if isinstance(job_state, str) and job_state.strip():
        facts.append(format_job_state_fact(job_state.strip()))

    last_summary = manifest.get("last_check_summary")
    if isinstance(last_summary, dict):
        safety = last_summary.get("safety_level")
        if isinstance(safety, str) and safety.strip():
            facts.append(format_safety_fact(safety.strip(), prefix="最近检查"))

    applied_at = manifest.get("applied_at")
    if isinstance(applied_at, str) and applied_at.strip():
        facts.append(f"已写回：{applied_at.strip()}")

    display_name = manifest.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        facts.append(f"显示名称：{display_name.strip()}")

    cost_estimate = manifest.get("cost_estimate")
    if isinstance(cost_estimate, dict):
        facts.extend(format_cost_estimate_facts(cost_estimate))

    if manifest_path and not (isinstance(job_name, str) and job_name.strip()):
        facts.extend(load_uncertain_submit_facts_from_manifest(manifest_path))

    non_chinese_rules = manifest.get("non_chinese_rules")
    if isinstance(non_chinese_rules, dict):
        for fact in format_non_chinese_rules_facts(non_chinese_rules):
            facts.append(f"非中文校验：{fact}")

    mode = manifest.get("mode")
    if isinstance(mode, str) and mode.strip():
        facts.append(f"任务类型：{manifest_mode_label(mode.strip())}")

    return facts


def idle_diagnostics_context() -> DiagnosticsContext:
    return DiagnosticsContext(
        status="idle",
        heading="暂无任务上下文",
        message="开始任务后，这里会显示任务记录、翻译包、云端任务和可复制命令。",
        facts=[],
        paths=[],
        commands=[],
        manifest_json_preview="",
    )


def sync_diagnostics_context(
    *,
    sync_script_path: str,
    python_exe: str = "python",
) -> DiagnosticsContext:
    command = format_cli_command(python_exe, sync_script_path, [])
    return DiagnosticsContext(
        status="ready",
        heading="同步翻译上下文",
        message="同步模式不生成批量任务记录；以下为可手动运行的同步命令。",
        facts=[],
        paths=[],
        commands=[DiagnosticsCommand(label="同步翻译", command=command)],
        manifest_json_preview="",
    )


def build_diagnostics_context(
    *,
    latest_manifest_path: str | None,
    manifest: dict[str, object] | None,
    batch_script_path: str,
    logs_dir: str,
    python_exe: str = "python",
    path_exists: Callable[[str], bool] | None = None,
    submit_max_cost: float | None = None,
) -> DiagnosticsContext:
    exists = path_exists or _default_path_exists

    if not latest_manifest_path and not manifest:
        return idle_diagnostics_context()

    manifest_path = ""
    if manifest:
        stored_path = manifest.get("_manifest_path")
        if isinstance(stored_path, str) and stored_path.strip():
            manifest_path = stored_path.strip()
    if not manifest_path and latest_manifest_path:
        manifest_path = latest_manifest_path

    if not manifest:
        if manifest_path and exists(manifest_path):
            return DiagnosticsContext(
                status="warning",
                heading="无法读取任务记录",
                message="找到了任务记录路径，但内容未能加载。请查看下方原始日志。",
                facts=[format_manifest_path_fact(manifest_path)],
                paths=[],
                commands=[],
                manifest_json_preview="",
            )
        return idle_diagnostics_context()

    package_dir = resolve_package_dir(manifest_path, manifest)
    facts = build_manifest_facts(manifest, manifest_path)

    latest_pointer = join_directory_file(logs_dir, "latest_manifest.txt")
    if exists(latest_pointer):
        facts.append(f"最近任务指针：{latest_pointer}")

    paths = collect_existing_report_paths(package_dir, manifest, path_exists=exists)
    commands = build_cli_commands(
        python_exe=python_exe,
        batch_script_path=batch_script_path,
        manifest_path=manifest_path,
        manifest=manifest,
        submit_max_cost=submit_max_cost,
    )
    preview = format_manifest_json_preview(manifest)

    status = "ready"
    message = "以下为当前任务记录对应的路径与可手动运行的命令。"
    if manifest_path and latest_manifest_path:
        if _canonical_compare_path(manifest_path) != _canonical_compare_path(
            latest_manifest_path
        ):
            status = "warning"
            message = "当前任务与最近任务记录不一致；下方预览以当前加载的记录为准。"

    return DiagnosticsContext(
        status=status,
        heading="当前任务上下文",
        message=message,
        facts=facts,
        paths=paths,
        commands=commands,
        manifest_json_preview=preview,
    )