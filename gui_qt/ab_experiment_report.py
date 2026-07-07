"""Translation A/B experiment summaries and CLI helpers for the diagnostics tab."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .diagnostics_context import DiagnosticsContext, DiagnosticsPathEntry
from .user_copy import format_manifest_path_fact

_MANIFEST_MODE_TRANSLATION = "translation"
_COMPARE_VARIANTS_LINE_RE = re.compile(
    r"^\s*-\s*(output_dir|chunks|variants|dry_run|report|results|settings):\s*(.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class AbExperimentSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    manifest_path: str = ""
    output_dir: str = ""
    report_path: str = ""
    results_path: str = ""
    settings_path: str = ""
    chunk_count: int | None = None
    variant_count: int | None = None
    dry_run: bool | None = None


def translation_ab_experiment_ready(
    manifest_path: str,
    manifest: dict[str, object] | None,
) -> tuple[bool, str]:
    if not manifest_path.strip():
        return False, "没有可对比的翻译任务记录。"
    if manifest is None:
        return False, "无法读取任务记录，请先在诊断页刷新上下文。"
    mode = manifest.get("mode")
    mode_text = mode.strip() if isinstance(mode, str) else _MANIFEST_MODE_TRANSLATION
    if mode_text != _MANIFEST_MODE_TRANSLATION:
        return False, "翻译 A/B 对比仅支持批量翻译任务记录。"
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return False, "任务记录中没有可采样的翻译块。"
    return True, ""


def build_compare_variants_cli_args(
    manifest_path: str,
    variants_file: str,
    *,
    limit: int = 3,
    offset: int = 0,
    output_dir: str = "",
    dry_run: bool = False,
    api_key_index: int | None = None,
) -> list[str]:
    args = [
        "compare-variants",
        manifest_path,
        "--variants-file",
        variants_file,
        "--limit",
        str(limit),
        "--offset",
        str(offset),
    ]
    if output_dir.strip():
        args.extend(["--output-dir", output_dir.strip()])
    if dry_run:
        args.append("--dry-run")
    if api_key_index is not None:
        args.extend(["--api-key-index", str(api_key_index)])
    return args


def parse_compare_variants_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for match in _COMPARE_VARIANTS_LINE_RE.finditer(output):
        key = match.group(1)
        value = match.group(2).strip()
        if key in {"chunks", "variants"}:
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = value
        elif key == "dry_run":
            parsed[key] = value.lower() in {"true", "1", "yes"}
        else:
            parsed[key] = value
    return parsed


def summarize_compare_variants_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
    variants_file: str = "",
) -> AbExperimentSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if variants_file:
        facts.append(f"变体文件：{variants_file}")

    if exit_code != 0:
        return AbExperimentSummary(
            status="failed",
            heading="翻译 A/B 对比失败",
            message="实验没有正常完成，请查看诊断日志中的配置或 API 错误。",
            facts=facts,
            findings=[],
            manifest_path=manifest_path,
        )

    parsed = parse_compare_variants_output(output)
    output_dir = parsed.get("output_dir")
    report_path = parsed.get("report")
    results_path = parsed.get("results")
    settings_path = parsed.get("settings")
    chunk_count = parsed.get("chunks")
    variant_count = parsed.get("variants")
    dry_run = parsed.get("dry_run")

    if isinstance(output_dir, str) and output_dir:
        facts.append(f"输出目录：{output_dir}")
    if isinstance(chunk_count, int):
        facts.append(f"采样块数：{chunk_count}")
    if isinstance(variant_count, int):
        facts.append(f"变体数：{variant_count}")
    if isinstance(dry_run, bool):
        facts.append(f"试跑模式：{'是' if dry_run else '否'}")
    if isinstance(report_path, str) and report_path:
        facts.append(f"报告：{report_path}")

    findings: list[str] = []
    if isinstance(report_path, str) and report_path and not os.path.isfile(report_path):
        findings.append("命令已结束，但报告文件尚未找到。")

    if isinstance(report_path, str) and report_path and os.path.isfile(report_path):
        if isinstance(dry_run, bool) and dry_run:
            message = (
                "试跑已完成：各变体 prompt 已重建并写入报告，未调用翻译 API。"
                "确认配置无误后可取消“仅试跑”再正式对比。"
            )
            heading = "翻译 A/B 试跑完成"
        else:
            message = (
                "实验已完成：请打开并排 Markdown 报告，人工比较各变体译文与配置差异。"
            )
            heading = "翻译 A/B 对比完成"
        return AbExperimentSummary(
            status="ok",
            heading=heading,
            message=message,
            facts=facts,
            findings=findings,
            manifest_path=manifest_path,
            output_dir=output_dir if isinstance(output_dir, str) else "",
            report_path=report_path,
            results_path=results_path if isinstance(results_path, str) else "",
            settings_path=settings_path if isinstance(settings_path, str) else "",
            chunk_count=chunk_count if isinstance(chunk_count, int) else None,
            variant_count=variant_count if isinstance(variant_count, int) else None,
            dry_run=dry_run if isinstance(dry_run, bool) else None,
        )

    return AbExperimentSummary(
        status="unknown",
        heading="翻译 A/B 结果不明确",
        message="命令已结束，但未能识别实验摘要，请查看诊断日志。",
        facts=facts,
        findings=findings,
        manifest_path=manifest_path,
    )


def running_ab_experiment_summary(
    *,
    manifest_path: str = "",
    variants_file: str = "",
    dry_run: bool = False,
) -> AbExperimentSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if variants_file:
        facts.append(f"变体文件：{variants_file}")
    if dry_run:
        facts.append("试跑模式：是")

    if dry_run:
        message = "正在重建各变体 prompt 并生成报告，不会调用翻译 API。"
        heading = "正在试跑翻译 A/B 对比"
    else:
        message = "正在对多个配置变体执行同步翻译；完成后这里会显示报告路径。"
        heading = "正在运行翻译 A/B 对比"

    return AbExperimentSummary(
        status="running",
        heading=heading,
        message=message,
        facts=facts,
        findings=[],
        manifest_path=manifest_path,
        dry_run=dry_run,
    )


def ab_experiment_summary_to_diagnostics_context(
    summary: AbExperimentSummary,
    base: DiagnosticsContext,
) -> DiagnosticsContext:
    paths = list(base.paths)
    if summary.report_path:
        paths.append(DiagnosticsPathEntry(label="A/B 报告", path=summary.report_path))
    if summary.results_path:
        paths.append(DiagnosticsPathEntry(label="A/B 结果", path=summary.results_path))
    if summary.settings_path:
        paths.append(DiagnosticsPathEntry(label="A/B 设置", path=summary.settings_path))
    if summary.output_dir and not any(entry.path == summary.output_dir for entry in paths):
        paths.append(DiagnosticsPathEntry(label="A/B 输出目录", path=summary.output_dir))

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