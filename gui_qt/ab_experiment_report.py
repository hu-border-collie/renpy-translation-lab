"""Translation A/B experiment summaries and CLI helpers for the diagnostics tab."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass

from .bootstrap_report import coerce_bool
from .diagnostics_context import DiagnosticsContext, DiagnosticsPathEntry
from .user_copy import format_manifest_path_fact

_MANIFEST_MODE_TRANSLATION = "translation"
_COMPARE_VARIANTS_LINE_RE = re.compile(
    r"^\s*-\s*(output_dir|chunks|variants|dry_run|report|results|settings|experiment_error):\s*(.+?)\s*$",
    re.MULTILINE,
)

BASELINE_VARIANT = {"name": "baseline", "overrides": {}}

AB_VARIANT_CHOICE_SKIP = "skip"
AB_VARIANT_CHOICE_FORCE_ON = "force_on"
AB_VARIANT_CHOICE_FORCE_OFF = "force_off"

AB_VARIANT_DIMENSIONS: tuple[dict[str, object], ...] = (
    {
        "id": "story_memory",
        "label": "Story Memory",
        "overrides_key": "story_memory",
    },
    {
        "id": "rag",
        "label": "RAG",
        "overrides_key": "rag",
    },
    {
        "id": "source_index",
        "label": "原文索引",
        "overrides_key": "source_index",
    },
)


def read_ab_dimension_enabled_states(config: dict[str, object]) -> dict[str, bool]:
    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
    states: dict[str, bool] = {}
    for dimension in AB_VARIANT_DIMENSIONS:
        dimension_id = str(dimension["id"])
        overrides_key = str(dimension["overrides_key"])
        section = batch.get(overrides_key)
        if not isinstance(section, dict):
            section = {}
        states[dimension_id] = coerce_bool(section.get("enabled"), False)
    return states


def build_variants_from_gui_selection(dimension_choices: dict[str, str]) -> list[dict]:
    variants = [dict(BASELINE_VARIANT)]
    for dimension in AB_VARIANT_DIMENSIONS:
        dimension_id = str(dimension["id"])
        choice = dimension_choices.get(dimension_id, AB_VARIANT_CHOICE_SKIP)
        if choice == AB_VARIANT_CHOICE_SKIP:
            continue
        overrides_key = str(dimension["overrides_key"])
        enabled = choice == AB_VARIANT_CHOICE_FORCE_ON
        variants.append(
            {
                "name": f"{dimension_id}_{'on' if enabled else 'off'}",
                "overrides": {"batch": {overrides_key: {"enabled": enabled}}},
            },
        )
    return variants


def validate_ab_experiment_variants(variants: list[dict]) -> tuple[bool, str]:
    if len(variants) < 2:
        return False, (
            "请至少选择一个对比项（将某维度设为「强制开启」或「强制关闭」）；"
            "baseline（当前配置）会自动包含。"
        )
    names = [str(entry.get("name") or "") for entry in variants]
    if len(set(names)) != len(names):
        return False, "变体名称重复，请调整勾选项。"
    return True, ""


def write_variants_to_temp_file(variants: list[dict]) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="ab_variants_",
        delete=False,
    )
    with handle:
        json.dump(variants, handle, ensure_ascii=False, indent=2)
        return handle.name


def format_variant_names(variants: list[dict]) -> str:
    return ", ".join(str(entry.get("name") or "") for entry in variants if entry.get("name"))


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


def manifest_chunk_count(manifest: dict[str, object]) -> int:
    chunks = manifest.get("chunks")
    if isinstance(chunks, list) and chunks:
        return len(chunks)
    summary = manifest.get("summary")
    if isinstance(summary, dict):
        chunk_count = summary.get("chunk_count")
        if isinstance(chunk_count, int) and chunk_count > 0:
            return chunk_count
    return 0


def translation_ab_experiment_ready(
    manifest_path: str,
    manifest: dict[str, object] | None,
) -> tuple[bool, str]:
    if not manifest_path.strip():
        return False, "没有可对比的翻译任务记录。"
    if manifest is None:
        return False, "无法读取任务记录，请先在诊断与工具刷新上下文。"
    mode = manifest.get("mode")
    mode_text = mode.strip() if isinstance(mode, str) else _MANIFEST_MODE_TRANSLATION
    if mode_text != _MANIFEST_MODE_TRANSLATION:
        return False, "翻译 A/B 对比仅支持批量翻译任务记录。"
    if manifest_chunk_count(manifest) <= 0:
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


def resolve_ab_settings_path(
    report_path: str,
    results_path: str,
    settings_path: str = "",
) -> str:
    candidates: list[str] = []
    if isinstance(settings_path, str) and settings_path.strip():
        candidates.append(settings_path.strip())
    for base_path in (report_path, results_path):
        if isinstance(base_path, str) and base_path.strip():
            candidates.append(os.path.join(os.path.dirname(base_path), "ab_settings.json"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return candidates[0] if candidates else ""


def collect_ab_experiment_issues(
    *,
    settings_path: str = "",
    results_path: str = "",
    expected_variant_count: int | None = None,
    stdout_experiment_error: str = "",
) -> tuple[str, list[str]]:
    findings: list[str] = []
    severity = ""

    if isinstance(stdout_experiment_error, str) and stdout_experiment_error.strip():
        findings.append(f"实验中断：{stdout_experiment_error.strip()}")
        severity = "failed"

    if settings_path and os.path.isfile(settings_path):
        try:
            with open(settings_path, encoding="utf-8") as handle:
                settings = json.load(handle)
            experiment_error = settings.get("experiment_error")
            if isinstance(experiment_error, str) and experiment_error.strip():
                message = f"实验中断：{experiment_error.strip()}"
                if message not in findings:
                    findings.append(message)
                severity = "failed"
        except (OSError, json.JSONDecodeError):
            findings.append("无法读取 ab_settings.json。")
            if severity != "failed":
                severity = "warn"

    if results_path and os.path.isfile(results_path):
        variant_errors: list[str] = []
        max_variants_seen = 0
        try:
            with open(results_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    variants = row.get("variants")
                    if not isinstance(variants, list):
                        continue
                    max_variants_seen = max(max_variants_seen, len(variants))
                    chunk_key = str(row.get("chunk_key") or "")
                    for variant in variants:
                        if not isinstance(variant, dict):
                            continue
                        error = variant.get("error")
                        name = str(variant.get("name") or "")
                        if isinstance(error, str) and error.strip():
                            label = "/".join(part for part in (chunk_key, name) if part) or "variant"
                            variant_errors.append(f"{label}: {error.strip()}")
        except OSError:
            findings.append("无法读取 ab_results.jsonl。")
            if severity != "failed":
                severity = "warn"
        else:
            if variant_errors:
                preview = "; ".join(variant_errors[:3])
                if len(variant_errors) > 3:
                    preview += f"；另有 {len(variant_errors) - 3} 条，见 ab_results.jsonl"
                findings.append(f"部分变体出错：{preview}")
                if severity != "failed":
                    severity = "warn"
            if (
                isinstance(expected_variant_count, int)
                and expected_variant_count > 0
                and max_variants_seen > 0
                and max_variants_seen < expected_variant_count
            ):
                findings.append(
                    f"仅完成 {max_variants_seen}/{expected_variant_count} 个变体，报告可能不完整。",
                )
                if severity != "failed":
                    severity = "warn"

    return severity, findings


def summarize_compare_variants_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
    variant_names: str = "",
) -> AbExperimentSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if variant_names:
        facts.append(f"对比变体：{variant_names}")

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
    stdout_experiment_error = parsed.get("experiment_error")
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
        return AbExperimentSummary(
            status="warn",
            heading="翻译 A/B 报告未找到",
            message="命令已结束，但报告文件尚未找到，请查看诊断日志。",
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

    if isinstance(report_path, str) and report_path and os.path.isfile(report_path):
        resolved_settings_path = resolve_ab_settings_path(
            report_path,
            results_path if isinstance(results_path, str) else "",
            settings_path if isinstance(settings_path, str) else "",
        )
        issue_severity, issue_findings = collect_ab_experiment_issues(
            settings_path=resolved_settings_path,
            results_path=results_path if isinstance(results_path, str) else "",
            expected_variant_count=variant_count if isinstance(variant_count, int) else None,
            stdout_experiment_error=(
                stdout_experiment_error
                if isinstance(stdout_experiment_error, str)
                else ""
            ),
        )
        findings.extend(issue_findings)

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

        status = "ok"
        if issue_severity == "failed":
            status = "failed"
            heading = "翻译 A/B 对比失败"
            message = "实验报告已生成，但部分变体未完成或实验中断，请查看诊断日志。"
        elif issue_severity == "warn":
            status = "warn"
            message = f"{message} 部分变体可能不完整，请查看下方详情。"

        return AbExperimentSummary(
            status=status,
            heading=heading,
            message=message,
            facts=facts,
            findings=findings,
            manifest_path=manifest_path,
            output_dir=output_dir if isinstance(output_dir, str) else "",
            report_path=report_path,
            results_path=results_path if isinstance(results_path, str) else "",
            settings_path=resolved_settings_path,
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
    variant_names: str = "",
    dry_run: bool = False,
) -> AbExperimentSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))
    if variant_names:
        facts.append(f"对比变体：{variant_names}")
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