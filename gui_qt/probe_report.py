"""Probe smoke-test summaries and CLI helpers for the diagnostics tab."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .diagnostics_context import DiagnosticsContext, DiagnosticsPathEntry
from .user_copy import format_manifest_path_fact

_MANIFEST_MODE_TRANSLATION = "translation"
_PROBE_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*-\s*(sample_count|parse_ok|full_item_match|max_tokens|missing_text|request_errors):\s*(-?\d+)\s*$",
    re.MULTILINE,
)
_PROBE_FILE_RE = re.compile(
    r"^\s*-\s*(summary_file|results_file):\s*(.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ProbeSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    manifest_path: str = ""
    sample_count: int | None = None
    parse_ok: int | None = None
    full_item_match: int | None = None
    request_errors: int | None = None
    summary_file: str = ""
    results_file: str = ""


def translation_probe_ready(
    manifest_path: str,
    manifest: dict[str, object] | None,
) -> tuple[bool, str]:
    if not manifest_path.strip():
        return False, "没有可探测的翻译任务记录。"
    if manifest is None:
        return False, "无法读取任务记录，请先在诊断页刷新上下文。"
    mode = manifest.get("mode")
    mode_text = mode.strip() if isinstance(mode, str) else _MANIFEST_MODE_TRANSLATION
    if mode_text != _MANIFEST_MODE_TRANSLATION:
        return False, "试跑样本请求仅支持批量翻译任务记录。"
    version = manifest.get("version", 1)
    if version != 1:
        return False, "当前仅支持 version 1 的翻译任务记录。"
    input_jsonl = manifest.get("input_jsonl_path")
    if not isinstance(input_jsonl, str) or not input_jsonl.strip():
        return False, "任务记录缺少 requests.jsonl 路径，无法试跑样本请求。"
    return True, ""


def build_probe_cli_args(
    manifest_path: str,
    *,
    limit: int = 3,
    offset: int = 0,
    api_key_index: int | None = None,
) -> list[str]:
    args = ["probe", manifest_path, "--limit", str(limit), "--offset", str(offset)]
    if api_key_index is not None:
        args.extend(["--api-key-index", str(api_key_index)])
    return args


def parse_probe_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for match in _PROBE_SUMMARY_PREFIX_RE.finditer(output):
        parsed[match.group(1)] = int(match.group(2))
    for match in _PROBE_FILE_RE.finditer(output):
        parsed[match.group(1)] = match.group(2).strip()
    return parsed


def summarize_probe_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
) -> ProbeSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    if exit_code != 0:
        return ProbeSummary(
            status="failed",
            heading="样本试跑失败",
            message="试跑样本请求没有正常完成，请查看诊断日志中的 API 或格式错误。",
            facts=facts,
            findings=[],
            manifest_path=manifest_path,
        )

    parsed = parse_probe_output(output)
    sample_count = parsed.get("sample_count")
    parse_ok = parsed.get("parse_ok")
    full_item_match = parsed.get("full_item_match")
    max_tokens = parsed.get("max_tokens")
    missing_text = parsed.get("missing_text")
    request_errors = parsed.get("request_errors")
    summary_file = parsed.get("summary_file")
    results_file = parsed.get("results_file")

    if isinstance(sample_count, int):
        facts.append(f"探测条数：{sample_count}")
    if isinstance(parse_ok, int):
        facts.append(f"解析成功：{parse_ok}")
    if isinstance(full_item_match, int):
        facts.append(f"完整匹配：{full_item_match}")
    if isinstance(max_tokens, int) and max_tokens > 0:
        facts.append(f"触发 MAX_TOKENS：{max_tokens}")
    if isinstance(missing_text, int) and missing_text > 0:
        facts.append(f"缺少响应文本：{missing_text}")
    if isinstance(request_errors, int) and request_errors > 0:
        facts.append(f"请求错误：{request_errors}")
    if isinstance(summary_file, str) and summary_file:
        facts.append(f"摘要文件：{summary_file}")
    if isinstance(results_file, str) and results_file:
        facts.append(f"结果文件：{results_file}")

    findings: list[str] = []
    if isinstance(request_errors, int) and request_errors > 0:
        findings.append("部分样本请求失败，请检查 API Key、模型配额或网络。")
    if isinstance(parse_ok, int) and isinstance(sample_count, int) and parse_ok < sample_count:
        findings.append("部分样本未能解析为有效 JSON，请检查请求格式或模型输出。")
    if isinstance(full_item_match, int) and isinstance(sample_count, int) and full_item_match < sample_count:
        findings.append("部分样本返回的条目数与预期不一致。")

    if (
        isinstance(sample_count, int)
        and sample_count > 0
        and isinstance(parse_ok, int)
        and parse_ok == sample_count
        and isinstance(request_errors, int)
        and request_errors == 0
    ):
        return ProbeSummary(
            status="ok",
            heading="样本试跑通过",
            message="同步样本请求已完成，API 与请求格式看起来正常，可继续提交批量任务。",
            facts=facts,
            findings=findings,
            manifest_path=manifest_path,
            sample_count=sample_count,
            parse_ok=parse_ok,
            full_item_match=full_item_match if isinstance(full_item_match, int) else None,
            request_errors=request_errors,
            summary_file=summary_file if isinstance(summary_file, str) else "",
            results_file=results_file if isinstance(results_file, str) else "",
        )

    if isinstance(sample_count, int) and sample_count > 0:
        return ProbeSummary(
            status="warn",
            heading="样本试跑需关注",
            message="试跑已完成，但部分样本存在问题；提交批量任务前请先查看诊断日志。",
            facts=facts,
            findings=findings,
            manifest_path=manifest_path,
            sample_count=sample_count,
            parse_ok=parse_ok if isinstance(parse_ok, int) else None,
            full_item_match=full_item_match if isinstance(full_item_match, int) else None,
            request_errors=request_errors if isinstance(request_errors, int) else None,
            summary_file=summary_file if isinstance(summary_file, str) else "",
            results_file=results_file if isinstance(results_file, str) else "",
        )

    return ProbeSummary(
        status="unknown",
        heading="样本试跑结果不明确",
        message="命令已结束，但未能识别探测摘要，请查看诊断日志。",
        facts=facts,
        findings=findings,
        manifest_path=manifest_path,
    )


def running_probe_summary(*, manifest_path: str = "") -> ProbeSummary:
    facts = [format_manifest_path_fact(manifest_path)] if manifest_path else []
    return ProbeSummary(
        status="running",
        heading="正在试跑样本请求",
        message="正在对少量请求做同步冒烟测试；完成后这里会显示摘要。",
        facts=facts,
        findings=[],
        manifest_path=manifest_path,
    )


def probe_summary_to_diagnostics_context(
    summary: ProbeSummary,
    base: DiagnosticsContext,
) -> DiagnosticsContext:
    paths = list(base.paths)
    if summary.summary_file:
        paths.append(DiagnosticsPathEntry(label="Probe 摘要", path=summary.summary_file))
    if summary.results_file:
        paths.append(DiagnosticsPathEntry(label="Probe 结果", path=summary.results_file))

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