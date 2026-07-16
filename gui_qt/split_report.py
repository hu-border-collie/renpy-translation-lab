"""Split package summaries and CLI helpers for the diagnostics tab."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .diagnostics_context import DiagnosticsContext, DiagnosticsPathEntry
from .user_copy import format_manifest_path_fact

_MANIFEST_MODE_TRANSLATION = "translation"
_CREATED_SPLIT_PACKAGE_RE = re.compile(
    r"^\s*Created split package:\s*(.+?)\s*$",
    re.MULTILINE,
)
_SOURCE_MANIFEST_RE = re.compile(
    r"^\s*Source manifest updated:\s*(.+?)\s*$",
    re.MULTILINE,
)
_LATEST_SPLIT_MANIFEST_RE = re.compile(
    r"^\s*Latest manifest set to first split package:\s*(.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SplitSummary:
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]
    manifest_path: str = ""
    source_manifest_path: str = ""
    latest_manifest_path: str = ""
    child_manifest_paths: list[str] | None = None


def translation_split_ready(
    manifest_path: str,
    manifest: dict[str, object] | None,
) -> tuple[bool, str]:
    if not manifest_path.strip():
        return False, "没有可拆分的翻译任务记录。"
    if manifest is None:
        return False, "无法读取任务记录，请先在诊断与工具刷新上下文。"
    mode = manifest.get("mode")
    mode_text = mode.strip() if isinstance(mode, str) else _MANIFEST_MODE_TRANSLATION
    if mode_text != _MANIFEST_MODE_TRANSLATION:
        return False, "拆分翻译包仅支持批量翻译任务记录。"
    chunks = manifest.get("chunks")
    summary = manifest.get("summary")
    summary_chunk_count = (
        summary.get("chunk_count", 0)
        if isinstance(summary, dict)
        else 0
    )
    has_chunks = isinstance(chunks, list) and bool(chunks)
    if not has_chunks and not (
        isinstance(summary_chunk_count, int) and summary_chunk_count > 0
    ):
        return False, "任务记录中没有可拆分的块。"
    input_jsonl = manifest.get("input_jsonl_path")
    if not isinstance(input_jsonl, str) or not input_jsonl.strip():
        return False, "任务记录缺少 requests.jsonl 路径，无法拆分翻译包。"
    return True, ""


def build_split_cli_args(
    manifest_path: str,
    *,
    max_chunks: int = 600,
    max_items: int = 0,
    display_name_prefix: str = "",
) -> list[str]:
    args = ["split", manifest_path, "--max-chunks", str(max_chunks)]
    if max_items > 0:
        args.extend(["--max-items", str(max_items)])
    if display_name_prefix.strip():
        args.extend(["--display-name-prefix", display_name_prefix.strip()])
    return args


def parse_split_output(output: str) -> dict[str, object]:
    child_dirs = [
        match.group(1).strip()
        for match in _CREATED_SPLIT_PACKAGE_RE.finditer(output)
    ]
    child_manifest_paths = [
        _join_manifest_path(part_dir)
        for part_dir in child_dirs
        if part_dir
    ]
    source_match = _SOURCE_MANIFEST_RE.search(output)
    latest_match = _LATEST_SPLIT_MANIFEST_RE.search(output)
    return {
        "unchanged": "Split not needed; current package already fits the requested limits." in output,
        "child_manifest_paths": child_manifest_paths,
        "source_manifest_path": source_match.group(1).strip() if source_match else "",
        "latest_manifest_path": latest_match.group(1).strip() if latest_match else "",
    }


def _join_manifest_path(package_dir: str) -> str:
    normalized = package_dir.rstrip("/\\")
    if normalized.lower().endswith("manifest.json"):
        return normalized
    separator = "\\" if "\\" in normalized and "/" not in normalized else "/"
    return f"{normalized}{separator}manifest.json"


def summarize_split_output(
    output: str,
    exit_code: int,
    *,
    manifest_path: str = "",
) -> SplitSummary:
    facts: list[str] = []
    if manifest_path:
        facts.append(format_manifest_path_fact(manifest_path))

    if exit_code != 0:
        return SplitSummary(
            status="failed",
            heading="拆分翻译包失败",
            message="拆分命令没有正常完成，请查看诊断日志。",
            facts=facts,
            findings=[],
            manifest_path=manifest_path,
        )

    parsed = parse_split_output(output)
    if parsed.get("unchanged"):
        return SplitSummary(
            status="unchanged",
            heading="无需拆分",
            message="当前翻译包已符合所选上限，无需生成子包。",
            facts=facts,
            findings=[],
            manifest_path=manifest_path,
        )

    child_paths = parsed.get("child_manifest_paths")
    child_manifest_paths = (
        [path for path in child_paths if isinstance(path, str) and path.strip()]
        if isinstance(child_paths, list)
        else []
    )
    source_manifest_path = parsed.get("source_manifest_path")
    latest_manifest_path = parsed.get("latest_manifest_path")

    if child_manifest_paths:
        facts.append(f"生成子包：{len(child_manifest_paths)} 个")
        for index, child_path in enumerate(child_manifest_paths, start=1):
            facts.append(f"子包 {index:02d}：{child_path}")
    if isinstance(source_manifest_path, str) and source_manifest_path:
        facts.append(f"源任务记录：{source_manifest_path}")
    if isinstance(latest_manifest_path, str) and latest_manifest_path:
        facts.append(f"最近任务指针：{latest_manifest_path}")

    findings = [
        "拆分后 RAG 记忆库为静态快照；各子包需分别 submit，不会自动提交。",
        "详见 docs/context_systems.md 中 split 与 RAG 说明。",
    ]

    if not child_manifest_paths:
        return SplitSummary(
            status="unknown",
            heading="拆分结果不明确",
            message="命令已结束，但未能识别子包路径，请查看诊断日志。",
            facts=facts,
            findings=findings,
            manifest_path=manifest_path,
            source_manifest_path=source_manifest_path if isinstance(source_manifest_path, str) else "",
            latest_manifest_path=latest_manifest_path if isinstance(latest_manifest_path, str) else "",
        )

    return SplitSummary(
        status="ok",
        heading="拆分翻译包完成",
        message=(
            f"已生成 {len(child_manifest_paths)} 个子包。"
            "可在命令参考中复制各子包的 submit 命令，或在工作台查看拆分包状态。"
        ),
        facts=facts,
        findings=findings,
        manifest_path=manifest_path,
        source_manifest_path=source_manifest_path if isinstance(source_manifest_path, str) else "",
        latest_manifest_path=latest_manifest_path if isinstance(latest_manifest_path, str) else "",
        child_manifest_paths=child_manifest_paths,
    )


def running_split_summary(*, manifest_path: str = "") -> SplitSummary:
    facts = [format_manifest_path_fact(manifest_path)] if manifest_path else []
    return SplitSummary(
        status="running",
        heading="正在拆分翻译包",
        message="正在按所选上限拆分子包；完成后这里会显示子包列表。",
        facts=facts,
        findings=[],
        manifest_path=manifest_path,
    )


def split_summary_to_diagnostics_context(
    summary: SplitSummary,
    base: DiagnosticsContext,
) -> DiagnosticsContext:
    paths = list(base.paths)
    for index, child_path in enumerate(summary.child_manifest_paths or [], start=1):
        paths.append(DiagnosticsPathEntry(label=f"拆分子包 {index:02d}", path=child_path))
    if summary.source_manifest_path:
        paths.append(
            DiagnosticsPathEntry(label="拆分源任务记录", path=summary.source_manifest_path)
        )

    facts = [*summary.facts, *base.facts]
    if summary.findings:
        facts.extend(summary.findings)

    status = summary.status
    if status == "ok":
        status = "ready"
    elif status in {"unchanged", "unknown"}:
        status = "warning"
    elif status == "running":
        status = "running"
    elif status == "failed":
        status = "failed"

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