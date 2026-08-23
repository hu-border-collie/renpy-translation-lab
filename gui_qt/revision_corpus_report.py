"""Machine-contract summary for the GUI revision-corpus export.

The GUI consumes the versioned ``export-revision-corpus --output json``
envelope and the structured corpus manifest it names.  It deliberately does
not scan Ren'Py files or infer export results from human-readable stdout.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cli_contract

from .translation_workflow import WorkflowUpdate
from .user_copy import REVISION_CORPUS_COPY


CORPUS_COMMAND = "export-revision-corpus"


@dataclass(frozen=True)
class RevisionCorpusExportResult:
    """Structured artifact summary rendered by the revision page."""

    status: str
    output_dir: str = ""
    jsonl_path: str = ""
    markdown_path: str = ""
    manifest_path: str = ""
    item_count: int = 0
    file_count: int = 0
    created_at: str = ""
    source_changed_during_scan: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_paths(self) -> bool:
        """Whether at least one named artifact path is available."""

        return bool(
            self.output_dir
            or self.jsonl_path
            or self.markdown_path
            or self.manifest_path
        )


def load_revision_corpus_manifest(path: str) -> dict[str, Any] | None:
    """Read a corpus manifest artifact for metadata such as ``created_at``.

    A missing or malformed manifest is returned as ``None`` so the GUI can
    still show the paths from the validated CLI envelope and tell the user
    what to repair.  No project source file is read here.
    """

    candidate = str(path or "").strip()
    if not candidate:
        return None
    try:
        with Path(candidate).open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _as_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _path_from(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _failure_update(
    heading: str,
    message: str,
    *,
    facts: list[str] | None = None,
    status: str = "failed",
) -> tuple[WorkflowUpdate, None]:
    return (
        WorkflowUpdate(
            status=status,
            heading=heading,
            message=message,
            facts=list(facts or []),
        ),
        None,
    )


def _error_update(
    envelope: Mapping[str, Any],
    exit_code: int,
) -> tuple[WorkflowUpdate, None]:
    error = envelope.get("error")
    error_map = error if isinstance(error, Mapping) else {}
    code = str(error_map.get("code") or "UNKNOWN_ERROR").strip()
    detail = str(error_map.get("message") or "导出命令未能完成。").strip()
    suggested_action = str(error_map.get("suggested_action") or "").strip()
    facts = [f"错误码：{code}"]
    if detail:
        facts.append(f"命令提示：{detail}")
    if suggested_action:
        facts.append(f"建议动作：{suggested_action}")
    if exit_code:
        facts.append(f"语义退出码：{exit_code}")
    return _failure_update(
        "润色语料导出失败",
        "导出没有完成；请先按提示修正项目或输出目录权限，再重试。",
        facts=facts,
    )


def summarize_revision_corpus_output(
    output: str,
    exit_code: int,
) -> tuple[WorkflowUpdate, RevisionCorpusExportResult | None]:
    """Summarize one CLI JSON envelope without parsing free-form stdout."""

    try:
        envelope = cli_contract.parse_result_envelope(output)
    except ValueError:
        return _failure_update(
            "润色语料导出失败",
            REVISION_CORPUS_COPY["invalid_result_message"],
            facts=["请检查运行日志中的 CLI 错误，并确认使用了 --output json。"],
        )

    if envelope.get("command") != CORPUS_COMMAND:
        return _failure_update(
            "润色语料导出失败",
            REVISION_CORPUS_COPY["invalid_result_message"],
            facts=[f"收到的命令：{envelope.get('command') or '未知'}"],
        )
    if exit_code != 0 or envelope.get("ok") is not True:
        return _error_update(envelope, exit_code)

    result = envelope.get("result")
    artifacts = envelope.get("artifacts")
    result_map = result if isinstance(result, Mapping) else {}
    artifact_map = artifacts if isinstance(artifacts, Mapping) else {}
    manifest_path = _path_from(
        result_map.get("corpus_manifest"),
        artifact_map.get("corpus_manifest"),
    )
    jsonl_path = _path_from(
        result_map.get("corpus_jsonl"),
        artifact_map.get("corpus_jsonl"),
    )
    markdown_path = _path_from(
        result_map.get("corpus_markdown"),
        artifact_map.get("corpus_markdown"),
    )
    output_dir = _path_from(result_map.get("output_dir"))
    if not output_dir and manifest_path:
        output_dir = str(Path(manifest_path).parent)

    missing = [
        label
        for label, path in (
            ("JSONL", jsonl_path),
            ("Markdown", markdown_path),
            ("manifest", manifest_path),
        )
        if not path
    ]
    if missing:
        return _failure_update(
            "润色语料导出失败",
            REVISION_CORPUS_COPY["missing_artifact_message"],
            facts=[f"缺少 artifact：{', '.join(missing)}"],
        )

    manifest = load_revision_corpus_manifest(manifest_path)
    source = manifest.get("source") if isinstance(manifest, Mapping) else {}
    source = source if isinstance(source, Mapping) else {}
    created_at = (
        str(manifest.get("created_at") or "").strip()
        if isinstance(manifest, Mapping)
        else ""
    )
    source_changed = bool(
        result_map.get("source_changed_during_scan")
        or source.get("source_changed_during_scan")
    )
    warnings = tuple(
        str(item).strip()
        for item in envelope.get("warnings") or []
        if str(item).strip()
    )
    export = RevisionCorpusExportResult(
        status=str(envelope.get("status") or "completed"),
        output_dir=output_dir,
        jsonl_path=jsonl_path,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
        item_count=_as_int(result_map.get("item_count")),
        file_count=_as_int(result_map.get("file_count")),
        created_at=created_at,
        source_changed_during_scan=source_changed,
        warnings=warnings,
    )

    facts = [
        f"条目数：{export.item_count}",
        f"文件数：{export.file_count}",
        f"JSONL：{export.jsonl_path}",
        f"Markdown：{export.markdown_path}",
        f"manifest：{export.manifest_path}",
        f"生成时间：{export.created_at or '未读取（请检查 manifest）'}",
    ]
    if export.warnings:
        facts.extend(f"警告：{warning}" for warning in export.warnings)

    if export.item_count <= 0:
        return (
            WorkflowUpdate(
                status="done",
                heading="没有可导出的润色译文",
                message=REVISION_CORPUS_COPY["empty_result"],
                facts=facts,
            ),
            export,
        )
    if export.source_changed_during_scan:
        return (
            WorkflowUpdate(
                status="warning",
                heading="润色语料已导出，但源文件发生变化",
                message=REVISION_CORPUS_COPY["source_changed_message"],
                facts=facts,
            ),
            export,
        )
    return (
        WorkflowUpdate(
            status="done",
            heading=REVISION_CORPUS_COPY["result_title"],
            message=REVISION_CORPUS_COPY["success_message"],
            facts=facts,
        ),
        export,
    )
