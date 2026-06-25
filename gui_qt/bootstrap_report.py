"""User-facing summaries for GUI bootstrap-rag / bootstrap-source-index commands."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .summary_helpers import extend_facts_with_notices
from .user_copy import format_bootstrap_fact
from typing import Any


@dataclass(frozen=True)
class BootstrapSummary:
    kind: str
    status: str
    heading: str
    message: str
    facts: list[str]
    findings: list[str]


RAG_SUMMARY_HEADER = "RAG bootstrap summary:"
SOURCE_INDEX_SUMMARY_HEADER = "Source Index bootstrap final summary:"


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def read_batch_context_flags(config: dict[str, Any]) -> dict[str, bool]:
    batch = config.get("batch")
    if not isinstance(batch, dict):
        batch = {}
    rag = batch.get("rag")
    if not isinstance(rag, dict):
        rag = {}
    source_index = batch.get("source_index")
    if not isinstance(source_index, dict):
        source_index = {}
    return {
        "rag_enabled": coerce_bool(rag.get("enabled"), False),
        "source_index_enabled": coerce_bool(source_index.get("enabled"), False),
        "bootstrap_on_build": coerce_bool(rag.get("bootstrap_on_build"), True),
    }


def _parse_summary_values(output: str, header: str) -> dict[str, str]:
    values: dict[str, str] = {}
    in_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == header:
            in_section = True
            continue
        if in_section and line.startswith("- "):
            match = re.match(r"-\s*([a-z_]+):\s*(.*)$", line)
            if match:
                values[match.group(1)] = match.group(2).strip()
            continue
        if in_section and not line.startswith("- "):
            break
    return values


def _parse_int_field(values: dict[str, str], key: str, default: int = 0) -> int:
    raw = values.get(key, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _format_fact_values(values: dict[str, str], keys: tuple[str, ...]) -> list[str]:
    facts: list[str] = []
    for key in keys:
        if key in values:
            facts.append(format_bootstrap_fact(key, values[key]))
    return facts


def idle_bootstrap_summary() -> BootstrapSummary:
    return BootstrapSummary(
        kind="",
        status="idle",
        heading="尚未预建上下文库",
        message="如果项目已有部分译文，可先预建 RAG 库；如果译文很少，可预建原文索引。",
        facts=[],
        findings=[],
    )


def running_bootstrap_summary(kind: str) -> BootstrapSummary:
    if kind == "source_index":
        return BootstrapSummary(
            kind=kind,
            status="running",
            heading="正在预建原文索引",
            message="正在扫描翻译模板原文并生成向量索引，请稍候。",
            facts=[],
            findings=[],
        )
    return BootstrapSummary(
        kind=kind or "rag",
        status="running",
        heading="正在预建 RAG 库",
        message="正在扫描已有译文并更新记忆库，请稍候。",
        facts=[],
        findings=[],
    )


def stale_bootstrap_summary() -> BootstrapSummary:
    return BootstrapSummary(
        kind="",
        status="stale",
        heading="预建状态已过期",
        message="项目或配置已切换，请针对当前项目重新运行预建库。",
        facts=[],
        findings=[],
    )


def summarize_rag_bootstrap_output(output: str, exit_code: int) -> BootstrapSummary:
    if "RAG is disabled" in output:
        return BootstrapSummary(
            kind="rag",
            status="warning",
            heading="RAG 未启用",
            message="请先在配置页启用 Batch RAG 并保存参数配置，再运行预建 RAG 库。",
            facts=[],
            findings=[],
        )

    values = _parse_summary_values(output, RAG_SUMMARY_HEADER)
    facts = _format_fact_values(
        values,
        (
            "store_dir",
            "scan_scope",
            "files_scanned",
            "scanned",
            "embedded",
            "upserted",
            "history_records_before",
            "history_records_after",
            "external_seed_records",
        ),
    )
    findings: list[str] = []
    if values.get("error"):
        findings.append(values["error"])

    if exit_code != 0:
        return BootstrapSummary(
            kind="rag",
            status="failed",
            heading="RAG 预建失败",
            message="预建 RAG 库未成功完成，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if values.get("error"):
        return BootstrapSummary(
            kind="rag",
            status="failed",
            heading="RAG 预建失败",
            message="预建过程中出现错误，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    scanned = _parse_int_field(values, "scanned")
    upserted = _parse_int_field(values, "upserted")
    embedded = _parse_int_field(values, "embedded")
    external_seed_records = _parse_int_field(values, "external_seed_records")
    if scanned == 0 and external_seed_records == 0:
        return BootstrapSummary(
            kind="rag",
            status="warning",
            heading="RAG 预建完成（无新记录）",
            message="未扫描到可写入的译文记录。若项目尚无译文，可先翻译一部分。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if upserted > 0 or embedded > 0 or external_seed_records > 0:
        return BootstrapSummary(
            kind="rag",
            status="ready",
            heading="RAG 预建完成",
            message="记忆库已刷新，可以开始翻译任务。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    return BootstrapSummary(
        kind="rag",
        status="ready",
        heading="RAG 预建完成",
        message="预建流程已完成；如需更新记忆库，可在译文变更后再次运行。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
    )


def summarize_source_index_bootstrap_output(output: str, exit_code: int) -> BootstrapSummary:
    values = _parse_summary_values(output, SOURCE_INDEX_SUMMARY_HEADER)
    facts = _format_fact_values(
        values,
        (
            "store_dir",
            "files_scanned",
            "scanned",
            "embedded",
            "upserted",
            "reused_embeddings",
            "stale_count",
            "pruned",
            "history_records_before",
            "history_records_after",
        ),
    )
    findings: list[str] = []
    if values.get("error"):
        findings.append(values["error"])

    if exit_code != 0:
        return BootstrapSummary(
            kind="source_index",
            status="failed",
            heading="原文索引预建失败",
            message="预建原文索引未成功完成，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if values.get("error"):
        return BootstrapSummary(
            kind="source_index",
            status="failed",
            heading="原文索引预建失败",
            message="预建过程中出现错误，请查看诊断日志。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    scanned = _parse_int_field(values, "scanned")
    upserted = _parse_int_field(values, "upserted")
    embedded = _parse_int_field(values, "embedded")
    if scanned == 0:
        return BootstrapSummary(
            kind="source_index",
            status="warning",
            heading="原文索引预建完成（无新记录）",
            message="未扫描到可索引的原文片段。请确认翻译模板已生成且项目路径正确。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    if upserted > 0 or embedded > 0:
        return BootstrapSummary(
            kind="source_index",
            status="ready",
            heading="原文索引预建完成",
            message="原文索引已刷新，后续翻译时可检索相关剧情原文。",
            facts=extend_facts_with_notices(facts, findings),
            findings=findings,
        )

    return BootstrapSummary(
        kind="source_index",
        status="ready",
        heading="原文索引预建完成",
        message="索引库已是最新状态，无需新增向量。",
        facts=extend_facts_with_notices(facts, findings),
        findings=findings,
    )