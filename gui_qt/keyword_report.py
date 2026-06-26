"""User-facing summaries for GUI keyword extraction runs."""
from __future__ import annotations

import re

from .translation_workflow import WorkflowUpdate


def summarize_keyword_export_output(output: str, exit_code: int) -> WorkflowUpdate:
    if exit_code != 0:
        return WorkflowUpdate(
            status="failed",
            heading="关键词导出中断",
            message="export-keywords 没有正常完成，请查看下方原始输出。",
            facts=_collect_export_facts(output),
        )

    facts = _collect_export_facts(output)
    candidate_match = re.search(
        r"^Keyword candidates:\s*(\d+)\s+deduped from\s+(\d+)\s+raw\s*$",
        output,
        re.MULTILINE,
    )
    if not candidate_match:
        return WorkflowUpdate(
            status="failed",
            heading="关键词导出结果异常",
            message="export-keywords 已结束，但输出中没有候选统计；请查看原始输出。",
            facts=facts,
        )

    deduped = int(candidate_match.group(1))
    raw = int(candidate_match.group(2))
    message = (
        f"已导出 {deduped} 个去重候选（原始 {raw} 个）。"
        "报告不会修改游戏 .rpy 文件，可在诊断页复制路径查看。"
    )
    return WorkflowUpdate(
        status="done",
        heading="关键词提取完成",
        message=message,
        facts=facts,
    )


def summarize_sync_keyword_output(output: str, exit_code: int) -> WorkflowUpdate:
    if exit_code != 0:
        return WorkflowUpdate(
            status="failed",
            heading="同步关键词提取中断",
            message=_sync_failure_message(output),
            facts=_collect_sync_facts(output),
        )

    if "No keyword source lines found." in output:
        return WorkflowUpdate(
            status="done",
            heading="没有可提取的关键词源行",
            message="当前项目没有可用于关键词提取的 TL 文本行。",
            facts=_collect_sync_facts(output),
        )

    facts = _collect_sync_facts(output)
    if not re.search(r"^Keyword candidates:\s*\d+\s+deduped from\s+\d+\s+raw\s*$", output, re.MULTILINE):
        return WorkflowUpdate(
            status="failed",
            heading="同步关键词提取结果异常",
            message="sync-keywords 已结束，但输出中没有候选统计；请查看原始输出。",
            facts=facts,
        )

    candidate_match = re.search(
        r"^Keyword candidates:\s*(\d+)\s+deduped from\s+(\d+)\s+raw\s*$",
        output,
        re.MULTILINE,
    )
    deduped = int(candidate_match.group(1)) if candidate_match else 0
    raw = int(candidate_match.group(2)) if candidate_match else 0
    message = (
        f"同步关键词提取已完成，导出 {deduped} 个去重候选（原始 {raw} 个）。"
        "报告不会修改游戏 .rpy 文件，可在诊断页复制路径查看。"
    )
    return WorkflowUpdate(
        status="done",
        heading="同步关键词提取完成",
        message=message,
        facts=facts,
    )


def _sync_failure_message(output: str) -> str:
    if "TL dir does not exist" in output:
        return "翻译目录不存在；请先运行环境检查或准备工作目录。"
    if "No keyword chunks available for the requested range." in output:
        return "当前范围没有可处理的关键词 chunk，请调整 limit/offset 后重试。"
    return "同步关键词提取没有正常完成，请查看下方原始输出。"


def _collect_export_facts(output: str) -> list[str]:
    facts: list[str] = []
    for pattern, label in (
        (r"^Keyword candidates:\s*(\d+)\s+deduped from\s+(\d+)\s+raw\s*$", "关键词候选"),
        (r"^Chunk summaries:\s*(\d+)\s*$", "剧情概要"),
        (r"^JSONL:\s*(.+?)\s*$", "候选 JSONL"),
        (r"^Markdown:\s*(.+?)\s*$", "候选 Markdown"),
        (r"^Summary JSONL:\s*(.+?)\s*$", "概要 JSONL"),
        (r"^Summary Markdown:\s*(.+?)\s*$", "概要 Markdown"),
    ):
        match = re.search(pattern, output, re.MULTILINE)
        if not match:
            continue
        if label == "关键词候选":
            facts.append(f"{label}：{match.group(1)} 个去重 / {match.group(2)} 个原始")
        elif label == "剧情概要":
            facts.append(f"{label}：{match.group(1)} 条")
        else:
            facts.append(f"{label}：{match.group(1).strip()}")
    return facts


def _collect_sync_facts(output: str) -> list[str]:
    facts = _collect_export_facts(output)
    run_match = re.search(r"^Sync keyword run:\s*(.+?)\s*$", output, re.MULTILINE)
    if run_match:
        facts.insert(0, f"同步输出目录：{run_match.group(1).strip()}")
    return facts