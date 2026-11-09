"""User-facing summaries for GUI keyword extraction runs."""
from __future__ import annotations

import re

from gui_qt.reports.check_report import WritebackSummary
from gui_qt.workflows.translation_workflow import WorkflowUpdate
from gui_qt.reports.sync_usage_report import collect_sync_usage_facts
from gui_qt.reports.workflow_progress import WorkflowProgressState
from gui_qt.user_copy import MODEL_CONTRACT_COPY, SYNC_KEYWORD_COPY


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _sync_keyword_manifest_counts(manifest: dict) -> tuple[int, int | None, int | None, int | None, int | None]:
    summary = manifest.get('sync_summary')
    summary = summary if isinstance(summary, dict) else {}
    source = manifest.get('summary')
    source = source if isinstance(source, dict) else {}
    total = _count(summary.get('request_count'))
    if total is None:
        total = _count(source.get('chunk_count')) or 0
    saved = _count(summary.get('checkpointed_request_count'))
    if saved is None:
        saved = _count(manifest.get('_keyword_saved_request_count'))
    successful = _count(summary.get('successful_request_count'))
    failed = _count(summary.get('failed_request_count'))
    valid = _count(summary.get('contract_final_complete_chunks'))
    if valid is None and successful is not None:
        valid = max(0, successful - (_count(summary.get('contract_partial_requests')) or 0))
    return total, saved, successful, failed, valid


def summarize_sync_keyword_manifest(manifest: dict[str, object]) -> WorkflowUpdate:
    """Restore saved extraction statistics without treating saved errors as success."""
    total, saved, successful, failed, valid = _sync_keyword_manifest_counts(manifest)
    summary = manifest.get('sync_summary')
    summary = summary if isinstance(summary, dict) else {}
    incomplete = _count(summary.get('contract_partial_requests')) or 0
    partial = manifest.get('job_state') == 'SYNC_PARTIAL' or bool(failed) or bool(incomplete)
    completed = manifest.get('job_state') == 'SYNC_COMPLETED'
    prefix = 'partial' if partial else 'completed' if completed else 'ready'
    facts: list[str] = []
    restored = _count(summary.get('restored_request_count'))
    if restored is not None:
        facts.extend((SYNC_KEYWORD_COPY['task_total'].format(count=total),
                      SYNC_KEYWORD_COPY['restored_valid'].format(count=restored)))
    if saved is not None and total:
        facts.append(SYNC_KEYWORD_COPY['saved_records'].format(current=saved, total=total))
    if successful is not None and failed is not None:
        facts.append(SYNC_KEYWORD_COPY['response_counts'].format(successful=successful, failed=failed))
    if valid is not None and total:
        facts.append(SYNC_KEYWORD_COPY['valid_progress'].format(
            current=valid, total=total, percent=100 * valid / total))
    if incomplete:
        facts.append(SYNC_KEYWORD_COPY['incomplete_responses'].format(count=incomplete))
    export = manifest.get('keyword_export')
    counts = export.get('summary') if isinstance(export, dict) else None
    if isinstance(counts, dict):
        for key, label in (('candidate_count_deduped', 'candidates'), ('chunk_summary_count', 'summaries')):
            count = _count(counts.get(key))
            if count is not None:
                facts.append(SYNC_KEYWORD_COPY[label].format(count=count))
    return WorkflowUpdate(
        status='warning' if partial else 'done' if completed else 'ready',
        heading=SYNC_KEYWORD_COPY[f'{prefix}_heading'],
        message=SYNC_KEYWORD_COPY[f'{prefix}_message'],
        facts=facts,
    )


def sync_keyword_progress_from_manifest(manifest: dict[str, object]) -> WorkflowProgressState:
    """Keep the latest pending-queue denominator; support pre-queue manifests."""
    summary = manifest.get('sync_summary')
    summary = summary if isinstance(summary, dict) else {}
    pending = _count(summary.get('pending_request_count'))
    processed = _count(summary.get('processed_pending_request_count'))
    if pending is not None and processed is not None:
        current = min(processed, pending)
        label = (SYNC_KEYWORD_COPY['pending_processed'].format(
            current=current, total=pending, percent=100 * current / pending)
            if pending else SYNC_KEYWORD_COPY['exporting_restored'])
        return WorkflowProgressState(
            kind='sync_keywords', current=current, total=pending, label=label,
            visible=pending > 0,
        )
    total, saved, _successful, _failed, valid = _sync_keyword_manifest_counts(manifest)
    current = valid if valid is not None else saved
    label = ''
    if current is not None and total:
        key = 'valid_progress' if valid is not None else 'saved_records'
        label = SYNC_KEYWORD_COPY[key].format(current=current, total=total, percent=100 * current / total)
    return WorkflowProgressState(
        kind='sync_keywords', current=current or 0, total=total, label=label,
        visible=current is not None and total > 0,
    )


def summarize_keyword_export_output(output: str, exit_code: int) -> WorkflowUpdate:
    if exit_code != 0:
        return WorkflowUpdate(
            status="failed",
            heading="关键词导出中断",
            message="关键词导出未完成，请查看诊断日志。",
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
            message="关键词导出已结束，但未能识别候选统计；请查看诊断日志。",
            facts=facts,
        )

    deduped = int(candidate_match.group(1))
    raw = int(candidate_match.group(2))
    message = (
        f"已导出 {deduped} 个去重候选（原始 {raw} 个）。"
        "报告包含历史首次译法与保留不译的人工作提示，不修改游戏脚本；完整路径可在诊断与工具复制。"
    )
    return WorkflowUpdate(
        status="done",
        heading="关键词提取完成",
        message=message,
        facts=facts,
    )


def summarize_keyword_result_from_manifest(manifest: dict[str, object]) -> WritebackSummary | None:
    export = manifest.get("keyword_export")
    if not isinstance(export, dict):
        return None

    manifest_path = manifest.get("_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        manifest_path = ""

    facts: list[str] = []
    if manifest_path:
        facts.append(f"任务记录：\n  {manifest_path}")

    summary = export.get("summary")
    if isinstance(summary, dict):
        deduped = summary.get("candidate_count_deduped")
        raw = summary.get("candidate_count_raw")
        if isinstance(deduped, int) and isinstance(raw, int):
            facts.append(f"关键词候选：{deduped} 个去重 / {raw} 个原始")
        chunk_summary_count = summary.get("chunk_summary_count")
        if isinstance(chunk_summary_count, int):
            facts.append(f"剧情概要：{chunk_summary_count} 条")
        history_counts = summary.get("history_candidate_status_counts")
        if isinstance(history_counts, dict):
            facts.append(f"历史首次译法证据：{history_counts}")

    for key, label in (
        ("markdown_path", "候选 Markdown"),
        ("jsonl_path", "候选 JSONL"),
        ("summary_markdown_path", "概要 Markdown"),
        ("summary_jsonl_path", "概要 JSONL"),
    ):
        path = export.get(key)
        if isinstance(path, str) and path.strip():
            facts.append(f"{label}：\n  {path.strip()}")

    return WritebackSummary(
        status="safe",
        heading="关键词报告已生成",
        message="批量关键词任务已完成；报告不会修改游戏脚本，可在诊断与工具复制完整路径。",
        facts=facts,
        findings=[],
        can_apply=False,
        manifest_path=manifest_path,
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
            message="当前项目没有可用于关键词提取的翻译文本。",
            facts=_collect_sync_facts(output),
        )

    facts = _collect_sync_facts(output)
    if not re.search(r"^Keyword candidates:\s*\d+\s+deduped from\s+\d+\s+raw\s*$", output, re.MULTILINE):
        return WorkflowUpdate(
            status="failed",
            heading="同步关键词提取结果异常",
            message="同步关键词提取已结束，但未能识别候选统计；请查看诊断日志。",
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
        "报告不修改游戏脚本，完整路径可在诊断与工具复制。"
    )
    unresolved = _contract_unresolved_count(output)
    if unresolved > 0 or _contract_partial_request_count(output) > 0:
        return WorkflowUpdate(
            status="warning",
            heading="同步关键词提取部分完成",
            message=f"{MODEL_CONTRACT_COPY['partial_keyword']}{message}",
            facts=facts,
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
        return "当前范围没有可提取的内容，请调整范围后重试。"
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
        (r"^Historical evidence:\s*(.+?)\s*$", "历史首次译法证据"),
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
    saved = re.findall(r"^Sync keyword saved: (\d+)/(\d+) requests; resumed=(\d+)\.$",
                       output, re.MULTILINE)
    if saved:
        current, total, resumed = saved[-1]
        facts.extend([
            SYNC_KEYWORD_COPY['saved_records'].format(current=current, total=total),
            SYNC_KEYWORD_COPY['resumed'].format(count=resumed),
        ])
    run_match = re.search(r"^Sync keyword run:\s*(.+?)\s*$", output, re.MULTILINE)
    if run_match:
        facts.insert(0, f"同步输出目录：{run_match.group(1).strip()}")
    completeness = re.search(
        r"^Model contract chunk completeness:\s*(\d+/\d+)\s*$",
        output,
        re.MULTILINE,
    )
    if completeness:
        facts.append(
            f"{MODEL_CONTRACT_COPY['chunk_completeness']}：{completeness.group(1)}"
        )
    retries = re.search(
        r"^Targeted retries:\s*(\d+) requests / (\d+) items\s*$",
        output,
        re.MULTILINE,
    )
    if retries:
        facts.append(
            f"{MODEL_CONTRACT_COPY['targeted_retries']}："
            f"{retries.group(1)} 次请求 / {retries.group(2)} 项"
        )
    unresolved = _contract_unresolved_count(output)
    if unresolved >= 0:
        facts.append(f"{MODEL_CONTRACT_COPY['unresolved_items']}：{unresolved} 个")
    partial_requests = _contract_partial_request_count(output)
    if partial_requests >= 0:
        facts.append(
            f"{MODEL_CONTRACT_COPY['partial_requests']}：{partial_requests} 个"
        )
    facts.extend(collect_sync_usage_facts(output))
    return facts


def _contract_unresolved_count(output: str) -> int:
    match = re.search(
        r"^Unresolved contract items:\s*(\d+)\s*$", output, re.MULTILINE
    )
    return int(match.group(1)) if match else -1


def _contract_partial_request_count(output: str) -> int:
    match = re.search(
        r"^Contract partial requests:\s*(\d+)\s*$", output, re.MULTILINE
    )
    return int(match.group(1)) if match else -1
