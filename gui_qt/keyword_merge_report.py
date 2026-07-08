"""GUI helpers for keyword candidate merge into glossary.json."""
from __future__ import annotations

import os
import re

import keyword_glossary_merge as merge_mod

_MANIFEST_MODE_KEYWORD = "keyword_extraction"
_SYNC_KEYWORD_JSONL_RE = re.compile(r"^JSONL:\s*(.+?)\s*$", re.MULTILINE)


def keyword_merge_candidates_path_from_manifest(
    manifest_path: str,
    manifest: dict[str, object] | None,
) -> str:
    if manifest is not None:
        export = manifest.get("keyword_export")
        if isinstance(export, dict):
            jsonl_path = export.get("jsonl_path")
            if isinstance(jsonl_path, str) and jsonl_path.strip() and os.path.isfile(jsonl_path):
                return jsonl_path.strip()
    if manifest_path.strip():
        try:
            return merge_mod.resolve_keyword_candidates_path(manifest_path.strip())
        except SystemExit:
            return ""
    return ""


def keyword_merge_candidates_path_from_sync_output(output: str) -> str:
    match = _SYNC_KEYWORD_JSONL_RE.search(output)
    if not match:
        return ""
    candidate = match.group(1).strip()
    return candidate if candidate and os.path.isfile(candidate) else ""


def keyword_merge_ready(
    *,
    manifest_path: str = "",
    manifest: dict[str, object] | None = None,
    candidates_path: str = "",
    glossary_path: str = "",
) -> tuple[bool, str]:
    resolved_candidates = candidates_path.strip()
    if not resolved_candidates:
        resolved_candidates = keyword_merge_candidates_path_from_manifest(
            manifest_path,
            manifest,
        )
    if not resolved_candidates:
        return False, "没有可合并的关键词候选 JSONL；请先完成关键词提取或选择候选文件。"
    if not os.path.isfile(resolved_candidates):
        return False, f"候选文件不存在：{resolved_candidates}"
    if not glossary_path.strip():
        return False, "未配置 glossary 路径，请在设置页保存项目术语表路径。"
    return True, ""


def load_keyword_merge_context(
    *,
    candidates_path: str,
    config: dict[str, object],
    game_root: str,
    tool_root: str,
    min_confidence: float = 0.0,
) -> tuple[list[merge_mod.CandidateMergeRow], list[dict], str, str]:
    glossary_path = merge_mod.resolve_glossary_path_from_config(
        config,
        game_root=game_root,
        tool_root=tool_root,
    )
    macro_path = merge_mod.resolve_macro_setting_path_from_config(
        config,
        game_root=game_root,
        tool_root=tool_root,
    )
    macro_text = merge_mod.load_macro_setting_text(macro_path)
    try:
        candidates = merge_mod.load_keyword_candidates_jsonl(candidates_path)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    glossary = merge_mod.load_glossary_file(glossary_path)
    rows = merge_mod.build_candidate_merge_rows(
        candidates,
        glossary,
        min_confidence=min_confidence,
        macro_setting_text=macro_text,
    )
    return rows, candidates, glossary_path, macro_path


def format_merge_preview_text(
    counts: dict[str, int],
    *,
    overwrite: bool,
) -> str:
    lines = [
        f"已勾选 {counts.get('selected', 0)} 条候选。",
        f"将新增 {counts.get('accept', 0)} 条，覆盖 {counts.get('overwrite', 0)} 条。",
    ]
    blocked = counts.get('blocked_duplicate', 0)
    if blocked:
        if overwrite:
            lines.append(f"另有 {counts.get('skipped', 0)} 条因置信度或内容为空被跳过。")
        else:
            lines.append(
                f"{blocked} 条与现有 glossary 冲突且未启用覆盖，写入时会被跳过。"
            )
    return "\n".join(lines)


def summarize_keyword_merge_result(summary: merge_mod.MergeSummary) -> dict[str, object]:
    facts: list[str] = []
    if summary.candidates_path:
        facts.append(f"候选文件：\n  {summary.candidates_path}")
    if summary.glossary_path:
        facts.append(f"术语表：\n  {summary.glossary_path}")
    if summary.backup_path:
        facts.append(f"备份：\n  {summary.backup_path}")

    findings: list[str] = []
    if summary.preview_lines:
        preview = summary.preview_lines[:8]
        findings.extend(preview)
        if len(summary.preview_lines) > 8:
            findings.append(f"…另有 {len(summary.preview_lines) - 8} 条预览行")

    if summary.dry_run:
        heading = "关键词合并预览完成"
        message = (
            f"预览完成：将写入 {summary.accepted} 条"
            f"（覆盖 {summary.overwritten} 条），未修改 glossary。"
        )
        status = "ready"
    elif summary.wrote_glossary:
        heading = "关键词已合并到 glossary"
        message = (
            f"已写入 {summary.accepted} 条到 glossary"
            f"（覆盖 {summary.overwritten} 条）。"
        )
        status = "ready"
    elif summary.accepted == 0:
        heading = "关键词合并未写入"
        message = "没有选中可写入的候选，glossary 未修改。"
        status = "warning"
    else:
        heading = "关键词合并结果不明确"
        message = "合并已结束，但 glossary 写入状态未知。"
        status = "warning"

    stats = (
        f"读取 {summary.candidates_read} 条；"
        f"写入 {summary.accepted} 条；"
        f"跳过重复 {summary.skipped_duplicate} 条；"
        f"用户未选 {summary.skipped_user} 条。"
    )
    facts.insert(0, stats)

    return {
        "status": status,
        "heading": heading,
        "message": message,
        "facts": facts,
        "findings": findings,
    }