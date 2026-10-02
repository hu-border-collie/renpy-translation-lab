"""GUI helpers for keyword candidate merge into glossary.json."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

import keyword_glossary_merge as merge_mod

from .path_utils import canonical_abs_path
from .user_copy import KEYWORD_CANDIDATE_COPY

_MANIFEST_MODE_KEYWORD = "keyword_extraction"
_SYNC_KEYWORD_JSONL_RE = re.compile(r"^JSONL:\s*(.+?)\s*$", re.MULTILINE)
_CANDIDATE_SOURCE_COPY = {
    "external": KEYWORD_CANDIDATE_COPY["source_external"],
    "extraction": KEYWORD_CANDIDATE_COPY["source_extraction"],
    "sync": KEYWORD_CANDIDATE_COPY["source_sync"],
}


def _sibling_keyword_candidates_jsonl(manifest_path: str) -> str:
    """Same-package keyword_candidates.jsonl next to manifest.json (O(1) stat).

    Full keyword manifests can be tens of MB (embedded chunks). GUI mode switches
    must not ``json.load`` them just to discover the export path.
    """
    path = (manifest_path or "").strip()
    if not path:
        return ""
    package_dir = path
    if path.lower().endswith("manifest.json") or path.lower().endswith(".json"):
        package_dir = os.path.dirname(os.path.abspath(path))
    elif os.path.isfile(path):
        package_dir = os.path.dirname(os.path.abspath(path))
    elif not os.path.isdir(path):
        return ""
    else:
        package_dir = os.path.abspath(path)
    candidate = os.path.join(package_dir, "keyword_candidates.jsonl")
    return candidate if os.path.isfile(candidate) else ""


def keyword_merge_candidates_path_from_manifest(
    manifest_path: str,
    manifest: dict[str, object] | None,
) -> str:
    # Explicit non-keyword modes must never pick up a sibling candidates file
    # from an unrelated package directory (e.g. batch_translation next to a
    # leftover keyword_candidates.jsonl).
    if (
        manifest is not None
        and isinstance(manifest.get("mode"), str)
        and manifest["mode"] != _MANIFEST_MODE_KEYWORD
    ):
        return ""

    if manifest is not None and manifest.get("mode") == _MANIFEST_MODE_KEYWORD:
        export = manifest.get("keyword_export")
        if isinstance(export, dict):
            jsonl_path = export.get("jsonl_path")
            if isinstance(jsonl_path, str) and jsonl_path.strip() and os.path.isfile(jsonl_path):
                return jsonl_path.strip()
        # Lite readers drop keyword_export; still resolve via package sibling.
        embedded = manifest.get("_manifest_path")
        for candidate_path in (manifest_path, embedded if isinstance(embedded, str) else ""):
            sibling = _sibling_keyword_candidates_jsonl(str(candidate_path or ""))
            if sibling:
                return sibling
    if manifest_path.strip():
        sibling = _sibling_keyword_candidates_jsonl(manifest_path.strip())
        if sibling:
            return sibling
        # Last resort: may parse full JSON (slow on multi-MB manifests).
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
        return False, (
            "没有可合并的关键词候选；请先完成关键词提取，"
            f"或用「{KEYWORD_CANDIDATE_COPY['open_action']}」加载已有候选。"
        )
    if not os.path.isfile(resolved_candidates):
        return False, f"候选文件不存在：{resolved_candidates}"
    if not glossary_path.strip():
        return False, "未配置术语表目标，请在设置页保存项目术语表路径。"
    return True, ""


@dataclass(frozen=True)
class KeywordCandidateSnapshot:
    """Candidate JSONL content as read for one review (#539).

    ``text`` and ``fingerprint`` always describe the same read: the fingerprint
    covers exactly the bytes handed to the shared parser, so a review can never
    display one file version while validating another.
    """

    text: str
    fingerprint: str


def _read_candidate_bytes(candidates_path: str) -> bytes:
    with open(candidates_path, "rb") as handle:
        return handle.read()


def keyword_candidate_content_fingerprint(candidates_path: str) -> str:
    """Return the sha256 of the candidate file bytes; empty when unreadable.

    Used to compare the content a review parsed against the file on disk right
    before a merge writes. Deliberately a content hash, not mtime or size.
    """
    path = str(candidates_path or "").strip()
    if not path:
        return ""
    try:
        return hashlib.sha256(_read_candidate_bytes(path)).hexdigest()
    except OSError:
        return ""


def read_keyword_candidate_snapshot(candidates_path: str) -> KeywordCandidateSnapshot:
    """Read candidate JSONL bytes once and fingerprint that same read.

    Raises ``ValueError`` for unreadable, non-UTF-8 input so GUI callers can
    report it without parsing a second, possibly different file state.
    """
    path = str(candidates_path or "").strip()
    if not path:
        raise ValueError("候选文件路径为空。")
    try:
        raw = _read_candidate_bytes(path)
    except OSError as exc:
        raise ValueError(f"无法读取候选文件：{path}（{exc}）") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"候选文件不是 UTF-8 文本：{path}（{exc}）") from exc
    return KeywordCandidateSnapshot(
        text=text,
        fingerprint=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class KeywordCandidateSelection:
    """One keyword candidate file loaded for human review (#539).

    ``game_root`` and ``glossary_path`` freeze the project identity captured when
    the file was opened. Comparing them against the live context lets the GUI
    drop a stale review instead of merging into another project's terms.
    """

    candidates_path: str
    game_root: str
    glossary_path: str
    macro_path: str
    candidate_total: int
    mergeable_total: int
    source: str = "external"


def format_keyword_candidate_selection(selection: KeywordCandidateSelection) -> str:
    """Render candidate source, counts and the current project glossary target."""
    source = _CANDIDATE_SOURCE_COPY.get(
        selection.source,
        _CANDIDATE_SOURCE_COPY["external"],
    )
    glossary = selection.glossary_path or "未配置（合并前请在设置页保存项目术语表路径）"
    return "\n".join(
        (
            f"候选来源：{source}（{KEYWORD_CANDIDATE_COPY['format_name']}）",
            f"候选文件：{selection.candidates_path}",
            f"候选条数：{selection.candidate_total} 条（可审核 {selection.mergeable_total} 条）",
            f"术语表目标：{glossary}",
            KEYWORD_CANDIDATE_COPY["info_hint"],
        )
    )


def keyword_candidate_open_ready(
    *,
    running: bool,
    game_root: str,
    project_ready: bool,
) -> tuple[bool, str]:
    """Return whether the standalone 打开候选文件 entry may ask for a file (#539).

    The second item is user copy explaining the restriction and is empty while
    the entry is usable.
    """
    if running:
        return False, KEYWORD_CANDIDATE_COPY["open_running"]
    if not str(game_root or "").strip():
        return False, KEYWORD_CANDIDATE_COPY["open_no_project"]
    if not project_ready:
        return False, KEYWORD_CANDIDATE_COPY["open_project_not_ready"]
    return True, ""


def _same_asset_path(left: str, right: str) -> bool:
    try:
        return (
            canonical_abs_path(left).casefold()
            == canonical_abs_path(right).casefold()
        )
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class KeywordReviewContext:
    """Frozen identity and content version of one open candidate review (#539).

    Built while the review actually reads and displays candidate content, then
    compared against the live context right before a merge writes anything.
    """

    candidates_path: str
    fingerprint: str
    game_root: str
    glossary_path: str
    generation: int = 0


def keyword_review_context_stale_reason(
    review: KeywordReviewContext,
    current: KeywordReviewContext,
) -> str:
    """Return why an open review no longer matches the live context (#539).

    Empty means the dialog may still write. Any other value is shown to the user
    and the write is refused, so a project switch, a candidate switch, replaced
    or removed candidate content, or a glossary retarget can never be applied to
    a review that was opened against the previous target.
    """
    if not _same_asset_path(review.game_root, current.game_root):
        return KEYWORD_CANDIDATE_COPY["stale_project"]
    if not review.candidates_path or not os.path.isfile(review.candidates_path):
        return KEYWORD_CANDIDATE_COPY["stale_candidates"]
    if not _same_asset_path(review.glossary_path, current.glossary_path):
        return KEYWORD_CANDIDATE_COPY["stale_glossary"]
    if not _same_asset_path(review.candidates_path, current.candidates_path):
        return KEYWORD_CANDIDATE_COPY["stale_selection"]
    if int(review.generation) != int(current.generation):
        return KEYWORD_CANDIDATE_COPY["stale_selection"]
    if (
        not review.fingerprint
        or not current.fingerprint
        or review.fingerprint != current.fingerprint
    ):
        return KEYWORD_CANDIDATE_COPY["stale_candidates"]
    return ""


def load_keyword_merge_context(
    *,
    candidates_path: str,
    config: dict[str, object],
    game_root: str,
    tool_root: str,
    min_confidence: float = 0.0,
    candidates_text: str | None = None,
) -> tuple[list[merge_mod.CandidateMergeRow], list[dict], str, str]:
    """Resolve glossary/macro targets and build review rows for one candidate file.

    ``candidates_text`` lets a caller supply content it already read and
    fingerprinted (#539) so the parsed rows always match the frozen review
    version instead of a possibly newer file state.
    """
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
        candidates = merge_mod.load_keyword_candidates_jsonl(
            candidates_path,
            text=candidates_text,
        )
        glossary = merge_mod.load_glossary_file(glossary_path)
    except SystemExit as exc:
        # Both loaders signal malformed input with SystemExit (CLI contract);
        # GUI callers must see a catchable ValueError instead of exiting.
        raise ValueError(str(exc)) from exc
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