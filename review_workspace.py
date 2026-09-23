"""GUI-neutral ordinary translation review operations (#427 S2).

The index remains derived; drafts are separate human work and never authorize
writeback. Proposal output is consumed by the existing revision import gate.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_jsonl

import review_index
import revision_proposals

DRAFTS_NAME = "review_drafts.json"
PROPOSALS_NAME = "review_proposals.jsonl"
PAGE_SIZE = 50


def _project(manifest: Mapping[str, Any]) -> str:
    project = manifest.get("project") or {}
    return str(project.get("identity_digest") or "") if isinstance(project, Mapping) else ""


def _entry(entries: Sequence[Mapping[str, Any]], occurrence_id: str) -> Mapping[str, Any]:
    found = [row for row in entries if row.get("occurrence_id") == occurrence_id]
    if len(found) != 1:
        raise ValueError("条目身份不存在或不唯一；请重新构建索引。")
    return found[0]


def open_workspace(
    corpus_path: str, *, expected_game_root: str, expected_tl_dir: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild a corpus or existing index while retaining its attached evidence."""
    supplied = Path(corpus_path)
    index_dir: Path | None = None
    if supplied.is_file() and supplied.suffix.lower() == ".json":
        document = json.loads(supplied.read_text(encoding="utf-8-sig"))
        if isinstance(document, dict) and document.get("kind") == "review_index":
            index_dir = supplied.parent
            corpus_meta = (document.get("inputs") or {}).get("corpus_manifest") or {}
            corpus_path = str(corpus_meta.get("path") or "")
    corpus = review_index.load_corpus_bundle(corpus_path)
    project = corpus["manifest"].get("project") or {}
    if not isinstance(project, Mapping):
        raise ValueError("语料缺少项目身份。")
    root = str(project.get("game_root") or "")
    if not root or os.path.normcase(os.path.realpath(root)) != os.path.normcase(os.path.realpath(expected_game_root)):
        raise ValueError("语料所属项目与当前项目不一致；请重新导出当前项目语料。")
    tl_dir = str(project.get("tl_dir") or "")
    if expected_tl_dir and (
        not tl_dir or os.path.normcase(os.path.realpath(tl_dir))
        != os.path.normcase(os.path.realpath(expected_tl_dir))
    ):
        raise ValueError("语料目标语言目录与当前项目不一致。")
    if bool((corpus["manifest"].get("source") or {}).get("source_changed_during_scan")):
        raise ValueError("语料导出期间源文件发生变化；请重新导出。")
    output_dir = str(index_dir or (Path(corpus["manifest_path"]).parent / "review_index"))
    review_index.build_review_index(corpus_path, output_dir=output_dir)
    return review_index.load_review_index(output_dir)


def filter_page(
    entries: Sequence[Mapping[str, Any]], *, page: int = 0, file: str = "",
    lifecycle: str = "", findings: str = "", severity: str = "",
    speaker: str = "", query: str = "",
) -> tuple[int, list[Mapping[str, Any]]]:
    """Filter complete entries and return only one bounded display page."""
    selected: list[Mapping[str, Any]] = []
    needle = query.casefold().strip()
    for row in entries:
        review = row.get("review") or {}
        issues = row.get("quality_findings") or []
        if file and file.casefold() not in str(row.get("file_rel_path") or "").casefold():
            continue
        if lifecycle and review.get("lifecycle") != lifecycle:
            continue
        if findings == "yes" and not issues:
            continue
        if findings == "no" and issues:
            continue
        if severity and not any(item.get("severity") == severity for item in issues):
            continue
        if speaker and speaker.casefold() not in str(row.get("speaker_id") or "").casefold():
            continue
        if needle and not any(needle in str(row.get(key) or "").casefold() for key in ("source", "current_translation")):
            continue
        selected.append(row)
    start = max(0, page) * PAGE_SIZE
    return len(selected), selected[start:start + PAGE_SIZE]


def load_drafts(index_path: str, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read persistent drafts; foreign-project data is rejected before editing."""
    path = Path(index_path).resolve().parent / DRAFTS_NAME
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("project_identity_digest") != _project(manifest):
        raise ValueError("草稿属于其他项目或格式无效，已拒绝加载。")
    drafts = raw.get("drafts")
    if not isinstance(drafts, dict):
        raise ValueError("草稿格式无效。")
    return drafts


def save_draft(
    index_path: str, manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
    occurrence_id: str, proposed_translation: str, reason: str,
) -> dict[str, Any]:
    """Persist full text and the current evidence binding outside the index."""
    row = _entry(entries, occurrence_id)
    drafts = load_drafts(index_path, manifest)
    draft = {
        "occurrence_id": occurrence_id,
        "binding": dict(row.get("binding") or {}),
        "proposed_translation": proposed_translation,
        "reason": reason,
    }
    drafts[occurrence_id] = draft
    atomic_write_json(
        Path(index_path).resolve().parent / DRAFTS_NAME,
        {"project_identity_digest": _project(manifest), "drafts": drafts},
        ensure_ascii=False, indent=2,
    )
    return draft


def draft_is_current(draft: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    return draft.get("binding") == row.get("binding")


def save_decision(
    index_path: str, entries: Sequence[Mapping[str, Any]], occurrence_id: str,
    lifecycle: str, reviewer: str, note: str,
) -> dict[str, Any]:
    """Append one human decision through the S1 import transaction."""
    row = _entry(entries, occurrence_id)
    decision = review_index.decision_template([row])[0]
    decision.update({
        "lifecycle": lifecycle,
        "reviewer": {"type": "human", "name": reviewer},
        "note": note,
    })
    review_index.normalize_decision(decision)
    package = Path(index_path).resolve().parent
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl", dir=package, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    try:
        return review_index.import_decisions_into_index(index_path, temporary)
    finally:
        temporary.unlink(missing_ok=True)


def export_proposals(
    index_path: str, manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
    occurrence_ids: Sequence[str],
) -> str:
    """Export explicitly selected current drafts in the #321 proposal schema."""
    if not occurrence_ids or len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("请明确选择至少一条且不重复的草稿。")
    drafts = load_drafts(index_path, manifest)
    inputs = manifest.get("inputs") or {}
    corpus_meta = inputs.get("corpus_manifest") or {}
    corpus_path = str(corpus_meta.get("path") or "")
    corpus = review_index.load_corpus_bundle(corpus_path)
    if review_index.project_identity(corpus["manifest"])["identity_digest"] != _project(manifest):
        raise ValueError("语料与索引项目身份不一致。")
    if corpus["manifest_digest"] != corpus_meta.get("digest"):
        raise ValueError("语料 manifest 已变化；请重新加载审校索引。")
    jsonl_meta = inputs.get("corpus_jsonl") or {}
    if corpus["jsonl_digest"] != jsonl_meta.get("digest"):
        raise ValueError("语料内容已变化；请重新加载审校索引。")
    source = corpus["manifest"].get("source") or {}
    if source.get("source_changed_during_scan"):
        raise ValueError("语料源快照不一致；请重新导出。")
    project = corpus["manifest"].get("project") or {}
    rows: list[dict[str, Any]] = []
    for occurrence_id in occurrence_ids:
        row = _entry(entries, occurrence_id)
        draft = drafts.get(occurrence_id)
        if not isinstance(draft, Mapping) or not draft_is_current(draft, row):
            raise ValueError(f"{occurrence_id} 的草稿缺失或证据已变化；请重新审校。")
        if not str(draft.get("proposed_translation") or "").strip() or not str(draft.get("reason") or "").strip():
            raise ValueError(f"{occurrence_id} 的草稿缺少建议译文或原因。")
        rows.append({
            "schema_version": 1,
            "occurrence_id": occurrence_id,
            "identity_v2": str(row.get("identity_v2") or occurrence_id),
            "file_rel_path": row.get("file_rel_path"),
            "source": row.get("source"),
            "current_translation": row.get("current_translation"),
            "proposed_translation": draft["proposed_translation"],
            "reason": draft["reason"],
            "selected": True,
            "disposition": "accepted",
            "producer": {"type": "human", "tool": "review-workspace"},
            "project_identity": {"game_root": project.get("game_root"), "tl_dir": project.get("tl_dir")},
            "snapshot_digest": row.get("snapshot_digest"),
            "corpus_snapshot_digest": source.get("snapshot_digest"),
        })
    # Structural validation here is advisory to the later live-project import;
    # reject malformed output before it can be presented as a candidate.
    live = {str(row["occurrence_id"]): row for row in corpus["rows"]}
    checked = revision_proposals.validate(
        rows, live, live_snapshot_digest=str(source.get("snapshot_digest") or ""),
        live_project_identity=project, corpus_manifest=corpus["manifest"],
    )
    if checked.status != "imported" or checked.selected_count != len(rows):
        raise ValueError(f"提案校验失败：{checked.diagnostics}")
    path = Path(index_path).resolve().parent / PROPOSALS_NAME
    atomic_write_jsonl(path, rows, ensure_ascii=False)
    return str(path)
