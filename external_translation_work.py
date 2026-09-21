"""Versioned external candidates using Batch results and the shared preview writer.

The manifest is the commit point for result generations and submission receipts.
This module never writes game files: apply delegates to sync_translation_preview.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from atomic_io import atomic_write_json, atomic_write_jsonl, exclusive_file_lock, file_sha256
from cli_contract import MachineContractError, EXIT_BLOCKED, EXIT_INVALID_STATE
from engine_adapters import RenPyAdapter, ProjectDiscoveryRequest, build_translation_snapshot
from revision_corpus import item_snapshot_digest
import sync_translation_preview
import translation_quality


VERSION = 1
COMMANDS = frozenset({
    "work-export", "work-submit", "work-status", "work-read", "work-preview", "work-apply",
})


def is_work_manifest(manifest):
    """Keep damaged work markers on the fail-closed external dispatch path."""
    return manifest.get("execution") == "external_work" or "external_work" in manifest


def digest(value):
    """Hash canonical JSON; receipt identity ignores JSON whitespace/key ordering."""
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _batch():
    main = sys.modules.get('__main__')
    if Path(getattr(main, '__file__', '')).name == 'gemini_translate_batch.py':
        return main
    import gemini_translate_batch
    return gemini_translate_batch


def _fail(code, message, **details):
    raise MachineContractError(
        message, code_name=code, suggested_action="inspect_work_status_and_inputs",
        semantic_exit_code=EXIT_BLOCKED if "CONFLICT" in code else EXIT_INVALID_STATE,
        details=details,
    )


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _quality_glossary_path():
    """Resolve the glossary used by the shared checker, including a missing file."""
    raw = str(os.environ.get('GLOSSARY_FILE') or getattr(_batch().legacy, 'GLOSSARY_FILE', '') or '')
    if not raw:
        return ''
    path = Path(raw)
    if not path.is_absolute():
        project_path = Path(_batch().legacy.BASE_DIR) / path
        path = project_path if project_path.is_file() or not path.is_file() else path
    return _canonical(path)


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        _fail("WORK_ARTIFACT_INVALID", "工作包制品无法读取。", error=str(exc))


def _artifact(manifest, path):
    root = Path(manifest["_package_dir"]).resolve()
    candidate = root / path
    if Path(path).is_absolute() or not candidate.resolve().is_relative_to(root):
        _fail("WORK_ARTIFACT_INVALID", "工作包制品路径越界。")
    return candidate


def _save(manifest):
    # Do not race the global latest pointer or advertise a partially received work.
    _batch().save_manifest(manifest, update_latest=False)


@contextmanager
def _locked(target):
    manifest = _batch().load_manifest(target)
    with exclusive_file_lock(Path(manifest["_package_dir"]) / ".external_work.lock"):
        yield _load(target)


def _load(target):
    manifest = _batch().load_manifest(target)
    state = manifest.get("external_work")
    if (not isinstance(state, dict) or type(state.get("schema_version")) is not int
            or state["schema_version"] != VERSION):
        _fail("WORK_MANIFEST_INVALID", "目标不是支持的外部初译工作包。")
    package = state.get("package")
    if not isinstance(package, dict) or digest(package) != state.get("package_digest"):
        _fail("WORK_PACKAGE_CHANGED", "工作包合同已变化；请重新导出。")
    if _json(_artifact(manifest, "work.json")) != {**package, "package_digest": state["package_digest"]}:
        _fail("WORK_PACKAGE_CHANGED", "工作包交换副本与 manifest 不一致。")
    if digest(_manifest_binding(manifest)) != state.get("manifest_binding"):
        _fail("WORK_PACKAGE_CHANGED", "工作包执行目标或配置已变化。")
    result_path = _artifact(manifest, manifest["result_jsonl_path"])
    if not result_path.is_file() or file_sha256(result_path) != state.get("results_sha256"):
        _fail("WORK_RESULTS_CHANGED", "当前成果代摘要不符；不能沿用接收或检查状态。")
    return manifest


def _manifest_binding(manifest):
    return {key: manifest.get(key) for key in (
        "mode", "execution", "base_dir", "tl_dir", "target_language", "tl_subdir",
        "generation_target", "engine", "adapter_version", "files", "chunks", "settings",
        "submit_disabled", "input_jsonl_path", "glossary_file", "non_chinese_rules",
    )}


def check_binding(manifest):
    """Include immutable work inputs and current candidate generation in Batch checks."""
    state = manifest["external_work"]
    return {**{key: state.get(key) for key in (
        "package_digest", "manifest_binding", "results_sha256", "generation", "check_id",
    )}, 'conflicts_digest': digest(state.get('conflicts', {}))}


def _candidates(manifest):
    rows = _batch().load_result_rows_by_key(manifest, "external candidates")[0]
    candidates = {}
    for row in rows:
        for item in (row.get("normalized_response") or {}).get("translations", []):
            candidates[item["occurrence_id"]] = item
    return candidates


def _file_versions(package):
    request = package["discovery"]
    discovered = RenPyAdapter(legacy_module=_batch().legacy).discover_project(
        ProjectDiscoveryRequest(**{**request,
            "include_files": tuple(request["include_files"]),
            "include_prefixes": tuple(request["include_prefixes"]),
        })
    )
    return {item.file_rel_path: item.sha256 for item in discovered.source_documents}


def _fresh(manifest, *, preview=None, references=True):
    batch = _batch()
    batch.require_manifest_project_match(manifest, "external-work")
    package = manifest["external_work"]["package"]
    if package["adapter_version"] != RenPyAdapter(legacy_module=batch.legacy).adapter_version:
        _fail("WORK_ADAPTER_STALE", "Adapter 版本变化，请重新导出工作包。")
    if package["target_language"] != batch.legacy.PREP_LANGUAGE:
        _fail("WORK_LANGUAGE_STALE", "目标语言变化，请重新导出工作包。")
    if manifest['generation_target'] != batch._manifest_target_language_fields()['generation_target']:
        _fail("WORK_LANGUAGE_STALE", "生成目标变化，请重新导出工作包。")
    if package['quality_glossary_path'] != _quality_glossary_path():
        _fail("WORK_REFERENCE_STALE", "质量检查使用的术语文件路径已变化。")
    expected = package["file_digests"]
    current = _file_versions(package)
    postimages = {item["relative_path"]: item["preview_sha256"] for item in (preview or {}).get("files", [])}
    if preview and (preview.get('state') == 'applied' or manifest['external_work'].get('applied')):
        expected = {**expected, **postimages}
    changed = sorted(key for key in expected.keys() | current.keys()
                     if key not in expected or key not in current
                     or current[key] not in {expected[key], postimages.get(key, "")})
    if changed:
        _fail("WORK_SOURCE_STALE", "源或现译文件已变化，请重新导出工作包。", files=changed)
    if references:
        for ref in package["references"]:
            if ref["kind"] == "file":
                path = Path(ref["path"])
                changed = (not path.is_file() or file_sha256(path) != ref['sha256']) if ref['sha256'] else path.exists()
                if changed:
                    _fail("WORK_REFERENCE_STALE", "参考文件已变化。", reference=ref["path"])
            else:
                other = _load(ref["manifest_path"])
                # Reference links form an export-time DAG; do not recursively walk it.
                _fresh(other, preview=_preview(other), references=False)
                if other["external_work"]["package_digest"] != ref["package_digest"]:
                    _fail("WORK_REFERENCE_STALE", "参考工作包已变化。")
                live = _candidates(other)
                for row in ref["items"]:
                    candidate = live.get(row["occurrence_id"], {})
                    if candidate.get("candidate_digest") != row["candidate_digest"]:
                        _fail("WORK_REFERENCE_STALE", "引用的已审校候选版本已变化。",
                              occurrence_id=row["occurrence_id"])


def _preview(manifest):
    ref = manifest["external_work"].get("preview")
    if not ref:
        return None
    preview = sync_translation_preview.load_sync_preview(_artifact(manifest, ref["path"]))
    if preview["preview_fingerprint"] != ref["fingerprint"]:
        _fail("WORK_PREVIEW_CHANGED", "绑定预览已变化。")
    return preview


def _writeback_state(manifest):
    state = manifest["external_work"]
    preview = _preview(manifest)
    if preview:
        if (Path(preview["_manifest_path"]).parent / ".sync_writeback_transaction.json").exists():
            return "recovery_required"
        changed_files = [item for item in preview["files"] if item["source_sha256"] != item["preview_sha256"]]
        committed = any(
            (Path(manifest["tl_dir"]) / item["relative_path"]).is_file()
            and file_sha256(Path(manifest["tl_dir"]) / item["relative_path"]) == item["preview_sha256"]
            for item in changed_files
        )
        if state.get("applied"):
            return "applied"
        if committed or preview.get("state") == "applied":
            return "recovery_required"
        return "previewed"
    if (manifest.get("last_check_summary", {}).get("writeback_gate") or {}).get("decision") == "allow":
        return "checked"
    return "not_applied"


def _invalidate(manifest):
    state = manifest["external_work"]
    state.pop("preview", None)
    state.pop("check_id", None)
    for key in ("last_check_at", "last_check_summary", "last_check_report_path"):
        manifest.pop(key, None)


def _conflict(manifest, submission, code, message, ids, **details):
    """Persist an unselected conflict as evidence and revoke prior writeback approval."""
    if _writeback_state(manifest) in {'applied', 'recovery_required'}:
        _fail(code, message, **details)
    key = digest({'submission': submission, 'code': code})
    conflicts = manifest['external_work'].setdefault('conflicts', {})
    conflicts.setdefault(key, {'conflict_id': key, 'submission_id': submission['submission_id'],
                              'submission_digest': digest(submission), 'code': code,
                              'occurrence_ids': sorted(ids), 'status': 'unresolved', **details})
    conflicts[key]['status'] = 'unresolved'
    conflicts[key].pop('resolved_by', None)
    _invalidate(manifest)
    _save(manifest)
    _fail(code, message, conflict=conflicts[key])


def _require_no_conflicts(manifest):
    unresolved = [row for row in manifest['external_work'].get('conflicts', {}).values()
                  if row['status'] == 'unresolved']
    if unresolved:
        _fail('WORK_UNRESOLVED_CONFLICT', '竞争提交尚未处置；用当前候选版本和新提交 ID 显式订正。', conflicts=unresolved)


def export_work(*, output_dir=None, occurrence_ids=(), reference_files=(), reference_works=()):
    """Freeze pending native-catalog occurrences without building a model plan."""
    batch = _batch()
    batch.legacy.require_supported_generation_target()
    jobs = batch.collect_pending_file_jobs()
    snapshot = jobs.adapter_snapshot
    pending_ids = {task["id"] for job in jobs for task in job["tasks"]}
    occurrences = [item for item in snapshot.occurrences if item.unit.id in pending_ids]
    selected = set(occurrence_ids)
    if selected - {item.occurrence_id for item in occurrences}:
        _fail("WORK_UNKNOWN_OCCURRENCE", "范围包含未知或非待译 occurrence。")
    if selected:
        occurrences = [item for item in occurrences if item.occurrence_id in selected]
    if not occurrences:
        _fail("WORK_EMPTY_SCOPE", "没有可导出的待译条目；请先准备原生翻译模板。")
    references = []
    glossary_path = _quality_glossary_path()
    if glossary_path:
        glossary = Path(glossary_path)
        content = glossary.read_bytes() if glossary.is_file() else None
        references.append({'kind': 'file', 'role': 'quality_glossary', 'path': glossary_path,
                           'sha256': hashlib.sha256(content).hexdigest() if content is not None else None,
                           'text': content.decode('utf-8-sig') if content is not None else ''})
    for path in reference_files:
        resolved = str(Path(path).resolve())
        content = Path(resolved).read_bytes()
        references.append({"kind": "file", "path": resolved, "sha256": hashlib.sha256(content).hexdigest(),
                           "text": content.decode('utf-8-sig')})
    for target in reference_works:
        with _locked(target) as other:
            _fresh(other, preview=_preview(other))
            rows = [copy.deepcopy(item) for item in _candidates(other).values()
                    if item["review"]["status"] == "reviewed"]
            if not rows:
                _fail("WORK_REFERENCE_UNREVIEWED", "参考包没有已审校候选。")
            if _canonical(other["tl_dir"]) != _canonical(batch.legacy.TL_DIR):
                _fail("WORK_PROJECT_MISMATCH", "不能引用其他项目的候选。")
            references.append({"kind": "work", "manifest_path": other["_manifest_path"],
                               "package_digest": other["external_work"]["package_digest"], "items": rows})
    items = []
    chunks = []
    by_unit = {item.unit.id: item for item in occurrences}
    for job in jobs:
        tasks = [copy.deepcopy(task) for task in job["tasks"] if task["id"] in by_unit]
        for task in tasks:
            task['line_number'] = task['line'] + 1
        if not tasks:
            continue
        chunks.append({"key": f"external-{len(chunks)}", "file_rel_path": job["file_rel_path"],
                       "chunk_index": len(chunks), "items": tasks})
        for task in tasks:
            occurrence = by_unit[task["id"]]
            unit = occurrence.unit
            source = unit.source_text
            context = job.get('context_items', [])
            position = next((i for i, item in enumerate(context) if item.get('id') == unit.id), 0)
            neighbors = context[max(0, position - 2):position] + context[position + 1:position + 3]
            row = {"occurrence_id": occurrence.occurrence_id, "identity_v2": unit.id,
                   "source": source, "current_translation": unit.current_translation,
                   "snapshot_digest": item_snapshot_digest(source, unit.current_translation),
                   "file_rel_path": unit.file_rel_path, "speaker_id": unit.speaker_id,
                   "locator": occurrence.locator.to_dict(),
                   "constraints": {"engine": "renpy", "preserve_interpolation_and_tags": True},
                   "context": neighbors, "context_truncated": len(context) > len(neighbors) + 1}
            items.append(row)
    project_id = digest({"base_dir": _canonical(batch.legacy.BASE_DIR), "tl_dir": _canonical(batch.legacy.TL_DIR)})
    package = {
        "schema_version": VERSION, "kind": "external_translation_work", "package_id": uuid4().hex,
        "project_id": project_id, "engine": "renpy", "engine_version": "unknown",
        "adapter_version": snapshot.project.adapter_version,
        "target_language": batch.legacy.PREP_LANGUAGE,
        "generation_target": batch._manifest_target_language_fields()['generation_target'],
        "quality_glossary_path": glossary_path,
        "discovery": {"project_root": _canonical(batch.legacy.BASE_DIR),
                      "localization_root": _canonical(batch.legacy.TL_DIR),
                      "target_language": batch.legacy.PREP_LANGUAGE,
                      "include_files": sorted(batch.legacy.INCLUDE_FILES),
                      "include_prefixes": sorted(batch.legacy.INCLUDE_PREFIXES)},
        "file_digests": {item.file_rel_path: item.sha256 for item in snapshot.project.source_documents},
        "items": items, "references": references, "reference_digest": digest(references),
        "coverage": snapshot.report.to_dict(),
    }
    manifest = {
        "version": 2, "manifest_version": 2, "core_schema_version": 2,
        "mode": batch.MANIFEST_MODE_TRANSLATION, "execution": "external_work",
        "engine": "renpy", "adapter_version": package["adapter_version"],
        "base_dir": batch.legacy.BASE_DIR, "tl_dir": batch.legacy.TL_DIR,
        "glossary_file": glossary_path,
        **batch._manifest_target_language_fields(),
        **batch.batch_non_chinese_rules.manifest_non_chinese_rules_fields(
            {'non_chinese_rules': batch.BATCH_NON_CHINESE_RULES}),
        **translation_quality.manifest_quality_policy_fields(runtime_policy=batch.BATCH_QUALITY_POLICY),
        "job_state": "LOCAL_CANDIDATES", "submit_disabled": True,
        "batch_model": "", "job_name": "", "input_jsonl_path": "",
        "result_jsonl_path": "results.empty.jsonl", "settings": {}, "chunks": chunks,
        "files": {job["file_rel_path"]: {"path": job["file_path"], "task_count": len(job["tasks"])}
                  for job in jobs if any(task["id"] in by_unit for task in job["tasks"])},
        "summary": {"file_count": len(chunks), "chunk_count": len(chunks), "item_count": len(items)},
    }
    if _file_versions(package) != package["file_digests"]:
        _fail("WORK_SOURCE_STALE", "导出扫描期间文件变化，请重试。")
    if output_dir:
        root = Path(output_dir).resolve()
        if root.is_relative_to(Path(batch.legacy.TL_DIR).resolve()):
            _fail("WORK_OUTPUT_INVALID", "工作包不能写入翻译源目录。")
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(batch.create_batch_package_dir(f"external_{package['package_id']}"))
    manifest.update(_package_dir=str(root), _manifest_path=str(root / "manifest.json"))
    atomic_write_jsonl(root / "results.empty.jsonl", [])
    manifest["external_work"] = {
        "schema_version": VERSION, "package": package, "package_digest": digest(package),
        "manifest_binding": digest(_manifest_binding(manifest)), "generation": 0,
        "results_sha256": file_sha256(root / "results.empty.jsonl"), "receipts": {},
        "usage": "unknown", "model": "unknown",
    }
    atomic_write_json(root / "work.json", {**package, "package_digest": digest(package)}, ensure_ascii=False)
    _save(manifest)
    _fresh(manifest)
    return {"status": "exported", "manifest_path": manifest["_manifest_path"],
            "work_path": str(root / "work.json"), "package_digest": digest(package), "item_count": len(items)}


def submission_template(work, *, submission_id, producer):
    """Create the envelope fields for a host-generated submission (no model call)."""
    return {"schema_version": VERSION, "kind": "external_translation_submission",
            "submission_id": submission_id, "producer": producer,
            **{key: work[key] for key in ("package_id", "project_id", "package_digest", "reference_digest")},
            "items": [], "reference_proposals": []}


def submit_work(target, submission):
    """Atomically accept a partial submission or return its durable original receipt."""
    if not isinstance(submission, dict):
        _fail("WORK_SUBMISSION_INVALID", "提交必须为 JSON 对象。")
    sid = submission.get("submission_id")
    if not isinstance(sid, str) or not sid.strip():
        _fail("WORK_SUBMISSION_INVALID", "submission_id 不能为空。")
    try:
        submission_digest = digest(submission)
    except (TypeError, ValueError):
        _fail("WORK_SUBMISSION_INVALID", "提交必须为有限值 JSON 数据。")
    if type(submission.get("schema_version")) is not int:
        _fail("WORK_SUBMISSION_INVALID", "schema_version 必须为整数。")
    with _locked(target) as manifest:
        state = manifest["external_work"]
        package = state["package"]
        _batch().require_manifest_project_match(manifest, "work-submit")
        old = state["receipts"].get(sid)
        if old:
            if old["submission_digest"] != submission_digest:
                _conflict(manifest, submission, 'WORK_SUBMISSION_CONFLICT', '同一提交 ID 的内容不同。', old['accepted_ids'])
            return copy.deepcopy(old)
        _fresh(manifest)
        if _writeback_state(manifest) in {"applied", "recovery_required"}:
            _fail("WORK_RECOVERY_REQUIRED", "先完成写回恢复；写回后请导出新的订正语料。")
        expected = {"schema_version": VERSION, "kind": "external_translation_submission",
                    "package_digest": state["package_digest"],
                    **{key: package[key] for key in ("project_id", "package_id", "reference_digest")}}
        for key, value in expected.items():
            if submission.get(key) != value:
                _fail("WORK_SUBMISSION_STALE", "提交不属于当前工作包及参考版本。", field=key)
        producer = submission.get("producer")
        if not isinstance(producer, dict) or producer.get("type") not in {"agent", "human"}:
            _fail("WORK_SUBMISSION_INVALID", "producer.type 必须为 agent 或 human。")
        producer = {"model": "unknown", "usage": "unknown", **producer}
        rows = submission.get("items")
        if not isinstance(rows, list) or not rows:
            _fail("WORK_SUBMISSION_INVALID", "提交必须包含非空 items 数组。")
        proposals = submission.get("reference_proposals", [])
        if not isinstance(proposals, list) or any(
            not isinstance(row, dict) or row.get("kind") not in {"term", "style"}
            or row.get("disposition") not in {"accepted", "rejected", "deferred"}
            or not isinstance(row.get("text"), str) or not row["text"].strip()
            or not isinstance(row.get("reason"), str) or not row["reason"].strip()
            for row in proposals
        ):
            _fail("WORK_SUBMISSION_INVALID", "参考建议需要类型、文本、明确处置和理由。")
        scope = {row["occurrence_id"]: row for row in package["items"]}
        candidates = _candidates(manifest)
        seen = set()
        live_snapshot = build_translation_snapshot(RenPyAdapter(legacy_module=_batch().legacy),
                                                   ProjectDiscoveryRequest(**package["discovery"]))
        live = {row.occurrence_id: row for row in live_snapshot.occurrences}
        policy_items = {item['id']: (chunk, item) for chunk in manifest['chunks'] for item in chunk['items']}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("occurrence_id"), str):
                _fail("WORK_SUBMISSION_INVALID", "成果条目缺少 occurrence_id。")
            oid = row["occurrence_id"]
            if oid not in scope:
                _fail("WORK_UNKNOWN_OCCURRENCE", "未知或范围外 occurrence。", occurrence_id=oid)
            if oid in seen:
                _fail("WORK_DUPLICATE_OCCURRENCE", "同一提交重复 occurrence。", occurrence_id=oid)
            seen.add(oid)
            if row.get("snapshot_digest") != scope[oid]["snapshot_digest"]:
                _fail("WORK_ITEM_STALE", "条目源/现译快照不符。", occurrence_id=oid)
            current_digest = candidates.get(oid, {}).get("candidate_digest", "")
            if row.get("expected_candidate_digest") != current_digest:
                _conflict(manifest, submission, 'WORK_CANDIDATE_CONFLICT',
                          '竞争译文未覆盖；请基于当前候选版本显式订正。', [oid],
                          occurrence_id=oid, current_candidate_digest=current_digest)
            text, reason = row.get("translation"), row.get("reason")
            if not isinstance(text, str) or not text.strip() or not isinstance(reason, str) or not reason.strip():
                _fail("WORK_SUBMISSION_INVALID", "译文和理由必须为非空字符串。", occurrence_id=oid)
            review = row.get("review", {"status": "unreviewed"})
            if not isinstance(review, dict) or review.get("status") not in {"unreviewed", "reviewed"}:
                _fail("WORK_SUBMISSION_INVALID", "审校状态无效。")
            if review["status"] == "reviewed" and (not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
                _fail("WORK_SUBMISSION_INVALID", "已审校候选须记录 reviewer。")
            occurrence = live.get(oid)
            if occurrence is None:
                _fail("WORK_ITEM_STALE", "条目在当前源中不存在。", occurrence_id=oid)
            validation = RenPyAdapter(legacy_module=_batch().legacy).validate_translation(occurrence, text)
            policy_chunk, policy_item = policy_items[scope[oid]['identity_v2']]
            if validation.status != "pass" and not _batch()._adapter_target_language_policy_allows(
                manifest, policy_chunk, policy_item, occurrence.unit.source_text, text,
                reason_codes=validation.reason_codes,
            ):
                _fail("WORK_STRUCTURE_BLOCKED", "外部译文未通过现有 Adapter 结构校验。",
                      occurrence_id=oid, reasons=list(validation.reason_codes))
            candidate = {"id": scope[oid]["identity_v2"], "occurrence_id": oid,
                         "translation": text, "reason": reason, "review": copy.deepcopy(review),
                         "producer": copy.deepcopy(producer), "submission_id": sid,
                         "snapshot_digest": row["snapshot_digest"], "previous_candidate_digest": current_digest}
            candidate["candidate_digest"] = digest(candidate)
            candidates[oid] = candidate
        result_rows = []
        by_id = {row["id"]: row for row in candidates.values()}
        for chunk in manifest["chunks"]:
            translated = [by_id[item["id"]] for item in chunk["items"] if item["id"] in by_id]
            if translated:
                result_rows.append({"key": chunk["key"], "normalized_response": {"translations": translated},
                                    "response_semantics": {"normalized_response": "external_candidates"}})
        result_name = f"results.{digest(result_rows)}.jsonl"
        atomic_write_jsonl(_artifact(manifest, result_name), result_rows, ensure_ascii=False)
        _load(target)
        _fresh(manifest)
        state["generation"] += 1
        receipt = {"status": "accepted", "submission_id": sid, "submission_digest": submission_digest,
                   "package_digest": state["package_digest"], "generation": state["generation"],
                   "accepted_ids": list(seen), "candidate_digests": {oid: candidates[oid]["candidate_digest"] for oid in sorted(seen)},
                   "remaining_ids": [oid for oid in scope if oid not in candidates],
                   "reference_proposals": copy.deepcopy(proposals),
                   "accepted_at": datetime.now(timezone.utc).isoformat()}
        receipt["accepted_ids"] = sorted(seen)
        state["receipts"][sid] = receipt
        for conflict in state.get('conflicts', {}).values():
            if conflict['status'] == 'unresolved' and set(conflict['occurrence_ids']) <= seen:
                conflict.update(status='resolved', resolved_by=sid)
        manifest["result_jsonl_path"] = result_name
        state["results_sha256"] = file_sha256(_artifact(manifest, result_name))
        _invalidate(manifest)
        _save(manifest)
        return copy.deepcopy(receipt)


def status_work(target, *, include_items=False, offset=0, limit=100, remaining=False):
    """Read persisted facts and live staleness; never repairs or writes state."""
    with _locked(target) as manifest:
        state = manifest["external_work"]
        scope = state["package"]["items"]
        candidates = _candidates(manifest)
        pending = [row["occurrence_id"] for row in scope if row["occurrence_id"] not in candidates]
        diagnostics = []
        try:
            _fresh(manifest, preview=_preview(manifest))
        except MachineContractError as exc:
            diagnostics.append({"code": exc.code_name, "message": str(exc), **exc.details})
        writeback = _writeback_state(manifest)
        conflicts = list(state.get('conflicts', {}).values())
        current_status = 'conflict' if any(row['status'] == 'unresolved' for row in conflicts) else 'current'
        result = {"status": "stale" if diagnostics else current_status, "manifest_path": manifest["_manifest_path"],
                  "package_digest": state["package_digest"], "generation": state["generation"],
                  "completeness": "empty" if not candidates else "partial" if pending else "complete",
                  "remaining_ids": pending, "received_count": len(candidates), "total_items": len(scope),
                  "unreviewed_ids": [oid for oid, row in candidates.items() if row["review"]["status"] != "reviewed"],
                  "writeback": writeback, "diagnostics": diagnostics, "model": "unknown", "usage": "unknown",
                  "conflicts": conflicts,
                  "receipts": list(state["receipts"].values())}
        if include_items:
            selected = [row for row in scope if not remaining or row["occurrence_id"] in pending]
            result.update(items=[{**row, "candidate": candidates.get(row["occurrence_id"])}
                                 for row in selected[offset:offset + limit]],
                          offset=offset, limit=limit, total_selected=len(selected),
                          truncated=offset + limit < len(selected), references=state["package"]["references"])
        return result


def check_work(target):
    """Revoke prior authorization first, then run the shared Batch check service."""
    with _locked(target) as manifest:
        if _writeback_state(manifest) in {"applied", "recovery_required"}:
            _fail("WORK_RECOVERY_REQUIRED", "已开始写回，请重放 apply 完成恢复。")
        _invalidate(manifest)
        manifest["external_work"]["check_id"] = uuid4().hex
        _save(manifest)
        try:
            _fresh(manifest)
            _require_no_conflicts(manifest)
            checked = _batch().check_translation_results(manifest)
            _fresh(checked)
            return checked
        except BaseException:
            _invalidate(manifest)
            _save(manifest)
            raise


def validate_work_manifest(manifest, *, operation):
    """Validate work identity in the shared checker; external work has no model plan."""
    if operation != 'check':
        _fail('WORK_PROVIDER_DISABLED', '外部工作包没有 Provider 执行计划，请使用 work-submit。')
    _load(manifest['_manifest_path'])
    _fresh(manifest)
    return {'code': 'EXTERNAL_WORK_CURRENT', 'mode': 'external_work',
            'message': 'Versioned external work; model execution plan is not applicable.'}


def _checked(manifest):
    _fresh(manifest)
    _require_no_conflicts(manifest)
    _batch().require_safe_check_for_apply(manifest, update_latest=False)


def preview_work(target):
    """Bind the latest allow check to the existing full-file preview artifacts."""
    with _locked(target) as manifest:
        _checked(manifest)
        replacements, translated, failures, summary = _batch().collect_result_actions(manifest, validate_sources=True)
        if failures or summary.get("skipped_items"):
            _fail("WORK_PREVIEW_BLOCKED", "预览复核失败，请重新 check。")
        binding = {"manifest_path": manifest["_manifest_path"], "work": check_binding(manifest),
                   "check_fingerprint": manifest["last_check_summary"]["check_fingerprint"],
                   "writeback_gate": manifest["last_check_summary"]["writeback_gate"]}
        path, preview = sync_translation_preview.create_sync_preview(
            log_dir=manifest["_package_dir"], project_root=manifest["base_dir"], tl_dir=manifest["tl_dir"],
            files=_batch()._translation_preview_files(manifest, replacements, translated),
            quality_policy=_batch().BATCH_QUALITY_POLICY, glossary_file=manifest.get("glossary_file", ""),
            external_check_binding=binding,
        )
        _checked(manifest)
        manifest["external_work"]["preview"] = {"path": str(Path(path).relative_to(manifest["_package_dir"])),
                                                  "fingerprint": preview["preview_fingerprint"]}
        _save(manifest)
        return {"status": "previewed", "preview_path": path, "manifest_path": manifest["_manifest_path"],
                "summary": preview["summary"]}


def validate_preview_binding(preview):
    """Enforce external input/check/source guards inside the shared preview service."""
    binding = preview.get("external_check_binding")
    if not isinstance(binding, dict):
        _fail("WORK_PREVIEW_INVALID", "外部检查绑定缺失。")
    manifest = _load(binding.get("manifest_path"))
    ref = manifest["external_work"].get("preview", {})
    if ref.get("fingerprint") != preview.get("preview_fingerprint"):
        _fail("WORK_PREVIEW_STALE", "预览不是最近一次绑定预览。")
    if binding.get("work") != check_binding(manifest):
        _fail("WORK_CHECK_STALE", "检查后工作包或成果版本变化。")
    if binding.get("check_fingerprint") != manifest.get("last_check_summary", {}).get("check_fingerprint"):
        _fail("WORK_CHECK_STALE", "最近检查已变化。")
    _require_no_conflicts(manifest)
    _batch().require_safe_check_for_apply(manifest, update_latest=False)
    _fresh(manifest, preview=preview)
    return manifest


def apply_work(target):
    """Consume the shared bound preview; replay only verified committed file facts."""
    with _locked(target) as manifest:
        preview = _preview(manifest)
        if preview is None:
            _fail("WORK_PREVIEW_REQUIRED", "先运行 check 和 work-preview。")
        batch = _batch()
        # One coordinator at a time across external packages in the same project.
        project_lock = Path(manifest["base_dir"]) / "translation_context" / ".external_work_apply.lock"
        project_lock.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(project_lock):
            batch.require_manifest_project_match(manifest, 'work-apply')
            applied = sync_translation_preview.apply_sync_preview(
                preview["_manifest_path"], active_project_root=batch.legacy.BASE_DIR,
                active_tl_dir=batch.legacy.TL_DIR, active_quality_policy=batch.BATCH_QUALITY_POLICY,
                active_glossary_file=manifest.get("glossary_file", ""), allow_external=True,
            )
            # A failure here leaves the preview + actual postimages as recoverable facts.
            manifest["external_work"].setdefault("applied", {
                "preview_fingerprint": applied["preview_fingerprint"], "applied_at": applied["applied_at"],
                "files": applied["applied_files"],
            })
            manifest["applied_at"] = manifest["external_work"]["applied"]["applied_at"]
            manifest["apply_summary"] = {"applied_files": len(applied["applied_files"]),
                                         "applied_lines": applied["summary"].get("translated_items", 0)}
            _save(manifest)
            return {"status": "applied", "manifest_path": manifest["_manifest_path"],
                    "receipt": manifest["external_work"]["applied"]}


def add_cli(subparsers, add_output):
    """Register the advanced script-only commands on the existing CLI parser."""
    export = subparsers.add_parser("work-export", help="Experimental: export a Ren'Py external translation work package (no model calls).")
    export.add_argument("--output-dir", default=None, help="New artifact directory; must not exist or be inside TL_DIR.")
    export.add_argument("--occurrence-id", action="append", default=[], help="Select an occurrence from work-read; repeat for a fixed scope.")
    export.add_argument("--reference-file", action="append", default=[], help="Freeze a UTF-8 reference file; changes make the work stale.")
    export.add_argument("--reference-work", action="append", default=[], help="Reference reviewed candidate versions in another work manifest.")
    add_output(export)
    for name, help_text in (
        ("work-submit", "Atomically receive partial external candidates or explicit candidate revisions."),
        ("work-status", "Inspect remaining occurrences, staleness and recoverable writeback facts."),
        ("work-read", "Read frozen material and received candidates with pagination."),
        ("work-preview", "Create the existing bound preview after check=allow."),
        ("work-apply", "Apply or recover the bound preview; requires latest check=allow."),
    ):
        parser = subparsers.add_parser(name, help="Experimental: " + help_text)
        parser.add_argument("target", help="Explicit external work manifest.json or package directory.")
        if name == "work-submit":
            parser.add_argument("submission", help="External submission JSON file; never a Provider response.")
        if name == "work-read":
            parser.add_argument("--offset", type=int, default=0)
            parser.add_argument("--limit", type=int, default=100)
            parser.add_argument("--remaining", action="store_true")
        add_output(parser)


def run_cli(args):
    """Dispatch after local-only runtime configuration has loaded."""
    if args.command == "work-export":
        return export_work(output_dir=args.output_dir, occurrence_ids=args.occurrence_id,
                           reference_files=args.reference_file, reference_works=args.reference_work)
    if args.command == "work-submit":
        return submit_work(args.target, _json(args.submission))
    if args.command in {"work-status", "work-read"}:
        if getattr(args, "offset", 0) < 0 or not 1 <= getattr(args, "limit", 100) <= 1000:
            _fail("WORK_PAGE_INVALID", "offset 必须非负，limit 必须在 1 至 1000 之间。")
        return status_work(args.target, include_items=args.command == "work-read",
                           offset=getattr(args, "offset", 0), limit=getattr(args, "limit", 100),
                           remaining=getattr(args, "remaining", False))
    if args.command == "work-preview":
        return preview_work(args.target)
    return apply_work(args.target)
