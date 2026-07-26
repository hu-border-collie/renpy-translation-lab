"""Safe hand-off from selected final-review findings to revision preview/apply."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Mapping, Sequence

import final_review as fr
from atomic_io import atomic_write_json, atomic_write_jsonl, atomic_write_text


def finding_digest(finding: Mapping[str, Any]) -> str:
    keys = ("finding_id", "identity_v2", "file_rel_path", "source", "current_translation",
            "suggested_revision", "review_unit_id", "review_unit_digest")
    return fr.stable_json_sha256({key: finding.get(key) or "" for key in keys})


def _write_findings(package: Mapping[str, Any], findings: Sequence[Mapping[str, Any]]) -> None:
    paths = package["paths"]
    manifest = dict(package["manifest"])
    manifest.pop("_manifest_path", None)
    manifest.pop("_package_dir", None)
    manifest["summary"] = {**dict(manifest.get("summary") or {}), "finding_count": len(findings)}
    atomic_write_jsonl(paths["findings"], findings, ensure_ascii=False)
    atomic_write_json(paths["manifest"], manifest, ensure_ascii=False, indent=2)
    atomic_write_text(paths["report"], fr.format_campaign_report_markdown(
        manifest, package.get("units") or [], findings,
    ))


def sync_linked_findings(manifest: Mapping[str, Any], state: str, identity_ids=None) -> int:
    """Advance finding state only after the corresponding revision step succeeds."""
    link = manifest.get("final_review_source")
    if not isinstance(link, Mapping):
        return 0
    allowed = (fr.REVISION_STATE_CANDIDATE, fr.REVISION_STATE_PREVIEWED, fr.REVISION_STATE_APPLIED)
    if state not in allowed:
        raise SystemExit(f"Unsupported final-review revision state: {state}")
    source = str(link.get("manifest_path") or "").strip()
    linked = link.get("findings")
    if not source or not isinstance(linked, list) or not linked:
        raise SystemExit("Final-review provenance is incomplete; refusing finding-state update.")
    try:
        package = fr.load_campaign_package(source)
    except fr.FinalReviewError as exc:
        raise SystemExit(f"Linked final-review campaign is unavailable: {exc}") from exc
    actual_snapshot = str(package.get("snapshot", {}).get("snapshot_digest") or package["manifest"].get("snapshot_digest") or "")
    if actual_snapshot != str(link.get("snapshot_digest") or ""):
        raise SystemExit("Linked final-review campaign snapshot changed; refusing finding-state update.")
    links = {str(row.get("finding_id") or ""): row for row in linked if isinstance(row, Mapping)}
    selected_identities = None if identity_ids is None else {
        str(value) for value in identity_ids if str(value)
    }
    findings = [dict(row) for row in package.get("findings") or []]
    found = set()
    rank = {fr.REVISION_STATE_NONE: 0, fr.REVISION_STATE_CANDIDATE: 1,
            fr.REVISION_STATE_PREVIEWED: 2, fr.REVISION_STATE_APPLIED: 3}
    changed = 0
    for finding in findings:
        finding_id = str(finding.get("finding_id") or "")
        row = links.get(finding_id)
        if row is None:
            continue
        found.add(finding_id)
        if finding_digest(finding) != str(row.get("digest") or ""):
            raise SystemExit(f"Linked final-review finding changed ({finding_id}); refusing state update.")
        should_update = (
            selected_identities is None
            or str(finding.get("identity_v2") or "") in selected_identities
        )
        if should_update and rank.get(
            str(finding.get("revision_state") or "none"), 0
        ) <= rank[state]:
            finding["selection_state"] = fr.SELECTION_STATE_SELECTED
            finding["revision_state"] = state
            changed += 1
    missing = set(links) - found
    if missing:
        raise SystemExit("Linked final-review findings are missing: " + ", ".join(sorted(missing)))
    _write_findings(package, findings)
    return changed


def _select(package: Mapping[str, Any], finding_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    if fr.collect_campaign_status(package=package).get("status") != fr.STATUS_DONE:
        raise SystemExit("Final-review campaign is not complete; resolve pending/stale/failed units first.")
    requested = {str(value).strip() for value in (finding_ids or []) if str(value).strip()}
    findings = [dict(row) for row in package.get("findings") or []]
    missing = requested - {str(row.get("finding_id") or "") for row in findings}
    if missing:
        raise SystemExit("Unknown final-review finding id(s): " + ", ".join(sorted(missing)))
    selected = []
    for finding in findings:
        use = (str(finding.get("finding_id") or "") in requested if requested
               else finding.get("selection_state") == fr.SELECTION_STATE_SELECTED)
        if not use:
            continue
        if finding.get("revision_state") == fr.REVISION_STATE_APPLIED:
            raise SystemExit(f"Finding was already applied: {finding.get('finding_id')}")
        if not str(finding.get("suggested_revision") or "").strip():
            raise SystemExit(f"Finding has no suggested revision: {finding.get('finding_id')}")
        selected.append(finding)
    if not selected:
        raise SystemExit("No findings selected. Pass --finding-id or select findings in the GUI first.")
    return selected


def create_revision_package(batch: Any, target: str | None, finding_ids: Sequence[str] | None):
    """Build local candidates and immediately use the existing revision preview gate."""
    try:
        package = fr.load_campaign_package(batch.manifest_path_for_target(target))
    except fr.FinalReviewError as exc:
        raise SystemExit(f"Unable to load final-review campaign: {exc}") from exc
    source_manifest = package["manifest"]
    batch.require_manifest_project_match(source_manifest, "final-review-create-revisions")
    selected = _select(package, finding_ids)
    units = {str(unit.get("unit_id") or ""): unit for unit in package.get("units") or []}
    live_jobs = batch.collect_revision_file_jobs()
    live_items = {str(item.get("id") or ""): item for job in live_jobs for item in job.get("items") or []}
    selected_by_identity: dict[str, list[dict[str, Any]]] = {}
    for finding in selected:
        finding_id = str(finding.get("finding_id") or "")
        identity = str(finding.get("identity_v2") or "")
        unit = units.get(str(finding.get("review_unit_id") or ""))
        item = live_items.get(identity)
        if not identity or unit is None or item is None:
            raise SystemExit(f"Finding no longer resolves in the active project: {finding_id}")
        if unit.get("status") != fr.STATUS_DONE or str(unit.get("input_digest") or "") != str(finding.get("review_unit_digest") or ""):
            raise SystemExit(f"Finding review unit is stale: {finding_id}")
        pairs = (("file_rel_path", "file_rel_path"), ("source", "source"),
                 ("current_translation", "current_translation"))
        if any(str(item.get(a) or "") != str(finding.get(b) or "") for a, b in pairs):
            raise SystemExit(f"Finding source/translation changed since review: {finding_id}. Resume final review first.")
        selected_by_identity.setdefault(identity, []).append(finding)
    for identity, rows in selected_by_identity.items():
        if len({str(row.get("suggested_revision") or "").strip() for row in rows}) != 1:
            raise SystemExit(f"Selected findings propose conflicting revisions for {identity}; choose one finding.")

    filtered = []
    for job in live_jobs:
        items = [item for item in job.get("items") or [] if str(item.get("id") or "") in selected_by_identity]
        if items:
            filtered.append({**job, "items": items, "task_count": len(items)})
    chunks = batch.build_revision_chunks(filtered, chunk_size=batch.REVISION_CHUNK_SIZE)
    if not chunks:
        raise SystemExit("No revision candidates could be built from the selection.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    package_dir = batch.create_batch_package_dir(f"{stamp}_{batch.guess_project_slug()}_final_review_revisions")
    atomic_write_text(os.path.join(package_dir, "requests.jsonl"), "")
    result_rows = []
    for chunk in chunks:
        results = []
        for item in chunk["items"]:
            linked = selected_by_identity[str(item.get("id") or "")]
            reasons = [str(row.get("reason") or "").strip() for row in linked if str(row.get("reason") or "").strip()]
            results.append({"id": item["id"], "should_update": True,
                            "revised_translation": str(linked[0]["suggested_revision"]).strip(),
                            "reason": "；".join(dict.fromkeys(reasons)) or "来自最终审查的人工选择"})
        text = json.dumps(results, ensure_ascii=False)
        result_rows.append({"key": chunk["key"], "response": {"candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}})
    atomic_write_jsonl(os.path.join(package_dir, "results.jsonl"), result_rows, ensure_ascii=False)
    snapshot = str(package.get("snapshot", {}).get("snapshot_digest") or source_manifest.get("snapshot_digest") or "")
    manifest = {
        "version": 2, "manifest_version": 2, "core_schema_version": 2,
        "mode": batch.MANIFEST_MODE_REVISION, "execution": "final_review_handoff",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "display_name": f"{batch.REVISION_DISPLAY_NAME_PREFIX}-{batch.guess_project_slug()}-{stamp}",
        "batch_model": source_manifest.get("model") or source_manifest.get("batch_model") or "",
        "base_dir": batch.legacy.BASE_DIR, "tl_dir": batch.legacy.TL_DIR,
        **batch._manifest_target_language_fields(),
        **batch.batch_non_chinese_rules.manifest_non_chinese_rules_fields(),
        "input_jsonl_path": "requests.jsonl", "result_jsonl_path": "results.jsonl",
        "job_name": "", "job_state": "LOCAL_CANDIDATES", "submit_disabled": True,
        "settings": {"revision_chunk_size": batch.REVISION_CHUNK_SIZE},
        "revision_settings": {"chunk_size": batch.REVISION_CHUNK_SIZE, "candidate_source": "final_review"},
        "summary": {"file_count": len(filtered), "chunk_count": len(chunks),
                    "item_count": sum(len(chunk["items"]) for chunk in chunks),
                    "finding_count": len(selected)},
        "files": {job["file_rel_path"]: {"path": job["file_path"], "task_count": job["task_count"]} for job in filtered},
        "chunks": chunks,
        "final_review_source": {"manifest_path": package["paths"]["manifest"],
            "snapshot_digest": snapshot,
            "findings": [{"finding_id": str(row.get("finding_id") or ""),
                          "identity_v2": str(row.get("identity_v2") or ""),
                          "digest": finding_digest(row)} for row in selected]},
    }
    manifest_path = os.path.join(package_dir, "manifest.json")
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    batch.remember_latest_manifest(manifest_path)
    loaded = batch.load_manifest(manifest_path)
    sync_linked_findings(loaded, fr.REVISION_STATE_CANDIDATE)
    print(f"Created final-review revision package: {package_dir}")
    print(f"Selected findings: {len(selected)}")
    print("No .rpy files were written; generating the required revision preview now.")
    return batch.preview_revisions(manifest_path)
