"""Preview artifacts and guarded apply for synchronous translation."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from atomic_io import (
    atomic_write_json,
    atomic_write_many_lines,
    atomic_write_text,
    file_sha256,
    recover_atomic_write_transaction,
    sha256_text,
)

import translation_quality


SCHEMA = "sync_translation_preview"
VERSION = 1


def _canonical(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _safe_relative_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"Unsafe sync preview relative path: {value!r}")
    return path.as_posix()


def _inside(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _artifact_path(package_dir: Path, value: Any) -> Path:
    relative = _safe_relative_path(value)
    candidate = package_dir.joinpath(*PurePosixPath(relative).parts)
    if not _inside(package_dir, candidate):
        raise ValueError(f"Sync preview artifact escapes its package: {value!r}")
    return candidate


def _fingerprint_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": manifest.get("schema"),
        "version": manifest.get("version"),
        "created_at": manifest.get("created_at"),
        "project_root": manifest.get("project_root"),
        "tl_dir": manifest.get("tl_dir"),
        "report_path": manifest.get("report_path"),
        "report_sha256": manifest.get("report_sha256"),
        "summary": manifest.get("summary"),
        "files": manifest.get("files"),
    }
    for key in (
        "quality_policy",
        "quality_policy_digest",
        "quality_glossary_file",
        "quality_glossary_digest",
        "last_quality_findings_path",
        "quality_findings_sha256",
        "quality_findings_digest",
        "quality_finding_schema_version",
        "quality_rule_schema_version",
    ):
        if key in manifest:
            payload[key] = manifest.get(key)
    if "prompt_context" in manifest:
        payload["prompt_context"] = manifest.get("prompt_context")
    if "failures" in manifest:
        payload["failures"] = manifest.get("failures")
    if "model_contract" in manifest:
        payload["model_contract"] = manifest.get("model_contract")
    return payload


def _fingerprint(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        _fingerprint_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deserialize_writeback_plan(payload: Any):
    """Reconstruct a validated writeback plan from persisted manifest JSON.

    The payload must contain every plan and operation field consumed by
    ``render_writeback_plan``, with numeric line and column coordinates. Shape
    and coercion failures are normalized to ``ValueError`` so preview apply can
    reject malformed or edited manifests through one public failure mode.
    """

    from engine_adapters.contracts import WritebackOperation, WritebackPlan

    if not isinstance(payload, dict):
        raise ValueError("Sync preview writeback plan must be an object.")
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list):
        raise ValueError("Sync preview writeback plan operations must be a list.")
    operations = []
    operation_fields = (
        "operation_id",
        "kind",
        "occurrence_id",
        "target_root",
        "target_rel_path",
        "expected_file_sha256",
        "line",
        "start_col",
        "end_col",
        "expected_fragment_sha256",
        "expected_text_digest",
        "replacement_fragment",
        "validation_digest",
    )
    try:
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, dict):
                raise ValueError("Sync preview writeback operation must be an object.")
            values = {field: raw_operation[field] for field in operation_fields}
            values["line"] = int(values["line"])
            values["start_col"] = int(values["start_col"])
            values["end_col"] = int(values["end_col"])
            operations.append(WritebackOperation(**values))
        return WritebackPlan(
            engine=str(payload["engine"]),
            adapter_version=str(payload["adapter_version"]),
            project_identity_digest=str(payload["project_identity_digest"]),
            source_snapshot_fingerprint=str(payload["source_snapshot_fingerprint"]),
            coverage_digest=str(payload.get("coverage_digest") or ""),
            coverage_review_digest=str(payload.get("coverage_review_digest") or ""),
            operations=tuple(operations),
            plan_digest=str(payload["plan_digest"]),
            writeback_plan_schema_version=int(payload["writeback_plan_schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sync preview writeback plan: {exc}") from exc


def create_sync_preview(
    *,
    log_dir: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    tl_dir: str | os.PathLike[str],
    files: Iterable[dict[str, Any]],
    failures: Iterable[dict[str, Any]] = (),
    contract_diagnostics: dict[str, Any] | None = None,
    prompt_context: dict[str, Any] | None = None,
    quality_policy: dict[str, Any] | None = None,
    glossary_file: str | os.PathLike[str] = "",
) -> tuple[str, dict[str, Any]]:
    """Persist source/proposed snapshots, a unified diff, and a bound manifest.

    ``contract_diagnostics`` stores final validation, retry, and unresolved-item
    details in the bound manifest; those details are covered by its fingerprint.
    ``prompt_context`` records the local-context settings, macro identity, and
    per-batch context construction facts used for the run; when present it is
    also covered by the fingerprint so a changed macro/settings file invalidates
    an old preview at apply time.

    Structurally validated sync candidates passed through each file's
    ``quality_subjects`` list are run through the shared mechanical quality
    rules.  Findings are persisted next to the preview diff as
    ``quality_findings.jsonl`` and summarized by the same ``quality_gate``
    contract as Batch check.
    """
    created = datetime.now(timezone.utc)
    run_name = created.strftime("%Y%m%dT%H%M%S.%fZ")
    failure_entries = []
    for item in failures:
        entry = {
            "relative_path": str(item.get("relative_path") or ""),
            "reason_code": str(item.get("reason_code") or "adapter.writeback.block"),
            "message": str(item.get("message") or ""),
        }
        if item.get("item_id"):
            entry["item_id"] = str(item.get("item_id"))
        failure_entries.append(entry)
    contract = dict(contract_diagnostics or {})
    package_dir = Path(log_dir) / "sync_runs" / run_name
    package_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, Any]] = []
    quality_subjects: list[dict[str, Any]] = []
    report_lines = [
        "# Synchronous translation preview\n\n",
        f"Created: {created.isoformat()}\n\n",
    ]
    total_items = 0

    for raw in files:
        relative_path = _safe_relative_path(raw.get("relative_path"))
        source_text = str(raw.get("source_text", ""))
        preview_text = str(raw.get("preview_text", ""))
        if source_text == preview_text:
            continue

        source_rel = f"source/{relative_path}"
        preview_rel = f"preview/{relative_path}"
        source_path = _artifact_path(package_dir, source_rel)
        preview_path = _artifact_path(package_dir, preview_rel)
        atomic_write_text(source_path, source_text, encoding="utf-8")
        atomic_write_text(preview_path, preview_text, encoding="utf-8")

        progress_entries = [str(item) for item in raw.get("progress_entries") or []]
        translated_items = int(raw.get("translated_items") or len(progress_entries))
        total_items += translated_items
        entries.append(
            {
                "relative_path": relative_path,
                "source_snapshot_path": source_rel,
                "preview_path": preview_rel,
                "source_sha256": str(raw.get("source_sha256") or sha256_text(source_text)),
                "source_snapshot_sha256": sha256_text(source_text),
                "preview_sha256": sha256_text(preview_text),
                "progress_entries": progress_entries,
                "translated_items": translated_items,
            }
        )
        if raw.get("writeback_plan") is not None:
            entries[-1]["writeback_plan"] = raw.get("writeback_plan")
        if raw.get("prompt_context") is not None:
            entries[-1]["prompt_context"] = raw.get("prompt_context")
        for subject in raw.get("quality_subjects") or []:
            if not isinstance(subject, dict):
                continue
            normalized_subject = dict(subject)
            normalized_subject.setdefault("file_rel_path", relative_path)
            quality_subjects.append(normalized_subject)
        report_lines.extend(
            difflib.unified_diff(
                source_text.splitlines(keepends=True),
                preview_text.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
        if report_lines and not report_lines[-1].endswith("\n"):
            report_lines[-1] += "\n"
        report_lines.append("\n")

    report_path = package_dir / "preview.diff"
    atomic_write_text(report_path, "".join(report_lines), encoding="utf-8")

    effective_quality_policy = translation_quality.normalize_policy(quality_policy)
    glossary_text = str(glossary_file or "").strip()
    quality_glossary_map = translation_quality.load_glossary_map(
        glossary_text,
        base_dir=os.path.realpath(os.path.abspath(os.fspath(project_root))),
    )
    if not quality_glossary_map and glossary_text:
        quality_glossary_map = translation_quality.load_glossary_map(glossary_text)
    quality_glossary_digest = translation_quality.glossary_digest(
        quality_glossary_map
    )
    quality_findings = translation_quality.check_quality(
        quality_subjects,
        policy=effective_quality_policy,
        glossary_map=quality_glossary_map,
    )
    quality_findings_path = package_dir / "quality_findings.jsonl"
    translation_quality.write_findings(quality_findings_path, quality_findings)
    quality_gate = translation_quality.summarize_quality_gate(quality_findings)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "state": "preview_ready",
        "created_at": created.isoformat(),
        "project_root": os.path.realpath(os.path.abspath(os.fspath(project_root))),
        "tl_dir": os.path.realpath(os.path.abspath(os.fspath(tl_dir))),
        "report_path": "preview.diff",
        "report_sha256": file_sha256(report_path),
        "summary": {
            "files_changed": len(entries),
            "translated_items": total_items,
            "failure_files": len(failure_entries),
            "adapter_writeback_status": "partial" if failure_entries else "pass",
            "model_contract_status": (
                "partial"
                if (
                    contract.get("unresolved_ids")
                    or contract.get("terminal_reason_counts")
                )
                else "pass"
            ),
            "model_contract_first_pass_valid": int(
                contract.get("first_pass_valid") or 0
            ),
            "model_contract_expected": int(contract.get("final_expected") or 0),
            "model_contract_final_valid": int(contract.get("final_valid") or 0),
            "model_contract_targeted_retries": int(
                contract.get("targeted_retry_requests") or 0
            ),
            "model_contract_unresolved": len(contract.get("unresolved_ids") or []),
            "quality_gate": quality_gate,
            "quality_findings_count": len(quality_findings),
            "quality_finding_schema_version": (
                translation_quality.QUALITY_FINDING_SCHEMA_VERSION
            ),
            "quality_rule_schema_version": (
                translation_quality.QUALITY_RULE_SCHEMA_VERSION
            ),
            "quality_policy_digest": translation_quality.policy_digest(
                effective_quality_policy
            ),
            "quality_findings_digest": translation_quality.findings_digest(
                quality_findings
            ),
            "quality_glossary_path": glossary_text,
            "quality_glossary_digest": quality_glossary_digest,
            "quality_glossary_entries": len(quality_glossary_map),
            "quality_glossary_loaded": bool(
                not glossary_text or quality_glossary_map
            ),
        },
        "files": entries,
        "failures": failure_entries,
        "model_contract": contract,
        "quality_policy": effective_quality_policy,
        "quality_policy_digest": translation_quality.policy_digest(
            effective_quality_policy
        ),
        "quality_glossary_file": glossary_text,
        "quality_glossary_digest": quality_glossary_digest,
        "last_quality_findings_path": "quality_findings.jsonl",
        "quality_findings_sha256": file_sha256(quality_findings_path),
        "quality_findings_digest": translation_quality.findings_digest(
            quality_findings
        ),
        "quality_finding_schema_version": (
            translation_quality.QUALITY_FINDING_SCHEMA_VERSION
        ),
        "quality_rule_schema_version": (
            translation_quality.QUALITY_RULE_SCHEMA_VERSION
        ),
    }
    if prompt_context is not None:
        manifest["prompt_context"] = dict(prompt_context)
    manifest["preview_fingerprint"] = _fingerprint(manifest)
    manifest_path = package_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    return str(manifest_path), manifest


def load_sync_preview(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read sync preview manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Sync preview manifest must be a JSON object.")
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise ValueError("Unsupported sync preview manifest schema or version.")
    if not isinstance(manifest.get("files"), list):
        raise ValueError("Sync preview manifest files must be a list.")
    if manifest.get("preview_fingerprint") != _fingerprint(manifest):
        raise ValueError("Sync preview manifest changed after preview generation.")
    manifest["_manifest_path"] = str(path)
    return manifest


def prepare_sync_preview_apply(
    manifest_path: str | os.PathLike[str],
    *,
    active_project_root: str | os.PathLike[str],
    active_tl_dir: str | os.PathLike[str],
    active_quality_policy: dict[str, Any] | None = None,
    active_glossary_file: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate every source and artifact before the first project write."""
    manifest = load_sync_preview(manifest_path)
    if manifest.get("state") == "applied":
        raise ValueError("Sync preview has already been applied.")
    if _canonical(manifest.get("project_root", "")) != _canonical(active_project_root):
        raise ValueError("Sync preview belongs to a different project.")
    if _canonical(manifest.get("tl_dir", "")) != _canonical(active_tl_dir):
        raise ValueError("Sync preview belongs to a different translation directory.")
    if manifest.get("quality_rule_schema_version") not in {
        None,
        translation_quality.QUALITY_RULE_SCHEMA_VERSION,
    }:
        raise ValueError(
            "Quality rules changed since sync preview; regenerate the preview."
        )
    summary = manifest.get("summary") or {}
    quality_gate = summary.get("quality_gate")
    if isinstance(quality_gate, dict) and int(
        quality_gate.get("blocker_count") or 0
    ) > 0:
        raise ValueError(
            "Sync preview has quality blockers; resolve them and regenerate "
            "the preview before applying."
        )
    if isinstance(active_quality_policy, dict):
        expected_digest = summary.get("quality_policy_digest")
        current_digest = translation_quality.policy_digest(
            translation_quality.normalize_policy(active_quality_policy)
        )
        if expected_digest and current_digest != expected_digest:
            raise ValueError(
                "Quality policy changed since sync preview; regenerate the preview."
            )
    active_glossary_text = str(active_glossary_file or "").strip()
    if (
        active_glossary_file is not None
        and "quality_glossary_file" in manifest
        and active_glossary_text != str(manifest.get("quality_glossary_file") or "")
    ):
        raise ValueError(
            "Quality glossary changed since sync preview; regenerate the preview."
        )
    if active_glossary_file is not None and "quality_glossary_digest" in manifest:
        current_glossary_map = translation_quality.load_glossary_map(
            active_glossary_text,
            base_dir=os.path.realpath(
                os.path.abspath(os.fspath(active_project_root))
            ),
        )
        if not current_glossary_map and active_glossary_text:
            current_glossary_map = translation_quality.load_glossary_map(
                active_glossary_text
            )
        current_glossary_digest = translation_quality.glossary_digest(
            current_glossary_map
        )
        expected_glossary_digest = (
            manifest.get("quality_glossary_digest")
            or summary.get("quality_glossary_digest")
        )
        if current_glossary_digest != expected_glossary_digest:
            raise ValueError(
                "Quality glossary content changed since sync preview; "
                "regenerate the preview."
            )

    package_dir = Path(manifest["_manifest_path"]).parent
    if manifest.get("last_quality_findings_path"):
        quality_report_path = _artifact_path(
            package_dir,
            manifest.get("last_quality_findings_path"),
        )
        if not quality_report_path.is_file():
            raise ValueError("Sync preview quality findings are missing.")
        if file_sha256(quality_report_path) != manifest.get("quality_findings_sha256"):
            raise ValueError("Sync preview quality findings changed after preview generation.")
    report_path = _artifact_path(package_dir, manifest.get("report_path"))
    if not report_path.is_file() or file_sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("Sync preview diff report changed after preview generation.")
    target_root = Path(active_tl_dir).resolve()
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Sync preview file entry must be an object.")
        relative_path = _safe_relative_path(entry.get("relative_path"))
        if relative_path in seen:
            raise ValueError(f"Duplicate sync preview path: {relative_path}")
        seen.add(relative_path)
        target = target_root.joinpath(*PurePosixPath(relative_path).parts)
        if not _inside(target_root, target):
            raise ValueError(f"Sync preview target escapes TL_DIR: {relative_path}")
        source_snapshot = _artifact_path(package_dir, entry.get("source_snapshot_path"))
        preview_path = _artifact_path(package_dir, entry.get("preview_path"))
        if not target.is_file() or not source_snapshot.is_file() or not preview_path.is_file():
            raise ValueError(f"Sync preview file is missing: {relative_path}")
        if file_sha256(source_snapshot) != entry.get("source_snapshot_sha256"):
            raise ValueError(f"Sync preview source snapshot changed: {relative_path}")
        if file_sha256(preview_path) != entry.get("preview_sha256"):
            raise ValueError(f"Sync preview proposed file changed: {relative_path}")
        current_sha = file_sha256(target)
        source_sha = str(entry.get("source_sha256") or "")
        preview_sha = str(entry.get("preview_sha256") or "")
        if current_sha not in {source_sha, preview_sha}:
            raise ValueError(f"Source changed after sync preview: {relative_path}")
        preview_text = preview_path.read_bytes().decode("utf-8")
        writeback_plan_payload = entry.get("writeback_plan")
        if writeback_plan_payload is not None:
            from engine_adapters.contracts import SourceDocument
            from engine_adapters.writeback import render_writeback_plan

            source_content = source_snapshot.read_bytes()
            source_document = SourceDocument(
                file_rel_path=relative_path,
                file_path=str(target),
                size=len(source_content),
                sha256=hashlib.sha256(source_content).hexdigest(),
                content=source_content,
            )
            plan = _deserialize_writeback_plan(writeback_plan_payload)
            rendered_by_file = render_writeback_plan(plan, (source_document,))
            rendered_text = "".join(rendered_by_file.get(relative_path, ()))
            if source_content.startswith(b"\xef\xbb\xbf") and not rendered_text.startswith("\ufeff"):
                rendered_text = "\ufeff" + rendered_text
            if rendered_text != preview_text:
                raise ValueError(
                    f"Sync preview writeback plan does not match proposed file: {relative_path}"
                )
        prepared.append(
            {
                "entry": entry,
                "target": target,
                "preview_text": preview_text,
                "already_applied": current_sha == preview_sha,
            }
        )
    return manifest, prepared


def apply_sync_preview(
    manifest_path: str | os.PathLike[str],
    *,
    active_project_root: str | os.PathLike[str],
    active_tl_dir: str | os.PathLike[str],
    on_file_applied: Callable[[dict[str, Any]], None] | None = None,
    active_quality_policy: dict[str, Any] | None = None,
    active_glossary_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    transaction_path = manifest_file.parent / ".sync_writeback_transaction.json"
    recover_atomic_write_transaction(transaction_path)
    manifest, prepared = prepare_sync_preview_apply(
        manifest_path,
        active_project_root=active_project_root,
        active_tl_dir=active_tl_dir,
        active_quality_policy=active_quality_policy,
        active_glossary_file=active_glossary_file,
    )
    applied_paths: list[str] = []
    try:
        writes = [
            (item["target"], item["preview_text"].splitlines(keepends=True))
            for item in prepared
            if not item["already_applied"]
        ]
        if writes:
            atomic_write_many_lines(
                writes,
                journal_path=transaction_path,
                encoding="utf-8",
            )
        for item in prepared:
            entry = item["entry"]
            applied_paths.append(entry["relative_path"])
            if on_file_applied is not None:
                on_file_applied(entry)
    except Exception as exc:
        manifest["state"] = "apply_failed"
        manifest["last_apply_failure"] = str(exc)
        atomic_write_json(manifest["_manifest_path"], _public_manifest(manifest), ensure_ascii=False, indent=2)
        raise

    manifest["state"] = "applied"
    manifest["applied_at"] = datetime.now(timezone.utc).isoformat()
    manifest["applied_files"] = applied_paths
    manifest.pop("last_apply_failure", None)
    atomic_write_json(manifest["_manifest_path"], _public_manifest(manifest), ensure_ascii=False, indent=2)
    return manifest


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}
