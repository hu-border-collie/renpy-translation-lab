"""MV bindings for the existing Gemini Batch plan and atomic writeback.

Event interpretation, result validation and JSON rendering belong to the MV
Adapter. Batch retains its request, check, recovery and apply contracts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

from engine_adapters.project_config import read_project_configuration
from engine_adapters.coverage import digest_json
from execution.mv_sync import live_material_identity, material_identity, require_material_fresh
from quality_review import translation_plan


def file_jobs(snapshot, *, include_complete_files=False, include_task_payloads=True):
    """Project the reviewed occurrences into Batch's shared job shape."""
    pending = snapshot.pending_tasks_by_file
    units_by_file = translation_plan.units_by_file_from_occurrences(snapshot.selected_occurrences)
    jobs = []
    for document in snapshot.project.target_documents:
        relative = document.file_rel_path
        tasks = [dict(item) for item in pending.get(relative, ())]
        total = len(units_by_file.get(relative, ()))
        translated = total - len(tasks)
        if not tasks and not (include_complete_files and translated):
            continue
        jobs.append({
            'file_rel_path': relative, 'file_path': document.file_path,
            'task_count': len(tasks), 'translated_count': translated,
            'tasks': tasks if include_task_payloads else [],
        })
    translation_plan.attach_context_items_from_units(jobs, units_by_file)
    return jobs


def require_supported_context(*, story_memory=False, project_analysis=False):
    """Reject unsupported MV context before any retrieval/provider work."""
    if story_memory or project_analysis:
        raise ValueError('MV Batch does not support Story Memory or Project Analysis injection; disable these layers.')


def receipt_path(manifest):
    return Path(manifest['_package_dir']) / 'mv_apply_receipt.json'


def read_receipt(manifest, check_validator):
    """Accept a committed receipt only for the same checked requests/results."""
    path = receipt_path(manifest)
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(receipt, dict):
        raise ValueError('MV Batch committed receipt must be an object.')
    check_validator(manifest)
    digest = receipt.pop('receipt_digest', '')
    if digest != digest_json(receipt):
        raise ValueError('MV Batch committed receipt was changed.')
    receipt['receipt_digest'] = digest
    if (receipt.get('schema_version') != 1
            or receipt.get('plan_fingerprint') != manifest['translation_plan']['plan_fingerprint']
            or receipt.get('check_fingerprint') != manifest['last_check_summary']['check_fingerprint']
            or not isinstance(receipt.get('files'), dict)
            or not receipt['files']
            or set(receipt['files']) - set(manifest.get('files') or {})):
        raise ValueError('MV Batch committed receipt does not match the current check.')
    translated = receipt.get('translated_by_file')
    allowed = {path: {str(item['id']) for chunk in manifest['chunks'] if chunk['file_rel_path'] == path
                      for item in chunk['items']} for path in receipt['files']}
    if (not isinstance(translated, dict) or set(translated) != set(receipt['files'])
            or any(not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids)
                   or len(ids) != len(set(ids)) or set(ids) - allowed[path]
                   for path, ids in translated.items())
            or receipt.get('apply_summary', {}).get('applied_lines') != sum(map(len, translated.values()))
            or receipt.get('apply_summary', {}).get('applied_files') != len(translated)):
        raise ValueError('MV Batch committed progress does not match its canonical items.')
    return receipt


def split_context(manifest, load_manifest):
    """Resolve a retry to its split part, retaining the frozen parent binding."""
    leaf = manifest
    seen = set()
    while leaf.get('retry_of_manifest'):
        path = str(Path(leaf['retry_of_manifest']).resolve())
        if path in seen or len(seen) >= 32:
            raise ValueError('MV Batch retry ancestry is invalid.')
        seen.add(path)
        leaf = load_manifest(path)
        translation_plan.validate_plan_fingerprint(leaf['translation_plan'])
        if leaf['translation_plan']['source_identity'] != manifest['translation_plan']['source_identity']:
            raise ValueError('MV Batch retry material binding changed.')
    if not leaf.get('split_from_manifest'):
        return None
    parent = load_manifest(leaf['split_from_manifest'])
    translation_plan.validate_plan_fingerprint(parent['translation_plan'])
    children = [str(Path(path).resolve()) for path in parent.get('split_children') or []]
    if (len(children) != len(set(children))
            or str(Path(leaf['_manifest_path']).resolve()) not in children
            or parent['translation_plan']['source_identity'] != manifest['translation_plan']['source_identity']):
        raise ValueError('MV Batch split parent binding changed.')
    return leaf, children


def prior_split_outputs(manifest, load_manifest, check_validator, *, require_previous=False):
    """Permit only earlier, committed and still-checked split writes.

    All parts retain the original material identity. Each part is applied in
    order, and the next part accepts the exact committed bytes of its siblings;
    arbitrary target edits, changed results and missing receipts remain stale.
    """
    context = split_context(manifest, load_manifest)
    if context is None:
        return {}
    leaf, children = context
    index = children.index(str(Path(leaf['_manifest_path']).resolve()))
    outputs = {}
    for path in children[:index]:
        sibling = load_manifest(path)
        translation_plan.validate_plan_fingerprint(sibling['translation_plan'])
        if (sibling['translation_plan']['source_identity'] != manifest['translation_plan']['source_identity']
                or str(Path(sibling.get('split_from_manifest') or '').resolve())
                != str(Path(leaf['split_from_manifest']).resolve())):
            raise ValueError('MV Batch sibling material binding changed.')
        receipt = read_receipt(sibling, check_validator)
        if receipt is None:
            if require_previous:
                raise ValueError('Apply earlier MV Batch split parts before this part.')
            continue
        outputs.update(receipt['files'])
    return outputs


def snapshot_for_manifest(runtime, manifest, load_manifest, check_validator, *, own_receipt=None):
    """Re-audit live JSON and retain the original review across known split writes."""
    from engine_adapters.coverage import (
        CoverageReport, default_coverage_review_path, load_review_record, validate_review_record,
    )

    outputs = prior_split_outputs(manifest, load_manifest, check_validator)
    if own_receipt is not None:
        outputs.update(own_receipt['files'])
    snapshot = runtime.current_mv_sync_snapshot(require_review=not bool(outputs))
    expected = manifest['translation_plan']['source_identity'].get('material_identity') or {}
    if outputs:
        payload = manifest.get('mv_coverage_report') or {}
        original = CoverageReport(**{field.name: payload[field.name]
            for field in fields(CoverageReport) if field.name in payload})
        if original.coverage_digest != expected.get('coverage_digest'):
            raise ValueError('MV Batch original coverage binding changed.')
        review = validate_review_record(
            load_review_record(default_coverage_review_path(snapshot.project.localization_root)),
            original, snapshot.inventory)
        if not review.policy_satisfied:
            raise ValueError('MV Batch original coverage review is no longer valid.')
        snapshot = replace(snapshot, review_error='', review_report=original,
            project=replace(snapshot.project, coverage_review_digest=review.coverage_review_digest))
    current = material_identity(snapshot)
    require_material_fresh(expected, current, applied_files=outputs)
    if any(current['target_files'].get(path) != digest for path, digest in outputs.items()):
        raise ValueError('MV Batch committed target bytes no longer match the receipt.')
    return snapshot


def build_receipt(manifest, writes, translated_by_file, summary, *, applied_at):
    """Write this receipt in the same existing transaction as the target JSON."""
    root = Path(manifest['tl_dir']).resolve()
    receipt = {
        'schema_version': 1,
        'plan_fingerprint': manifest['translation_plan']['plan_fingerprint'],
        'check_fingerprint': manifest['last_check_summary']['check_fingerprint'],
        'applied_at': applied_at,
        'files': {Path(path).resolve().relative_to(root).as_posix():
                  hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest() for path, lines in writes},
        'translated_by_file': {path: sorted(ids) for path, ids in translated_by_file.items()},
        'apply_summary': {
            'applied_files': len(translated_by_file),
            'applied_lines': sum(len(ids) for ids in translated_by_file.values()),
            'candidate_items': summary.get('candidate_valid_items', summary['valid_items']),
            'recoverable_items': summary['valid_items'],
            'skipped_items': summary.get('skipped_items', 0),
            'source_mismatch_items': summary.get('source_mismatch_items', 0),
            'failure_count': summary.get('failure_items', 0), 'rag': {},
        },
    }
    receipt['receipt_digest'] = digest_json(receipt)
    return receipt


def commit_guards(manifest, snapshot, writes, *, config_file, check_validator, prior_outputs=None):
    """Bind exact target preimages and roll back a commit on material drift."""
    root = Path(snapshot.project.localization_root).resolve()
    documents = {doc.file_rel_path: doc for doc in snapshot.project.target_documents}
    expected = manifest['translation_plan']['source_identity']['material_identity']
    preimages, applied = {}, dict(prior_outputs or {})
    for path, lines in writes:
        relative = Path(path).resolve().relative_to(root).as_posix()
        preimages[path] = documents[relative].sha256
        applied[relative] = hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()

    def validate_commit():
        # Normalize SystemExit-based CLI errors so the atomic transaction
        # immediately rolls back every target, rather than leaving a journal.
        try:
            check_validator()
            current = live_material_identity(snapshot, read_project_configuration(config_file), config_file)
            require_material_fresh(expected, current, applied_files=applied)
            if any(current['target_files'].get(path) != digest for path, digest in applied.items()):
                raise ValueError('MV Batch committed target bytes changed during commit.')
        except SystemExit as exc:
            raise ValueError(str(exc)) from exc

    return {'expected_preimages': preimages, 'post_commit_validator': validate_commit}
