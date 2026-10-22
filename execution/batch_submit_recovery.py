# -*- coding: utf-8 -*-
"""Submit journal and recovery helpers for Batch manifest idempotency."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime
from typing import Any

SUBMIT_JOURNAL_FILENAME = 'submit_journal.jsonl'

SUBMIT_STATE_STARTING = 'starting'
SUBMIT_STATE_UPLOADED = 'uploaded'
SUBMIT_STATE_JOB_CREATED = 'job_created'
SUBMIT_STATE_COMMITTED = 'committed'

EVENT_ATTEMPT_STARTED = 'submit_attempt_started'
EVENT_UPLOAD_COMPLETED = 'upload_completed'
EVENT_JOB_CREATED = 'job_created'
EVENT_MANIFEST_COMMITTED = 'manifest_committed'

BLOCKED_MESSAGE_PREFIX = 'Submit blocked:'
RECOVER_HINT = 'Run recover-submit on this package before submitting again.'


def submit_journal_path(package_dir: str) -> str:
    return os.path.join(package_dir, SUBMIT_JOURNAL_FILENAME)


def new_submit_attempt_id() -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{stamp}_{secrets.token_hex(4)}'


def compute_request_checksum(manifest: dict[str, Any]) -> str:
    jsonl_path = manifest.get('input_jsonl_path')
    if not isinstance(jsonl_path, str) or not jsonl_path.strip():
        raise FileNotFoundError('Manifest is missing input_jsonl_path.')
    path = jsonl_path.strip()
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def append_submit_journal_entry(package_dir: str, entry: dict[str, Any]) -> None:
    payload = dict(entry)
    payload.setdefault('at', datetime.now().isoformat(timespec='seconds'))
    journal_path = submit_journal_path(package_dir)
    with open(journal_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def read_submit_journal_entries(package_dir: str) -> list[dict[str, Any]]:
    journal_path = submit_journal_path(package_dir)
    if not os.path.isfile(journal_path):
        return []

    entries: list[dict[str, Any]] = []
    with open(journal_path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
    return entries


def begin_submit_attempt(manifest: dict[str, Any], *, package_dir: str) -> str:
    attempt_id = new_submit_attempt_id()
    checksum = compute_request_checksum(manifest)
    manifest['submit_attempt_id'] = attempt_id
    manifest['request_checksum'] = checksum
    manifest['submit_state'] = SUBMIT_STATE_STARTING
    manifest.pop('submit_recovery_hint', None)
    append_submit_journal_entry(
        package_dir,
        {
            'event': EVENT_ATTEMPT_STARTED,
            'submit_attempt_id': attempt_id,
            'display_name': manifest.get('display_name', ''),
            'batch_model': manifest.get('batch_model', ''),
            'request_checksum': checksum,
        },
    )
    return attempt_id


def record_upload_completed(
    manifest: dict[str, Any],
    *,
    package_dir: str,
    uploaded_file_name: str,
) -> None:
    manifest['uploaded_file_name'] = uploaded_file_name
    manifest.setdefault('uploaded_file_names', [])
    if uploaded_file_name and uploaded_file_name not in manifest['uploaded_file_names']:
        manifest['uploaded_file_names'].append(uploaded_file_name)
    manifest['submit_state'] = SUBMIT_STATE_UPLOADED
    append_submit_journal_entry(
        package_dir,
        {
            'event': EVENT_UPLOAD_COMPLETED,
            'submit_attempt_id': manifest.get('submit_attempt_id', ''),
            'display_name': manifest.get('display_name', ''),
            'batch_model': manifest.get('batch_model', ''),
            'request_checksum': manifest.get('request_checksum', ''),
            'uploaded_file_name': uploaded_file_name,
        },
    )


def record_job_created(
    manifest: dict[str, Any],
    *,
    package_dir: str,
    job_name: str,
    job_state: str,
    uploaded_file_name: str,
) -> None:
    append_submit_journal_entry(
        package_dir,
        {
            'event': EVENT_JOB_CREATED,
            'submit_attempt_id': manifest.get('submit_attempt_id', ''),
            'display_name': manifest.get('display_name', ''),
            'batch_model': manifest.get('batch_model', ''),
            'request_checksum': manifest.get('request_checksum', ''),
            'uploaded_file_name': uploaded_file_name,
            'job_name': job_name,
            'job_state': job_state,
        },
    )
    manifest['submit_state'] = SUBMIT_STATE_JOB_CREATED


def record_manifest_committed(manifest: dict[str, Any], *, package_dir: str) -> None:
    manifest['submit_state'] = SUBMIT_STATE_COMMITTED
    append_submit_journal_entry(
        package_dir,
        {
            'event': EVENT_MANIFEST_COMMITTED,
            'submit_attempt_id': manifest.get('submit_attempt_id', ''),
            'job_name': manifest.get('job_name', ''),
            'request_checksum': manifest.get('request_checksum', ''),
        },
    )


def _latest_attempt_entries(entries: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    attempt_id = ''
    for entry in reversed(entries):
        candidate = entry.get('submit_attempt_id')
        if isinstance(candidate, str) and candidate.strip():
            attempt_id = candidate.strip()
            break
    if not attempt_id:
        return '', []
    return attempt_id, [entry for entry in entries if entry.get('submit_attempt_id') == attempt_id]


def find_uncommitted_job_created(
    entries: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if manifest.get('job_name'):
        return None

    attempt_id, attempt_entries = _latest_attempt_entries(entries)
    if not attempt_id:
        return None

    committed = any(
        entry.get('event') == EVENT_MANIFEST_COMMITTED
        for entry in attempt_entries
    )
    if committed:
        return None

    job_created_entries = [
        entry
        for entry in attempt_entries
        if entry.get('event') == EVENT_JOB_CREATED and entry.get('job_name')
    ]
    if not job_created_entries:
        return None
    return job_created_entries[-1]


def has_upload_pending_job_create(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> bool:
    if manifest.get('job_name'):
        return False
    if manifest.get('submit_state') != SUBMIT_STATE_UPLOADED:
        return False
    uploaded_file_name = manifest.get('uploaded_file_name')
    if not isinstance(uploaded_file_name, str) or not uploaded_file_name.strip():
        return False

    attempt_id = manifest.get('submit_attempt_id')
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        return False

    attempt_entries = [
        entry for entry in entries if entry.get('submit_attempt_id') == attempt_id
    ]
    if any(entry.get('event') == EVENT_JOB_CREATED for entry in attempt_entries):
        return False
    if any(entry.get('event') == EVENT_MANIFEST_COMMITTED for entry in attempt_entries):
        return False
    return True


def get_uncertain_submit_state(
    manifest: dict[str, Any],
    *,
    package_dir: str | None = None,
) -> dict[str, Any] | None:
    if manifest.get('job_name'):
        return None

    resolved_package_dir = package_dir or manifest.get('_package_dir') or ''
    entries = read_submit_journal_entries(resolved_package_dir) if resolved_package_dir else []

    pending_job = find_uncommitted_job_created(entries, manifest)
    if pending_job:
        return {
            'kind': 'job_created_uncommitted',
            'job_name': pending_job.get('job_name', ''),
            'uploaded_file_name': pending_job.get('uploaded_file_name', ''),
            'display_name': pending_job.get('display_name', ''),
            'request_checksum': pending_job.get('request_checksum', ''),
            'message': (
                f'{BLOCKED_MESSAGE_PREFIX} a remote batch job was created but manifest was not '
                f"updated ({pending_job.get('job_name', '')})."
            ),
            'recovery_hint': RECOVER_HINT,
        }

    if has_upload_pending_job_create(manifest, entries):
        return {
            'kind': 'upload_pending_job_create',
            'uploaded_file_name': manifest.get('uploaded_file_name', ''),
            'display_name': manifest.get('display_name', ''),
            'request_checksum': manifest.get('request_checksum', ''),
            'message': (
                f'{BLOCKED_MESSAGE_PREFIX} input JSONL was uploaded '
                f"({manifest.get('uploaded_file_name', '')}) but no batch job was created yet."
            ),
            'recovery_hint': 'Re-run submit with --resume to continue job creation, or --force to start over.',
        }

    return None


def clear_incomplete_submit_state(manifest: dict[str, Any]) -> None:
    manifest.pop('submit_attempt_id', None)
    manifest.pop('request_checksum', None)
    manifest.pop('submit_state', None)
    manifest.pop('submit_recovery_hint', None)
    manifest.pop('uploaded_file_name', None)


def format_uncertain_submit_hints(uncertain_state: dict[str, Any] | None) -> list[str]:
    if not uncertain_state:
        return []

    hints: list[str] = []
    kind = uncertain_state.get('kind')
    if kind == 'job_created_uncommitted':
        job_name = uncertain_state.get('job_name')
        if isinstance(job_name, str) and job_name.strip():
            hints.append(f'Recoverable remote job: {job_name.strip()}')
    elif kind == 'upload_pending_job_create':
        uploaded_file_name = uncertain_state.get('uploaded_file_name')
        if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
            hints.append(f'Uploaded input file: {uploaded_file_name.strip()}')
    display_name = uncertain_state.get('display_name')
    if isinstance(display_name, str) and display_name.strip():
        hints.append(f'Display name: {display_name.strip()}')
    checksum = uncertain_state.get('request_checksum')
    if isinstance(checksum, str) and checksum.strip():
        hints.append(f'Request checksum: {checksum.strip()}')
    recovery_hint = uncertain_state.get('recovery_hint')
    if isinstance(recovery_hint, str) and recovery_hint.strip():
        hints.append(recovery_hint.strip())
    return hints


def apply_recovered_job_to_manifest(
    manifest: dict[str, Any],
    pending_job: dict[str, Any],
    *,
    package_dir: str,
    submitted_api_key_index: int | None = None,
) -> None:
    manifest['job_name'] = pending_job.get('job_name', '')
    manifest['job_state'] = pending_job.get('job_state') or 'JOB_STATE_PENDING'
    manifest['submitted_at'] = pending_job.get('at') or datetime.now().isoformat(timespec='seconds')
    manifest['last_status_checked_at'] = manifest['submitted_at']
    if submitted_api_key_index is not None:
        manifest['submitted_api_key_index'] = submitted_api_key_index
        manifest['submitted_api_key_number'] = submitted_api_key_index + 1
        manifest['last_status_api_key_index'] = submitted_api_key_index

    uploaded_file_name = pending_job.get('uploaded_file_name')
    if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
        manifest['uploaded_file_name'] = uploaded_file_name.strip()
        manifest.setdefault('uploaded_file_names', [])
        if manifest['uploaded_file_name'] not in manifest['uploaded_file_names']:
            manifest['uploaded_file_names'].append(manifest['uploaded_file_name'])

    attempt_id = pending_job.get('submit_attempt_id')
    if isinstance(attempt_id, str) and attempt_id.strip():
        manifest['submit_attempt_id'] = attempt_id.strip()
    checksum = pending_job.get('request_checksum')
    if isinstance(checksum, str) and checksum.strip():
        manifest['request_checksum'] = checksum.strip()

    manifest['last_submit_error'] = ''
    manifest.pop('last_submit_error_type', None)
    manifest.pop('split_recommended', None)
    manifest.pop('last_submit_quota_recommendation', None)
    record_manifest_committed(manifest, package_dir=package_dir)