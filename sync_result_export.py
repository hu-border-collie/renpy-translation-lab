# -*- coding: utf-8 -*-
"""Deterministic durable Sync result export and usage-outbox delivery (#347 P3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from atomic_io import atomic_write_text, file_sha256, sha256_text
import model_usage_ledger
import sync_run_contracts as contracts
from sync_run_contracts import (
    AttemptStatus,
    ErrorCode,
    RunStatus,
    SyncRunError,
    canonical_json,
)
from sync_run_store import SyncRunStore


DURABLE_SYNC_RESULT_SCHEMA_VERSION = 1
RESULTS_FILENAME = 'results.jsonl'
RESULTS_HASH_FILENAME = 'results.jsonl.sha256'
RUN_MANIFEST_FILENAME = 'run_manifest.json'
PLAN_FILENAME = 'plan.json'
REQUESTS_FILENAME = 'requests.jsonl'
EVENTS_FILENAME = 'events.jsonl'


def _json_payload(value: Any, default: Any = None) -> Any:
    if value in (None, ''):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _attempt_projection(attempt: Mapping[str, Any], lineage_kind: str) -> dict[str, Any]:
    attempt = dict(attempt)
    status = str(attempt.get('status') or '')
    response = _json_payload(attempt.get('response_payload_json'), {})
    diagnostics = _json_payload(attempt.get('contract_diagnostics_json'), {})
    usage = _json_payload(attempt.get('usage_metadata_json'), {})
    projected = {
        'attempt_id': attempt.get('attempt_id'),
        'lineage_request_id': attempt.get('request_id'),
        'kind': (
            'first_pass'
            if lineage_kind == contracts.LineageKind.ROOT.value
            else lineage_kind
        ),
        'status': status,
        'provider': attempt.get('provider') or '',
        'model': attempt.get('model') or '',
        'finish_time': attempt.get('finish_time'),
        'error_category': attempt.get('error_category'),
        'error_reason_code': attempt.get('error_reason_code'),
        'usage_metadata': usage,
        'contract_diagnostics': diagnostics,
    }
    if response not in ({}, None):
        projected['response'] = response
    return projected


def _translation_text(payload: Any) -> str:
    """Return the shared result-row translation scalar from a stored winner."""
    value = _json_payload(payload, '')
    if isinstance(value, Mapping):
        value = value.get('translation', '')
    return str(value or '')


def build_result_rows(store: SyncRunStore) -> list[dict[str, Any]]:
    """Project authoritative DB facts into Batch/Sync-compatible result rows."""
    violations = store.verify_integrity()
    if violations:
        raise SyncRunError(
            ErrorCode.SYNC_RUN_STORAGE_ERROR,
            'cannot export a durable run with integrity violations',
            safe_details={'run_id': store.run_id, 'violations': violations[:20]},
        )
    run = store.get_run()
    if RunStatus(str(run['status'])) not in contracts.RUN_TERMINAL_STATES:
        raise SyncRunError(
            ErrorCode.SYNC_RUN_STORAGE_ERROR,
            'durable results can only be exported from a terminal run',
            safe_details={'run_id': store.run_id, 'run_status': run['status']},
        )

    rows: list[dict[str, Any]] = []
    with store._conn() as conn:
        root_requests = conn.execute(
            'SELECT * FROM requests WHERE run_id = ? AND parent_request_id IS NULL '
            'ORDER BY rowid',
            (store.run_id,),
        ).fetchall()
        for root in root_requests:
            root_id = str(root['request_id'])
            root_payload = json.loads(root['payload_json'])
            expected_ids = list(root_payload.get('expected_ids') or [])
            lineage_requests = conn.execute(
                'SELECT * FROM requests WHERE run_id = ? AND root_request_id = ? '
                'ORDER BY rowid',
                (store.run_id, root_id),
            ).fetchall()
            lineage_by_id = {
                str(request['request_id']): request for request in lineage_requests
            }
            attempts = conn.execute(
                'SELECT attempts.* FROM attempts JOIN requests '
                'ON requests.run_id = attempts.run_id '
                'AND requests.request_id = attempts.request_id '
                'WHERE attempts.run_id = ? AND requests.root_request_id = ? '
                'ORDER BY requests.rowid, attempts.ordinal',
                (store.run_id, root_id),
            ).fetchall()
            winners = conn.execute(
                'SELECT item_results.* FROM item_results JOIN requests '
                'ON requests.run_id = item_results.run_id '
                'AND requests.request_id = item_results.winner_request_id '
                'WHERE item_results.run_id = ? AND requests.root_request_id = ?',
                (store.run_id, root_id),
            ).fetchall()
            winner_by_id = {str(winner['item_id']): winner for winner in winners}
            accepted_ids = [item_id for item_id in expected_ids if item_id in winner_by_id]
            unresolved_ids = [item_id for item_id in expected_ids if item_id not in winner_by_id]
            translations = [
                {
                    'id': item_id,
                    'translation': _translation_text(
                        winner_by_id[item_id]['translation_payload_json']
                    ),
                }
                for item_id in accepted_ids
            ]
            attempt_rows = [
                _attempt_projection(
                    attempt,
                    str(lineage_by_id[str(attempt['request_id'])]['lineage_kind']),
                )
                for attempt in attempts
            ]
            first_response = {}
            for attempt in attempts:
                if str(attempt['request_id']) != root_id:
                    continue
                response = _json_payload(attempt['response_payload_json'], None)
                if response is not None:
                    first_response = response
                    break
            late_count = conn.execute(
                'SELECT COUNT(*) AS n FROM late_receipts WHERE run_id = ? '
                'AND attempt_id IN ('
                'SELECT attempts.attempt_id FROM attempts JOIN requests '
                'ON requests.run_id = attempts.run_id '
                'AND requests.request_id = attempts.request_id '
                'WHERE attempts.run_id = ? AND requests.root_request_id = ?)',
                (store.run_id, store.run_id, root_id),
            ).fetchone()['n']
            status_values = {str(attempt['status']) for attempt in attempts}
            normalized_response = {'translations': translations}
            contract_diagnostics = {
                'mode': 'translation',
                'complete': not unresolved_ids,
                'expected_count': len(expected_ids),
                'valid_count': len(accepted_ids),
                'valid_ids': accepted_ids,
                'retry_ids': unresolved_ids,
                'reason_counts': (
                    {} if not unresolved_ids else {'durable_unresolved_ids': len(unresolved_ids)}
                ),
            }
            row = {
                'schema_version': DURABLE_SYNC_RESULT_SCHEMA_VERSION,
                'key': root_payload.get('chunk_id') or root_id,
                'run_id': store.run_id,
                'plan_id': run['plan_id'],
                'request_id': root_id,
                'chunk_id': root_payload.get('chunk_id') or '',
                'request_fingerprint': root_payload.get('request_fingerprint') or '',
                'prompt_fingerprint': root_payload.get('prompt_fingerprint') or '',
                'response': first_response,
                'provider_response_attempts': attempt_rows,
                'normalized_response': normalized_response,
                'response_semantics': {
                    'response': 'first_pass_provider_payload',
                    'provider_response_attempts': 'durable_attempt_audit',
                    'normalized_response': 'final_merged_contract',
                },
                'contract_diagnostics': contract_diagnostics,
                'accepted_ids': accepted_ids,
                'unresolved_ids': unresolved_ids,
                'late_receipt_count': int(late_count),
                'outcome_unknown': AttemptStatus.OUTCOME_UNKNOWN.value in status_values,
                'cancelled': (
                    str(run['status']) == RunStatus.CANCELLED.value
                    or AttemptStatus.CANCELLED.value in status_values
                ),
                'execution_mode': 'durable_sync',
            }
            row['row_sha256'] = sha256_text(canonical_json(row))
            rows.append(row)
    return rows


def render_results_jsonl(rows) -> str:
    return ''.join(canonical_json(row) + '\n' for row in rows)


def export_run_artifacts(store: SyncRunStore) -> dict[str, Any]:
    """Atomically and deterministically materialize every durable audit artifact."""
    rows = build_result_rows(store)
    payloads = {
        RESULTS_FILENAME: render_results_jsonl(rows),
        RUN_MANIFEST_FILENAME: store.export_run_manifest_json() + '\n',
        PLAN_FILENAME: canonical_json(store.get_plan()['plan']) + '\n',
        REQUESTS_FILENAME: store.export_requests_jsonl(),
        EVENTS_FILENAME: store.export_events_jsonl(),
    }
    artifacts = {}
    for filename, text in payloads.items():
        path = store.run_dir / filename
        atomic_write_text(path, text, newline='\n')
        digest = file_sha256(path)
        kind = filename.replace('.', '_')
        store.put_artifact(
            kind=kind,
            relative_path=filename,
            sha256_digest=digest,
            schema_version=DURABLE_SYNC_RESULT_SCHEMA_VERSION,
        )
        artifacts[kind] = {
            'path': str(path),
            'sha256': digest,
        }
    hash_path = store.run_dir / RESULTS_HASH_FILENAME
    result_digest = artifacts['results_jsonl']['sha256']
    atomic_write_text(hash_path, result_digest + '\n', newline='\n')
    sidecar_digest = file_sha256(hash_path)
    store.put_artifact(
        kind='results_sha256',
        relative_path=RESULTS_HASH_FILENAME,
        sha256_digest=sidecar_digest,
        schema_version=DURABLE_SYNC_RESULT_SCHEMA_VERSION,
    )
    artifacts['results_sha256'] = {
        'path': str(hash_path),
        'sha256': sidecar_digest,
        'content_sha256': result_digest,
    }
    store.checkpoint()
    return {
        'run_id': store.run_id,
        'result_rows': len(rows),
        'artifacts': artifacts,
    }


def deliver_usage_outbox(
    store: SyncRunStore,
    *,
    game_root: str | Path,
    pricing_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay pending usage records with ``usage:<attempt_id>`` dedupe keys."""
    ledger = model_usage_ledger.UsageLedger(game_root)
    inserted = 0
    duplicates = 0
    failed = 0
    for outbox in store.pending_usage_outbox():
        with store._conn() as conn:
            attempt = conn.execute(
                'SELECT * FROM attempts WHERE run_id = ? AND attempt_id = ?',
                (store.run_id, outbox['attempt_id']),
            ).fetchone()
        if attempt is None:
            store.ack_usage_outbox(
                usage_event_id=outbox['usage_event_id'],
                delivery_error='attempt_not_found',
            )
            failed += 1
            continue
        usage = _json_payload(attempt['usage_metadata_json'], {})
        if not usage:
            usage = dict(_json_payload(outbox['record_json'], {}) or {})
            usage.pop('attempt_id', None)
            usage.pop('run_id', None)
        response = _json_payload(attempt['response_payload_json'], {})
        reservation = _json_payload(attempt['reservation_json'], {})
        try:
            record = model_usage_ledger.build_usage_record(
                game_root=game_root,
                task_mode='translation',
                stage='durable_sync_translation',
                provider=attempt['provider'] or 'unknown',
                model=attempt['model'] or 'unknown',
                usage_metadata=usage,
                response_payload=response,
                operation_id=store.run_id,
                run_id=store.run_id,
                execution_mode='durable_sync',
                source_key=attempt['attempt_id'],
                source={
                    'attempt_id': attempt['attempt_id'],
                    'request_id': attempt['request_id'],
                    'status': attempt['status'],
                },
                pricing_config=pricing_config,
                estimated_cost=reservation.get('estimated_cost'),
                dedupe_key=outbox['usage_event_id'],
            )
            outcome = ledger.add_records([record])
            inserted += int(outcome['inserted_records'])
            duplicates += int(outcome['duplicate_records'])
            store.ack_usage_outbox(usage_event_id=outbox['usage_event_id'])
        except (OSError, ValueError, model_usage_ledger.UsageLedgerError) as exc:
            failed += 1
            store.ack_usage_outbox(
                usage_event_id=outbox['usage_event_id'],
                delivery_error=type(exc).__name__,
            )
    return {
        'pending_before': inserted + duplicates + failed,
        'inserted': inserted,
        'duplicates': duplicates,
        'failed': failed,
        'pending_after': len(store.pending_usage_outbox()),
    }
