# -*- coding: utf-8 -*-
"""SQLite durability store for the #347 durable sync executor (P1).

This module owns the on-disk run database and the transactional boundaries
described in the P0 design contract.  It deliberately contains no scheduler
loop, no provider call and no project write-back logic.

The SQLite file is the authoritative run fact.  JSON/JSONL exports generated
by :meth:`SyncRunStore.export_*` are hash-bound projections, never the source
of truth.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from atomic_io import AtomicFileLockTimeoutError, exclusive_file_lock, file_sha256
import sync_run_contracts as contracts
from sync_run_contracts import (
    AttemptStatus,
    ErrorCategory,
    ErrorCode,
    EventType,
    RequestStatus,
    RunStatus,
    SyncRunError,
    assert_valid_run_id,
    canonical_json,
    client_token_digest,
    sha256_hex,
    utcnow_iso,
)
from sync_retry_policy import ExecutorPolicy

DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_LEASE_TTL_SECONDS = 300.0
DEFAULT_START_LOCK_TIMEOUT_SECONDS = 30.0


def _now_iso() -> str:
    return utcnow_iso()


def _lease_expiry(now_iso: str, ttl_seconds: float) -> str:
    try:
        dt = datetime.fromisoformat(now_iso.replace('Z', '+00:00'))
    except ValueError:
        dt = datetime.now(timezone.utc)
    return (dt + timedelta(seconds=ttl_seconds)).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _source_identity_digest(plan_payload: Mapping[str, Any]) -> str:
    identity = plan_payload.get('source_identity')
    if identity is None:
        return ''
    return sha256_hex(canonical_json(dict(identity)))


def _normalized_root_request_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one immutable root ``TranslationRequest`` payload."""
    payload = dict(request or {})
    request_id = str(payload.get('request_id') or '')
    plan_id = str(payload.get('plan_id') or '')
    expected_ids = list(payload.get('expected_ids') or [])
    if not request_id or not plan_id:
        raise ValueError('root request requires request_id and plan_id')
    if not expected_ids or any(not isinstance(item_id, str) or not item_id for item_id in expected_ids):
        raise ValueError(f'request {request_id} requires non-empty string expected_ids')
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f'request {request_id} contains duplicate expected_ids')
    if not str(payload.get('prompt_fingerprint') or ''):
        raise ValueError(f'request {request_id} requires prompt_fingerprint')
    if not str(payload.get('request_fingerprint') or ''):
        raise ValueError(f'request {request_id} requires request_fingerprint')
    return payload


def _bootstrap_input_snapshot(
    *,
    plan_payload: Mapping[str, Any],
    request_payloads: Sequence[Mapping[str, Any]],
    policy_payload: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    token_digest: str | None,
) -> dict[str, Any]:
    normalized_requests = [
        _normalized_root_request_payload(request) for request in request_payloads
    ]
    if not normalized_requests:
        raise ValueError('a durable run requires at least one root request')
    plan_id = str(plan_payload.get('plan_id') or '')
    if any(str(request.get('plan_id') or '') != plan_id for request in normalized_requests):
        raise ValueError('every root request must reference the stored plan_id')
    meta = dict(run_meta or {})
    return {
        'client_token_digest': token_digest,
        'plan_sha256': sha256_hex(canonical_json(dict(plan_payload))),
        'policy_digest': sha256_hex(canonical_json(dict(policy_payload))),
        'source_identity_digest': str(
            meta.get('source_identity_digest') or _source_identity_digest(plan_payload)
        ),
        'profile_digest': str(meta.get('profile_digest') or ''),
        'config_digest': str(
            meta.get('config_digest') or str(plan_payload.get('config_fingerprint') or '')
        ),
        'resume_compatibility_fingerprint': str(
            meta.get('resume_compatibility_fingerprint') or ''
        ),
        'derived_from_run_id': meta.get('derived_from_run_id') or None,
        'derivation_json': canonical_json(meta.get('derivation') or {}),
        'request_payloads': normalized_requests,
        'request_payload_hashes': [
            sha256_hex(canonical_json(request)) for request in normalized_requests
        ],
    }




class SyncRunStore:
    """Access object for one durable run directory.

    The caller selects the run directory root (``<log_dir>/sync_runs``) and
    the run id.  The store opens ``<root>/<run_id>/state.sqlite3``.
    """

    def __init__(self, root_dir: Path | str, run_id: str):
        assert_valid_run_id(run_id)
        self.root_dir = Path(root_dir)
        self.run_id = str(run_id)
        self.run_dir = self.root_dir / self.run_id
        self.db_path = self.run_dir / 'state.sqlite3'

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise SyncRunError(
                ErrorCode.SYNC_RUN_NOT_FOUND,
                f'durable sync run not found: {self.run_id}',
                retryable=False,
                safe_details={'run_id': self.run_id, 'path': str(self.db_path)},
            )
        conn = sqlite3.connect(str(self.db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('PRAGMA busy_timeout=%d' % DEFAULT_BUSY_TIMEOUT_MS)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=FULL')
            self._ensure_schema(conn)
            return conn
        except Exception:
            conn.close()
            raise

    @contextmanager
    def _conn(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone()
        if exists is None:
            SyncRunStore._create_schema(conn)
            return
        row = conn.execute(
            'SELECT version FROM schema_meta ORDER BY version DESC LIMIT 1'
        ).fetchone()
        version = int(row['version']) if row else 0
        if version != contracts.SYNC_RUN_SCHEMA_VERSION:
            if version > contracts.SYNC_RUN_SCHEMA_VERSION:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_SCHEMA_UNSUPPORTED,
                    f'sync run schema is newer than supported: {version}',
                    retryable=False,
                    safe_details={
                        'stored_version': version,
                        'supported_version': contracts.SYNC_RUN_SCHEMA_VERSION,
                    },
                )
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                f'unsupported forward migration from schema {version}',
                retryable=False,
                safe_details={
                    'stored_version': version,
                    'supported_version': contracts.SYNC_RUN_SCHEMA_VERSION,
                },
            )

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            'CREATE TABLE schema_meta ('
            ' version INTEGER PRIMARY KEY,'
            ' created_by TEXT NOT NULL,'
            ' created_at TEXT NOT NULL'
            ')'
        )
        conn.execute(
            'INSERT INTO schema_meta(version, created_by, created_at) VALUES (?, ?, ?)',
            (contracts.SYNC_RUN_SCHEMA_VERSION, 'sync_run_store.py', _now_iso()),
        )
        conn.execute(
            'CREATE TABLE runs ('
            ' run_id TEXT PRIMARY KEY,'
            ' client_token_digest TEXT,'
            ' status TEXT NOT NULL,'
            ' revision INTEGER NOT NULL DEFAULT 0,'
            ' cancel_epoch INTEGER NOT NULL DEFAULT 0,'
            ' plan_id TEXT NOT NULL,'
            ' plan_fingerprint TEXT NOT NULL,'
            ' source_identity_digest TEXT NOT NULL DEFAULT \'\','
            ' profile_digest TEXT NOT NULL DEFAULT \'\','
            ' config_digest TEXT NOT NULL DEFAULT \'\','
            ' policy_digest TEXT NOT NULL DEFAULT \'\','
            ' resume_compatibility_fingerprint TEXT NOT NULL DEFAULT \'\','
            ' policy_json TEXT NOT NULL DEFAULT \'{}\','
            ' derived_from_run_id TEXT,'
            ' derivation_json TEXT NOT NULL DEFAULT \'{}\','
            ' created_at TEXT NOT NULL,'
            ' updated_at TEXT NOT NULL,'
            ' first_dispatched_at TEXT,'
            ' finished_at TEXT'
            ')'
        )
        conn.execute(
            'CREATE TABLE plans ('
            ' run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,'
            ' canonical_json TEXT NOT NULL,'
            ' payload_sha256 TEXT NOT NULL'
            ')'
        )
        conn.execute(
            'CREATE TABLE requests ('
            ' run_id TEXT NOT NULL REFERENCES runs(run_id),'
            ' request_id TEXT NOT NULL,'
            ' root_request_id TEXT NOT NULL,'
            ' parent_request_id TEXT,'
            ' lineage_kind TEXT NOT NULL DEFAULT \'root\','
            ' lineage_depth INTEGER NOT NULL DEFAULT 0,'
            ' status TEXT NOT NULL,'
            ' expected_ids_json TEXT NOT NULL,'
            ' payload_json TEXT NOT NULL,'
            ' payload_sha256 TEXT NOT NULL,'
            ' prompt_fingerprint TEXT NOT NULL DEFAULT \'\','
            ' request_fingerprint TEXT NOT NULL DEFAULT \'\','
            ' attempt_count INTEGER NOT NULL DEFAULT 0,'
            ' next_eligible_at TEXT,'
            ' created_at TEXT NOT NULL,'
            ' updated_at TEXT NOT NULL,'
            ' PRIMARY KEY (run_id, request_id),'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE'
            ')'
        )
        conn.execute(
            'CREATE INDEX idx_requests_run_status '
            'ON requests(run_id, status)'
        )
        conn.execute(
            'CREATE TABLE attempts ('
            ' attempt_id TEXT PRIMARY KEY,'
            ' run_id TEXT NOT NULL,'
            ' request_id TEXT NOT NULL,'
            ' ordinal INTEGER NOT NULL,'
            ' status TEXT NOT NULL,'
            ' provider TEXT NOT NULL DEFAULT \'\','
            ' model TEXT NOT NULL DEFAULT \'\','
            ' profile_digest TEXT NOT NULL DEFAULT \'\','
            ' credential_identity TEXT NOT NULL DEFAULT \'\','
            ' claim_owner_token TEXT NOT NULL,'
            ' claim_cancel_epoch INTEGER NOT NULL DEFAULT 0,'
            ' reservation_json TEXT NOT NULL DEFAULT \'{}\','
            ' dispatch_time TEXT,'
            ' finish_time TEXT,'
            ' error_category TEXT,'
            ' error_reason_code TEXT,'
            ' error_safe_details_json TEXT,'
            ' response_payload_json TEXT,'
            ' normalized_payload_json TEXT,'
            ' contract_diagnostics_json TEXT,'
            ' usage_metadata_json TEXT,'
            ' next_eligible_at TEXT,'
            ' created_at TEXT NOT NULL,'
            ' UNIQUE (run_id, request_id, ordinal),'
            ' FOREIGN KEY (run_id, request_id) REFERENCES requests(run_id, request_id)'
            ' ON DELETE CASCADE'
            ')'
        )
        conn.execute(
            'CREATE INDEX idx_attempts_run_request '
            'ON attempts(run_id, request_id)'
        )
        conn.execute(
            'CREATE INDEX idx_attempts_run_status '
            'ON attempts(run_id, status)'
        )
        conn.execute(
            'CREATE TABLE item_results ('
            ' run_id TEXT NOT NULL,'
            ' item_id TEXT NOT NULL,'
            ' winner_request_id TEXT NOT NULL,'
            ' winner_attempt_id TEXT,'
            ' reused_from_run_id TEXT,'
            ' reused_from_attempt_id TEXT,'
            ' translation_payload_json TEXT NOT NULL DEFAULT \'{}\','
            ' translation_digest TEXT NOT NULL DEFAULT \'\','
            ' validation_diagnostics_json TEXT NOT NULL DEFAULT \'{}\','
            ' created_at TEXT NOT NULL,'
            ' PRIMARY KEY (run_id, item_id),'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,'
            ' FOREIGN KEY (winner_attempt_id) REFERENCES attempts(attempt_id),'
            ' FOREIGN KEY (run_id, winner_request_id) '
            'REFERENCES requests(run_id, request_id) ON DELETE CASCADE,'
            ' CHECK ((winner_attempt_id IS NOT NULL AND reused_from_run_id IS NULL) OR '
            '(winner_attempt_id IS NULL AND reused_from_run_id IS NOT NULL))'
            ')'
        )
        conn.execute(
            'CREATE TABLE late_receipts ('
            ' receipt_id TEXT PRIMARY KEY,'
            ' run_id TEXT NOT NULL,'
            ' attempt_id TEXT NOT NULL,'
            ' observed_owner_token TEXT,'
            ' observed_cancel_epoch INTEGER NOT NULL DEFAULT 0,'
            ' response_payload_json TEXT,'
            ' error_payload_json TEXT,'
            ' usage_payload_json TEXT,'
            ' ignored_reason TEXT NOT NULL,'
            ' received_at TEXT NOT NULL,'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,'
            ' FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)'
            ')'
        )
        conn.execute(
            'CREATE TABLE usage_outbox ('
            ' usage_event_id TEXT PRIMARY KEY,'
            ' run_id TEXT NOT NULL,'
            ' attempt_id TEXT NOT NULL UNIQUE,'
            ' record_json TEXT NOT NULL,'
            ' delivered_at TEXT,'
            ' delivery_error TEXT,'
            ' created_at TEXT NOT NULL,'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,'
            ' FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)'
            ')'
        )
        conn.execute(
            'CREATE TABLE events ('
            ' event_seq INTEGER PRIMARY KEY AUTOINCREMENT,'
            ' run_id TEXT NOT NULL,'
            ' entity_type TEXT NOT NULL,'
            ' entity_id TEXT NOT NULL,'
            ' event_type TEXT NOT NULL,'
            ' old_status TEXT,'
            ' new_status TEXT,'
            ' safe_details_json TEXT NOT NULL DEFAULT \'{}\','
            ' committed_at TEXT NOT NULL,'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE'
            ')'
        )
        conn.execute(
            'CREATE INDEX idx_events_run_seq '
            'ON events(run_id, event_seq)'
        )
        conn.execute(
            'CREATE TABLE leases ('
            ' run_id TEXT PRIMARY KEY,'
            ' owner_token TEXT NOT NULL,'
            ' pid INTEGER NOT NULL DEFAULT 0,'
            ' acquired_at TEXT NOT NULL,'
            ' heartbeat_at TEXT NOT NULL,'
            ' expires_at TEXT NOT NULL,'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE'
            ')'
        )
        conn.execute(
            'CREATE TABLE artifacts ('
            ' run_id TEXT NOT NULL,'
            ' kind TEXT NOT NULL,'
            ' relative_path TEXT NOT NULL,'
            ' sha256 TEXT NOT NULL,'
            ' schema_version INTEGER NOT NULL,'
            ' created_at TEXT NOT NULL,'
            ' UNIQUE (run_id, kind),'
            ' FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE'
        ')'
        )

    # ------------------------------------------------------------------
    # Bootstrap (T0)
    # ------------------------------------------------------------------
    @classmethod
    def bootstrap(
        cls,
        root_dir: Path | str,
        run_id: str,
        *,
        plan: Mapping[str, Any],
        requests: Sequence[Mapping[str, Any]],
        client_token: str | None = None,
        executor_policy: Mapping[str, Any] | None = None,
        run_meta: Mapping[str, Any] | None = None,
    ) -> tuple['SyncRunStore', bool]:
        """Create or reopen a durable run with T0 atomicity.

        ``plan`` and ``requests`` are stored as immutable canonical JSON and
        hashed before any model call can exist.  Returns ``(store, created)``.
        A second bootstrap with the same non-empty client token and same plan
        hash reopens the same run; with a different plan hash it raises
        ``SYNC_RUN_CLIENT_TOKEN_CONFLICT``.
        """
        assert_valid_run_id(run_id)
        run_id = str(run_id)
        plan_payload = dict(plan or {})
        request_payloads = [dict(item) for item in (requests or [])]
        plan_json = canonical_json(plan_payload)
        plan_hash = sha256_hex(plan_json)
        policy_payload = ExecutorPolicy.from_mapping(executor_policy).to_dict()
        meta = dict(run_meta or {})
        token_digest = client_token_digest(client_token)
        if token_digest is not None and run_id != contracts.build_run_id(client_token):
            raise ValueError('token-backed run_id does not match the client token digest')
        input_snapshot = _bootstrap_input_snapshot(
            plan_payload=plan_payload,
            request_payloads=request_payloads,
            policy_payload=policy_payload,
            run_meta=meta,
            token_digest=token_digest,
        )
        request_payloads = input_snapshot['request_payloads']

        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        try:
            with exclusive_file_lock(
                root / '.start.lock',
                timeout=DEFAULT_START_LOCK_TIMEOUT_SECONDS,
            ):
                run_dir = root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                if not run_dir.is_dir() or not os.access(run_dir, os.W_OK):
                    raise SyncRunError(
                        ErrorCode.SYNC_RUN_STORAGE_ERROR,
                        f'run directory is not writable: {run_dir}',
                        safe_details={'run_id': run_id},
                    )
                db_path = run_dir / 'state.sqlite3'
                created_db = not db_path.is_file()
                if created_db:
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    try:
                        conn.execute('PRAGMA foreign_keys=ON')
                        conn.execute('PRAGMA busy_timeout=%d' % DEFAULT_BUSY_TIMEOUT_MS)
                        conn.execute('PRAGMA journal_mode=WAL')
                        conn.execute('PRAGMA synchronous=FULL')
                        conn.execute('BEGIN IMMEDIATE')
                        cls._ensure_schema(conn)
                        cls._insert_run_tx(
                            conn,
                            run_id=run_id,
                            token_digest=token_digest,
                            plan_payload=plan_payload,
                            plan_json=plan_json,
                            plan_hash=plan_hash,
                            request_payloads=request_payloads,
                            policy_payload=policy_payload,
                            run_meta=meta,
                        )
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        conn.close()
                    return (cls(root, run_id), True)

                store = cls(root, run_id)
                with store._tx() as conn:
                    run_row = conn.execute(
                        'SELECT * FROM runs WHERE run_id = ?', (run_id,)
                    ).fetchone()
                    if run_row is None:
                        cls._insert_run_tx(
                            conn,
                            run_id=run_id,
                            token_digest=token_digest,
                            plan_payload=plan_payload,
                            plan_json=plan_json,
                            plan_hash=plan_hash,
                            request_payloads=request_payloads,
                            policy_payload=policy_payload,
                            run_meta=meta,
                        )
                        return (store, True)
                    cls._validate_existing_bootstrap_tx(
                        conn,
                        run_row=run_row,
                        input_snapshot=input_snapshot,
                    )
                return (store, False)
        except AtomicFileLockTimeoutError as exc:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_BUSY,
                f'another process is starting a durable sync run under {root}',
                retryable=True,
                safe_details={'run_id': run_id},
            ) from exc

    @staticmethod
    def _validate_existing_bootstrap_tx(
        conn: sqlite3.Connection,
        *,
        run_row: sqlite3.Row,
        input_snapshot: Mapping[str, Any],
    ) -> None:
        run_id = str(run_row['run_id'])
        plan_row = conn.execute(
            'SELECT payload_sha256 FROM plans WHERE run_id = ?', (run_id,)
        ).fetchone()
        request_rows = conn.execute(
            'SELECT payload_sha256 FROM requests '
            'WHERE run_id = ? AND parent_request_id IS NULL ORDER BY rowid',
            (run_id,),
        ).fetchall()
        actual = {
            'client_token_digest': run_row['client_token_digest'],
            'plan_sha256': plan_row['payload_sha256'] if plan_row else None,
            'policy_digest': run_row['policy_digest'],
            'source_identity_digest': run_row['source_identity_digest'],
            'profile_digest': run_row['profile_digest'],
            'config_digest': run_row['config_digest'],
            'resume_compatibility_fingerprint': run_row['resume_compatibility_fingerprint'],
            'derived_from_run_id': run_row['derived_from_run_id'],
            'derivation_json': run_row['derivation_json'],
            'request_payload_hashes': [row['payload_sha256'] for row in request_rows],
        }
        expected = {
            key: input_snapshot[key]
            for key in actual
        }
        if actual == expected:
            return
        differing = sorted(key for key in actual if actual[key] != expected[key])
        code = (
            ErrorCode.SYNC_RUN_CLIENT_TOKEN_CONFLICT
            if input_snapshot.get('client_token_digest') is not None
            else ErrorCode.SYNC_RUN_STORAGE_ERROR
        )
        raise SyncRunError(
            code,
            'durable run already exists with incompatible bootstrap inputs',
            retryable=False,
            safe_details={'run_id': run_id, 'differing_fields': differing},
        )

    @staticmethod
    def _insert_run_tx(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        token_digest: str | None,
        plan_payload: Mapping[str, Any],
        plan_json: str,
        plan_hash: str,
        request_payloads: Sequence[Mapping[str, Any]],
        policy_payload: Mapping[str, Any],
        run_meta: Mapping[str, Any],
    ) -> None:
        now = _now_iso()
        plan_id = str(plan_payload.get('plan_id') or '')
        plan_fingerprint = str(plan_payload.get('plan_fingerprint') or '')
        if not plan_id or not plan_fingerprint:
            raise ValueError('plan must contain non-empty plan_id and plan_fingerprint')
        policy_json = canonical_json(policy_payload)
        policy_digest = sha256_hex(policy_json)
        source_digest = str(
            run_meta.get('source_identity_digest') or _source_identity_digest(plan_payload)
        )
        profile_digest = str(run_meta.get('profile_digest') or '')
        config_digest = str(
            run_meta.get('config_digest') or str(plan_payload.get('config_fingerprint') or '')
        )
        resume_fp = str(run_meta.get('resume_compatibility_fingerprint') or '')

        conn.execute(
            'INSERT INTO runs('
            ' run_id, client_token_digest, status, revision, cancel_epoch,'
            ' plan_id, plan_fingerprint, source_identity_digest, profile_digest,'
            ' config_digest, policy_digest, resume_compatibility_fingerprint,'
            ' policy_json, derived_from_run_id, created_at, updated_at'
            ', derivation_json) VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_id,
                token_digest,
                RunStatus.PLANNED.value,
                plan_id,
                plan_fingerprint,
                source_digest,
                profile_digest,
                config_digest,
                policy_digest,
                resume_fp,
                policy_json,
                run_meta.get('derived_from_run_id') or None,
                now,
                now,
                canonical_json(run_meta.get('derivation') or {}),
            ),
        )
        conn.execute(
            'INSERT INTO plans(run_id, canonical_json, payload_sha256) VALUES (?, ?, ?)',
            (run_id, plan_json, plan_hash),
        )
        for request in request_payloads:
            SyncRunStore._insert_root_request_tx(conn, run_id=run_id, request=request, now=now)
        SyncRunStore._write_event_tx(
            conn,
            run_id=run_id,
            entity_type='run',
            entity_id=run_id,
            event_type=EventType.RUN_CREATED,
            old_status=None,
            new_status=RunStatus.PLANNED.value,
            safe_details={'plan_id': plan_id, 'plan_fingerprint': plan_fingerprint},
        )

    @staticmethod
    def _insert_root_request_tx(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        request: Mapping[str, Any],
        now: str | None = None,
    ) -> None:
        now = now or _now_iso()
        payload = _normalized_root_request_payload(request)
        request_id = str(payload.get('request_id') or '')
        payload_json = canonical_json(payload)
        conn.execute(
            'INSERT INTO requests('
            ' run_id, request_id, root_request_id, parent_request_id, lineage_kind,'
            ' lineage_depth, status, expected_ids_json, payload_json, payload_sha256,'
            ' prompt_fingerprint, request_fingerprint, attempt_count, created_at, updated_at'
            ') VALUES (?, ?, ?, NULL, ?, 0, ?, ?, ?, ?, ?, ?, 0, ?, ?)',
            (
                run_id,
                request_id,
                request_id,
                contracts.LineageKind.ROOT.value,
                RequestStatus.PENDING.value,
                canonical_json(list(payload.get('expected_ids') or [])),
                payload_json,
                sha256_hex(payload_json),
                str(payload.get('prompt_fingerprint') or ''),
                str(payload.get('request_fingerprint') or ''),
                now,
                now,
            ),
        )

    @staticmethod
    def _insert_derived_request_tx(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        request: Mapping[str, Any],
        parent_request: sqlite3.Row,
        now: str | None = None,
    ) -> str:
        now = now or _now_iso()
        payload = dict(request)
        request_id = str(payload.get('request_id') or '')
        if not request_id:
            raise ValueError('request_id is required for derived requests')
        parent_id = str(parent_request['request_id'])
        root_id = str(parent_request['root_request_id'])
        transport = dict(payload.get('transport_metadata') or {})
        lineage_kind = str(
            transport.get('retry_lineage_kind') or contracts.LineageKind.MISSING_IDS.value
        )
        known_lineage = {kind.value for kind in contracts.LineageKind}
        if lineage_kind not in known_lineage:
            raise ValueError(f'unsupported derived lineage kind: {lineage_kind}')
        lineage_depth = int(parent_request['lineage_depth']) + 1
        payload_json = canonical_json(payload)
        conn.execute(
            'INSERT INTO requests('
            ' run_id, request_id, root_request_id, parent_request_id, lineage_kind,'
            ' lineage_depth, status, expected_ids_json, payload_json, payload_sha256,'
            ' prompt_fingerprint, request_fingerprint, attempt_count, created_at, updated_at'
            ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)',
            (
                run_id,
                request_id,
                root_id,
                parent_id,
                lineage_kind,
                lineage_depth,
                RequestStatus.PENDING.value,
                canonical_json(list(payload.get('expected_ids') or [])),
                payload_json,
                sha256_hex(payload_json),
                str(payload.get('prompt_fingerprint') or ''),
                str(payload.get('request_fingerprint') or ''),
                now,
                now,
            ),
        )
        return request_id

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def get_run(self, run_id: str | None = None) -> dict:
        rid = str(run_id or self.run_id)
        with self._conn() as conn:
            row = conn.execute('SELECT * FROM runs WHERE run_id = ?', (rid,)).fetchone()
            if row is None:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_NOT_FOUND,
                    f'durable sync run not found: {rid}',
                    safe_details={'run_id': rid},
                )
            return dict(row)

    def get_plan(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                'SELECT canonical_json, payload_sha256 FROM plans WHERE run_id = ?',
                (self.run_id,),
            ).fetchone()
            if row is None:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_NOT_FOUND,
                    f'plan not found for run: {self.run_id}',
                    safe_details={'run_id': self.run_id},
                )
            return {
                'plan': json.loads(row['canonical_json']),
                'canonical_json': row['canonical_json'],
                'payload_sha256': row['payload_sha256'],
            }

    def get_request(self, request_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                'SELECT * FROM requests WHERE run_id = ? AND request_id = ?',
                (self.run_id, request_id),
            ).fetchone()
            return _row_dict(row)

    def list_requests(self, *, status: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if status is None:
                rows = conn.execute(
                    'SELECT * FROM requests WHERE run_id = ? ORDER BY rowid',
                    (self.run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT * FROM requests WHERE run_id = ? AND status = ? ORDER BY rowid',
                    (self.run_id, status),
                ).fetchall()
            return [dict(row) for row in rows]

    def list_attempts(self, *, request_id: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if request_id is None:
                rows = conn.execute(
                    'SELECT * FROM attempts WHERE run_id = ? ORDER BY rowid',
                    (self.run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT * FROM attempts WHERE run_id = ? AND request_id = ? '
                    'ORDER BY ordinal',
                    (self.run_id, request_id),
                ).fetchall()
            return [dict(row) for row in rows]

    def get_attempt(self, attempt_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                'SELECT * FROM attempts WHERE run_id = ? AND attempt_id = ?',
                (self.run_id, str(attempt_id)),
            ).fetchone()
            return _row_dict(row)

    def get_request_payload(self, request_id: str) -> dict[str, Any]:
        """Return the immutable stored request payload after hash verification."""
        with self._conn() as conn:
            row = self._load_request_tx(conn, request_id)
            payload_json = str(row['payload_json'])
            if sha256_hex(payload_json) != str(row['payload_sha256']):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'request payload hash mismatch: {request_id}',
                    safe_details={
                        'run_id': self.run_id,
                        'request_id': str(request_id),
                    },
                )
            return json.loads(payload_json)

    def list_active_attempts(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM attempts WHERE run_id = ? AND status IN (?, ?, ?) '
                'ORDER BY created_at, attempt_id',
                (
                    self.run_id,
                    AttemptStatus.PREPARED.value,
                    AttemptStatus.DISPATCHED.value,
                    AttemptStatus.CANCEL_REQUESTED.value,
                ),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_eligible_requests(
        self,
        *,
        now: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        deadline = str(now or _now_iso())
        row_limit = None if limit is None else int(limit)
        if row_limit is not None and row_limit < 1:
            return []
        with self._conn() as conn:
            sql = (
                'SELECT * FROM requests WHERE run_id = ? AND '
                '(status = ? OR (status = ? AND next_eligible_at <= ?)) '
                'ORDER BY rowid'
            )
            params: list[Any] = [
                self.run_id,
                RequestStatus.PENDING.value,
                RequestStatus.RETRYABLE_FAILED.value,
                deadline,
            ]
            if row_limit is not None:
                sql += ' LIMIT ?'
                params.append(row_limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def lineage_budget_reason(
        self,
        *,
        request_id: str,
        child_count: int,
        policy: Mapping[str, Any],
    ) -> str | None:
        """Return the request-scoped reason that blocks a proposed T5 split."""
        child_count = int(child_count)
        if child_count < 1:
            raise ValueError('child_count must be positive')
        limits = dict(policy or {})
        with self._conn() as conn:
            request = self._load_request_tx(conn, request_id)
            if int(request['lineage_depth']) + 1 > int(limits['max_lineage_depth']):
                return contracts.REASON_LINEAGE_DEPTH_EXHAUSTED
            derived_count = conn.execute(
                'SELECT COUNT(*) AS n FROM requests WHERE run_id = ? '
                'AND root_request_id = ? AND parent_request_id IS NOT NULL',
                (self.run_id, str(request['root_request_id'])),
            ).fetchone()['n']
            if int(derived_count) + child_count > int(
                limits['max_derived_requests_per_root']
            ):
                return contracts.REASON_DERIVED_REQUESTS_EXHAUSTED
            return None

    def terminalize_request(
        self, *, request_id: str, owner_token: str, reason_code: str
    ) -> bool:
        """Terminalize an eligible leaf without creating a Provider attempt."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            request = self._load_request_tx(conn, request_id)
            before = str(request['status'])
            self._terminalize_request_tx(
                conn, request=request, reason_code=reason_code
            )
            changed = before != str(
                self._load_request_tx(conn, request_id)['status']
            )
            if changed:
                self._touch_run_tx(conn)
            return changed

    def stop_run_dispatch(
        self, *, owner_token: str, reason_code: str
    ) -> dict[str, Any]:
        """Stop new dispatches and terminalize every safe local leaf."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            self._close_not_dispatched_for_run_stop_tx(
                conn, reason_code=str(reason_code)
            )
            finished = self._finish_run_if_quiescent_tx(conn)
            self._touch_run_tx(conn)
            return {
                'finished': finished,
                'run': dict(self._load_run_tx(conn, self.run_id)),
            }

    def list_events(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM events WHERE run_id = ? ORDER BY event_seq',
                (self.run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_item_results(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM item_results WHERE run_id = ? ORDER BY rowid',
                (self.run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Lease helpers
    # ------------------------------------------------------------------
    def acquire_lease(
        self,
        *,
        owner_token: str,
        pid: int | None = None,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    ) -> dict:
        now = _now_iso()
        pid = int(pid if pid is not None else os.getpid())
        with self._tx() as conn:
            self._load_run_tx(conn, self.run_id)
            row = conn.execute('SELECT * FROM leases WHERE run_id = ?', (self.run_id,)).fetchone()
            if row is not None and str(row['expires_at']) > now and row['owner_token'] != owner_token:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_BUSY,
                    f'run is leased by another scheduler: {self.run_id}',
                    retryable=True,
                    safe_details={
                        'run_id': self.run_id,
                        'expires_at': row['expires_at'],
                    },
                )
            if row is None:
                conn.execute(
                    'INSERT INTO leases(run_id, owner_token, pid, acquired_at, heartbeat_at, expires_at)'
                    ' VALUES (?, ?, ?, ?, ?, ?)',
                    (self.run_id, owner_token, pid, now, now, _lease_expiry(now, ttl_seconds)),
                )
                lease_event = 'acquired'
            else:
                active_same_owner = (
                    str(row['expires_at']) > now
                    and str(row['owner_token']) == str(owner_token)
                )
                conn.execute(
                    'UPDATE leases SET owner_token = ?, pid = ?, acquired_at = ?,'
                    ' heartbeat_at = ?, expires_at = ?'
                    ' WHERE run_id = ?',
                    (
                        owner_token,
                        pid,
                        row['acquired_at'] if active_same_owner else now,
                        now,
                        _lease_expiry(now, ttl_seconds),
                        self.run_id,
                    ),
                )
                lease_event = 'renewed' if active_same_owner else 'taken_over'
            self._write_event_tx(
                conn,
                run_id=self.run_id,
                entity_type='lease',
                entity_id=self.run_id,
                event_type=EventType.LEASE,
                old_status=None if row is None else 'held',
                new_status=lease_event,
                safe_details={
                    'previous_owner_present': row is not None,
                    'pid': pid,
                },
            )
            self._touch_run_tx(conn)
            updated = conn.execute(
                'SELECT * FROM leases WHERE run_id = ?', (self.run_id,)
            ).fetchone()
            return dict(updated)

    def release_lease(self, *, owner_token: str) -> bool:
        with self._tx() as conn:
            row = conn.execute('SELECT * FROM leases WHERE run_id = ?', (self.run_id,)).fetchone()
            if row is None:
                return False
            if row['owner_token'] != owner_token:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_BUSY,
                    'lease is not held by the requested owner',
                    retryable=True,
                    safe_details={'run_id': self.run_id},
                )
            conn.execute('DELETE FROM leases WHERE run_id = ?', (self.run_id,))
            self._write_event_tx(
                conn,
                run_id=self.run_id,
                entity_type='lease',
                entity_id=self.run_id,
                event_type=EventType.LEASE,
                old_status='held',
                new_status='released',
            )
            self._touch_run_tx(conn)
            return True

    def heartbeat_lease(
        self,
        *,
        owner_token: str,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    ) -> dict:
        now = _now_iso()
        with self._tx() as conn:
            self._load_run_tx(conn, self.run_id)
            row = conn.execute('SELECT * FROM leases WHERE run_id = ?', (self.run_id,)).fetchone()
            if row is None or row['owner_token'] != owner_token:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_BUSY,
                    'lease is not held by the requested owner',
                    retryable=True,
                    safe_details={'run_id': self.run_id},
                )
            if str(row['expires_at']) <= now:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_BUSY,
                    'cannot revive an expired run lease; acquire it again',
                    retryable=True,
                    safe_details={
                        'run_id': self.run_id,
                        'expires_at': row['expires_at'],
                    },
                )
            conn.execute(
                'UPDATE leases SET heartbeat_at = ?, expires_at = ? WHERE run_id = ?',
                (now, _lease_expiry(now, ttl_seconds), self.run_id),
            )
            updated = conn.execute(
                'SELECT * FROM leases WHERE run_id = ?', (self.run_id,)
            ).fetchone()
            return dict(updated)

    # ------------------------------------------------------------------
    # Core transaction helpers
    # ------------------------------------------------------------------
    def _load_run_tx(self, conn: sqlite3.Connection, run_id: str | None = None) -> dict:
        rid = str(run_id or self.run_id)
        row = conn.execute('SELECT * FROM runs WHERE run_id = ?', (rid,)).fetchone()
        if row is None:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_NOT_FOUND,
                f'durable sync run not found: {rid}',
                safe_details={'run_id': rid},
            )
        return dict(row)

    def _load_request_tx(self, conn: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        row = conn.execute(
            'SELECT * FROM requests WHERE run_id = ? AND request_id = ?',
            (self.run_id, str(request_id)),
        ).fetchone()
        if row is None:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_NOT_FOUND,
                f'request not found: {request_id}',
                safe_details={'run_id': self.run_id, 'request_id': str(request_id)},
            )
        return row

    def _require_lease_owner_tx(self, conn: sqlite3.Connection, owner_token: str) -> sqlite3.Row:
        row = conn.execute('SELECT * FROM leases WHERE run_id = ?', (self.run_id,)).fetchone()
        if row is None or str(row['owner_token']) != str(owner_token):
            raise SyncRunError(
                ErrorCode.SYNC_RUN_BUSY,
                'run lease is not held by the requested owner',
                retryable=True,
                safe_details={'run_id': self.run_id},
            )
        if str(row['expires_at']) <= _now_iso():
            raise SyncRunError(
                ErrorCode.SYNC_RUN_BUSY,
                'run lease has expired',
                retryable=True,
                safe_details={'run_id': self.run_id, 'expires_at': row['expires_at']},
            )
        return row

    def _touch_run_tx(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            'UPDATE runs SET revision = revision + 1, updated_at = ? WHERE run_id = ?',
            (_now_iso(), self.run_id),
        )

    def _transition_run_tx(
        self,
        conn: sqlite3.Connection,
        run: Mapping[str, Any],
        next_status: RunStatus,
        *,
        event_type: EventType = EventType.RUN_STATUS,
        safe_details: Mapping[str, Any] | None = None,
    ) -> None:
        current = RunStatus(str(run['status']))
        contracts.ensure_run_transition(current, next_status)
        conn.execute(
            'UPDATE runs SET status = ?, revision = revision + 1, updated_at = ? WHERE run_id = ?',
            (next_status.value, _now_iso(), self.run_id),
        )
        self._write_event_tx(
            conn,
            run_id=self.run_id,
            entity_type='run',
            entity_id=self.run_id,
            event_type=event_type,
            old_status=current.value,
            new_status=next_status.value,
            safe_details=safe_details,
        )

    def _transition_request_tx(
        self,
        conn: sqlite3.Connection,
        request: sqlite3.Row,
        next_status: RequestStatus,
        *,
        safe_details: Mapping[str, Any] | None = None,
    ) -> None:
        current = RequestStatus(str(request['status']))
        contracts.ensure_request_transition(current, next_status)
        conn.execute(
            'UPDATE requests SET status = ?, updated_at = ? WHERE run_id = ? AND request_id = ?',
            (next_status.value, _now_iso(), self.run_id, request['request_id']),
        )
        self._write_event_tx(
            conn,
            run_id=self.run_id,
            entity_type='request',
            entity_id=str(request['request_id']),
            event_type=EventType.REQUEST_STATUS,
            old_status=current.value,
            new_status=next_status.value,
            safe_details=safe_details,
        )
        self._touch_run_tx(conn)

    def _transition_attempt_tx(
        self,
        conn: sqlite3.Connection,
        attempt: sqlite3.Row,
        next_status: AttemptStatus,
        *,
        event_type: EventType | None = None,
        safe_details: Mapping[str, Any] | None = None,
    ) -> None:
        current = AttemptStatus(str(attempt['status']))
        contracts.ensure_attempt_transition(current, next_status)
        conn.execute(
            'UPDATE attempts SET status = ? WHERE attempt_id = ?',
            (next_status.value, attempt['attempt_id']),
        )
        self._write_event_tx(
            conn,
            run_id=self.run_id,
            entity_type='attempt',
            entity_id=str(attempt['attempt_id']),
            event_type=event_type or self._attempt_event_for(next_status),
            old_status=current.value,
            new_status=next_status.value,
            safe_details=safe_details,
        )
        self._touch_run_tx(conn)

    @staticmethod
    def _attempt_event_for(next_status: AttemptStatus) -> EventType:
        if next_status is AttemptStatus.SUCCEEDED:
            return EventType.ATTEMPT_SUCCEEDED
        if next_status is AttemptStatus.DISPATCHED:
            return EventType.ATTEMPT_DISPATCHED
        if next_status is AttemptStatus.RETRYABLE_FAILED:
            return EventType.ATTEMPT_FAILED
        if next_status is AttemptStatus.TERMINAL_FAILED:
            return EventType.ATTEMPT_FAILED
        if next_status is AttemptStatus.CANCELLED:
            return EventType.ATTEMPT_CANCELLED
        if next_status is AttemptStatus.OUTCOME_UNKNOWN:
            return EventType.ATTEMPT_UNKNOWN
        if next_status in (
            AttemptStatus.LATE_SUCCEEDED_IGNORED,
            AttemptStatus.LATE_FAILED_IGNORED,
        ):
            return EventType.ATTEMPT_LATE_IGNORED
        return EventType.NOTICE

    @staticmethod
    def _write_event_tx(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        entity_type: str,
        entity_id: str,
        event_type: EventType,
        old_status: str | None,
        new_status: str | None,
        safe_details: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            'INSERT INTO events('
            ' run_id, entity_type, entity_id, event_type,'
            ' old_status, new_status, safe_details_json, committed_at'
            ') VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_id,
                str(entity_type),
                str(entity_id),
                event_type.value,
                None if old_status is None else str(old_status),
                None if new_status is None else str(new_status),
                canonical_json(dict(safe_details or {})),
                _now_iso(),
            ),
        )

    def _receipt_allowed_tx(
        self,
        conn: sqlite3.Connection,
        attempt: sqlite3.Row,
        run: Mapping[str, Any],
        owner_token: str,
    ) -> bool:
        """T3/T4 guard from P0 section 5.4.

        A normal receipt requires the current lease owner to match the claim
        owner, the run cancel epoch to be unchanged since the claim, and the
        attempt to still be ``dispatched``.
        """
        lease = conn.execute('SELECT * FROM leases WHERE run_id = ?', (self.run_id,)).fetchone()
        if lease is None or str(lease['owner_token']) != str(owner_token):
            return False
        if str(lease['expires_at']) <= _now_iso():
            return False
        if str(attempt['claim_owner_token']) != str(owner_token):
            return False
        if int(run['cancel_epoch']) != int(attempt['claim_cancel_epoch']):
            return False
        if str(attempt['status']) != AttemptStatus.DISPATCHED.value:
            return False
        return True

    # ------------------------------------------------------------------
    # T1/T2: attempt claim and dispatch intent
    # ------------------------------------------------------------------
    def prepare_attempt(
        self,
        *,
        request_id: str,
        owner_token: str,
        provider: str = '',
        model: str = '',
        profile_digest: str = '',
        credential_identity: str = '',
        reservation: Mapping[str, Any] | None = None,
    ) -> str:
        """T1: atomically create a ``prepared`` attempt and claim the request."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            return self._prepare_attempt_tx(
                conn,
                request_id=request_id,
                owner_token=owner_token,
                provider=provider,
                model=model,
                profile_digest=profile_digest,
                credential_identity=credential_identity,
                reservation=reservation,
            )

    def prepare_attempt_guarded(
        self,
        *,
        request_id: str,
        owner_token: str,
        policy: Mapping[str, Any],
        provider: str = '',
        model: str = '',
        profile_digest: str = '',
        credential_identity: str = '',
        reservation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """T1 with atomic request/root/run/time/cost budget enforcement.

        The return value has ``prepared=True`` and an ``attempt_id`` when a
        Provider call may proceed.  A request-scoped limit terminalizes only
        that leaf.  A run-scoped limit closes every not-yet-dispatched leaf
        in the same transaction and returns ``scope='run'``.
        """
        limits = dict(policy or {})
        reservation_payload = dict(reservation or {})
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            run = self._load_run_tx(conn, self.run_id)
            request = self._load_request_tx(conn, request_id)
            request_reason = self._request_budget_reason_tx(
                conn, request=request, policy=limits
            )
            if request_reason:
                self._terminalize_request_tx(
                    conn, request=request, reason_code=request_reason
                )
                self._touch_run_tx(conn)
                return {
                    'prepared': False,
                    'scope': 'request',
                    'reason_code': request_reason,
                    'request_id': str(request_id),
                }

            run_reason = self._run_budget_reason_tx(
                conn,
                run=run,
                policy=limits,
                reservation=reservation_payload,
            )
            if run_reason:
                self._close_not_dispatched_for_run_stop_tx(
                    conn, reason_code=run_reason
                )
                self._finish_run_if_quiescent_tx(conn)
                self._touch_run_tx(conn)
                return {
                    'prepared': False,
                    'scope': 'run',
                    'reason_code': run_reason,
                    'request_id': str(request_id),
                }

            attempt_id = self._prepare_attempt_tx(
                conn,
                request_id=request_id,
                owner_token=owner_token,
                provider=provider,
                model=model,
                profile_digest=profile_digest,
                credential_identity=credential_identity,
                reservation=reservation_payload,
            )
            return {
                'prepared': True,
                'scope': None,
                'reason_code': None,
                'request_id': str(request_id),
                'attempt_id': attempt_id,
            }

    def _request_budget_reason_tx(
        self,
        conn: sqlite3.Connection,
        *,
        request: sqlite3.Row,
        policy: Mapping[str, Any],
    ) -> str | None:
        root_id = str(request['root_request_id'])
        if int(request['attempt_count']) >= int(policy['max_attempts_per_request']):
            return contracts.REASON_REQUEST_ATTEMPTS_EXHAUSTED
        root_attempts = conn.execute(
            'SELECT COUNT(*) AS n FROM attempts WHERE run_id = ? AND request_id IN '
            '(SELECT request_id FROM requests WHERE run_id = ? AND root_request_id = ?)',
            (self.run_id, self.run_id, root_id),
        ).fetchone()['n']
        if int(root_attempts) >= int(policy['max_attempts_per_root']):
            return contracts.REASON_ROOT_ATTEMPTS_EXHAUSTED
        if int(request['lineage_depth']) > int(policy['max_lineage_depth']):
            return contracts.REASON_LINEAGE_DEPTH_EXHAUSTED
        derived_count = conn.execute(
            'SELECT COUNT(*) AS n FROM requests WHERE run_id = ? '
            'AND root_request_id = ? AND parent_request_id IS NOT NULL',
            (self.run_id, root_id),
        ).fetchone()['n']
        if int(derived_count) > int(policy['max_derived_requests_per_root']):
            return contracts.REASON_DERIVED_REQUESTS_EXHAUSTED
        return None

    @staticmethod
    def _numeric_cost(payload: Mapping[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0:
                    return parsed
        return None

    def _run_budget_reason_tx(
        self,
        conn: sqlite3.Connection,
        *,
        run: sqlite3.Row,
        policy: Mapping[str, Any],
        reservation: Mapping[str, Any],
    ) -> str | None:
        total_attempts = conn.execute(
            'SELECT COUNT(*) AS n FROM attempts WHERE run_id = ?', (self.run_id,)
        ).fetchone()['n']
        if int(total_attempts) >= int(policy['max_total_attempts_per_run']):
            return contracts.REASON_RUN_POLICY_ATTEMPTS

        first_dispatched_at = run['first_dispatched_at']
        if first_dispatched_at:
            try:
                started = datetime.fromisoformat(
                    str(first_dispatched_at).replace('Z', '+00:00')
                )
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            except (TypeError, ValueError):
                elapsed = float('inf')
            if elapsed >= float(policy['max_elapsed_seconds']):
                return contracts.REASON_RUN_BUDGET_EXHAUSTED_TIME

        attempt_rows = conn.execute(
            'SELECT reservation_json, usage_metadata_json FROM attempts WHERE run_id = ?',
            (self.run_id,),
        ).fetchall()
        estimated_total = 0.0
        actual_total = 0.0
        for attempt in attempt_rows:
            reservation_payload = json.loads(attempt['reservation_json'] or '{}')
            usage_payload = json.loads(attempt['usage_metadata_json'] or '{}')
            estimated = self._numeric_cost(
                reservation_payload, 'estimated_cost', 'cost_upper_bound', 'cost'
            )
            actual = self._numeric_cost(
                usage_payload, 'actual_cost', 'cost', 'estimated_cost'
            )
            if estimated is not None:
                estimated_total += estimated
            if actual is not None:
                actual_total += actual

        max_estimated = policy.get('max_estimated_cost')
        if max_estimated is not None:
            next_estimated = self._numeric_cost(
                reservation, 'estimated_cost', 'cost_upper_bound', 'cost'
            )
            if next_estimated is None:
                return contracts.REASON_RUN_BUDGET_EXHAUSTED_COST
            if estimated_total + next_estimated > float(max_estimated):
                return contracts.REASON_RUN_BUDGET_EXHAUSTED_COST
        max_actual = policy.get('max_actual_cost')
        if max_actual is not None and actual_total >= float(max_actual):
            return contracts.REASON_RUN_BUDGET_EXHAUSTED_COST
        return None

    def _terminalize_request_tx(
        self,
        conn: sqlite3.Connection,
        *,
        request: sqlite3.Row,
        reason_code: str,
    ) -> None:
        status = RequestStatus(str(request['status']))
        if status in (RequestStatus.PENDING, RequestStatus.RETRYABLE_FAILED):
            self._transition_request_tx(
                conn,
                request,
                RequestStatus.TERMINAL_FAILED,
                safe_details={'reason_code': str(reason_code)},
            )

    def _close_not_dispatched_for_run_stop_tx(
        self, conn: sqlite3.Connection, *, reason_code: str
    ) -> None:
        prepared_attempts = conn.execute(
            'SELECT * FROM attempts WHERE run_id = ? AND status = ?',
            (self.run_id, AttemptStatus.PREPARED.value),
        ).fetchall()
        for attempt in prepared_attempts:
            self._transition_attempt_tx(
                conn,
                attempt,
                AttemptStatus.TERMINAL_FAILED,
                event_type=EventType.ATTEMPT_FAILED,
                safe_details={'reason_code': str(reason_code)},
            )
            conn.execute(
                'UPDATE attempts SET finish_time = ?, error_category = ?, '
                'error_reason_code = ? WHERE attempt_id = ?',
                (
                    _now_iso(),
                    ErrorCategory.LOCAL_VALIDATION.value,
                    str(reason_code),
                    attempt['attempt_id'],
                ),
            )
            request = self._load_request_tx(conn, str(attempt['request_id']))
            if str(request['status']) == RequestStatus.IN_FLIGHT.value:
                self._transition_request_tx(
                    conn,
                    request,
                    RequestStatus.TERMINAL_FAILED,
                    safe_details={'reason_code': str(reason_code)},
                )
        remaining = conn.execute(
            'SELECT * FROM requests WHERE run_id = ? AND status IN (?, ?)',
            (
                self.run_id,
                RequestStatus.PENDING.value,
                RequestStatus.RETRYABLE_FAILED.value,
            ),
        ).fetchall()
        for request in remaining:
            self._terminalize_request_tx(
                conn, request=request, reason_code=reason_code
            )

    def _finish_run_if_quiescent_tx(self, conn: sqlite3.Connection) -> bool:
        if self._has_active_attempts_tx(conn):
            return False
        active_requests = conn.execute(
            'SELECT COUNT(*) AS n FROM requests WHERE run_id = ? '
            'AND status IN (?, ?, ?)',
            (
                self.run_id,
                RequestStatus.PENDING.value,
                RequestStatus.IN_FLIGHT.value,
                RequestStatus.RETRYABLE_FAILED.value,
            ),
        ).fetchone()['n']
        if int(active_requests):
            return False
        run = self._load_run_tx(conn, self.run_id)
        if RunStatus(str(run['status'])) in contracts.RUN_TERMINAL_STATES:
            return False
        accepted_count = conn.execute(
            'SELECT COUNT(*) AS n FROM item_results WHERE run_id = ?', (self.run_id,)
        ).fetchone()['n']
        target = (
            RunStatus.COMPLETED_WITH_ERRORS if int(accepted_count) else RunStatus.FAILED
        )
        self._transition_run_tx(
            conn,
            run,
            target,
            event_type=EventType.RUN_STATUS,
            safe_details={'accepted_count': int(accepted_count)},
        )
        conn.execute(
            'UPDATE runs SET finished_at = ? WHERE run_id = ?',
            (_now_iso(), self.run_id),
        )
        return True

    def _prepare_attempt_tx(
        self,
        conn: sqlite3.Connection,
        *,
        request_id: str,
        owner_token: str,
        provider: str,
        model: str,
        profile_digest: str,
        credential_identity: str,
        reservation: Mapping[str, Any] | None,
    ) -> str:
        run = self._load_run_tx(conn, self.run_id)
        if run['status'] == RunStatus.PLANNED.value:
            self._transition_run_tx(conn, run, RunStatus.RUNNING)
            run['status'] = RunStatus.RUNNING.value
        if run['status'] != RunStatus.RUNNING.value:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                f'cannot prepare attempt while run is {run["status"]}',
                safe_details={'run_id': self.run_id, 'run_status': run['status']},
            )
        if int(run['cancel_epoch']) > 0:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                'cannot prepare attempt after cancellation was requested',
                safe_details={'run_id': self.run_id},
            )
        request = self._load_request_tx(conn, request_id)
        current_status = RequestStatus(str(request['status']))
        if current_status not in (
            RequestStatus.PENDING,
            RequestStatus.RETRYABLE_FAILED,
        ):
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                f'request {request_id} is not eligible: {request["status"]}',
                safe_details={'run_id': self.run_id, 'request_id': str(request_id)},
            )
        if current_status is RequestStatus.RETRYABLE_FAILED:
            eligible_at = request['next_eligible_at']
            if not eligible_at or str(eligible_at) > _now_iso():
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'request {request_id} is in backoff until {eligible_at}',
                    safe_details={'run_id': self.run_id, 'request_id': str(request_id)},
                )
        existing_active = conn.execute(
            'SELECT attempt_id FROM attempts WHERE run_id = ? AND request_id = ? '
            'AND status IN (?, ?, ?)',
            (
                self.run_id,
                str(request_id),
                AttemptStatus.PREPARED.value,
                AttemptStatus.DISPATCHED.value,
                AttemptStatus.CANCEL_REQUESTED.value,
            ),
        ).fetchone()
        if existing_active is not None:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                f'request already has an active attempt: {request_id}',
                safe_details={'run_id': self.run_id, 'request_id': str(request_id)},
            )

        ordinal = int(request['attempt_count']) + 1
        attempt_id = contracts.build_attempt_id(self.run_id, str(request_id), ordinal)
        now = _now_iso()
        conn.execute(
            'INSERT INTO attempts('
            ' attempt_id, run_id, request_id, ordinal, status, provider, model,'
            ' profile_digest, credential_identity, claim_owner_token, claim_cancel_epoch,'
            ' reservation_json, created_at'
            ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                attempt_id,
                self.run_id,
                str(request_id),
                ordinal,
                AttemptStatus.PREPARED.value,
                provider,
                model,
                profile_digest,
                credential_identity,
                owner_token,
                int(run['cancel_epoch']),
                canonical_json(dict(reservation or {})),
                now,
            ),
        )
        self._transition_request_tx(conn, request, RequestStatus.IN_FLIGHT)
        conn.execute(
            'UPDATE requests SET attempt_count = ? WHERE run_id = ? AND request_id = ?',
            (ordinal, self.run_id, str(request_id)),
        )
        self._write_event_tx(
            conn,
            run_id=self.run_id,
            entity_type='attempt',
            entity_id=attempt_id,
            event_type=EventType.ATTEMPT_PREPARED,
            old_status=None,
            new_status=AttemptStatus.PREPARED.value,
            safe_details={'request_id': str(request_id), 'ordinal': ordinal},
        )
        self._touch_run_tx(conn)
        return attempt_id

    def dispatch_attempt(self, *, attempt_id: str, owner_token: str) -> dict:
        """T2: persist dispatch intent before any network byte is sent."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            run = self._load_run_tx(conn, self.run_id)
            if run['status'] not in (RunStatus.RUNNING.value, RunStatus.CANCEL_REQUESTED.value):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'cannot dispatch attempt while run is {run["status"]}',
                    safe_details={'run_id': self.run_id, 'run_status': run['status']},
                )
            attempt = conn.execute(
                'SELECT * FROM attempts WHERE attempt_id = ? AND run_id = ?',
                (str(attempt_id), self.run_id),
            ).fetchone()
            if attempt is None:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_NOT_FOUND,
                    f'attempt not found: {attempt_id}',
                    safe_details={'run_id': self.run_id, 'attempt_id': str(attempt_id)},
                )
            if str(attempt['status']) != AttemptStatus.PREPARED.value:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'attempt {attempt_id} is not prepared',
                    safe_details={
                        'run_id': self.run_id,
                        'attempt_id': str(attempt_id),
                        'attempt_status': attempt['status'],
                    },
                )
            if run['status'] == RunStatus.CANCEL_REQUESTED.value:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'cannot dispatch attempt after cancellation was requested',
                    safe_details={'run_id': self.run_id},
                )
            if int(run['cancel_epoch']) != int(attempt['claim_cancel_epoch']):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'attempt claim epoch is stale',
                    safe_details={'run_id': self.run_id, 'attempt_id': str(attempt_id)},
                )
            if str(attempt['claim_owner_token']) != str(owner_token):
                conn.execute(
                    'UPDATE attempts SET claim_owner_token = ? WHERE attempt_id = ?',
                    (str(owner_token), attempt['attempt_id']),
                )
                attempt = conn.execute(
                    'SELECT * FROM attempts WHERE attempt_id = ?',
                    (attempt['attempt_id'],),
                ).fetchone()
            now = _now_iso()
            self._transition_attempt_tx(
                conn,
                attempt,
                AttemptStatus.DISPATCHED,
                event_type=EventType.ATTEMPT_DISPATCHED,
                safe_details={'request_id': attempt['request_id']},
            )
            conn.execute(
                'UPDATE attempts SET dispatch_time = ? WHERE attempt_id = ?',
                (now, attempt['attempt_id']),
            )
            conn.execute(
                'UPDATE runs SET first_dispatched_at = COALESCE(first_dispatched_at, ?),'
                ' revision = revision + 1, updated_at = ? WHERE run_id = ?',
                (now, now, self.run_id),
            )
            updated = conn.execute(
                'SELECT * FROM attempts WHERE attempt_id = ?', (attempt['attempt_id'],)
            ).fetchone()
            return dict(updated)

    # ------------------------------------------------------------------
    # T3/T4: receipt handling
    # ------------------------------------------------------------------
    def seed_reused_results(
        self,
        *,
        request_id: str,
        owner_token: str,
        source_run_id: str,
        reused_items: Mapping[str, Mapping[str, Any]],
        derived_requests: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        """Seed strictly validated winners from an immutable source run.

        Reuse is not a Provider attempt and therefore does not consume attempt,
        cost, or elapsed-time budgets.  A partial reuse atomically supersedes
        the current root and creates children for every non-reused ID.
        """
        if str(source_run_id) == self.run_id:
            raise ValueError('a run cannot reuse results from itself')
        reused_map = {str(item_id): dict(value or {}) for item_id, value in reused_items.items()}
        if not reused_map:
            return False
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            run = self._load_run_tx(conn, self.run_id)
            if str(run['status']) not in (
                RunStatus.PLANNED.value,
                RunStatus.RUNNING.value,
            ) or int(run['cancel_epoch']):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'cannot seed reused results into a non-dispatchable run',
                    safe_details={'run_id': self.run_id, 'run_status': run['status']},
                )
            if str(run['status']) == RunStatus.PLANNED.value:
                self._transition_run_tx(conn, run, RunStatus.RUNNING)
                run = self._load_run_tx(conn, self.run_id)
            request = self._load_request_tx(conn, request_id)
            if str(request['status']) != RequestStatus.PENDING.value:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'reuse target request is not pending: {request_id}',
                    safe_details={'run_id': self.run_id, 'request_id': request_id},
                )
            expected_ids = json.loads(request['expected_ids_json'] or '[]')
            extra_ids = set(reused_map) - set(expected_ids)
            if extra_ids:
                raise ValueError(f'reused items are outside request scope: {sorted(extra_ids)}')
            accepted_ids = [item_id for item_id in expected_ids if item_id in reused_map]
            missing_ids = [item_id for item_id in expected_ids if item_id not in reused_map]
            if not missing_ids and derived_requests:
                raise ValueError('complete reuse must not create derived requests')
            children = []
            if missing_ids:
                children = self._validate_derived_children_tx(
                    request,
                    derived_requests,
                    remaining_ids=missing_ids,
                )
            now = _now_iso()
            for item_id in accepted_ids:
                payload = dict(reused_map[item_id])
                source_attempt_id = str(payload.pop('source_attempt_id', '') or '')
                translation = payload.get('translation', payload)
                digest = str(
                    payload.get('translation_digest')
                    or sha256_hex(canonical_json(translation))
                )
                diagnostics = payload.get('validation_diagnostics') or {}
                conn.execute(
                    'INSERT INTO item_results('
                    ' run_id, item_id, winner_request_id, winner_attempt_id, '
                    'reused_from_run_id, reused_from_attempt_id, '
                    'translation_payload_json, translation_digest, '
                    'validation_diagnostics_json, created_at'
                    ') VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)',
                    (
                        self.run_id,
                        item_id,
                        str(request_id),
                        str(source_run_id),
                        source_attempt_id or None,
                        canonical_json(translation),
                        digest,
                        canonical_json(diagnostics),
                        now,
                    ),
                )
            if missing_ids:
                self._transition_request_tx(
                    conn,
                    request,
                    RequestStatus.SUPERSEDED,
                    safe_details={
                        'reused_count': len(accepted_ids),
                        'missing_count': len(missing_ids),
                        'source_run_id': str(source_run_id),
                    },
                )
                self._insert_derived_children_tx(
                    conn,
                    parent_request=request,
                    children=children,
                    now=now,
                )
            else:
                self._transition_request_tx(
                    conn,
                    request,
                    RequestStatus.SUCCEEDED,
                    safe_details={
                        'reused_count': len(accepted_ids),
                        'source_run_id': str(source_run_id),
                    },
                )
            self._write_event_tx(
                conn,
                run_id=self.run_id,
                entity_type='request',
                entity_id=str(request_id),
                event_type=EventType.NOTICE,
                old_status=None,
                new_status='results_reused',
                safe_details={
                    'source_run_id': str(source_run_id),
                    'reused_count': len(accepted_ids),
                },
            )
            self._touch_run_tx(conn)
            return True

    def record_success(
        self,
        *,
        attempt_id: str,
        owner_token: str,
        accepted_items: Mapping[str, Any] | Sequence[str],
        response_payload: Any = None,
        normalized_payload: Any = None,
        contract_diagnostics: Any = None,
        usage_metadata: Mapping[str, Any] | None = None,
        derived_requests: Sequence[Mapping[str, Any]] = (),
        partial_terminal_reason: str | None = None,
    ) -> bool:
        """T3: atomically commit a successful attempt and accepted item winners.

        ``accepted_items`` may be a mapping of item IDs to payload mappings or
        a sequence of item IDs.  Full coverage marks the request ``succeeded``;
        a strict subset requires ``derived_requests`` children for the missing
        IDs so parent ``superseded`` and children are committed together.

        Returns ``True`` when the receipt was accepted as a normal closeout.
        A late receipt (cancel epoch / lease owner / status guard failed) is
        only audited and returns ``False``.
        """
        with self._tx() as conn:
            run = self._load_run_tx(conn, self.run_id)
            attempt = self._load_attempt_tx(conn, attempt_id)
            if not self._receipt_allowed_tx(conn, attempt, run, owner_token):
                return self._record_late_receipt_tx(
                    conn,
                    attempt=attempt,
                    run=run,
                    observed_owner_token=owner_token,
                    response_payload=response_payload,
                    error_payload=None,
                    usage_payload=usage_metadata,
                    ignored_reason='late.success',
                    late_status=AttemptStatus.LATE_SUCCEEDED_IGNORED,
                )
            request = self._load_request_tx(conn, str(attempt['request_id']))
            accepted_map = self._normalize_accepted_items(accepted_items)
            expected_ids = json.loads(request['expected_ids_json'] or '[]')
            accepted_ids = set(accepted_map.keys())
            missing_ids = [item_id for item_id in expected_ids if item_id not in accepted_ids]
            extra_ids = accepted_ids - set(expected_ids)
            if extra_ids:
                raise ValueError(
                    f'accepted items not present in request expected_ids: {sorted(extra_ids)}'
                )
            if not missing_ids and (derived_requests or partial_terminal_reason):
                raise ValueError('complete success must not create derived requests')
            if missing_ids and not accepted_ids:
                raise ValueError(
                    'a zero-progress response must close as failed before it can be split'
                )
            if missing_ids and derived_requests and partial_terminal_reason:
                raise ValueError(
                    'partial success cannot both derive children and terminalize the parent'
                )
            validated_children = []
            if missing_ids and not partial_terminal_reason:
                validated_children = self._validate_derived_children_tx(
                    request,
                    derived_requests,
                    remaining_ids=missing_ids,
                )

            now = _now_iso()
            self._insert_item_winners_tx(
                conn,
                request=request,
                attempt_id=attempt['attempt_id'],
                accepted_map=accepted_map,
                now=now,
            )
            self._update_attempt_receipt_tx(
                conn,
                attempt=attempt,
                status=AttemptStatus.SUCCEEDED,
                now=now,
                response_payload=response_payload,
                normalized_payload=normalized_payload,
                contract_diagnostics=contract_diagnostics,
                usage_metadata=usage_metadata,
                error_category=None,
                error_reason_code=None,
                error_safe_details=None,
            )
            if usage_metadata:
                self._enqueue_usage_tx(conn, attempt['attempt_id'], usage_metadata, now=now)

            if not missing_ids:
                self._transition_request_tx(
                    conn, request, RequestStatus.SUCCEEDED,
                    safe_details={'accepted_count': len(accepted_ids)},
                )
            elif partial_terminal_reason:
                self._transition_request_tx(
                    conn,
                    request,
                    RequestStatus.TERMINAL_FAILED,
                    safe_details={
                        'accepted_count': len(accepted_ids),
                        'missing_count': len(missing_ids),
                        'reason_code': str(partial_terminal_reason),
                    },
                )
            else:
                self._transition_request_tx(
                    conn, request, RequestStatus.SUPERSEDED,
                    safe_details={
                        'accepted_count': len(accepted_ids),
                        'missing_count': len(missing_ids),
                    },
                )
                self._insert_derived_children_tx(
                    conn,
                    parent_request=request,
                    children=validated_children,
                    now=now,
                )
            self._touch_run_tx(conn)
            return True

    def record_failure(
        self,
        *,
        attempt_id: str,
        owner_token: str,
        error_category: ErrorCategory | str,
        error_reason_code: str = '',
        error_safe_details: Mapping[str, Any] | None = None,
        next_eligible_at: str | None = None,
        terminal: bool | None = None,
        usage_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """T4: atomically commit a failed attempt and its retry/terminal decision.

        Returns ``True`` for a normal receipt; a late receipt is only audited.
        """
        category = (
            error_category
            if isinstance(error_category, ErrorCategory)
            else ErrorCategory(str(error_category))
        )
        category_value = category.value
        with self._tx() as conn:
            run = self._load_run_tx(conn, self.run_id)
            attempt = self._load_attempt_tx(conn, attempt_id)
            if not self._receipt_allowed_tx(conn, attempt, run, owner_token):
                return self._record_late_receipt_tx(
                    conn,
                    attempt=attempt,
                    run=run,
                    observed_owner_token=owner_token,
                    response_payload=None,
                    error_payload={
                        'category': category_value,
                        'reason_code': str(error_reason_code),
                        'details': dict(error_safe_details or {}),
                    },
                    usage_payload=usage_metadata,
                    ignored_reason='late.failure',
                    late_status=AttemptStatus.LATE_FAILED_IGNORED,
                )
            if terminal is None:
                terminal = category in contracts.TERMINAL_ERROR_CATEGORIES
            now = _now_iso()
            attempt_status = (
                AttemptStatus.TERMINAL_FAILED if terminal else AttemptStatus.RETRYABLE_FAILED
            )
            request_status = (
                RequestStatus.TERMINAL_FAILED if terminal else RequestStatus.RETRYABLE_FAILED
            )
            self._update_attempt_receipt_tx(
                conn,
                attempt=attempt,
                status=attempt_status,
                now=now,
                error_category=category,
                error_reason_code=str(error_reason_code),
                error_safe_details=dict(error_safe_details or {}),
                usage_metadata=usage_metadata,
            )
            if usage_metadata:
                self._enqueue_usage_tx(conn, attempt['attempt_id'], usage_metadata, now=now)
            request = self._load_request_tx(conn, str(attempt['request_id']))
            self._transition_request_tx(conn, request, request_status)
            if not terminal:
                eligible_at = next_eligible_at or now
                conn.execute(
                    'UPDATE requests SET next_eligible_at = ? WHERE run_id = ? AND request_id = ?',
                    (eligible_at, self.run_id, str(request['request_id'])),
                )
                conn.execute(
                    'UPDATE attempts SET next_eligible_at = ? WHERE attempt_id = ?',
                    (eligible_at, attempt['attempt_id']),
                )
            self._touch_run_tx(conn)
            return True

    def mark_outcome_unknown(
        self,
        *,
        attempt_id: str,
        owner_token: str,
        reason_code: str = 'outcome_unknown',
        usage_metadata: Mapping[str, Any] | None = None,
        response_payload: Any = None,
    ) -> bool:
        """Crash-closeout path for an orphaned dispatched attempt."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            self._load_run_tx(conn, self.run_id)
            attempt = self._load_attempt_tx(conn, attempt_id)
            if str(attempt['status']) not in (
                AttemptStatus.DISPATCHED.value,
                AttemptStatus.CANCEL_REQUESTED.value,
            ):
                return False
            if str(attempt['status']) == AttemptStatus.CANCEL_REQUESTED.value:
                # Cancel-closeout path can end unknown after best-effort cancel.
                pass
            now = _now_iso()
            self._transition_attempt_tx(
                conn,
                attempt,
                AttemptStatus.OUTCOME_UNKNOWN,
                event_type=EventType.ATTEMPT_UNKNOWN,
                safe_details={'reason_code': str(reason_code)},
            )
            conn.execute(
                'UPDATE attempts SET finish_time = ?, error_reason_code = ?,'
                ' response_payload_json = ? WHERE attempt_id = ?',
                (now, str(reason_code), canonical_json(response_payload), attempt['attempt_id']),
            )
            if usage_metadata:
                self._enqueue_usage_tx(conn, attempt['attempt_id'], usage_metadata, now=now)
            request = self._load_request_tx(conn, str(attempt['request_id']))
            if str(request['status']) == RequestStatus.IN_FLIGHT.value:
                self._transition_request_tx(conn, request, RequestStatus.OUTCOME_UNKNOWN)
            self._touch_run_tx(conn)
            return True

    # ------------------------------------------------------------------
    # Derivation (T5)
    # ------------------------------------------------------------------
    def supersede_with_children(
        self,
        *,
        request_id: str,
        children: Sequence[Mapping[str, Any]],
        owner_token: str,
    ) -> list[str]:
        """T5: atomically supersede a parent and insert all derived children.

        The parent must not have any active attempt.  This is the single
        atomic boundary for lineage splits.
        """
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            parent = self._load_request_tx(conn, request_id)
            active = conn.execute(
                'SELECT attempt_id FROM attempts WHERE run_id = ? AND request_id = ? '
                'AND status IN (?, ?, ?)',
                (
                    self.run_id,
                    str(request_id),
                    AttemptStatus.PREPARED.value,
                    AttemptStatus.DISPATCHED.value,
                    AttemptStatus.CANCEL_REQUESTED.value,
                ),
            ).fetchone()
            if active is not None:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'request {request_id} still has an active attempt',
                    safe_details={
                        'run_id': self.run_id,
                        'request_id': str(request_id),
                        'attempt_id': active['attempt_id'],
                    },
                )
            if str(parent['status']) in (
                RequestStatus.SUCCEEDED.value,
                RequestStatus.SUPERSEDED.value,
                RequestStatus.CANCELLED.value,
                RequestStatus.OUTCOME_UNKNOWN.value,
            ):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'request {request_id} cannot be superseded from {parent["status"]}',
                    safe_details={
                        'run_id': self.run_id,
                        'request_id': str(request_id),
                        'request_status': parent['status'],
                    },
                )
            if not children:
                raise ValueError('children must not be empty')
            expected_ids = json.loads(parent['expected_ids_json'] or '[]')
            validated_children = self._validate_derived_children_tx(
                parent,
                children,
                remaining_ids=expected_ids,
            )
            now = _now_iso()
            self._transition_request_tx(
                conn, parent, RequestStatus.SUPERSEDED,
                safe_details={'derived_count': len(children)},
            )
            created = self._insert_derived_children_tx(
                conn,
                parent_request=parent,
                children=validated_children,
                now=now,
            )
            self._touch_run_tx(conn)
            return created

    def _validate_derived_children_tx(
        self,
        parent_request: sqlite3.Row,
        children: Sequence[Mapping[str, Any]],
        *,
        remaining_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not children:
            raise ValueError('derived children must cover every remaining item')
        parent_id = str(parent_request['request_id'])
        parent_payload = json.loads(parent_request['payload_json'])
        plan_id = str(parent_payload.get('plan_id') or '')
        remaining_order = list(remaining_ids)
        remaining_set = set(remaining_order)
        if len(remaining_set) != len(remaining_order):
            raise ValueError('remaining item IDs must be unique')
        covered: set[str] = set()
        flattened: list[str] = []
        normalized_children: list[dict[str, Any]] = []
        for raw_child in children:
            child = _normalized_root_request_payload(raw_child)
            child_id = str(child['request_id'])
            if not child_id.startswith(parent_id + '--'):
                raise ValueError(
                    f'derived request {child_id} is not a child of {parent_id}'
                )
            if str(child.get('plan_id') or '') != plan_id:
                raise ValueError(f'derived request {child_id} changed plan_id')
            child_ids = list(child['expected_ids'])
            child_set = set(child_ids)
            if not child_set <= remaining_set:
                raise ValueError(
                    f'derived request {child_id} contains accepted or unknown item IDs'
                )
            if covered & child_set:
                raise ValueError('derived request children overlap')
            expected_child_order = [
                item_id for item_id in remaining_order if item_id in child_set
            ]
            if child_ids != expected_child_order:
                raise ValueError(
                    f'derived request {child_id} changed parent item order'
                )
            transport = dict(child.get('transport_metadata') or {})
            if str(transport.get('retry_parent_request_id') or '') != parent_id:
                raise ValueError(
                    f'derived request {child_id} lacks its retry parent identity'
                )
            if list(transport.get('retry_item_ids') or []) != child_ids:
                raise ValueError(
                    f'derived request {child_id} retry item identity mismatch'
                )
            covered.update(child_set)
            flattened.extend(child_ids)
            normalized_children.append(child)
        if covered != remaining_set or flattened != remaining_order:
            missing = [item_id for item_id in remaining_order if item_id not in covered]
            raise ValueError(
                f'derived request children do not exactly cover remaining IDs: {missing}'
            )
        return normalized_children

    def _insert_derived_children_tx(
        self,
        conn: sqlite3.Connection,
        *,
        parent_request: sqlite3.Row,
        children: Sequence[Mapping[str, Any]],
        now: str | None = None,
    ) -> list[str]:
        now = now or _now_iso()
        created = []
        for child in children:
            child_id = self._insert_derived_request_tx(
                conn,
                run_id=self.run_id,
                request=child,
                parent_request=parent_request,
                now=now,
            )
            created.append(child_id)
        return created

    def _update_attempt_receipt_tx(
        self,
        conn: sqlite3.Connection,
        *,
        attempt: sqlite3.Row,
        status: AttemptStatus,
        now: str,
        error_category: ErrorCategory | None = None,
        error_reason_code: str | None = None,
        error_safe_details: Mapping[str, Any] | None = None,
        response_payload: Any = None,
        normalized_payload: Any = None,
        contract_diagnostics: Any = None,
        usage_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._transition_attempt_tx(conn, attempt, status)
        conn.execute(
            'UPDATE attempts SET'
            ' finish_time = ?, error_category = ?, error_reason_code = ?,'
            ' error_safe_details_json = ?, response_payload_json = ?,'
            ' normalized_payload_json = ?, contract_diagnostics_json = ?,'
            ' usage_metadata_json = ?'
            ' WHERE attempt_id = ?',
            (
                now,
                None if error_category is None else error_category.value,
                error_reason_code,
                canonical_json(dict(error_safe_details or {})),
                canonical_json(response_payload),
                canonical_json(normalized_payload),
                canonical_json(contract_diagnostics),
                canonical_json(dict(usage_metadata or {})),
                attempt['attempt_id'],
            ),
        )

    def _insert_item_winners_tx(
        self,
        conn: sqlite3.Connection,
        *,
        request: sqlite3.Row,
        attempt_id: str,
        accepted_map: Mapping[str, Any],
        now: str,
    ) -> None:
        for item_id, payload in accepted_map.items():
            payload = dict(payload or {}) if isinstance(payload, Mapping) else {}
            translation = payload.get('translation', payload)
            digest = str(payload.get('translation_digest') or sha256_hex(canonical_json(translation)))
            diagnostics = payload.get('validation_diagnostics') or {}
            existing = conn.execute(
                'SELECT winner_attempt_id FROM item_results WHERE run_id = ? AND item_id = ?',
                (self.run_id, str(item_id)),
            ).fetchone()
            if existing is not None:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'item {item_id} already has an accepted winner',
                    safe_details={
                        'run_id': self.run_id,
                        'item_id': str(item_id),
                        'existing_winner_attempt_id': existing['winner_attempt_id'],
                    },
                )
            conn.execute(
                'INSERT INTO item_results('
                ' run_id, item_id, winner_request_id, winner_attempt_id, '
                'translation_payload_json,'
                ' translation_digest, validation_diagnostics_json, created_at'
                ') VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    self.run_id,
                    str(item_id),
                    str(request['request_id']),
                    attempt_id,
                    canonical_json(translation),
                    digest,
                    canonical_json(diagnostics),
                    now,
                ),
            )

    def _normalize_accepted_items(
        self, accepted_items: Mapping[str, Any] | Sequence[str]
    ) -> dict[str, Any]:
        if isinstance(accepted_items, Mapping):
            return {str(item_id): item for item_id, item in accepted_items.items()}
        item_ids = [str(item_id) for item_id in accepted_items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError('accepted_items contains duplicate item IDs')
        return {item_id: {} for item_id in item_ids}

    # ------------------------------------------------------------------
    # Late receipts
    # ------------------------------------------------------------------
    def _record_late_receipt_tx(
        self,
        conn: sqlite3.Connection,
        *,
        attempt: sqlite3.Row,
        run: Mapping[str, Any],
        observed_owner_token: str,
        response_payload: Any = None,
        error_payload: Any = None,
        usage_payload: Mapping[str, Any] | None = None,
        ignored_reason: str,
        late_status: AttemptStatus | None = None,
    ) -> bool:
        now = _now_iso()
        receipt_id = contracts.sha256_hex(
            canonical_json([
                'late-receipt',
                attempt['attempt_id'],
                now,
                canonical_json(response_payload),
                canonical_json(error_payload),
            ])
        )[:32]
        conn.execute(
            'INSERT INTO late_receipts('
            ' receipt_id, run_id, attempt_id, observed_owner_token,'
            ' observed_cancel_epoch, response_payload_json, error_payload_json,'
            ' usage_payload_json, ignored_reason, received_at'
            ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                receipt_id,
                self.run_id,
                attempt['attempt_id'],
                observed_owner_token,
                int(run['cancel_epoch']),
                canonical_json(response_payload),
                canonical_json(error_payload),
                canonical_json(dict(usage_payload or {})),
                str(ignored_reason),
                now,
            ),
        )
        if usage_payload:
            self._enqueue_usage_tx(conn, attempt['attempt_id'], usage_payload, now=now)
        current = AttemptStatus(str(attempt['status']))
        attempt_closed = False
        if current in (AttemptStatus.DISPATCHED, AttemptStatus.CANCEL_REQUESTED):
            next_status = late_status or AttemptStatus.LATE_FAILED_IGNORED
            if contracts.can_transition(current, next_status, contracts.ATTEMPT_TRANSITIONS):
                self._transition_attempt_tx(conn, attempt, next_status)
                conn.execute(
                    'UPDATE attempts SET finish_time = ? WHERE attempt_id = ?',
                    (now, attempt['attempt_id']),
                )
                attempt_closed = True
        if attempt_closed:
            request = self._load_request_tx(conn, str(attempt['request_id']))
            if str(request['status']) == RequestStatus.IN_FLIGHT.value:
                request_closeout = (
                    RequestStatus.CANCELLED
                    if str(run['status']) == RunStatus.CANCEL_REQUESTED.value
                    else RequestStatus.OUTCOME_UNKNOWN
                )
                self._transition_request_tx(
                    conn,
                    request,
                    request_closeout,
                    safe_details={'reason': str(ignored_reason)},
                )
        self._touch_run_tx(conn)
        return False

    def _load_attempt_tx(self, conn: sqlite3.Connection, attempt_id: str) -> sqlite3.Row:
        row = conn.execute(
            'SELECT * FROM attempts WHERE attempt_id = ? AND run_id = ?',
            (str(attempt_id), self.run_id),
        ).fetchone()
        if row is None:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_NOT_FOUND,
                f'attempt not found: {attempt_id}',
                safe_details={'run_id': self.run_id, 'attempt_id': str(attempt_id)},
            )
        return row

    # ------------------------------------------------------------------
    # Cancellation (T6a / T6b)
    # ------------------------------------------------------------------
    def cancel_intent(self, *, reason: str = 'user') -> bool:
        """T6a: durable cancellation intent.  Idempotent after first call."""
        with self._tx() as conn:
            run = self._load_run_tx(conn, self.run_id)
            current = RunStatus(str(run['status']))
            if current in contracts.RUN_TERMINAL_STATES:
                return False
            if current is RunStatus.CANCEL_REQUESTED:
                return False
            if not contracts.can_transition(current, RunStatus.CANCEL_REQUESTED, contracts.RUN_TRANSITIONS):
                return False
            conn.execute(
                'UPDATE runs SET status = ?, cancel_epoch = cancel_epoch + 1,'
                ' revision = revision + 1, updated_at = ? WHERE run_id = ?',
                (RunStatus.CANCEL_REQUESTED.value, _now_iso(), self.run_id),
            )
            self._write_event_tx(
                conn,
                run_id=self.run_id,
                entity_type='run',
                entity_id=self.run_id,
                event_type=EventType.CANCEL_INTENT,
                old_status=current.value,
                new_status=RunStatus.CANCEL_REQUESTED.value,
                safe_details={'reason': str(reason)},
            )
            return True

    def cancel_closeout(self, *, owner_token: str) -> bool:
        """T6b: close out cancellable leaves as the current lease owner."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            run = self._load_run_tx(conn, self.run_id)
            changed = False
            if str(run['status']) == RunStatus.CANCELLED.value:
                return False
            if str(run['status']) != RunStatus.CANCEL_REQUESTED.value:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'cancel closeout requires a committed cancel intent',
                    retryable=False,
                    safe_details={
                        'run_id': self.run_id,
                        'run_status': run['status'],
                    },
                )

            pending = conn.execute(
                'SELECT * FROM requests WHERE run_id = ? AND status IN (?, ?)',
                (self.run_id, RequestStatus.PENDING.value, RequestStatus.RETRYABLE_FAILED.value),
            ).fetchall()
            for request in pending:
                self._transition_request_tx(
                    conn, request, RequestStatus.CANCELLED,
                    safe_details={'reason': 'cancel_closeout'},
                )
                changed = True

            in_flight = conn.execute(
                'SELECT * FROM requests WHERE run_id = ? AND status = ?',
                (self.run_id, RequestStatus.IN_FLIGHT.value),
            ).fetchall()
            for request in in_flight:
                active_attempts = conn.execute(
                    'SELECT * FROM attempts WHERE run_id = ? AND request_id = ? '
                    'AND status IN (?, ?, ?)',
                    (
                        self.run_id,
                        request['request_id'],
                        AttemptStatus.PREPARED.value,
                        AttemptStatus.DISPATCHED.value,
                        AttemptStatus.CANCEL_REQUESTED.value,
                    ),
                ).fetchall()
                for attempt in active_attempts:
                    current = AttemptStatus(str(attempt['status']))
                    if current is AttemptStatus.PREPARED:
                        self._transition_attempt_tx(conn, attempt, AttemptStatus.CANCELLED)
                        self._transition_request_tx(
                            conn, request, RequestStatus.CANCELLED,
                            safe_details={'reason': 'cancel_closeout.prepared'},
                        )
                        changed = True
                    elif current is AttemptStatus.DISPATCHED:
                        self._transition_attempt_tx(
                            conn, attempt, AttemptStatus.CANCEL_REQUESTED,
                            safe_details={'reason': 'cancel_closeout.dispatched'},
                        )
                        changed = True
                    elif current is AttemptStatus.CANCEL_REQUESTED:
                        pass
            if not self._has_active_leaves_tx(conn):
                run_after = self._load_run_tx(conn, self.run_id)
                if RunStatus(str(run_after['status'])) is RunStatus.CANCEL_REQUESTED:
                    self._transition_run_tx(
                        conn, run_after, RunStatus.CANCELLED,
                        safe_details={'reason': 'cancel_closeout'},
                    )
                    conn.execute(
                        'UPDATE runs SET finished_at = ? WHERE run_id = ?',
                        (_now_iso(), self.run_id),
                    )
                    changed = True
            return changed

    def confirm_attempt_cancelled(
        self,
        *,
        attempt_id: str,
        owner_token: str,
    ) -> bool:
        """Close a provider-confirmed cancellation without accepting a result."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            run = self._load_run_tx(conn, self.run_id)
            if str(run['status']) != RunStatus.CANCEL_REQUESTED.value:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'attempt cancellation requires a committed cancel intent',
                    safe_details={'run_id': self.run_id, 'run_status': run['status']},
                )
            attempt = self._load_attempt_tx(conn, attempt_id)
            if str(attempt['status']) == AttemptStatus.CANCELLED.value:
                return False
            if str(attempt['status']) != AttemptStatus.CANCEL_REQUESTED.value:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    f'attempt {attempt_id} is not awaiting cancellation',
                    safe_details={
                        'run_id': self.run_id,
                        'attempt_id': str(attempt_id),
                        'attempt_status': attempt['status'],
                    },
                )
            self._transition_attempt_tx(
                conn,
                attempt,
                AttemptStatus.CANCELLED,
                safe_details={'reason': 'provider_cancel_confirmed'},
            )
            conn.execute(
                'UPDATE attempts SET finish_time = ? WHERE attempt_id = ?',
                (_now_iso(), attempt['attempt_id']),
            )
            request = self._load_request_tx(conn, str(attempt['request_id']))
            if str(request['status']) == RequestStatus.IN_FLIGHT.value:
                self._transition_request_tx(
                    conn,
                    request,
                    RequestStatus.CANCELLED,
                    safe_details={'reason': 'provider_cancel_confirmed'},
                )
            self._touch_run_tx(conn)
            return True

    def _has_active_leaves_tx(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            'SELECT COUNT(*) AS n FROM requests WHERE run_id = ? AND status IN (?, ?, ?)',
            (
                self.run_id,
                RequestStatus.PENDING.value,
                RequestStatus.IN_FLIGHT.value,
                RequestStatus.RETRYABLE_FAILED.value,
            ),
        ).fetchone()
        return int(row['n']) > 0

    def _has_active_attempts_tx(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            'SELECT COUNT(*) AS n FROM attempts WHERE run_id = ? AND status IN (?, ?, ?)',
            (
                self.run_id,
                AttemptStatus.PREPARED.value,
                AttemptStatus.DISPATCHED.value,
                AttemptStatus.CANCEL_REQUESTED.value,
            ),
        ).fetchone()
        return int(row['n']) > 0

    def _root_expected_ids_tx(self, conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            'SELECT expected_ids_json FROM requests WHERE run_id = ? '
            'AND parent_request_id IS NULL ORDER BY rowid',
            (self.run_id,),
        ).fetchall()
        expected: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for item_id in json.loads(row['expected_ids_json'] or '[]'):
                if item_id in seen:
                    raise SyncRunError(
                        ErrorCode.SYNC_RUN_STORAGE_ERROR,
                        f'duplicate root item identity: {item_id}',
                        safe_details={'run_id': self.run_id, 'item_id': str(item_id)},
                    )
                seen.add(item_id)
                expected.append(item_id)
        return expected

    # ------------------------------------------------------------------
    # Finalization (T8)
    # ------------------------------------------------------------------
    def finalize_run(self, *, owner_token: str) -> dict:
        """T8: compute and persist the terminal run state from transaction facts."""
        with self._tx() as conn:
            self._require_lease_owner_tx(conn, owner_token)
            run = self._load_run_tx(conn, self.run_id)
            current = RunStatus(str(run['status']))
            if current in contracts.RUN_TERMINAL_STATES:
                return dict(run)
            if self._has_active_leaves_tx(conn) or self._has_active_attempts_tx(conn):
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_STORAGE_ERROR,
                    'cannot finalize run with active requests or attempts',
                    retryable=False,
                    safe_details={'run_id': self.run_id, 'run_status': run['status']},
                )
            accepted_rows = conn.execute(
                'SELECT item_id FROM item_results WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            accepted_ids = {str(row['item_id']) for row in accepted_rows}
            expected_ids = set(self._root_expected_ids_tx(conn))
            accepted_count = len(accepted_ids)
            leaf_rows = conn.execute(
                'SELECT status, COUNT(*) AS n FROM requests WHERE run_id = ? '
                'AND status != ? GROUP BY status',
                (self.run_id, RequestStatus.SUPERSEDED.value),
            ).fetchall()
            leaf_statuses = {str(row['status']) for row in leaf_rows}

            if current is RunStatus.CANCEL_REQUESTED:
                next_status = RunStatus.CANCELLED
            elif leaf_statuses <= {RequestStatus.SUCCEEDED.value}:
                if accepted_ids != expected_ids:
                    raise SyncRunError(
                        ErrorCode.SYNC_RUN_STORAGE_ERROR,
                        'all request leaves succeeded without complete item winners',
                        safe_details={
                            'run_id': self.run_id,
                            'missing_item_count': len(expected_ids - accepted_ids),
                            'unexpected_item_count': len(accepted_ids - expected_ids),
                        },
                    )
                next_status = RunStatus.COMPLETED
            elif int(accepted_count) > 0:
                next_status = RunStatus.COMPLETED_WITH_ERRORS
            else:
                next_status = RunStatus.FAILED

            self._transition_run_tx(
                conn,
                run,
                next_status,
                event_type=EventType.RUN_STATUS,
                safe_details={'accepted_count': int(accepted_count)},
            )
            conn.execute(
                'UPDATE runs SET finished_at = ? WHERE run_id = ?',
                (_now_iso(), self.run_id),
            )
            updated = self._load_run_tx(conn, self.run_id)
            return updated

    # ------------------------------------------------------------------
    # Usage outbox (T7)
    # ------------------------------------------------------------------
    def _enqueue_usage_tx(
        self,
        conn: sqlite3.Connection,
        attempt_id: str,
        usage_metadata: Mapping[str, Any],
        *,
        now: str | None = None,
    ) -> str:
        now = now or _now_iso()
        usage_event_id = contracts.build_usage_event_id(attempt_id)
        existing = conn.execute(
            'SELECT usage_event_id FROM usage_outbox WHERE attempt_id = ?',
            (str(attempt_id),),
        ).fetchone()
        if existing is not None:
            return str(existing['usage_event_id'])
        record = {
            'attempt_id': str(attempt_id),
            'run_id': self.run_id,
            **dict(usage_metadata or {}),
        }
        conn.execute(
            'INSERT INTO usage_outbox('
            ' usage_event_id, run_id, attempt_id, record_json, created_at'
            ') VALUES (?, ?, ?, ?, ?)',
            (
                usage_event_id,
                self.run_id,
                str(attempt_id),
                canonical_json(record),
                now,
            ),
        )
        self._write_event_tx(
            conn,
            run_id=self.run_id,
            entity_type='usage_outbox',
            entity_id=usage_event_id,
            event_type=EventType.OUTBOX,
            old_status=None,
            new_status='pending',
            safe_details={'attempt_id': str(attempt_id)},
        )
        return usage_event_id

    def pending_usage_outbox(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM usage_outbox WHERE run_id = ? AND delivered_at IS NULL '
                'ORDER BY created_at',
                (self.run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def ack_usage_outbox(self, *, usage_event_id: str, delivery_error: str | None = None) -> None:
        with self._tx() as conn:
            row = conn.execute(
                'SELECT * FROM usage_outbox WHERE usage_event_id = ? AND run_id = ?',
                (str(usage_event_id), self.run_id),
            ).fetchone()
            if row is None:
                raise SyncRunError(
                    ErrorCode.SYNC_RUN_NOT_FOUND,
                    f'usage outbox event not found: {usage_event_id}',
                    safe_details={'run_id': self.run_id, 'usage_event_id': str(usage_event_id)},
                )
            conn.execute(
                'UPDATE usage_outbox SET delivered_at = ?, delivery_error = ? '
                'WHERE usage_event_id = ?',
                (None if delivery_error else _now_iso(), delivery_error, str(usage_event_id)),
            )

    # ------------------------------------------------------------------
    # Projections / exports
    # ------------------------------------------------------------------
    def request_projection(self, request: Mapping[str, Any]) -> dict:
        expected = json.loads(request.get('expected_ids_json') or '[]')
        return {
            'request_id': request.get('request_id'),
            'root_request_id': request.get('root_request_id'),
            'parent_request_id': request.get('parent_request_id'),
            'lineage_kind': request.get('lineage_kind'),
            'lineage_depth': request.get('lineage_depth'),
            'status': request.get('status'),
            'expected_count': len(expected),
            'prompt_fingerprint': request.get('prompt_fingerprint'),
            'request_fingerprint': request.get('request_fingerprint'),
            'attempt_count': request.get('attempt_count'),
            'next_eligible_at': request.get('next_eligible_at'),
        }

    def build_snapshot(self) -> dict:
        """Return the store-backed run snapshot defined in #347 section 11.2."""
        with self._conn() as conn:
            run = self._load_run_tx(conn, self.run_id)
            request_rows = conn.execute(
                'SELECT * FROM requests WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            attempts = conn.execute(
                'SELECT * FROM attempts WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            item_count = conn.execute(
                'SELECT COUNT(*) AS n FROM item_results WHERE run_id = ?', (self.run_id,)
            ).fetchone()['n']
            pending = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.PENDING.value
            )
            in_flight = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.IN_FLIGHT.value
            )
            succeeded = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.SUCCEEDED.value
            )
            retryable_failed = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.RETRYABLE_FAILED.value
            )
            terminal_failed = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.TERMINAL_FAILED.value
            )
            superseded = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.SUPERSEDED.value
            )
            outcome_unknown = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.OUTCOME_UNKNOWN.value
            )
            cancelled = sum(
                1 for r in request_rows
                if str(r['status']) == RequestStatus.CANCELLED.value
            )
            total = len(request_rows)
            expected_count = len(self._root_expected_ids_tx(conn))
            usage_rows = conn.execute(
                'SELECT * FROM usage_outbox WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            usage_pending = sum(
                1 for row in usage_rows if row['delivered_at'] is None
            )
            total_tokens = 0
            total_tokens_known = 0
            estimated_cost = 0.0
            estimated_cost_known = 0
            actual_cost = 0.0
            actual_cost_known = 0
            currencies: set[str] = set()
            for attempt in attempts:
                usage = json.loads(attempt['usage_metadata_json'] or '{}')
                reservation = json.loads(attempt['reservation_json'] or '{}')
                tokens = usage.get('total_tokens', usage.get('totalTokenCount'))
                try:
                    parsed_tokens = int(tokens)
                except (TypeError, ValueError):
                    parsed_tokens = None
                if parsed_tokens is not None and parsed_tokens >= 0:
                    total_tokens += parsed_tokens
                    total_tokens_known += 1
                estimated = self._numeric_cost(
                    usage, 'estimated_cost'
                )
                if estimated is None:
                    estimated = self._numeric_cost(
                        reservation, 'estimated_cost', 'cost_upper_bound', 'cost'
                    )
                if estimated is not None:
                    estimated_cost += estimated
                    estimated_cost_known += 1
                actual = self._numeric_cost(usage, 'actual_cost', 'cost')
                if actual is not None:
                    actual_cost += actual
                    actual_cost_known += 1
                currency = str(
                    usage.get('actual_cost_currency')
                    or usage.get('estimated_cost_currency')
                    or reservation.get('currency')
                    or ''
                ).strip()
                if currency:
                    currencies.add(currency)
            run_status = RunStatus(str(run['status']))
            if run_status is RunStatus.COMPLETED:
                next_action = 'check'
            elif run_status in (
                RunStatus.COMPLETED_WITH_ERRORS,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            ):
                next_action = 'derive'
            elif run_status is RunStatus.CANCEL_REQUESTED:
                next_action = 'wait_cancel'
            else:
                next_action = 'resume'
            return {
                'run_id': run['run_id'],
                'run_status': run['status'],
                'revision': run['revision'],
                'plan': {
                    'plan_id': run['plan_id'],
                    'plan_fingerprint': run['plan_fingerprint'],
                },
                'cancellation': {'requested': int(run['cancel_epoch']) > 0},
                'progress': {
                    'requests': {
                        'total': total,
                        'active_leaf_total': total - superseded,
                        'pending': pending,
                        'in_flight': in_flight,
                        'succeeded': succeeded,
                        'retryable_failed': retryable_failed,
                        'terminal_failed': terminal_failed,
                        'superseded': superseded,
                        'outcome_unknown': outcome_unknown,
                        'cancelled': cancelled,
                    },
                    'items': {
                        'expected': expected_count,
                        'accepted': int(item_count),
                        'unresolved': max(0, expected_count - int(item_count)),
                    },
                    'attempts': {
                        'total': len(attempts),
                        'unknown': sum(
                            1 for a in attempts
                            if str(a['status']) == AttemptStatus.OUTCOME_UNKNOWN.value
                        ),
                        'late_ignored': sum(
                            1 for a in attempts
                            if str(a['status']) in (
                                AttemptStatus.LATE_SUCCEEDED_IGNORED.value,
                                AttemptStatus.LATE_FAILED_IGNORED.value,
                            )
                        ),
                    },
                    'usage': {
                        'known_calls': len(usage_rows),
                        'billing_unknown_attempts': sum(
                            1 for a in attempts
                            if str(a['status']) == AttemptStatus.OUTCOME_UNKNOWN.value
                        ),
                        'total_tokens': total_tokens if total_tokens_known else None,
                        'estimated_cost': (
                            estimated_cost if estimated_cost_known else None
                        ),
                        'actual_cost': actual_cost if actual_cost_known else None,
                        'currency': (
                            next(iter(currencies)) if len(currencies) == 1 else None
                        ),
                        'delivery_pending': usage_pending,
                    },
                },
                'next_action': next_action,
            }

    def export_requests_jsonl(self) -> str:
        lines = [
            canonical_json(self.request_projection(row))
            for row in self.list_requests()
        ]
        return ''.join(line + '\n' for line in lines)

    def export_run_manifest_json(self) -> str:
        with self._conn() as conn:
            run = self._load_run_tx(conn, self.run_id)
        manifest = {
            'schema_version': contracts.SYNC_RUN_SCHEMA_VERSION,
            'run_id': run['run_id'],
            'run_status': run['status'],
            'revision': run['revision'],
            'plan_id': run['plan_id'],
            'plan_fingerprint': run['plan_fingerprint'],
            'source_identity_digest': run['source_identity_digest'],
            'profile_digest': run['profile_digest'],
            'config_digest': run['config_digest'],
            'policy_digest': run['policy_digest'],
            'resume_compatibility_fingerprint': run['resume_compatibility_fingerprint'],
            'derived_from_run_id': run['derived_from_run_id'],
            'derivation': json.loads(run['derivation_json'] or '{}'),
            'created_at': run['created_at'],
            'updated_at': run['updated_at'],
            'first_dispatched_at': run['first_dispatched_at'],
            'finished_at': run['finished_at'],
        }
        return canonical_json(manifest)

    def export_events_jsonl(self) -> str:
        lines = []
        with self._conn() as conn:
            for row in conn.execute(
                'SELECT * FROM events WHERE run_id = ? ORDER BY event_seq', (self.run_id,)
            ).fetchall():
                line = {
                    'event_seq': row['event_seq'],
                    'run_id': row['run_id'],
                    'entity_type': row['entity_type'],
                    'entity_id': row['entity_id'],
                    'event_type': row['event_type'],
                    'old_status': row['old_status'],
                    'new_status': row['new_status'],
                    'safe_details': json.loads(row['safe_details_json'] or '{}'),
                    'committed_at': row['committed_at'],
                }
                lines.append(canonical_json(line))
        return ''.join(line + '\n' for line in lines)

    def verify_integrity(self) -> list[str]:
        """Return integrity violations; empty means the store is internally valid."""
        violations: list[str] = []
        with self._conn() as conn:
            quick_rows = conn.execute('PRAGMA quick_check').fetchall()
            for row in quick_rows:
                if str(row[0]).lower() != 'ok':
                    violations.append(f'sqlite quick_check failed: {row[0]}')
            for row in conn.execute('PRAGMA foreign_key_check').fetchall():
                violations.append(
                    'foreign key violation: '
                    f'table={row[0]} rowid={row[1]} parent={row[2]}'
                )

            run = self._load_run_tx(conn, self.run_id)
            if sha256_hex(run['policy_json']) != run['policy_digest']:
                violations.append('runs.policy_digest mismatch')
            plan = conn.execute(
                'SELECT canonical_json, payload_sha256 FROM plans WHERE run_id = ?',
                (self.run_id,),
            ).fetchone()
            if plan is None:
                violations.append('missing plans row')
            elif sha256_hex(plan['canonical_json']) != plan['payload_sha256']:
                violations.append('plan.payload_sha256 mismatch')
            request_rows = conn.execute(
                'SELECT * FROM requests WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            requests_by_id = {
                str(request['request_id']): request for request in request_rows
            }
            root_expected_ids: list[str] = []
            for request in request_rows:
                request_id = str(request['request_id'])
                payload_json = str(request['payload_json'])
                if sha256_hex(payload_json) != request['payload_sha256']:
                    violations.append(f'request {request_id} payload_sha256 mismatch')
                    continue
                try:
                    payload = json.loads(payload_json)
                    expected_ids = json.loads(request['expected_ids_json'] or '[]')
                except (TypeError, ValueError, json.JSONDecodeError):
                    violations.append(f'request {request_id} has invalid JSON payload')
                    continue
                if list(payload.get('expected_ids') or []) != list(expected_ids):
                    violations.append(f'request {request_id} expected_ids mismatch')
                if str(payload.get('request_id') or '') != request_id:
                    violations.append(f'request {request_id} payload identity mismatch')
                if str(payload.get('plan_id') or '') != str(run['plan_id']):
                    violations.append(f'request {request_id} plan identity mismatch')

                parent_id = request['parent_request_id']
                root_id = str(request['root_request_id'])
                depth = int(request['lineage_depth'])
                lineage_kind = str(request['lineage_kind'])
                if parent_id is None:
                    root_expected_ids.extend(expected_ids)
                    if (
                        root_id != request_id
                        or depth != 0
                        or lineage_kind != contracts.LineageKind.ROOT.value
                    ):
                        violations.append(f'request {request_id} has invalid root lineage')
                else:
                    parent = requests_by_id.get(str(parent_id))
                    root = requests_by_id.get(root_id)
                    transport = dict(payload.get('transport_metadata') or {})
                    if parent is None:
                        violations.append(f'request {request_id} has missing parent')
                    elif depth != int(parent['lineage_depth']) + 1:
                        violations.append(f'request {request_id} lineage depth mismatch')
                    if (
                        root is None
                        or root['parent_request_id'] is not None
                        or str(root['root_request_id']) != root_id
                    ):
                        violations.append(f'request {request_id} has invalid root reference')
                    if str(transport.get('retry_parent_request_id') or '') != str(parent_id):
                        violations.append(f'request {request_id} retry parent mismatch')
                    if list(transport.get('retry_item_ids') or []) != list(expected_ids):
                        violations.append(f'request {request_id} retry item identity mismatch')

            if len(root_expected_ids) != len(set(root_expected_ids)):
                violations.append('root requests contain duplicate expected item IDs')
            root_expected_set = set(root_expected_ids)
            for request in request_rows:
                request_id = str(request['request_id'])
                attempt_count = conn.execute(
                    'SELECT COUNT(*) AS n FROM attempts WHERE run_id = ? AND request_id = ?',
                    (self.run_id, request_id),
                ).fetchone()['n']
                if int(request['attempt_count']) != int(attempt_count):
                    violations.append(
                        f'request {request_id} attempt_count mismatch'
                    )
                attempt_rows = conn.execute(
                    'SELECT * FROM attempts WHERE run_id = ? AND request_id = ? '
                    'ORDER BY ordinal',
                    (self.run_id, request_id),
                ).fetchall()
                expected_ordinals = list(range(1, len(attempt_rows) + 1))
                actual_ordinals = [int(row['ordinal']) for row in attempt_rows]
                if actual_ordinals != expected_ordinals:
                    violations.append(f'request {request_id} attempt ordinals are not contiguous')
                for attempt in attempt_rows:
                    expected_attempt_id = contracts.build_attempt_id(
                        self.run_id, request_id, int(attempt['ordinal'])
                    )
                    if str(attempt['attempt_id']) != expected_attempt_id:
                        violations.append(
                            f'attempt {attempt["attempt_id"]} identity mismatch'
                        )
                active = conn.execute(
                    'SELECT COUNT(*) AS n FROM attempts WHERE run_id = ? AND request_id = ? '
                    'AND status IN (?, ?, ?)',
                    (
                        self.run_id,
                        request_id,
                        AttemptStatus.PREPARED.value,
                        AttemptStatus.DISPATCHED.value,
                        AttemptStatus.CANCEL_REQUESTED.value,
                    ),
                ).fetchone()['n']
                if int(active) > 1:
                    violations.append(
                        f'request {request_id} has multiple active attempts'
                    )
                if str(request['status']) == RequestStatus.IN_FLIGHT.value and int(active) != 1:
                    violations.append(f'request {request_id} in_flight without active attempt')
                if (
                    str(request['status']) != RequestStatus.IN_FLIGHT.value
                    and int(active) != 0
                ):
                    violations.append(f'request {request_id} has stranded active attempt')

                expected_set = set(json.loads(request['expected_ids_json'] or '[]'))
                winner_rows = conn.execute(
                    'SELECT item_id FROM item_results WHERE run_id = ? '
                    'AND winner_request_id = ?',
                    (self.run_id, request_id),
                ).fetchall()
                winner_set = {str(row['item_id']) for row in winner_rows}
                if not winner_set <= expected_set:
                    violations.append(f'request {request_id} has winner outside expected IDs')
                if str(request['status']) in (
                    RequestStatus.SUCCEEDED.value,
                    RequestStatus.SUPERSEDED.value,
                ):
                    if (
                        str(request['status']) == RequestStatus.SUCCEEDED.value
                        and winner_set != expected_set
                    ):
                        violations.append(
                            f'request {request_id} succeeded without exact winners'
                        )
                    if str(request['status']) == RequestStatus.SUPERSEDED.value:
                        child_rows = conn.execute(
                            'SELECT expected_ids_json FROM requests '
                            'WHERE run_id = ? AND parent_request_id = ? ORDER BY rowid',
                            (self.run_id, request_id),
                        ).fetchall()
                        child_ids = [
                            item_id
                            for child in child_rows
                            for item_id in json.loads(child['expected_ids_json'] or '[]')
                        ]
                        remaining = [
                            item_id
                            for item_id in json.loads(request['expected_ids_json'] or '[]')
                            if item_id not in winner_set
                        ]
                        if child_ids != remaining or len(child_ids) != len(set(child_ids)):
                            violations.append(
                                f'request {request_id} children do not exactly cover remaining IDs'
                            )

            item_rows = conn.execute(
                'SELECT * FROM item_results WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            accepted_ids = {str(row['item_id']) for row in item_rows}
            if not accepted_ids <= root_expected_set:
                violations.append('item results contain IDs outside root request universe')
            for item in item_rows:
                attempt_id = item['winner_attempt_id']
                source_run_id = item['reused_from_run_id']
                if bool(attempt_id) == bool(source_run_id):
                    violations.append(
                        f'item result {item["item_id"]} has invalid winner provenance'
                    )
                winner_request = requests_by_id.get(str(item['winner_request_id']))
                if winner_request is None or str(item['item_id']) not in set(
                    json.loads(winner_request['expected_ids_json'] or '[]')
                ):
                    violations.append(
                        f'item result {item["item_id"]} has invalid winner request'
                    )

            event_rows = conn.execute(
                'SELECT event_seq FROM events WHERE run_id = ? ORDER BY event_seq',
                (self.run_id,),
            ).fetchall()
            if event_rows:
                event_seqs = [int(row['event_seq']) for row in event_rows]
                if event_seqs != list(range(event_seqs[0], event_seqs[-1] + 1)):
                    violations.append('event sequence contains a gap')

            if str(run['status']) in (
                RunStatus.COMPLETED.value,
                RunStatus.COMPLETED_WITH_ERRORS.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            ):
                for request in request_rows:
                    if str(request['status']) in (
                        RequestStatus.PENDING.value,
                        RequestStatus.IN_FLIGHT.value,
                        RequestStatus.RETRYABLE_FAILED.value,
                    ):
                        violations.append(
                            f'terminal run still has active leaf {request["request_id"]}'
                        )
                active_attempts = conn.execute(
                    'SELECT COUNT(*) AS n FROM attempts WHERE run_id = ? '
                    'AND status IN (?, ?, ?)',
                    (
                        self.run_id,
                        AttemptStatus.PREPARED.value,
                        AttemptStatus.DISPATCHED.value,
                        AttemptStatus.CANCEL_REQUESTED.value,
                    ),
                ).fetchone()['n']
                if int(active_attempts):
                    violations.append('terminal run still has active attempt')
                if str(run['status']) == RunStatus.COMPLETED.value:
                    if accepted_ids != root_expected_set:
                        violations.append('completed run does not exactly cover root item universe')

            artifact_rows = conn.execute(
                'SELECT * FROM artifacts WHERE run_id = ?', (self.run_id,)
            ).fetchall()
            run_root = self.run_dir.resolve(strict=False)
            for artifact in artifact_rows:
                relative = Path(str(artifact['relative_path']))
                candidate = (self.run_dir / relative).resolve(strict=False)
                try:
                    candidate.relative_to(run_root)
                except ValueError:
                    violations.append(
                        f'artifact {artifact["kind"]} escapes the run directory'
                    )
                    continue
                if not candidate.is_file():
                    violations.append(f'artifact {artifact["kind"]} is missing')
                elif file_sha256(candidate) != str(artifact['sha256']):
                    violations.append(f'artifact {artifact["kind"]} sha256 mismatch')
        return violations

    # ------------------------------------------------------------------
    # Artifacts and checkpoint
    # ------------------------------------------------------------------
    def resolve_artifact_path(self, relative_path: str | Path) -> Path:
        """Resolve one artifact path while enforcing run-directory containment."""
        relative = Path(str(relative_path))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('artifact relative_path must stay inside the run directory')
        root = self.run_dir.resolve(strict=False)
        candidate = (self.run_dir / relative).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                'artifact relative_path must stay inside the run directory'
            ) from exc
        return candidate

    def put_artifact(
        self,
        *,
        kind: str,
        relative_path: str,
        sha256_digest: str,
        schema_version: int,
    ) -> None:
        self.resolve_artifact_path(relative_path)
        with self._tx() as conn:
            self._load_run_tx(conn, self.run_id)
            conn.execute(
                'INSERT INTO artifacts('
                ' run_id, kind, relative_path, sha256, schema_version, created_at'
                ') VALUES (?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(run_id, kind) DO UPDATE SET '
                'relative_path = excluded.relative_path, sha256 = excluded.sha256, '
                'schema_version = excluded.schema_version, created_at = excluded.created_at',
                (
                    self.run_id,
                    str(kind),
                    str(relative_path),
                    str(sha256_digest),
                    int(schema_version),
                    _now_iso(),
                ),
            )

    def get_artifact(self, *, kind: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                'SELECT * FROM artifacts WHERE run_id = ? AND kind = ?',
                (self.run_id, str(kind)),
            ).fetchone()
            return _row_dict(row)

    def delete_artifact(self, *, kind: str) -> bool:
        """Remove a derived artifact binding while leaving its audit file intact."""
        with self._tx() as conn:
            self._load_run_tx(conn, self.run_id)
            cursor = conn.execute(
                'DELETE FROM artifacts WHERE run_id = ? AND kind = ?',
                (self.run_id, str(kind)),
            )
            return cursor.rowcount > 0

    def checkpoint(self) -> dict:
        """Run a truncating WAL checkpoint and return the PRAGMA result row."""
        with self._conn() as conn:
            row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            return dict(row)
