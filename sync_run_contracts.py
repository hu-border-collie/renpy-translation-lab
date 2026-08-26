# -*- coding: utf-8 -*-
"""Pure contracts for the durable sync executor (issue #347, P1).

This module owns the stable enums, legal state transitions, error codes and
ID/digest helpers used by the SQLite store.  It intentionally has no I/O or
scheduler logic so tests can pin every transition without touching a file
system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
import uuid
from typing import Mapping


#: Schema version managed by :mod:`sync_run_store`.  Only forward migrations
#: are supported; a newer on-disk version is rejected with
#: ``SYNC_RUN_SCHEMA_UNSUPPORTED`` rather than opened read-write.
SYNC_RUN_SCHEMA_VERSION = 1

#: Prefix used by all durable sync run directories and run IDs.
SYNC_RUN_DIR_PREFIX = 'sync-run-v1-'

RUN_ID_TOKEN_PREFIX = 'sync-run-v1-token-'


class ErrorCode(str, Enum):
    """Stable error codes from #347 section 11.2."""

    SYNC_RUN_NOT_FOUND = 'SYNC_RUN_NOT_FOUND'
    SYNC_RUN_BUSY = 'SYNC_RUN_BUSY'
    SYNC_RUN_FRESHNESS_MISMATCH = 'SYNC_RUN_FRESHNESS_MISMATCH'
    SYNC_RUN_CLIENT_TOKEN_CONFLICT = 'SYNC_RUN_CLIENT_TOKEN_CONFLICT'
    SYNC_RUN_STORAGE_ERROR = 'SYNC_RUN_STORAGE_ERROR'
    SYNC_RUN_SCHEMA_UNSUPPORTED = 'SYNC_RUN_SCHEMA_UNSUPPORTED'
    SYNC_RUN_BUDGET_EXHAUSTED = 'SYNC_RUN_BUDGET_EXHAUSTED'
    SYNC_RUN_OUTCOME_UNKNOWN = 'SYNC_RUN_OUTCOME_UNKNOWN'


class SyncRunError(RuntimeError):
    """Raised for expected durable-sync contract failures.

    ``code`` is one of :class:`ErrorCode`.  ``retryable`` mirrors the CLI
    contract from #347 section 11.2; it is false except for ``SYNC_RUN_BUSY``.
    ``safe_details`` must already be denormalized / credential-free.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        retryable: bool | None = None,
        safe_details: Mapping | None = None,
    ):
        code = ErrorCode(code)
        super().__init__(message)
        self.code = code
        self.retryable = (code is ErrorCode.SYNC_RUN_BUSY) if retryable is None else bool(retryable)
        self.safe_details = dict(safe_details or {})

    def to_dict(self) -> dict:
        return {
            'code': self.code.value,
            'message': str(self),
            'retryable': self.retryable,
            'details': self.safe_details,
        }


class RunStatus(str, Enum):
    """Run lifecycle from #347 section 4.1."""

    PLANNED = 'planned'
    RUNNING = 'running'
    CANCEL_REQUESTED = 'cancel_requested'
    CANCELLED = 'cancelled'
    COMPLETED = 'completed'
    COMPLETED_WITH_ERRORS = 'completed_with_errors'
    FAILED = 'failed'


class RequestStatus(str, Enum):
    """Request lifecycle from #347 section 4.2."""

    PENDING = 'pending'
    IN_FLIGHT = 'in_flight'
    RETRYABLE_FAILED = 'retryable_failed'
    SUCCEEDED = 'succeeded'
    TERMINAL_FAILED = 'terminal_failed'
    SUPERSEDED = 'superseded'
    CANCELLED = 'cancelled'
    OUTCOME_UNKNOWN = 'outcome_unknown'


class AttemptStatus(str, Enum):
    """Attempt lifecycle from #347 section 4.3."""

    PREPARED = 'prepared'
    DISPATCHED = 'dispatched'
    CANCEL_REQUESTED = 'cancel_requested'
    SUCCEEDED = 'succeeded'
    RETRYABLE_FAILED = 'retryable_failed'
    TERMINAL_FAILED = 'terminal_failed'
    CANCELLED = 'cancelled'
    OUTCOME_UNKNOWN = 'outcome_unknown'
    LATE_SUCCEEDED_IGNORED = 'late_succeeded_ignored'
    LATE_FAILED_IGNORED = 'late_failed_ignored'


class LineageKind(str, Enum):
    ROOT = 'root'
    MISSING_IDS = 'missing_ids'  # stable child id suffix ``--M-``
    SPLIT_LEFT = 'split_left'    # stable child id suffix ``--L``
    SPLIT_RIGHT = 'split_right'  # stable child id suffix ``--R``


class ErrorCategory(str, Enum):
    """v1 error classification from #347 section 8.1."""

    AUTHENTICATION = 'authentication'
    CONFIGURATION = 'configuration'
    MISSING_DEPENDENCY = 'missing_dependency'
    UNSUPPORTED_CAPABILITY = 'unsupported_capability'
    UNSUPPORTED_REQUEST = 'unsupported_request'
    RATE_LIMIT = 'rate_limit'
    QUOTA_EXHAUSTED = 'quota_exhausted'
    TIMEOUT = 'timeout'
    TRANSPORT = 'transport'
    PROVIDER_SERVER = 'provider_server'
    INVALID_STRUCTURED_RESPONSE = 'invalid_structured_response'
    INCOMPLETE_IDS = 'incomplete_ids'
    CONTENT_POLICY = 'content_policy'
    LOCAL_VALIDATION = 'local_validation'
    LOCAL_PERSISTENCE = 'local_persistence'
    LOCAL_ARTIFACT_WRITE = 'local_artifact_write'
    CANCELLED = 'cancelled'
    UNKNOWN_PROVIDER = 'unknown_provider'


class EventType(str, Enum):
    RUN_CREATED = 'run_created'
    RUN_STATUS = 'run_status'
    REQUEST_STATUS = 'request_status'
    ATTEMPT_PREPARED = 'attempt_prepared'
    ATTEMPT_DISPATCHED = 'attempt_dispatched'
    ATTEMPT_SUCCEEDED = 'attempt_succeeded'
    ATTEMPT_FAILED = 'attempt_failed'
    ATTEMPT_UNKNOWN = 'attempt_unknown'
    ATTEMPT_CANCELLED = 'attempt_cancelled'
    ATTEMPT_LATE_IGNORED = 'attempt_late_ignored'
    CANCEL_INTENT = 'cancel_intent'
    LEASE = 'lease'
    OUTBOX = 'outbox'
    NOTICE = 'notice'


# Frozen transition tables from #347 sections 4.1-4.3.  Services, schedulers,
# CLI and GUI must route through these tables instead of inferring transitions.
RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PLANNED: frozenset({
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.FAILED,
    }),
    RunStatus.RUNNING: frozenset({
        RunStatus.CANCEL_REQUESTED,
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_ERRORS,
        RunStatus.FAILED,
    }),
    RunStatus.CANCEL_REQUESTED: frozenset({RunStatus.CANCELLED}),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.COMPLETED_WITH_ERRORS: frozenset(),
    RunStatus.FAILED: frozenset(),
}

REQUEST_TRANSITIONS: dict[RequestStatus, frozenset[RequestStatus]] = {
    RequestStatus.PENDING: frozenset({
        RequestStatus.IN_FLIGHT,
        RequestStatus.CANCELLED,
    }),
    RequestStatus.IN_FLIGHT: frozenset({
        RequestStatus.SUCCEEDED,
        RequestStatus.RETRYABLE_FAILED,
        RequestStatus.TERMINAL_FAILED,
        RequestStatus.SUPERSEDED,
        RequestStatus.CANCELLED,
        RequestStatus.OUTCOME_UNKNOWN,
    }),
    RequestStatus.RETRYABLE_FAILED: frozenset({
        RequestStatus.IN_FLIGHT,
        RequestStatus.SUPERSEDED,
        RequestStatus.TERMINAL_FAILED,
        RequestStatus.CANCELLED,
    }),
    RequestStatus.SUCCEEDED: frozenset(),
    RequestStatus.TERMINAL_FAILED: frozenset(),
    RequestStatus.SUPERSEDED: frozenset(),
    RequestStatus.CANCELLED: frozenset(),
    RequestStatus.OUTCOME_UNKNOWN: frozenset(),
}

ATTEMPT_TRANSITIONS: dict[AttemptStatus, frozenset[AttemptStatus]] = {
    AttemptStatus.PREPARED: frozenset({
        AttemptStatus.DISPATCHED,
        AttemptStatus.CANCELLED,
        AttemptStatus.TERMINAL_FAILED,
    }),
    AttemptStatus.DISPATCHED: frozenset({
        AttemptStatus.SUCCEEDED,
        AttemptStatus.RETRYABLE_FAILED,
        AttemptStatus.TERMINAL_FAILED,
        AttemptStatus.CANCEL_REQUESTED,
        AttemptStatus.OUTCOME_UNKNOWN,
        AttemptStatus.LATE_SUCCEEDED_IGNORED,
        AttemptStatus.LATE_FAILED_IGNORED,
    }),
    AttemptStatus.CANCEL_REQUESTED: frozenset({
        AttemptStatus.CANCELLED,
        AttemptStatus.LATE_SUCCEEDED_IGNORED,
        AttemptStatus.LATE_FAILED_IGNORED,
        AttemptStatus.OUTCOME_UNKNOWN,
    }),
    AttemptStatus.SUCCEEDED: frozenset(),
    AttemptStatus.RETRYABLE_FAILED: frozenset(),
    AttemptStatus.TERMINAL_FAILED: frozenset(),
    AttemptStatus.CANCELLED: frozenset(),
    AttemptStatus.OUTCOME_UNKNOWN: frozenset(),
    AttemptStatus.LATE_SUCCEEDED_IGNORED: frozenset(),
    AttemptStatus.LATE_FAILED_IGNORED: frozenset(),
}

RUN_TERMINAL_STATES = frozenset({
    RunStatus.CANCELLED,
    RunStatus.COMPLETED,
    RunStatus.COMPLETED_WITH_ERRORS,
    RunStatus.FAILED,
})

#: Request statuses that still represent active delivery scope in scheduler
#: terms.  ``superseded`` is terminal history, not an active leaf.
REQUEST_ACTIVE_STATES = frozenset({
    RequestStatus.PENDING,
    RequestStatus.IN_FLIGHT,
    RequestStatus.RETRYABLE_FAILED,
})

#: Request/attempt statuses that must be zero before a run may be finalized.
REQUEST_NON_TERMINAL_STATES = frozenset({
    RequestStatus.PENDING,
    RequestStatus.IN_FLIGHT,
    RequestStatus.RETRYABLE_FAILED,
})

ATTEMPT_ACTIVE_STATES = frozenset({
    AttemptStatus.PREPARED,
    AttemptStatus.DISPATCHED,
    AttemptStatus.CANCEL_REQUESTED,
})

ATTEMPT_TERMINAL_STATES = frozenset({
    AttemptStatus.SUCCEEDED,
    AttemptStatus.RETRYABLE_FAILED,
    AttemptStatus.TERMINAL_FAILED,
    AttemptStatus.CANCELLED,
    AttemptStatus.OUTCOME_UNKNOWN,
    AttemptStatus.LATE_SUCCEEDED_IGNORED,
    AttemptStatus.LATE_FAILED_IGNORED,
})

SUPERSEDED_ATTEMPT_STATUSES = frozenset({
    AttemptStatus.SUCCEEDED,
    AttemptStatus.CANCELLED,
    AttemptStatus.OUTCOME_UNKNOWN,
    AttemptStatus.LATE_SUCCEEDED_IGNORED,
})

#: Error classifications that never retry a model call.
TERMINAL_ERROR_CATEGORIES = frozenset({
    ErrorCategory.AUTHENTICATION,
    ErrorCategory.CONFIGURATION,
    ErrorCategory.MISSING_DEPENDENCY,
    ErrorCategory.UNSUPPORTED_CAPABILITY,
    ErrorCategory.UNSUPPORTED_REQUEST,
    ErrorCategory.QUOTA_EXHAUSTED,
    ErrorCategory.CONTENT_POLICY,
    ErrorCategory.LOCAL_PERSISTENCE,
    ErrorCategory.LOCAL_ARTIFACT_WRITE,
    ErrorCategory.CANCELLED,
    ErrorCategory.UNKNOWN_PROVIDER,
})

#: Error classifications that can produce derived children instead of a plain
#: retry (see #347 section 8.1).  Split policy remains owned by #346 helpers.
DERIVED_ERROR_CATEGORIES = frozenset({
    ErrorCategory.INVALID_STRUCTURED_RESPONSE,
    ErrorCategory.INCOMPLETE_IDS,
    ErrorCategory.LOCAL_VALIDATION,
    ErrorCategory.CONTENT_POLICY,
})

#: Attempts that may safely continue/resume without a model call when they
#: are found in a reopened database.
RESUMABLE_ATTEMPT_STATUSES = frozenset({AttemptStatus.PREPARED})
SAFE_ERROR_REASON_EVENT = 'safe_details_json'

# Reason codes.
REASON_RUN_BUDGET_EXHAUSTED_COST = 'run_budget_exhausted.cost'
REASON_RUN_BUDGET_EXHAUSTED_TIME = 'run_budget_exhausted.time'
REASON_RUN_POLICY_ATTEMPTS = 'run_policy.max_attempts'
REASON_REQUEST_ATTEMPTS_EXHAUSTED = 'request.max_attempts_exhausted'


def utcnow_iso() -> str:
    """Return a timezone-aware UTC timestamp with ``Z`` suffix (ISO 8601)."""
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def utcnow_run_id_timestamp() -> str:
    """Return the path-safe UTC timestamp used inside a run ID."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def canonical_json(value) -> str:
    """Return deterministic, NaN-free UTF-8 JSON text.

    The implementation matches :func:`translation_plan.canonical_json` so a
    plan serialized by #346 and the copy stored by this module hash identically.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


def sha256_hex(text) -> str:
    """Hex SHA-256 of ``text`` (used for content-derived identities)."""
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def payload_sha256(value) -> str:
    """Hex SHA-256 of the canonical serialization of ``value``."""
    return sha256_hex(canonical_json(value))


def normalize_client_token(token: str | None) -> str | None:
    """Normalize the #347 client token contract.

    Missing, blank and whitespace-only tokens mean "create a new run" and are
    represented as SQL ``NULL``.  Original token text must never be persisted.
    """
    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    return text


def client_token_digest(token: str | None) -> str | None:
    """Return a one-way digest for a non-empty client token, else ``None``.

    The digest is stored in ``runs.client_token_digest`` and participates in
    deterministic run IDs.  The token itself is never persisted.
    """
    normalized = normalize_client_token(token)
    if normalized is None:
        return None
    return sha256_hex('sync-run-client-token\0' + normalized)


def build_run_id(client_token: str | None = None, *, now: str | None = None) -> str:
    """Build a stable, path-safe run ID per #347 section 3.1.

    Non-empty tokens produce ``sync-run-v1-token-<digest>``; empty/omitted
    tokens produce ``sync-run-v1-<UTC timestamp>-<uuid4hex>``.
    """
    digest = client_token_digest(client_token)
    if digest is not None:
        return f'{RUN_ID_TOKEN_PREFIX}{digest}'
    timestamp = now or utcnow_run_id_timestamp()
    return f'{SYNC_RUN_DIR_PREFIX}{timestamp}-{uuid.uuid4().hex}'


def build_attempt_id(run_id: str, request_id: str, ordinal: int) -> str:
    """Build a deterministic attempt ID per #347 section 3.1."""
    ordinal = int(ordinal)
    if ordinal < 0:
        raise ValueError('attempt ordinal must be non-negative')
    seed = canonical_json([str(run_id), str(request_id), ordinal])
    return sha256_hex(seed)[:24]


def build_usage_event_id(attempt_id: str) -> str:
    """Build the usage outbox primary key / external dedupe key seed."""
    return f'usage:{attempt_id}'


def derive_unknown_ids(accepted_ids: set[str] | list[str] | tuple[str], expected_ids) -> list[str]:
    """Return expected IDs not present in accepted IDs, preserving plan order."""
    accepted = set(accepted_ids or ())
    expected = list(expected_ids or ())
    return [item_id for item_id in expected if item_id not in accepted]


def validate_run_id(run_id: str) -> bool:
    """Return True when ``run_id`` is a path-safe durable-sync run id."""
    if not isinstance(run_id, str):
        return False
    if not run_id.startswith(SYNC_RUN_DIR_PREFIX):
        return False
    if '/' in run_id or '\\' in run_id:
        return False
    if run_id in ('.', '..'):
        return False
    return True


_RUN_ID_LOOSE_RE = re.compile(r'^sync-run-v1-[A-Za-z0-9_.-]+$')


def assert_valid_run_id(run_id: str) -> None:
    if not validate_run_id(run_id) or not _RUN_ID_LOOSE_RE.match(run_id):
        raise ValueError(f'unsafe or malformed run id: {run_id!r}')


def can_transition(status: Enum, next_status: Enum, table: dict[Enum, frozenset[Enum]]) -> bool:
    return next_status in table.get(status, frozenset())


def ensure_run_transition(current: RunStatus, next_: RunStatus) -> None:
    if not can_transition(current, next_, RUN_TRANSITIONS):
        raise ValueError(
            f'illegal run transition: {current.value} -> {next_.value}'
        )


def ensure_request_transition(current: RequestStatus, next_: RequestStatus) -> None:
    if not can_transition(current, next_, REQUEST_TRANSITIONS):
        raise ValueError(
            f'illegal request transition: {current.value} -> {next_.value}'
        )


def ensure_attempt_transition(current: AttemptStatus, next_: AttemptStatus) -> None:
    if not can_transition(current, next_, ATTEMPT_TRANSITIONS):
        raise ValueError(
            f'illegal attempt transition: {current.value} -> {next_.value}'
        )
