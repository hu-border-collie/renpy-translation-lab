# -*- coding: utf-8 -*-
"""Frozen retry and budget policy for the durable Sync executor (#347 P2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import random
from typing import Any, Mapping

from sync_run_contracts import (
    ErrorCategory,
    RetryDecision,
    canonical_json,
    retry_decision_for,
)


DEFAULT_MAX_ATTEMPTS_PER_REQUEST = 3
DEFAULT_MAX_ATTEMPTS_PER_ROOT = 8
DEFAULT_MAX_LINEAGE_DEPTH = 5
DEFAULT_MAX_DERIVED_REQUESTS_PER_ROOT = 15
DEFAULT_MAX_TOTAL_ATTEMPTS_PER_RUN = 100
DEFAULT_MAX_ELAPSED_SECONDS = 3600.0
DEFAULT_MAX_UNKNOWN_BILLING_ATTEMPTS = 0
DEFAULT_MAX_IN_FLIGHT = 1
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 120.0
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_CAP_SECONDS = 60.0


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return parsed


def _positive_float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    parsed = float(value)
    if parsed <= minimum:
        raise ValueError(f'{name} must be > {minimum}')
    return parsed


def _optional_cost(value: Any, name: str) -> float | None:
    if value is None or value == '':
        return None
    return _positive_float(value, name)


@dataclass(frozen=True)
class ExecutorPolicy:
    """Validated policy frozen into a run before its first dispatch."""

    max_attempts_per_request: int = DEFAULT_MAX_ATTEMPTS_PER_REQUEST
    max_attempts_per_root: int = DEFAULT_MAX_ATTEMPTS_PER_ROOT
    max_lineage_depth: int = DEFAULT_MAX_LINEAGE_DEPTH
    max_derived_requests_per_root: int = DEFAULT_MAX_DERIVED_REQUESTS_PER_ROOT
    max_total_attempts_per_run: int = DEFAULT_MAX_TOTAL_ATTEMPTS_PER_RUN
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS
    max_estimated_cost: float | None = None
    max_actual_cost: float | None = None
    max_unknown_billing_attempts: int = DEFAULT_MAX_UNKNOWN_BILLING_ATTEMPTS
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    attempt_timeout_seconds: float = DEFAULT_ATTEMPT_TIMEOUT_SECONDS
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> 'ExecutorPolicy':
        raw = dict(value or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f'unknown durable executor policy fields: {unknown}')
        policy = cls(
            max_attempts_per_request=_positive_int(
                raw.get('max_attempts_per_request', DEFAULT_MAX_ATTEMPTS_PER_REQUEST),
                'max_attempts_per_request',
            ),
            max_attempts_per_root=_positive_int(
                raw.get('max_attempts_per_root', DEFAULT_MAX_ATTEMPTS_PER_ROOT),
                'max_attempts_per_root',
            ),
            max_lineage_depth=_positive_int(
                raw.get('max_lineage_depth', DEFAULT_MAX_LINEAGE_DEPTH),
                'max_lineage_depth',
            ),
            max_derived_requests_per_root=_positive_int(
                raw.get(
                    'max_derived_requests_per_root',
                    DEFAULT_MAX_DERIVED_REQUESTS_PER_ROOT,
                ),
                'max_derived_requests_per_root',
            ),
            max_total_attempts_per_run=_positive_int(
                raw.get(
                    'max_total_attempts_per_run',
                    DEFAULT_MAX_TOTAL_ATTEMPTS_PER_RUN,
                ),
                'max_total_attempts_per_run',
            ),
            max_elapsed_seconds=_positive_float(
                raw.get('max_elapsed_seconds', DEFAULT_MAX_ELAPSED_SECONDS),
                'max_elapsed_seconds',
            ),
            max_estimated_cost=_optional_cost(
                raw.get('max_estimated_cost'), 'max_estimated_cost'
            ),
            max_actual_cost=_optional_cost(
                raw.get('max_actual_cost'), 'max_actual_cost'
            ),
            max_unknown_billing_attempts=_positive_int(
                raw.get(
                    'max_unknown_billing_attempts',
                    DEFAULT_MAX_UNKNOWN_BILLING_ATTEMPTS,
                ),
                'max_unknown_billing_attempts',
                minimum=0,
            ),
            max_in_flight=_positive_int(
                raw.get('max_in_flight', DEFAULT_MAX_IN_FLIGHT),
                'max_in_flight',
            ),
            attempt_timeout_seconds=_positive_float(
                raw.get('attempt_timeout_seconds', DEFAULT_ATTEMPT_TIMEOUT_SECONDS),
                'attempt_timeout_seconds',
            ),
            backoff_base_seconds=_positive_float(
                raw.get('backoff_base_seconds', DEFAULT_BACKOFF_BASE_SECONDS),
                'backoff_base_seconds',
            ),
            backoff_cap_seconds=_positive_float(
                raw.get('backoff_cap_seconds', DEFAULT_BACKOFF_CAP_SECONDS),
                'backoff_cap_seconds',
            ),
        )
        if policy.max_attempts_per_root < policy.max_attempts_per_request:
            raise ValueError(
                'max_attempts_per_root must be >= max_attempts_per_request'
            )
        if policy.backoff_cap_seconds < policy.backoff_base_seconds:
            raise ValueError('backoff_cap_seconds must be >= backoff_base_seconds')
        return policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureDisposition:
    decision: RetryDecision
    terminal: bool
    may_split: bool = False


def classify_failure(
    category: ErrorCategory | str,
    *,
    retry_after_seconds: float | None = None,
    isolatable: bool = False,
    repairable: bool = False,
) -> FailureDisposition:
    """Apply the frozen §8.1 table, including its two explicit exceptions."""
    parsed = category if isinstance(category, ErrorCategory) else ErrorCategory(str(category))
    if parsed is ErrorCategory.QUOTA_EXHAUSTED and retry_after_seconds is not None:
        return FailureDisposition(RetryDecision.RETRYABLE, terminal=False)
    if parsed is ErrorCategory.CONTENT_POLICY and isolatable:
        return FailureDisposition(RetryDecision.DERIVED, terminal=False, may_split=True)
    if parsed is ErrorCategory.LOCAL_VALIDATION:
        return FailureDisposition(
            RetryDecision.DERIVED if repairable else RetryDecision.TERMINAL,
            terminal=not repairable,
            may_split=False,
        )
    decision = retry_decision_for(parsed)
    return FailureDisposition(
        decision=decision,
        terminal=decision in {
            RetryDecision.TERMINAL,
            RetryDecision.RUN_LEVEL_STOP,
        },
        may_split=decision is RetryDecision.DERIVED,
    )


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec='microseconds'
    ).replace('+00:00', 'Z')


def compute_next_eligible_at(
    *,
    failure_time: str,
    same_request_retry_index: int,
    policy: ExecutorPolicy,
    retry_after_seconds: float | None = None,
    rng: random.Random | None = None,
) -> str:
    """Compute one persisted exponential full-jitter deadline."""
    retry_index = _positive_int(
        same_request_retry_index, 'same_request_retry_index'
    )
    cap = policy.backoff_cap_seconds
    if retry_after_seconds is not None:
        delay = min(cap, max(0.0, float(retry_after_seconds)))
    else:
        ceiling = min(
            cap,
            policy.backoff_base_seconds * (2 ** (retry_index - 1)),
        )
        delay = (rng or random.SystemRandom()).uniform(0.0, ceiling)
    return format_utc(parse_utc(failure_time) + timedelta(seconds=delay))


def missing_lineage_suffix(item_ids) -> str:
    """Return the frozen targeted-retry suffix without exposing item text."""
    digest = hashlib.sha256(canonical_json(list(item_ids or ())).encode('utf-8')).hexdigest()
    return f'--M-{digest[:12]}'
