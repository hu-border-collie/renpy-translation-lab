# -*- coding: utf-8 -*-
"""Lease-owned fake/production-neutral scheduler for durable Sync (#347 P2).

The scheduler is the only request/attempt writer.  Backends receive immutable
stored request payloads and return facts; they never mutate the run store.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import json
import os
import secrets
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import sync_run_contracts as contracts
from sync_run_contracts import (
    AttemptStatus,
    ErrorCategory,
    ErrorCode,
    RequestStatus,
    RetryDecision,
    RunStatus,
    SyncRunError,
    utcnow_iso,
)
from sync_retry_policy import (
    ExecutorPolicy,
    classify_failure,
    compute_next_eligible_at,
    missing_lineage_suffix,
    parse_utc,
)
from sync_run_store import DEFAULT_LEASE_TTL_SECONDS, SyncRunStore


class SyncProviderBackend(Protocol):
    """Provider-neutral model call seam consumed by the durable scheduler."""

    def send(
        self,
        request: Mapping[str, Any],
        *,
        attempt: Mapping[str, Any],
        timeout_seconds: float,
    ) -> 'ProviderOutcome': ...

    def cancel(self, *, attempt: Mapping[str, Any]) -> bool: ...


@dataclass(frozen=True)
class ProviderOutcome:
    """Successful Provider receipt after adapter/local contract validation."""

    accepted_items: Mapping[str, Any] | Sequence[str]
    response_payload: Any = None
    normalized_payload: Any = None
    contract_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    usage_metadata: Mapping[str, Any] = field(default_factory=dict)


class ProviderFailure(RuntimeError):
    """Stable, already-redacted Provider or adapter failure."""

    def __init__(
        self,
        category: ErrorCategory | str,
        reason_code: str,
        *,
        safe_details: Mapping[str, Any] | None = None,
        retry_after_seconds: float | None = None,
        usage_metadata: Mapping[str, Any] | None = None,
        isolatable: bool = False,
        repairable: bool = False,
    ):
        self.category = (
            category
            if isinstance(category, ErrorCategory)
            else ErrorCategory(str(category))
        )
        self.reason_code = str(reason_code or self.category.value)
        self.safe_details = dict(safe_details or {})
        self.retry_after_seconds = (
            None if retry_after_seconds is None else float(retry_after_seconds)
        )
        self.usage_metadata = dict(usage_metadata or {})
        self.isolatable = bool(isolatable)
        self.repairable = bool(repairable)
        super().__init__(self.reason_code)


DerivedRequestBuilder = Callable[
    [Mapping[str, Any], Sequence[str], str, contracts.LineageKind],
    Mapping[str, Any],
]
ReservationProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]
FreshnessCheck = Callable[[SyncRunStore], bool | None]


class TranslationPlanDerivedRequestBuilder:
    """Adapter that delegates all child request rendering to #346's helper."""

    def __init__(
        self,
        item_resolver: Callable[[Mapping[str, Any], Sequence[str]], Sequence[Mapping]],
        *,
        context_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ):
        self.item_resolver = item_resolver
        self.context_resolver = context_resolver or (lambda _request: {})

    def __call__(
        self,
        parent_payload: Mapping[str, Any],
        target_ids: Sequence[str],
        suffix: str,
        lineage_kind: contracts.LineageKind,
    ) -> Mapping[str, Any]:
        import translation_plan

        parent = translation_plan.TranslationRequest.from_dict(dict(parent_payload))
        target_items = list(self.item_resolver(parent_payload, list(target_ids)))
        actual_ids = [str(item.get('id') or '') for item in target_items]
        if actual_ids != list(target_ids):
            raise ValueError('derived item resolver changed target ID order or coverage')
        kwargs = dict(self.context_resolver(parent_payload) or {})
        child = translation_plan.derive_translation_request(
            parent,
            target_items,
            lineage_suffix=suffix,
            lineage_kind=lineage_kind.value,
            **kwargs,
        )
        return child.to_dict()


def _accepted_ids(value: Mapping[str, Any] | Sequence[str]) -> list[str]:
    if isinstance(value, Mapping):
        return [str(item_id) for item_id in value]
    return [str(item_id) for item_id in value]


class DurableSyncExecutor:
    """Run one durable queue until terminal, cancelled, stopped, or backoff."""

    def __init__(
        self,
        store: SyncRunStore,
        backend: SyncProviderBackend,
        *,
        derived_request_builder: DerivedRequestBuilder,
        policy: ExecutorPolicy | Mapping[str, Any] | None = None,
        reservation_provider: ReservationProvider | None = None,
        freshness_check: FreshnessCheck | None = None,
        owner_token: str | None = None,
        lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        provider: str = '',
        model: str = '',
        profile_digest: str = '',
        credential_identity: str = '',
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.store = store
        self.backend = backend
        self.derived_request_builder = derived_request_builder
        stored_policy = ExecutorPolicy.from_mapping(
            json.loads(store.get_run()['policy_json'] or '{}')
        )
        requested_policy = (
            stored_policy
            if policy is None
            else (
                policy
                if isinstance(policy, ExecutorPolicy)
                else ExecutorPolicy.from_mapping(policy)
            )
        )
        if requested_policy.to_dict() != stored_policy.to_dict():
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                'executor policy differs from the policy frozen at T0',
                safe_details={'run_id': store.run_id},
            )
        self.policy = stored_policy
        self.reservation_provider = reservation_provider or (lambda _request: {})
        self.freshness_check = freshness_check or (lambda _store: True)
        self.owner_token = str(owner_token or f'worker-{secrets.token_hex(16)}')
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self.provider = str(provider)
        self.model = str(model)
        self.profile_digest = str(profile_digest)
        self.credential_identity = str(credential_identity)
        self.sleep = sleep

    def run(self, *, wait_for_backoff: bool = False) -> dict[str, Any]:
        """Resume safely and return a store-backed snapshot.

        ``wait_for_backoff=False`` returns with ``run_status=running`` once
        only future retry deadlines remain.  Production foreground service
        uses ``True``; tests and polling callers can resume explicitly.
        """
        self.store.acquire_lease(
            owner_token=self.owner_token,
            pid=os.getpid(),
            ttl_seconds=self.lease_ttl_seconds,
        )
        try:
            self._verify_before_dispatch()
            recovered = deque(self._recover_active_attempts())
            futures: dict[Future, Mapping[str, Any]] = {}
            cancel_attempted_ids: set[str] = set()
            worker_error: BaseException | None = None
            with ThreadPoolExecutor(
                max_workers=self.policy.max_in_flight,
                thread_name_prefix='durable-sync',
            ) as pool:
                while True:
                    self.store.heartbeat_lease(
                        owner_token=self.owner_token,
                        ttl_seconds=self.lease_ttl_seconds,
                    )
                    new_worker_error = self._collect_worker_results(futures)
                    worker_error = worker_error or new_worker_error
                    if worker_error is not None:
                        if futures:
                            self._wait_for_worker(futures)
                            continue
                        raise worker_error

                    run = self.store.get_run()
                    status = RunStatus(str(run['status']))
                    if status in contracts.RUN_TERMINAL_STATES:
                        if futures:
                            self._wait_for_worker(futures)
                            continue
                        return self.store.build_snapshot()
                    if status is RunStatus.CANCEL_REQUESTED:
                        self.store.cancel_closeout(owner_token=self.owner_token)
                        self._request_worker_cancellation(
                            futures, cancel_attempted_ids
                        )
                        if futures:
                            self._wait_for_worker(futures)
                            continue
                        self._close_cancelled_run(
                            cancel_attempted_ids=cancel_attempted_ids
                        )
                        continue

                    self._verify_before_dispatch()
                    if not recovered and not futures and self._derive_failed_parents():
                        continue
                    self._fill_dispatch_slots(pool, futures, recovered)
                    if futures:
                        self._wait_for_worker(futures)
                        continue
                    if recovered:
                        continue
                    if self.store.list_active_attempts():
                        recovered.extend(self._recover_active_attempts())
                        continue
                    future_deadline = self._next_backoff_deadline()
                    if future_deadline is not None:
                        if not wait_for_backoff:
                            return self.store.build_snapshot()
                        delay = max(
                            0.0,
                            (
                                future_deadline - parse_utc(utcnow_iso())
                            ).total_seconds(),
                        )
                        self.sleep(
                            min(
                                delay,
                                max(0.05, self.lease_ttl_seconds / 3.0),
                            )
                        )
                        continue
                    self.store.finalize_run(owner_token=self.owner_token)
        finally:
            self.store.release_lease(owner_token=self.owner_token)

    def _collect_worker_results(
        self, futures: dict[Future, Mapping[str, Any]]
    ) -> BaseException | None:
        first_error = None
        for future in [item for item in futures if item.done()]:
            futures.pop(future, None)
            try:
                attempt, request_payload, receipt = future.result()
                if isinstance(receipt, ProviderOutcome):
                    self._record_outcome(attempt, request_payload, receipt)
                elif isinstance(receipt, ProviderFailure):
                    self._record_failure(attempt, request_payload, receipt)
                else:
                    self._record_failure(
                        attempt,
                        request_payload,
                        ProviderFailure(
                            ErrorCategory.LOCAL_VALIDATION,
                            'backend.invalid_outcome',
                            safe_details={
                                'outcome_type': type(receipt).__name__
                            },
                        ),
                    )
            except BaseException as exc:
                first_error = first_error or exc
        return first_error

    def _wait_for_worker(
        self, futures: Mapping[Future, Mapping[str, Any]]
    ) -> None:
        if not futures:
            return
        wait(
            tuple(futures),
            timeout=max(0.01, min(0.25, self.lease_ttl_seconds / 4.0)),
            return_when=FIRST_COMPLETED,
        )

    def _fill_dispatch_slots(
        self,
        pool: ThreadPoolExecutor,
        futures: dict[Future, Mapping[str, Any]],
        recovered: deque[tuple[str, Mapping[str, Any]]],
    ) -> None:
        while recovered and len(futures) < self.policy.max_in_flight:
            attempt_id, request_payload = recovered.popleft()
            self._dispatch_to_pool(
                pool,
                futures,
                attempt_id=attempt_id,
                request_payload=request_payload,
            )

        slots = self.policy.max_in_flight - len(futures)
        if slots <= 0:
            return
        for request in self.store.list_eligible_requests(limit=slots):
            request_payload = self.store.get_request_payload(request['request_id'])
            reservation = dict(self.reservation_provider(request_payload) or {})
            prepared = self.store.prepare_attempt_guarded(
                request_id=request['request_id'],
                owner_token=self.owner_token,
                policy=self.policy.to_dict(),
                provider=self.provider,
                model=self.model,
                profile_digest=self.profile_digest,
                credential_identity=self.credential_identity,
                reservation=reservation,
            )
            if not prepared['prepared']:
                continue
            self._dispatch_to_pool(
                pool,
                futures,
                attempt_id=prepared['attempt_id'],
                request_payload=request_payload,
            )

    def _dispatch_to_pool(
        self,
        pool: ThreadPoolExecutor,
        futures: dict[Future, Mapping[str, Any]],
        *,
        attempt_id: str,
        request_payload: Mapping[str, Any],
    ) -> bool:
        try:
            attempt = self.store.dispatch_attempt(
                attempt_id=attempt_id, owner_token=self.owner_token
            )
        except SyncRunError:
            run_status = RunStatus(str(self.store.get_run()['status']))
            attempt_row = self.store.get_attempt(attempt_id)
            if attempt_row is None:
                raise
            attempt_status = AttemptStatus(str(attempt_row['status']))
            if run_status in {
                RunStatus.CANCEL_REQUESTED,
                RunStatus.CANCELLED,
            } and attempt_status in {
                AttemptStatus.PREPARED,
                AttemptStatus.CANCELLED,
            }:
                return False
            raise
        future = pool.submit(
            self._send_to_provider,
            attempt=attempt,
            request_payload=request_payload,
        )
        futures[future] = attempt
        return True

    def _request_worker_cancellation(
        self,
        futures: Mapping[Future, Mapping[str, Any]],
        attempted_ids: set[str],
    ) -> None:
        cancel = getattr(self.backend, 'cancel', None)
        for attempt in futures.values():
            attempt_id = str(attempt['attempt_id'])
            if attempt_id in attempted_ids:
                continue
            attempted_ids.add(attempt_id)
            cancelled = False
            if callable(cancel):
                try:
                    cancelled = bool(cancel(attempt=attempt))
                except Exception:
                    cancelled = False
            if cancelled:
                self.store.confirm_attempt_cancelled(
                    attempt_id=attempt_id,
                    owner_token=self.owner_token,
                )

    def _verify_before_dispatch(self) -> None:
        violations = self.store.verify_integrity()
        if violations:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_STORAGE_ERROR,
                'durable run integrity verification failed',
                safe_details={
                    'run_id': self.store.run_id,
                    'violations': violations[:20],
                },
            )
        if self.freshness_check(self.store) is False:
            raise SyncRunError(
                ErrorCode.SYNC_RUN_FRESHNESS_MISMATCH,
                'stored durable request inputs are no longer fresh',
                safe_details={'run_id': self.store.run_id},
            )

    def _recover_active_attempts(
        self,
    ) -> list[tuple[str, Mapping[str, Any]]]:
        prepared = []
        for attempt in self.store.list_active_attempts():
            status = AttemptStatus(str(attempt['status']))
            if status is AttemptStatus.PREPARED:
                self._verify_before_dispatch()
                request_payload = self.store.get_request_payload(attempt['request_id'])
                prepared.append((str(attempt['attempt_id']), request_payload))
            elif status in (
                AttemptStatus.DISPATCHED,
                AttemptStatus.CANCEL_REQUESTED,
            ):
                self.store.mark_outcome_unknown(
                    attempt_id=attempt['attempt_id'],
                    owner_token=self.owner_token,
                    reason_code='orphaned_after_lease_takeover',
                )
        return prepared

    def _send_to_provider(
        self,
        *,
        attempt: Mapping[str, Any],
        request_payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
        try:
            outcome = self.backend.send(
                request_payload,
                attempt=attempt,
                timeout_seconds=self.policy.attempt_timeout_seconds,
            )
        except ProviderFailure as failure:
            receipt: ProviderOutcome | ProviderFailure = failure
        except Exception as exc:
            receipt = ProviderFailure(
                ErrorCategory.UNKNOWN_PROVIDER,
                'backend.unclassified_exception',
                safe_details={'exception_type': type(exc).__name__},
            )
        else:
            receipt = outcome
        return attempt, request_payload, receipt

    def _record_outcome(
        self,
        attempt: Mapping[str, Any],
        request_payload: Mapping[str, Any],
        outcome: ProviderOutcome,
    ) -> None:
        expected = list(request_payload.get('expected_ids') or [])
        accepted = _accepted_ids(outcome.accepted_items)
        missing = [item_id for item_id in expected if item_id not in set(accepted)]
        if not accepted:
            self._record_failure(
                attempt,
                request_payload,
                ProviderFailure(
                    ErrorCategory.INCOMPLETE_IDS,
                    'response.zero_valid_items',
                    usage_metadata=outcome.usage_metadata,
                ),
            )
            return
        children = []
        terminal_reason = None
        if missing:
            terminal_reason = self.store.lineage_budget_reason(
                request_id=attempt['request_id'],
                child_count=1,
                policy=self.policy.to_dict(),
            )
            if terminal_reason is None:
                children = [
                    self.derived_request_builder(
                        request_payload,
                        missing,
                        missing_lineage_suffix(missing),
                        contracts.LineageKind.MISSING_IDS,
                    )
                ]
        self.store.record_success(
            attempt_id=attempt['attempt_id'],
            owner_token=self.owner_token,
            accepted_items=outcome.accepted_items,
            response_payload=outcome.response_payload,
            normalized_payload=outcome.normalized_payload,
            contract_diagnostics=outcome.contract_diagnostics,
            usage_metadata=outcome.usage_metadata,
            derived_requests=children,
            partial_terminal_reason=terminal_reason,
        )

    def _record_failure(
        self,
        attempt: Mapping[str, Any],
        request_payload: Mapping[str, Any],
        failure: ProviderFailure,
    ) -> None:
        disposition = classify_failure(
            failure.category,
            retry_after_seconds=failure.retry_after_seconds,
            isolatable=failure.isolatable,
            repairable=failure.repairable,
        )
        if disposition.decision is RetryDecision.RETRYABLE:
            next_eligible_at = compute_next_eligible_at(
                failure_time=utcnow_iso(),
                same_request_retry_index=int(attempt['ordinal']),
                policy=self.policy,
                retry_after_seconds=failure.retry_after_seconds,
            )
            self.store.record_failure(
                attempt_id=attempt['attempt_id'],
                owner_token=self.owner_token,
                error_category=failure.category,
                error_reason_code=failure.reason_code,
                error_safe_details=failure.safe_details,
                next_eligible_at=next_eligible_at,
                terminal=False,
                usage_metadata=failure.usage_metadata,
            )
            return

        if disposition.decision is RetryDecision.DERIVED:
            accepted = self.store.record_failure(
                attempt_id=attempt['attempt_id'],
                owner_token=self.owner_token,
                error_category=failure.category,
                error_reason_code=failure.reason_code,
                error_safe_details=failure.safe_details,
                next_eligible_at=utcnow_iso(),
                terminal=False,
                usage_metadata=failure.usage_metadata,
            )
            if accepted:
                self._derive_request(request_payload, failure.reason_code)
            return

        accepted = self.store.record_failure(
            attempt_id=attempt['attempt_id'],
            owner_token=self.owner_token,
            error_category=failure.category,
            error_reason_code=failure.reason_code,
            error_safe_details=failure.safe_details,
            terminal=True,
            usage_metadata=failure.usage_metadata,
        )
        if accepted and disposition.decision is RetryDecision.RUN_LEVEL_STOP:
            self.store.stop_run_dispatch(
                owner_token=self.owner_token,
                reason_code=failure.reason_code,
            )

    def _derive_failed_parents(self) -> bool:
        """Finish a T4->T5 split interrupted after the failed-attempt commit."""
        for request in self.store.list_requests(status=RequestStatus.RETRYABLE_FAILED.value):
            attempts = self.store.list_attempts(request_id=request['request_id'])
            if not attempts:
                continue
            latest = attempts[-1]
            category = latest.get('error_category')
            if category not in {
                ErrorCategory.INVALID_STRUCTURED_RESPONSE.value,
                ErrorCategory.INCOMPLETE_IDS.value,
                ErrorCategory.CONTENT_POLICY.value,
                ErrorCategory.LOCAL_VALIDATION.value,
            }:
                continue
            self._derive_request(
                self.store.get_request_payload(request['request_id']),
                str(latest.get('error_reason_code') or category),
            )
            return True
        return False

    def _derive_request(
        self, request_payload: Mapping[str, Any], reason_code: str
    ) -> None:
        request_id = str(request_payload['request_id'])
        expected = list(request_payload.get('expected_ids') or [])
        if len(expected) < 2:
            self.store.terminalize_request(
                request_id=request_id,
                owner_token=self.owner_token,
                reason_code=f'{reason_code}.unsplittable',
            )
            return
        budget_reason = self.store.lineage_budget_reason(
            request_id=request_id,
            child_count=2,
            policy=self.policy.to_dict(),
        )
        if budget_reason:
            self.store.terminalize_request(
                request_id=request_id,
                owner_token=self.owner_token,
                reason_code=budget_reason,
            )
            return
        midpoint = len(expected) // 2
        left_ids = expected[:midpoint]
        right_ids = expected[midpoint:]
        children = [
            self.derived_request_builder(
                request_payload,
                left_ids,
                '--L',
                contracts.LineageKind.SPLIT_LEFT,
            ),
            self.derived_request_builder(
                request_payload,
                right_ids,
                '--R',
                contracts.LineageKind.SPLIT_RIGHT,
            ),
        ]
        self.store.supersede_with_children(
            request_id=request_id,
            children=children,
            owner_token=self.owner_token,
        )

    def _next_backoff_deadline(self):
        deadlines = []
        for request in self.store.list_requests(
            status=RequestStatus.RETRYABLE_FAILED.value
        ):
            value = request.get('next_eligible_at')
            if value:
                deadlines.append(parse_utc(value))
        return min(deadlines) if deadlines else None

    def _close_cancelled_run(
        self, *, cancel_attempted_ids: set[str] | None = None
    ) -> None:
        attempted_ids = set(cancel_attempted_ids or ())
        self.store.cancel_closeout(owner_token=self.owner_token)
        for attempt in self.store.list_active_attempts():
            status = AttemptStatus(str(attempt['status']))
            if status is not AttemptStatus.CANCEL_REQUESTED:
                continue
            cancelled = False
            cancel = getattr(self.backend, 'cancel', None)
            if attempt['attempt_id'] not in attempted_ids and callable(cancel):
                try:
                    cancelled = bool(cancel(attempt=attempt))
                except Exception:
                    cancelled = False
            if cancelled:
                self.store.confirm_attempt_cancelled(
                    attempt_id=attempt['attempt_id'], owner_token=self.owner_token
                )
            else:
                self.store.mark_outcome_unknown(
                    attempt_id=attempt['attempt_id'],
                    owner_token=self.owner_token,
                    reason_code='cancel_unconfirmed',
                )
        self.store.cancel_closeout(owner_token=self.owner_token)
