# -*- coding: utf-8 -*-
"""Scheduler, retry, budget, lineage and recovery tests for issue #347 P2."""

from __future__ import annotations

from collections import deque
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path

import sync_run_contracts as contracts
from durable_sync_executor import (
    DurableSyncExecutor,
    ProviderFailure,
    ProviderOutcome,
    TranslationPlanDerivedRequestBuilder,
)
from sync_retry_policy import (
    ExecutorPolicy,
    classify_failure,
    compute_next_eligible_at,
    missing_lineage_suffix,
)
from sync_run_contracts import (
    AttemptStatus,
    ErrorCategory,
    ErrorCode,
    RequestStatus,
    RetryDecision,
    RunStatus,
    SyncRunError,
    build_run_id,
)
from sync_run_store import SyncRunStore


def make_plan(plan_id='plan-1111111111111111'):
    return {
        'schema_version': 1,
        'plan_id': plan_id,
        'plan_fingerprint': 'b' * 16,
        'source_identity': {'source_snapshot_fingerprint': 'source-1'},
        'config_fingerprint': 'config-1',
        'model_profile_snapshot': {'provider': 'fake', 'model': 'fake-model'},
        'execution_strategy': 'sync',
        'chunk_policy': {},
        'context_policy': {},
        'chunks': [],
        'request_summaries': [],
        'artifacts': {},
    }


def make_request(request_id='req-1', expected_ids=('item-1', 'item-2')):
    return {
        'request_id': request_id,
        'plan_id': 'plan-1111111111111111',
        'chunk_id': request_id.replace('req-', 'chunk-'),
        'system_instruction': 'system',
        'user_prompt': 'user',
        'response_schema': {},
        'expected_ids': list(expected_ids),
        'capability_requirements': {},
        'generation_config': {},
        'transport_metadata': {},
        'context_assembly': {},
        'prompt_fingerprint': f'prompt-{request_id}',
        'request_fingerprint': f'fingerprint-{request_id}',
    }


def simple_child_builder(parent, target_ids, suffix, lineage_kind):
    child = dict(parent)
    child['request_id'] = f'{parent["request_id"]}{suffix}'
    child['chunk_id'] = f'{parent["chunk_id"]}{suffix}'
    child['expected_ids'] = list(target_ids)
    child['prompt_fingerprint'] = f'prompt-{child["request_id"]}'
    child['request_fingerprint'] = f'fingerprint-{child["request_id"]}'
    transport = dict(parent.get('transport_metadata') or {})
    transport.update({
        'retry_parent_request_id': parent['request_id'],
        'retry_parent_chunk_id': parent['chunk_id'],
        'retry_lineage_kind': lineage_kind.value,
        'retry_item_ids': list(target_ids),
    })
    child['transport_metadata'] = transport
    return child


class ScriptedBackend:
    def __init__(self, scripts=None, default=None):
        self.scripts = {
            request_id: deque(outcomes)
            for request_id, outcomes in dict(scripts or {}).items()
        }
        self.default = default
        self.calls = []
        self.cancelled = []

    def send(self, request, *, attempt, timeout_seconds):
        self.calls.append((request['request_id'], attempt['attempt_id'], timeout_seconds))
        script = self.scripts.get(request['request_id'])
        result = script.popleft() if script else self.default
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            result = result(request, attempt)
        if result is None:
            return ProviderOutcome(accepted_items=request['expected_ids'])
        return result

    def cancel(self, *, attempt):
        self.cancelled.append(attempt['attempt_id'])
        return True


class BlockingBackend:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def send(self, request, *, attempt, timeout_seconds):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise AssertionError('test backend was not released')
        return ProviderOutcome(
            accepted_items=request['expected_ids'],
            usage_metadata={'total_tokens': 1},
        )

    def cancel(self, *, attempt):
        self.cancelled.set()
        self.release.set()
        return True


class ConcurrentBlockingBackend:
    def __init__(self, expected_started, *, release_on_cancel=False):
        self.expected_started = int(expected_started)
        self.release_on_cancel = bool(release_on_cancel)
        self.lock = threading.Lock()
        self.started = []
        self.started_event = threading.Event()
        self.release = threading.Event()
        self.cancelled = []
        self.active = 0
        self.max_active = 0

    def send(self, request, *, attempt, timeout_seconds):
        with self.lock:
            self.started.append(request['request_id'])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if len(self.started) >= self.expected_started:
                self.started_event.set()
        try:
            if not self.release.wait(timeout=3.0):
                raise AssertionError('concurrent test backend was not released')
            return ProviderOutcome(accepted_items=request['expected_ids'])
        finally:
            with self.lock:
                self.active -= 1

    def cancel(self, *, attempt):
        with self.lock:
            self.cancelled.append(attempt['attempt_id'])
            release = len(set(self.cancelled)) >= self.expected_started
        if self.release_on_cancel and release:
            self.release.set()
        return True


class PolicyTests(unittest.TestCase):
    def test_policy_is_validated_and_round_trips(self):
        policy = ExecutorPolicy.from_mapping({'max_attempts_per_request': 2})
        self.assertEqual(policy.max_attempts_per_request, 2)
        self.assertEqual(ExecutorPolicy.from_mapping(policy.to_dict()), policy)
        with self.assertRaises(ValueError):
            ExecutorPolicy.from_mapping({'unknown': 1})
        with self.assertRaises(ValueError):
            ExecutorPolicy.from_mapping({
                'max_attempts_per_request': 4,
                'max_attempts_per_root': 3,
            })

    def test_failure_table_exceptions_are_explicit(self):
        self.assertEqual(
            classify_failure(ErrorCategory.AUTHENTICATION).decision,
            RetryDecision.TERMINAL,
        )
        self.assertEqual(
            classify_failure(
                ErrorCategory.QUOTA_EXHAUSTED, retry_after_seconds=5
            ).decision,
            RetryDecision.RETRYABLE,
        )
        self.assertEqual(
            classify_failure(ErrorCategory.CONTENT_POLICY, isolatable=True).decision,
            RetryDecision.DERIVED,
        )
        self.assertEqual(
            classify_failure(ErrorCategory.LOCAL_VALIDATION, repairable=False).decision,
            RetryDecision.TERMINAL,
        )

    def test_backoff_is_bounded_and_suffix_is_stable(self):
        policy = ExecutorPolicy.from_mapping({
            'backoff_base_seconds': 2,
            'backoff_cap_seconds': 3,
        })
        deadline = compute_next_eligible_at(
            failure_time='2026-01-01T00:00:00.000000Z',
            same_request_retry_index=3,
            policy=policy,
            rng=random.Random(7),
        )
        self.assertGreaterEqual(deadline, '2026-01-01T00:00:00.000000Z')
        self.assertLessEqual(deadline, '2026-01-01T00:00:03.000000Z')
        self.assertEqual(
            missing_lineage_suffix(['a', 'b']),
            missing_lineage_suffix(['a', 'b']),
        )
        self.assertNotEqual(
            missing_lineage_suffix(['a', 'b']),
            missing_lineage_suffix(['b', 'a']),
        )


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def bootstrap(self, *, requests=None, policy=None):
        policy = policy or ExecutorPolicy()
        store, _ = SyncRunStore.bootstrap(
            self.root,
            build_run_id(),
            plan=make_plan(),
            requests=requests or [make_request()],
            executor_policy=policy.to_dict(),
        )
        return store, policy

    def executor(self, store, backend, policy, **kwargs):
        return DurableSyncExecutor(
            store,
            backend,
            derived_request_builder=simple_child_builder,
            policy=policy,
            provider='fake',
            model='fake-model',
            **kwargs,
        )

    def test_full_success_dispatches_once_and_finalizes(self):
        store, policy = self.bootstrap()
        backend = ScriptedBackend()
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual([call[0] for call in backend.calls], ['req-1'])
        self.assertEqual(snapshot['progress']['items']['accepted'], 2)
        self.assertEqual(store.verify_integrity(), [])

    def test_partial_success_only_retries_missing_ids(self):
        store, policy = self.bootstrap()
        child_id = 'req-1' + missing_lineage_suffix(['item-2'])
        backend = ScriptedBackend({
            'req-1': [ProviderOutcome(
                accepted_items={'item-1': {'translation': '一'}},
                usage_metadata={'total_tokens': 3},
            )],
            child_id: [ProviderOutcome(
                accepted_items={'item-2': {'translation': '二'}},
                usage_metadata={'total_tokens': 2},
            )],
        })
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual([call[0] for call in backend.calls], ['req-1', child_id])
        self.assertEqual(
            {row['item_id'] for row in store.list_item_results()},
            {'item-1', 'item-2'},
        )

    def test_invalid_response_splits_then_children_succeed(self):
        store, policy = self.bootstrap()
        backend = ScriptedBackend({
            'req-1': [ProviderFailure(
                ErrorCategory.INVALID_STRUCTURED_RESPONSE,
                'invalid_json',
            )],
            'req-1--L': [ProviderOutcome(accepted_items=['item-1'])],
            'req-1--R': [ProviderOutcome(accepted_items=['item-2'])],
        })
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual(
            [call[0] for call in backend.calls],
            ['req-1', 'req-1--L', 'req-1--R'],
        )
        self.assertEqual(store.get_request('req-1')['status'], RequestStatus.SUPERSEDED.value)

    def test_retry_deadline_is_persisted_and_then_succeeds(self):
        store, policy = self.bootstrap()
        backend = ScriptedBackend({
            'req-1': [
                ProviderFailure(
                    ErrorCategory.RATE_LIMIT,
                    'rate_limit',
                    retry_after_seconds=0,
                ),
                ProviderOutcome(accepted_items=['item-1', 'item-2']),
            ],
        })
        snapshot = self.executor(store, backend, policy).run(wait_for_backoff=True)
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual(len(backend.calls), 2)
        first = store.list_attempts(request_id='req-1')[0]
        self.assertTrue(first['next_eligible_at'])

    def test_authentication_failure_is_not_retried_or_split(self):
        store, policy = self.bootstrap()
        backend = ScriptedBackend(default=ProviderFailure(
            ErrorCategory.AUTHENTICATION, 'bad_api_key'
        ))
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.FAILED.value)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(store.list_requests()), 1)

    def test_request_attempt_cap_terminalizes_leaf(self):
        policy = ExecutorPolicy.from_mapping({
            'max_attempts_per_request': 1,
            'max_attempts_per_root': 1,
        })
        store, _ = self.bootstrap(policy=policy)
        backend = ScriptedBackend(default=ProviderFailure(
            ErrorCategory.TRANSPORT,
            'transport',
            retry_after_seconds=0,
        ))
        snapshot = self.executor(store, backend, policy).run(wait_for_backoff=True)
        self.assertEqual(snapshot['run_status'], RunStatus.FAILED.value)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(
            store.get_request('req-1')['status'],
            RequestStatus.TERMINAL_FAILED.value,
        )

    def test_run_attempt_cap_stops_remaining_leaves(self):
        policy = ExecutorPolicy.from_mapping({'max_total_attempts_per_run': 1})
        store, _ = self.bootstrap(
            policy=policy,
            requests=[
                make_request('req-1', ['item-1']),
                make_request('req-2', ['item-2']),
            ],
        )
        backend = ScriptedBackend()
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED_WITH_ERRORS.value)
        self.assertEqual([call[0] for call in backend.calls], ['req-1'])
        self.assertEqual(
            store.get_request('req-2')['status'], RequestStatus.TERMINAL_FAILED.value
        )

    def test_cost_cap_requires_trustworthy_reservation(self):
        policy = ExecutorPolicy.from_mapping({'max_estimated_cost': 1.0})
        store, _ = self.bootstrap(policy=policy)
        backend = ScriptedBackend()
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.FAILED.value)
        self.assertEqual(backend.calls, [])

    def test_cost_reservation_cannot_cross_run_cap(self):
        policy = ExecutorPolicy.from_mapping({'max_estimated_cost': 1.0})
        store, _ = self.bootstrap(
            policy=policy,
            requests=[
                make_request('req-1', ['item-1']),
                make_request('req-2', ['item-2']),
            ],
        )
        backend = ScriptedBackend()
        snapshot = self.executor(
            store,
            backend,
            policy,
            reservation_provider=lambda _request: {'estimated_cost': 0.6},
        ).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED_WITH_ERRORS.value)
        self.assertEqual([call[0] for call in backend.calls], ['req-1'])

    def test_concurrent_cost_reservations_cannot_cross_run_cap(self):
        policy = ExecutorPolicy.from_mapping({
            'max_estimated_cost': 1.0,
            'max_in_flight': 2,
        })
        store, _ = self.bootstrap(
            policy=policy,
            requests=[
                make_request('req-1', ['item-1']),
                make_request('req-2', ['item-2']),
            ],
        )
        backend = ConcurrentBlockingBackend(expected_started=1)
        result = {}
        worker = threading.Thread(
            target=lambda: result.setdefault(
                'snapshot',
                self.executor(
                    store,
                    backend,
                    policy,
                    reservation_provider=lambda _request: {
                        'estimated_cost': 0.6
                    },
                ).run(),
            )
        )
        worker.start()
        self.assertTrue(backend.started_event.wait(timeout=1.0))
        backend.release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(backend.started, ['req-1'])
        self.assertEqual(
            result['snapshot']['run_status'],
            RunStatus.COMPLETED_WITH_ERRORS.value,
        )

    def test_prepared_attempt_is_resumed_without_new_ordinal(self):
        store, policy = self.bootstrap()
        store.acquire_lease(owner_token='old-owner')
        attempt_id = store.prepare_attempt(request_id='req-1', owner_token='old-owner')
        with store._tx() as conn:
            conn.execute(
                'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                ('2000-01-01T00:00:00.000000Z', store.run_id),
            )
        backend = ScriptedBackend()
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual(backend.calls[0][1], attempt_id)
        self.assertEqual(len(store.list_attempts(request_id='req-1')), 1)

    def test_orphaned_dispatched_is_unknown_but_other_request_continues(self):
        store, policy = self.bootstrap(requests=[
            make_request('req-1', ['item-1']),
            make_request('req-2', ['item-2']),
        ])
        store.acquire_lease(owner_token='old-owner')
        attempt_id = store.prepare_attempt(request_id='req-1', owner_token='old-owner')
        store.dispatch_attempt(attempt_id=attempt_id, owner_token='old-owner')
        with store._tx() as conn:
            conn.execute(
                'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                ('2000-01-01T00:00:00.000000Z', store.run_id),
            )
        backend = ScriptedBackend()
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED_WITH_ERRORS.value)
        self.assertEqual([call[0] for call in backend.calls], ['req-2'])
        self.assertEqual(
            store.get_request('req-1')['status'], RequestStatus.OUTCOME_UNKNOWN.value
        )

    def test_stale_prepared_attempt_is_not_dispatched(self):
        store, policy = self.bootstrap()
        store.acquire_lease(owner_token='old-owner')
        store.prepare_attempt(request_id='req-1', owner_token='old-owner')
        with store._tx() as conn:
            conn.execute(
                'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                ('2000-01-01T00:00:00.000000Z', store.run_id),
            )
        backend = ScriptedBackend()
        executor = self.executor(
            store, backend, policy, freshness_check=lambda _store: False
        )
        with self.assertRaises(SyncRunError) as ctx:
            executor.run()
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_FRESHNESS_MISMATCH)
        self.assertEqual(backend.calls, [])
        self.assertEqual(
            store.list_attempts()[0]['status'], AttemptStatus.PREPARED.value
        )

    def test_committed_cancel_never_dispatches(self):
        store, policy = self.bootstrap()
        store.cancel_intent(reason='user')
        backend = ScriptedBackend()
        snapshot = self.executor(store, backend, policy).run()
        self.assertEqual(snapshot['run_status'], RunStatus.CANCELLED.value)
        self.assertEqual(backend.calls, [])

    def test_cancel_between_prepare_and_dispatch_never_calls_provider(self):
        store, policy = self.bootstrap()
        backend = ScriptedBackend()
        entered = threading.Event()
        release = threading.Event()
        original_dispatch = store.dispatch_attempt

        def delayed_dispatch(**kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=2.0))
            return original_dispatch(**kwargs)

        store.dispatch_attempt = delayed_dispatch
        result = {}
        worker = threading.Thread(
            target=lambda: result.setdefault(
                'snapshot', self.executor(store, backend, policy).run()
            )
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=1.0))
        self.assertTrue(store.cancel_intent(reason='user'))
        release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result['snapshot']['run_status'], RunStatus.CANCELLED.value)
        self.assertEqual(backend.calls, [])

    def test_heartbeat_keeps_lease_alive_during_provider_io(self):
        store, policy = self.bootstrap()
        backend = BlockingBackend()

        def release_later():
            self.assertTrue(backend.started.wait(timeout=1.0))
            time.sleep(1.1)
            backend.release.set()

        releaser = threading.Thread(target=release_later)
        releaser.start()
        snapshot = self.executor(
            store, backend, policy, lease_ttl_seconds=0.5
        ).run()
        releaser.join(timeout=2.0)
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)

    def test_max_in_flight_is_a_real_bounded_dispatch_limit(self):
        policy = ExecutorPolicy.from_mapping({'max_in_flight': 2})
        requests = [
            make_request(f'req-{index}', [f'item-{index}'])
            for index in range(1, 5)
        ]
        store, _ = self.bootstrap(requests=requests, policy=policy)
        backend = ConcurrentBlockingBackend(expected_started=2)
        result = {}

        worker = threading.Thread(
            target=lambda: result.setdefault(
                'snapshot', self.executor(store, backend, policy).run()
            )
        )
        worker.start()
        self.assertTrue(backend.started_event.wait(timeout=1.0))
        time.sleep(0.1)
        with backend.lock:
            self.assertEqual(len(backend.started), 2)
            self.assertEqual(backend.max_active, 2)
        backend.release.set()
        worker.join(timeout=3.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result['snapshot']['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual(len(backend.started), 4)
        self.assertEqual(backend.max_active, 2)

    def test_cancel_closes_multiple_in_flight_and_never_starts_pending(self):
        policy = ExecutorPolicy.from_mapping({'max_in_flight': 2})
        requests = [
            make_request(f'req-{index}', [f'item-{index}'])
            for index in range(1, 4)
        ]
        store, _ = self.bootstrap(requests=requests, policy=policy)
        backend = ConcurrentBlockingBackend(
            expected_started=2,
            release_on_cancel=True,
        )
        result = {}

        worker = threading.Thread(
            target=lambda: result.setdefault(
                'snapshot',
                self.executor(
                    store, backend, policy, lease_ttl_seconds=5.0
                ).run(),
            )
        )
        worker.start()
        worker_alive = True
        try:
            self.assertTrue(backend.started_event.wait(timeout=2.0))
            self.assertTrue(store.cancel_intent(reason='user'))
            worker.join(timeout=5.0)
            worker_alive = worker.is_alive()
        finally:
            # Never leave provider workers holding the SQLite file open when
            # an assertion fails; the pre-cleanup value still detects a hang.
            backend.release.set()
            worker.join(timeout=2.0)
        self.assertFalse(worker_alive)
        self.assertEqual(result['snapshot']['run_status'], RunStatus.CANCELLED.value)
        self.assertEqual(set(backend.started), {'req-1', 'req-2'})
        self.assertEqual(len(set(backend.cancelled)), 2)
        self.assertEqual(store.list_item_results(), [])
        self.assertEqual(
            store.get_request('req-3')['status'], RequestStatus.CANCELLED.value
        )

    def test_cancel_during_provider_io_is_confirmed_and_late_result_ignored(self):
        store, policy = self.bootstrap()
        backend = BlockingBackend()
        result = {}

        def execute():
            result['snapshot'] = self.executor(
                store, backend, policy, lease_ttl_seconds=0.15
            ).run()

        worker = threading.Thread(target=execute)
        worker.start()
        self.assertTrue(backend.started.wait(timeout=1.0))
        self.assertTrue(store.cancel_intent(reason='user'))
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(backend.cancelled.is_set())
        self.assertEqual(
            result['snapshot']['run_status'], RunStatus.CANCELLED.value
        )
        self.assertEqual(store.list_item_results(), [])
        self.assertEqual(
            store.list_attempts()[0]['status'],
            AttemptStatus.CANCELLED.value,
        )

    def test_executor_rejects_policy_drift(self):
        store, policy = self.bootstrap()
        changed = ExecutorPolicy.from_mapping({'max_attempts_per_request': 2})
        with self.assertRaises(SyncRunError):
            self.executor(store, ScriptedBackend(), changed)

    def test_real_translation_plan_child_helper_is_consumed(self):
        parent = make_request(expected_ids=['item-1'])
        item = {
            'id': 'item-1',
            'text': 'Hello',
            'line': 1,
            'speaker_id': '',
            'speaker_name': '',
            'block_name': '',
        }
        builder = TranslationPlanDerivedRequestBuilder(
            lambda _parent, ids: [item for item_id in ids if item_id == 'item-1']
        )
        child = builder(
            parent,
            ['item-1'],
            '--L',
            contracts.LineageKind.SPLIT_LEFT,
        )
        self.assertEqual(child['request_id'], 'req-1--L')
        self.assertEqual(child['expected_ids'], ['item-1'])
        self.assertEqual(
            child['transport_metadata']['retry_parent_request_id'], 'req-1'
        )


if __name__ == '__main__':
    unittest.main()
