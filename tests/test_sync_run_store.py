# -*- coding: utf-8 -*-
"""Tests for the #347 P1 pure state and SQLite store.

P1 deliberately has no network, no scheduler loop and no production wiring.
These tests pin the state machines, transactional boundaries, lease/event
durability and redacted projections.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sync_run_contracts as contracts
from sync_run_contracts import (
    AttemptStatus,
    ErrorCategory,
    ErrorCode,
    RequestStatus,
    RunStatus,
    SyncRunError,
    build_run_id,
)
from sync_run_store import SyncRunStore


def make_plan(plan_id='plan-1111111111111111'):
    return {
        'schema_version': 1,
        'plan_id': plan_id,
        'plan_fingerprint': 'a' * 16,
        'run_id': '',
        'source_identity': {'engine': 'renpy', 'adapter_version': 'v1'},
        'config_fingerprint': 'b' * 16,
        'model_profile_snapshot': {'provider': 'fake'},
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
        'chunk_id': 'chunk-1',
        'system_instruction': 'system',
        'user_prompt': 'user',
        'response_schema': {},
        'expected_ids': list(expected_ids),
        'capability_requirements': {},
        'generation_config': {},
        'transport_metadata': {},
        'context_assembly': {},
        'prompt_fingerprint': 'c' * 16,
        'request_fingerprint': 'd' * 16,
    }


def make_child_request(
    *,
    parent_request_id='req-1',
    request_id=None,
    expected_ids=('item-2',),
    lineage_kind=contracts.LineageKind.MISSING_IDS,
):
    request_id = request_id or f'{parent_request_id}--M-' + 'e' * 12
    child = make_request(request_id=request_id, expected_ids=expected_ids)
    child['transport_metadata'] = {
        'retry_parent_request_id': parent_request_id,
        'retry_parent_chunk_id': 'chunk-1',
        'retry_lineage_kind': lineage_kind.value,
        'retry_item_ids': list(expected_ids),
    }
    return child


def bootstrap_run(root, run_id=None, *, plan=None, requests=None, client_token=None):
    run_id = run_id or build_run_id()
    plan = plan if plan is not None else make_plan()
    requests = requests if requests is not None else [make_request()]
    return SyncRunStore.bootstrap(
        root,
        run_id,
        plan=plan,
        requests=requests,
        client_token=client_token,
    )


class ContractTests(unittest.TestCase):
    def test_frozen_run_transitions(self):
        self.assertIn(RunStatus.CANCELLED, contracts.RUN_TERMINAL_STATES)
        self.assertNotIn(RunStatus.RUNNING, contracts.RUN_TERMINAL_STATES)
        self.assertTrue(contracts.can_transition(
            RunStatus.RUNNING, RunStatus.CANCEL_REQUESTED, contracts.RUN_TRANSITIONS
        ))
        self.assertFalse(contracts.can_transition(
            RunStatus.CANCELLED, RunStatus.RUNNING, contracts.RUN_TRANSITIONS
        ))

    def test_frozen_request_transitions(self):
        self.assertFalse(contracts.can_transition(
            RequestStatus.SUCCEEDED, RequestStatus.IN_FLIGHT, contracts.REQUEST_TRANSITIONS
        ))
        self.assertTrue(contracts.can_transition(
            RequestStatus.IN_FLIGHT, RequestStatus.SUPERSEDED, contracts.REQUEST_TRANSITIONS
        ))

    def test_frozen_attempt_transitions(self):
        self.assertTrue(contracts.can_transition(
            AttemptStatus.PREPARED, AttemptStatus.DISPATCHED, contracts.ATTEMPT_TRANSITIONS
        ))
        self.assertTrue(contracts.can_transition(
            AttemptStatus.DISPATCHED, AttemptStatus.LATE_SUCCEEDED_IGNORED,
            contracts.ATTEMPT_TRANSITIONS,
        ))
        self.assertFalse(contracts.can_transition(
            AttemptStatus.SUCCEEDED, AttemptStatus.RETRYABLE_FAILED,
            contracts.ATTEMPT_TRANSITIONS,
        ))

    def test_run_id_shape_and_token_stability(self):
        fresh = build_run_id()
        self.assertRegex(fresh, r'^sync-run-v1-\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}$')
        token_run = build_run_id('my-project-token')
        self.assertEqual(token_run, build_run_id('my-project-token'))
        self.assertTrue(token_run.startswith('sync-run-v1-token-'))
        self.assertNotIn('my-project-token', token_run)

    def test_malformed_prefixed_run_id_is_rejected(self):
        self.assertFalse(contracts.validate_run_id('sync-run-v1-garbage'))
        with self.assertRaises(ValueError):
            contracts.assert_valid_run_id('sync-run-v1-garbage')

    def test_retry_decision_table_covers_every_error_category(self):
        self.assertEqual(set(contracts.ERROR_RETRY_DECISIONS), set(ErrorCategory))
        self.assertEqual(
            contracts.retry_decision_for(ErrorCategory.TIMEOUT),
            contracts.RetryDecision.RETRYABLE,
        )

    def test_attempt_id_is_stable(self):
        first = contracts.build_attempt_id('run-1', 'req-1', 1)
        second = contracts.build_attempt_id('run-1', 'req-1', 1)
        other = contracts.build_attempt_id('run-1', 'req-1', 2)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 24)

    def test_usage_event_id_is_stable(self):
        attempt_id = contracts.build_attempt_id('run-1', 'req-1', 1)
        self.assertEqual(
            contracts.build_usage_event_id(attempt_id),
            f'usage:{attempt_id}',
        )


class BootstrapStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_t0_creates_run_plan_requests_and_event_atomically(self):
        run_id = build_run_id()
        store, created = bootstrap_run(self.root, run_id)
        self.assertTrue(created)
        self.assertTrue((self.root / run_id / 'state.sqlite3').is_file())

        run = store.get_run()
        self.assertEqual(run['status'], RunStatus.PLANNED.value)
        self.assertEqual(run['plan_id'], 'plan-1111111111111111')
        self.assertEqual(run['revision'], 0)

        plan = store.get_plan()
        self.assertEqual(
            plan['payload_sha256'],
            contracts.sha256_hex(plan['canonical_json']),
        )

        requests = store.list_requests()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['status'], RequestStatus.PENDING.value)
        self.assertEqual(
            json.loads(requests[0]['expected_ids_json']),
            ['item-1', 'item-2'],
        )

        events = store.list_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], contracts.EventType.RUN_CREATED.value)
        self.assertEqual(store.verify_integrity(), [])

    def test_t0_is_idempotent_for_same_inputs(self):
        run_id = build_run_id('project-token')
        store1, created1 = bootstrap_run(
            self.root, run_id,
            client_token='project-token',
            requests=[make_request()],
        )
        self.assertTrue(created1)

        store2, created2 = bootstrap_run(
            self.root, run_id,
            client_token='project-token',
            requests=[make_request()],
        )
        self.assertFalse(created2)
        self.assertEqual(store1.get_run()['run_id'], store2.get_run()['run_id'])

    def test_t0_token_reopen_ignores_derived_requests(self):
        run_id = build_run_id('project-token')
        store1, _created = bootstrap_run(
            self.root,
            run_id,
            client_token='project-token',
            requests=[make_request()],
        )
        store1.acquire_lease(owner_token='owner-1')
        attempt_id = store1.prepare_attempt(
            request_id='req-1', owner_token='owner-1'
        )
        store1.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        store1.record_success(
            attempt_id=attempt_id,
            owner_token='owner-1',
            accepted_items=['item-1'],
            derived_requests=[make_child_request(expected_ids=['item-2'])],
        )
        store1.release_lease(owner_token='owner-1')

        store2, created2 = bootstrap_run(
            self.root,
            run_id,
            client_token='project-token',
            requests=[make_request()],
        )

        self.assertFalse(created2)
        self.assertEqual(len(store2.list_requests()), 2)

    def test_token_conflict_detects_different_plan(self):
        run_id = build_run_id('project-token')
        bootstrap_run(self.root, run_id, client_token='project-token')
        other_request = make_request()
        other_request['plan_id'] = 'other-plan-999999999999'
        with self.assertRaises(SyncRunError) as ctx:
            bootstrap_run(
                self.root,
                run_id,
                client_token='project-token',
                plan=make_plan(plan_id='other-plan-999999999999'),
                requests=[other_request],
            )
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_CLIENT_TOKEN_CONFLICT)

    def test_token_conflict_detects_different_policy(self):
        run_id = build_run_id('project-token')
        SyncRunStore.bootstrap(
            self.root,
            run_id,
            plan=make_plan(),
            requests=[make_request()],
            executor_policy={'max_attempts_per_request': 3},
            client_token='project-token',
        )
        with self.assertRaises(SyncRunError) as ctx:
            SyncRunStore.bootstrap(
                self.root,
                run_id,
                plan=make_plan(),
                requests=[make_request()],
                executor_policy={'max_attempts_per_request': 4},
                client_token='project-token',
            )
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_CLIENT_TOKEN_CONFLICT)

    def test_malformed_root_request_is_rejected(self):
        request = make_request()
        request['expected_ids'] = ['item-1', 'item-1']
        with self.assertRaises(ValueError):
            bootstrap_run(self.root, requests=[request])

    def test_schema_newer_version_rejected(self):
        run_id = build_run_id()
        store, _ = bootstrap_run(self.root, run_id)
        conn = sqlite3.connect(str(store.db_path))
        try:
            with conn:
                conn.execute('DELETE FROM schema_meta')
                conn.execute(
                    'INSERT INTO schema_meta(version, created_by, created_at) VALUES (?, ?, ?)',
                    (99, 'test', '2026-01-01T00:00:00Z'),
                )
        finally:
            conn.close()
        with self.assertRaises(SyncRunError) as ctx:
            store.get_run()
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_SCHEMA_UNSUPPORTED)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store, _ = bootstrap_run(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_acquire_release_and_expiry(self):
        lease = self.store.acquire_lease(owner_token='owner-a', ttl_seconds=1.0)
        self.assertEqual(lease['owner_token'], 'owner-a')
        with self.assertRaises(SyncRunError) as ctx:
            self.store.acquire_lease(owner_token='owner-b', ttl_seconds=1.0)
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_BUSY)
        self.assertTrue(ctx.exception.retryable)

        self.store.heartbeat_lease(owner_token='owner-a', ttl_seconds=1.0)
        self.assertTrue(
            self.store.release_lease(owner_token='owner-a')
        )
        lease_b = self.store.acquire_lease(owner_token='owner-b', ttl_seconds=1.0)
        self.assertEqual(lease_b['owner_token'], 'owner-b')

    def test_lease_owner_required_for_attempt_changes(self):
        with self.assertRaises(SyncRunError) as ctx:
            self.store.prepare_attempt(
                request_id='req-1',
                owner_token='no-such-owner',
            )
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_BUSY)

    def test_expired_owner_must_reacquire_before_heartbeat(self):
        self.store.acquire_lease(owner_token='owner-a', ttl_seconds=0.001)
        with self.store._tx() as conn:
            conn.execute(
                'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                ('2000-01-01T00:00:00.000000Z', self.store.run_id),
            )
        with self.assertRaises(SyncRunError) as ctx:
            self.store.heartbeat_lease(owner_token='owner-a')
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_BUSY)


class AttemptLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store, _ = bootstrap_run(self.root)
        self.store.acquire_lease(owner_token='owner-1')

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_dispatch_success_finalize(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.assertEqual(self.store.get_request('req-1')['status'], RequestStatus.IN_FLIGHT.value)
        self.assertEqual(self.store.get_run()['status'], RunStatus.RUNNING.value)

        dispatch = self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        self.assertEqual(dispatch['status'], AttemptStatus.DISPATCHED.value)
        self.assertIsNotNone(dispatch['dispatch_time'])

        accepted = self.store.record_success(
            attempt_id=attempt_id,
            owner_token='owner-1',
            accepted_items=['item-1', 'item-2'],
            usage_metadata={'total_tokens': 10},
        )
        self.assertTrue(accepted)
        self.assertEqual(self.store.get_request('req-1')['status'], RequestStatus.SUCCEEDED.value)
        self.assertEqual(len(self.store.list_item_results()), 2)
        self.assertEqual(len(self.store.pending_usage_outbox()), 1)

        run = self.store.finalize_run(owner_token='owner-1')
        self.assertEqual(run['status'], RunStatus.COMPLETED.value)
        self.assertEqual(self.store.verify_integrity(), [])

    def test_only_one_active_attempt_per_request(self):
        self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        with self.assertRaises(SyncRunError):
            self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')

    def test_failure_retryable_persists_backoff(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        accepted = self.store.record_failure(
            attempt_id=attempt_id,
            owner_token='owner-1',
            error_category=ErrorCategory.PROVIDER_SERVER,
            error_reason_code='server_5xx',
            next_eligible_at='2099-01-01T00:00:00.000000Z',
        )
        self.assertTrue(accepted)
        request = self.store.get_request('req-1')
        self.assertEqual(request['status'], RequestStatus.RETRYABLE_FAILED.value)
        self.assertEqual(
            request['next_eligible_at'],
            '2099-01-01T00:00:00.000000Z',
        )
        attempt = self.store.list_attempts(request_id='req-1')[0]
        self.assertEqual(attempt['status'], AttemptStatus.RETRYABLE_FAILED.value)

    def test_failure_terminal_category_never_retries(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        self.store.record_failure(
            attempt_id=attempt_id,
            owner_token='owner-1',
            error_category=ErrorCategory.AUTHENTICATION,
            error_reason_code='bad_apikey',
        )
        request = self.store.get_request('req-1')
        self.assertEqual(request['status'], RequestStatus.TERMINAL_FAILED.value)
        attempt = self.store.list_attempts(request_id='req-1')[0]
        self.assertEqual(attempt['status'], AttemptStatus.TERMINAL_FAILED.value)
        self.assertEqual(attempt['error_category'], ErrorCategory.AUTHENTICATION.value)

    def test_partial_success_is_atomic_with_children(self):
        child_ids = ['req-1--M-' + 'e' * 12]
        child = make_child_request(request_id=child_ids[0], expected_ids=['item-2'])
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        accepted = self.store.record_success(
            attempt_id=attempt_id,
            owner_token='owner-1',
            accepted_items=['item-1'],
            derived_requests=[child],
        )
        self.assertTrue(accepted)
        parent = self.store.get_request('req-1')
        self.assertEqual(parent['status'], RequestStatus.SUPERSEDED.value)
        child_row = self.store.get_request(child_ids[0])
        self.assertEqual(child_row['status'], RequestStatus.PENDING.value)
        self.assertEqual(child_row['parent_request_id'], 'req-1')
        winners = {row['item_id'] for row in self.store.list_item_results()}
        self.assertEqual(winners, {'item-1'})
        self.assertEqual(self.store.verify_integrity(), [])

    def test_partial_success_rejects_incomplete_child_coverage_atomically(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        with self.assertRaises(ValueError):
            self.store.record_success(
                attempt_id=attempt_id,
                owner_token='owner-1',
                accepted_items=['item-1'],
                derived_requests=[],
            )
        self.assertEqual(self.store.get_request('req-1')['status'], RequestStatus.IN_FLIGHT.value)
        self.assertEqual(self.store.list_item_results(), [])

    def test_partial_success_rejects_overlapping_children(self):
        child_a = make_child_request(
            request_id='req-1--M-' + 'a' * 12,
            expected_ids=['item-2'],
        )
        child_b = make_child_request(
            request_id='req-1--M-' + 'b' * 12,
            expected_ids=['item-2'],
        )
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        with self.assertRaises(ValueError):
            self.store.record_success(
                attempt_id=attempt_id,
                owner_token='owner-1',
                accepted_items=['item-1'],
                derived_requests=[child_a, child_b],
            )

    def test_zero_progress_success_must_be_recorded_as_failure(self):
        child = make_child_request(expected_ids=['item-1', 'item-2'])
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        with self.assertRaises(ValueError):
            self.store.record_success(
                attempt_id=attempt_id,
                owner_token='owner-1',
                accepted_items=[],
                derived_requests=[child],
            )

    def test_takeover_can_dispatch_prepared_attempt(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        with self.store._tx() as conn:
            conn.execute(
                'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                ('2000-01-01T00:00:00.000000Z', self.store.run_id),
            )
        self.store.acquire_lease(owner_token='owner-2')
        attempt = self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-2')
        self.assertEqual(attempt['claim_owner_token'], 'owner-2')

    def test_takeover_can_close_old_dispatched_attempt_as_unknown(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        with self.store._tx() as conn:
            conn.execute(
                'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                ('2000-01-01T00:00:00.000000Z', self.store.run_id),
            )
        self.store.acquire_lease(owner_token='owner-2')
        self.assertTrue(
            self.store.mark_outcome_unknown(attempt_id=attempt_id, owner_token='owner-2')
        )
        self.assertEqual(
            self.store.get_request('req-1')['status'],
            RequestStatus.OUTCOME_UNKNOWN.value,
        )

    def test_finalize_refuses_stranded_active_leaves(self):
        self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        with self.assertRaises(SyncRunError) as ctx:
            self.store.finalize_run(owner_token='owner-1')
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_STORAGE_ERROR)

    def test_supersede_with_children_requires_no_active_attempt(self):
        child_ids = ['req-1--M-' + 'f' * 12]
        child = make_request(request_id=child_ids[0], expected_ids=['item-2'])
        child.update({
            'root_request_id': 'req-1',
            'parent_request_id': 'req-1',
            'lineage_kind': contracts.LineageKind.MISSING_IDS.value,
            'lineage_depth': 1,
        })
        self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        with self.assertRaises(SyncRunError):
            self.store.supersede_with_children(
                request_id='req-1',
                children=[child],
                owner_token='owner-1',
            )


class CancellationAndLateReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store, _ = bootstrap_run(self.root)
        self.store.acquire_lease(owner_token='owner-1')

    def tearDown(self):
        self.tmp.cleanup()

    def test_cancel_intent_is_idempotent(self):
        self.assertTrue(self.store.cancel_intent(reason='user'))
        self.assertEqual(
            self.store.get_run()['status'],
            RunStatus.CANCEL_REQUESTED.value,
        )
        epoch = self.store.get_run()['cancel_epoch']
        self.assertFalse(self.store.cancel_intent())
        self.assertEqual(self.store.get_run()['cancel_epoch'], epoch)

    def test_canceled_prepared_attempt_never_dispatches(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.cancel_intent()
        with self.assertRaises(SyncRunError):
            self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        with self.assertRaises(SyncRunError):
            self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        changed = self.store.cancel_closeout(owner_token='owner-1')
        self.assertTrue(changed)
        self.assertEqual(self.store.get_run()['status'], RunStatus.CANCELLED.value)
        self.assertEqual(self.store.get_request('req-1')['status'], RequestStatus.CANCELLED.value)

    def test_late_success_after_cancel_only_audits(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')

        self.assertTrue(self.store.cancel_intent())
        accepted = self.store.record_success(
            attempt_id=attempt_id,
            owner_token='owner-1',
            accepted_items=['item-1', 'item-2'],
            usage_metadata={'total_tokens': 5},
        )
        self.assertFalse(accepted)

        request = self.store.get_request('req-1')
        self.assertEqual(request['status'], RequestStatus.CANCELLED.value)
        attempt = self.store.list_attempts(request_id='req-1')[0]
        self.assertEqual(attempt['status'], AttemptStatus.LATE_SUCCEEDED_IGNORED.value)
        late_receipts = self._late_receipts()
        self.assertEqual(len(late_receipts), 1)
        self.assertEqual(self.store.list_item_results(), [])
        self.assertEqual(len(self.store.pending_usage_outbox()), 1)
        self.assertTrue(self.store.cancel_closeout(owner_token='owner-1'))
        self.assertEqual(self.store.get_run()['status'], RunStatus.CANCELLED.value)

    def test_cancel_closeout_requires_committed_intent(self):
        with self.assertRaises(SyncRunError):
            self.store.cancel_closeout(owner_token='owner-1')

    def test_provider_cancel_confirmation_closes_attempt_and_request(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        self.store.cancel_intent()
        self.store.cancel_closeout(owner_token='owner-1')
        self.assertTrue(
            self.store.confirm_attempt_cancelled(
                attempt_id=attempt_id,
                owner_token='owner-1',
            )
        )
        self.assertEqual(self.store.get_request('req-1')['status'], RequestStatus.CANCELLED.value)
        self.assertTrue(self.store.cancel_closeout(owner_token='owner-1'))

    def _late_receipts(self):
        with self.store._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM late_receipts WHERE run_id = ?', (self.store.run_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def test_cancel_closeout_leaves_dispatched_until_receipt(self):
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        self.store.cancel_intent()
        self.store.cancel_closeout(owner_token='owner-1')
        attempt = self.store.list_attempts(request_id='req-1')[0]
        self.assertEqual(attempt['status'], AttemptStatus.CANCEL_REQUESTED.value)
        self.assertEqual(self.store.get_run()['status'], RunStatus.CANCEL_REQUESTED.value)
        self.assertEqual(self.store.get_request('req-1')['status'], RequestStatus.IN_FLIGHT.value)


class ProjectionAndIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store, _ = bootstrap_run(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exports_do_not_leak_sensitive_payload(self):
        requests_jsonl = self.store.export_requests_jsonl()
        run_manifest = self.store.export_run_manifest_json()
        events_jsonl = self.store.export_events_jsonl()
        for export in (requests_jsonl, run_manifest, events_jsonl):
            self.assertNotIn('user_prompt', export)
            self.assertNotIn('secret-prompt', export)
            self.assertNotIn('system_instruction', export)
            self.assertNotIn('translation_payload', export)
        self.assertIn('req-1', requests_jsonl)
        self.assertIn('plan_fingerprint', run_manifest)

    def test_event_jsonl_is_valid_jsonl(self):
        for line in self.store.export_events_jsonl().splitlines():
            self.assertIsInstance(json.loads(line), dict)

    def test_delete_artifact_removes_binding_but_preserves_file(self):
        artifact_path = self.store.run_dir / 'preview.json'
        artifact_path.write_text('{}\n', encoding='utf-8')
        self.store.put_artifact(
            kind='preview_manifest',
            relative_path=artifact_path.name,
            sha256_digest='a' * 64,
            schema_version=1,
        )

        self.assertTrue(self.store.delete_artifact(kind='preview_manifest'))
        self.assertIsNone(self.store.get_artifact(kind='preview_manifest'))
        self.assertTrue(artifact_path.is_file())
        self.assertFalse(self.store.delete_artifact(kind='preview_manifest'))

    def test_integrity_detects_event_and_attempt_disagreement(self):
        self.store.acquire_lease(owner_token='owner-1')
        self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        conn = sqlite3.connect(str(self.store.db_path))
        try:
            with conn:
                conn.execute(
                    'UPDATE requests SET attempt_count = 99 WHERE run_id = ? AND request_id = ?',
                    (self.store.run_id, 'req-1'),
                )
        finally:
            conn.close()
        violations = self.store.verify_integrity()
        self.assertTrue(any('attempt_count mismatch' in v for v in violations))

    def test_snapshot_counts_root_item_universe_once_after_derivation(self):
        self.store.acquire_lease(owner_token='owner-1')
        child = make_child_request(expected_ids=['item-2'])
        attempt_id = self.store.prepare_attempt(request_id='req-1', owner_token='owner-1')
        self.store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner-1')
        self.store.record_success(
            attempt_id=attempt_id,
            owner_token='owner-1',
            accepted_items=['item-1'],
            derived_requests=[child],
        )
        snapshot = self.store.build_snapshot()
        self.assertEqual(snapshot['progress']['items']['expected'], 2)
        self.assertEqual(snapshot['progress']['items']['accepted'], 1)

    def test_checkpoint_runs(self):
        self.store.checkpoint()

    def test_snapshot_buckets_sum_to_total(self):
        snapshot = self.store.build_snapshot()
        request_counts = snapshot['progress']['requests']
        total = request_counts['total']
        bucket_sum = sum(
            request_counts[field]
            for field in (
                'pending', 'in_flight', 'succeeded', 'retryable_failed',
                'terminal_failed', 'superseded', 'outcome_unknown', 'cancelled',
            )
        )
        self.assertEqual(total, bucket_sum)


if __name__ == '__main__':
    unittest.main()
