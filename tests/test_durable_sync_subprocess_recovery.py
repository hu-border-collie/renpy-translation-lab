# -*- coding: utf-8 -*-
"""Real subprocess-abort recovery tests for #347 F03/F04/F06."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from durable_sync_executor import DurableSyncExecutor, ProviderOutcome
from sync_retry_policy import ExecutorPolicy
from sync_run_contracts import RequestStatus, RunStatus, build_run_id
from sync_run_store import SyncRunStore


def _plan():
    return {
        'schema_version': 1,
        'plan_id': 'plan-1111111111111111',
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


def _request(request_id, item_id):
    return {
        'request_id': request_id,
        'plan_id': 'plan-1111111111111111',
        'chunk_id': request_id.replace('req-', 'chunk-'),
        'system_instruction': 'system',
        'user_prompt': 'user',
        'response_schema': {},
        'expected_ids': [item_id],
        'capability_requirements': {},
        'generation_config': {},
        'transport_metadata': {},
        'context_assembly': {},
        'prompt_fingerprint': f'prompt-{request_id}',
        'request_fingerprint': f'fingerprint-{request_id}',
    }


def _child_builder(parent, target_ids, suffix, lineage_kind):
    child = dict(parent)
    child['request_id'] = parent['request_id'] + suffix
    child['chunk_id'] = parent['chunk_id'] + suffix
    child['expected_ids'] = list(target_ids)
    child['prompt_fingerprint'] += suffix
    child['request_fingerprint'] += suffix
    return child


class _Backend:
    def __init__(self):
        self.calls = []

    def send(self, request, *, attempt, timeout_seconds):
        self.calls.append((request['request_id'], attempt['attempt_id']))
        return ProviderOutcome(accepted_items={
            item_id: {'translation': item_id}
            for item_id in request['expected_ids']
        })

    def cancel(self, *, attempt):
        return False


class SubprocessRecoveryTests(unittest.TestCase):
    worker = Path(__file__).parent / 'fixtures' / 'durable_sync_crash_worker.py'

    def _run_case(self, mode):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / 'runs'
        run_id = build_run_id()
        policy = ExecutorPolicy()
        store, _created = SyncRunStore.bootstrap(
            root,
            run_id,
            plan=_plan(),
            requests=[_request('req-1', 'item-1'), _request('req-2', 'item-2')],
            executor_policy=policy.to_dict(),
        )
        completed = subprocess.run(
            [sys.executable, str(self.worker), str(root), run_id, mode],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertIn(completed.returncode, {91, 92, 93}, completed.stderr)
        time.sleep(0.08)
        backend = _Backend()
        snapshot = DurableSyncExecutor(
            store,
            backend,
            derived_request_builder=_child_builder,
            policy=policy,
        ).run()
        return store, backend, snapshot

    def test_prepared_attempt_reuses_identity_after_abrupt_exit(self):
        store, backend, snapshot = self._run_case('prepared')
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        attempts = store.list_attempts(request_id='req-1')
        self.assertEqual(len(attempts), 1)
        self.assertEqual(backend.calls[0][1], attempts[0]['attempt_id'])
        self.assertEqual([call[0] for call in backend.calls], ['req-1', 'req-2'])

    def test_dispatched_attempt_becomes_unknown_without_duplicate_call(self):
        store, backend, snapshot = self._run_case('dispatched')
        self.assertEqual(
            snapshot['run_status'], RunStatus.COMPLETED_WITH_ERRORS.value
        )
        self.assertEqual([call[0] for call in backend.calls], ['req-2'])
        self.assertEqual(
            store.get_request('req-1')['status'],
            RequestStatus.OUTCOME_UNKNOWN.value,
        )

    def test_committed_success_is_not_recalled_after_abrupt_exit(self):
        store, backend, snapshot = self._run_case('committed')
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual([call[0] for call in backend.calls], ['req-2'])
        self.assertEqual(
            store.get_request('req-1')['status'],
            RequestStatus.SUCCEEDED.value,
        )


if __name__ == '__main__':
    unittest.main()
