# -*- coding: utf-8 -*-
"""Durable result artifact and usage outbox tests for issue #347 P3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atomic_io import atomic_write_text, file_sha256
from durable_sync_executor import DurableSyncExecutor, ProviderOutcome
import model_usage_ledger
from sync_result_export import (
    RESULTS_FILENAME,
    build_result_rows,
    deliver_usage_outbox,
    export_run_artifacts,
    render_results_jsonl,
)
from sync_retry_policy import ExecutorPolicy
from sync_run_contracts import LineageKind, RunStatus, build_run_id
from sync_run_store import SyncRunStore


def make_plan():
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


def make_request():
    return {
        'request_id': 'req-1',
        'plan_id': 'plan-1111111111111111',
        'chunk_id': 'chunk-1',
        'system_instruction': 'system secret prompt',
        'user_prompt': 'user secret prompt',
        'response_schema': {},
        'expected_ids': ['item-1', 'item-2'],
        'capability_requirements': {},
        'generation_config': {},
        'transport_metadata': {},
        'context_assembly': {},
        'prompt_fingerprint': 'prompt-1',
        'request_fingerprint': 'request-1',
    }


def child_builder(parent, target_ids, suffix, lineage_kind):
    child = dict(parent)
    child['request_id'] = parent['request_id'] + suffix
    child['chunk_id'] = parent['chunk_id'] + suffix
    child['expected_ids'] = list(target_ids)
    child['prompt_fingerprint'] = 'prompt-' + suffix
    child['request_fingerprint'] = 'request-' + suffix
    child['transport_metadata'] = {
        'retry_parent_request_id': parent['request_id'],
        'retry_parent_chunk_id': parent['chunk_id'],
        'retry_lineage_kind': lineage_kind.value,
        'retry_item_ids': list(target_ids),
    }
    return child


class PartialBackend:
    def __init__(self):
        self.calls = []

    def send(self, request, *, attempt, timeout_seconds):
        self.calls.append(request['request_id'])
        if request['request_id'] == 'req-1':
            return ProviderOutcome(
                accepted_items={'item-1': {'translation': '一'}},
                response_payload={'id': 'provider-first', 'text': 'safe first'},
                normalized_payload={'translations': [{'id': 'item-1', 'translation': '一'}]},
                contract_diagnostics={'complete': False},
                usage_metadata={'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
            )
        return ProviderOutcome(
            accepted_items={'item-2': {'translation': '二'}},
            response_payload={'id': 'provider-retry', 'text': 'safe retry'},
            normalized_payload={'translations': [{'id': 'item-2', 'translation': '二'}]},
            contract_diagnostics={'complete': True},
            usage_metadata={'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        )

    def cancel(self, *, attempt):
        return True


class SyncResultExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.run_root = self.base / 'runs'
        self.game_root = self.base / 'game'
        self.game_root.mkdir()
        self.policy = ExecutorPolicy()
        self.store, _ = SyncRunStore.bootstrap(
            self.run_root,
            build_run_id(),
            plan=make_plan(),
            requests=[make_request()],
            executor_policy=self.policy.to_dict(),
        )
        self.backend = PartialBackend()
        self.snapshot = DurableSyncExecutor(
            self.store,
            self.backend,
            derived_request_builder=child_builder,
            policy=self.policy,
            provider='fake',
            model='fake-model',
            reservation_provider=lambda _request: {'estimated_cost': 0.01},
        ).run()

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_merge_lineage_into_one_root_result(self):
        self.assertEqual(self.snapshot['run_status'], RunStatus.COMPLETED.value)
        rows = build_result_rows(self.store)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['key'], 'chunk-1')
        self.assertEqual(row['accepted_ids'], ['item-1', 'item-2'])
        self.assertEqual(row['unresolved_ids'], [])
        self.assertEqual(
            row['normalized_response']['translations'],
            [
                {'id': 'item-1', 'translation': '一'},
                {'id': 'item-2', 'translation': '二'},
            ],
        )
        self.assertEqual(len(row['provider_response_attempts']), 2)
        self.assertEqual(
            row['provider_response_attempts'][1]['kind'],
            LineageKind.MISSING_IDS.value,
        )
        self.assertEqual(len(row['row_sha256']), 64)

    def test_export_is_byte_stable_and_hash_bound(self):
        first = export_run_artifacts(self.store)
        first_bytes = (self.store.run_dir / RESULTS_FILENAME).read_bytes()
        second = export_run_artifacts(self.store)
        second_bytes = (self.store.run_dir / RESULTS_FILENAME).read_bytes()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            first['artifacts']['results_jsonl']['sha256'],
            second['artifacts']['results_jsonl']['sha256'],
        )
        self.assertEqual(self.store.verify_integrity(), [])
        sidecar = (self.store.run_dir / 'results.jsonl.sha256').read_text(
            encoding='utf-8'
        ).strip()
        self.assertEqual(sidecar, file_sha256(self.store.run_dir / RESULTS_FILENAME))

    def test_replace_before_artifact_registration_is_repaired(self):
        expected_text = render_results_jsonl(build_result_rows(self.store))
        atomic_write_text(self.store.run_dir / RESULTS_FILENAME, expected_text)
        self.assertIsNone(self.store.get_artifact(kind='results_jsonl'))
        exported = export_run_artifacts(self.store)
        self.assertEqual(
            exported['artifacts']['results_jsonl']['sha256'],
            file_sha256(self.store.run_dir / RESULTS_FILENAME),
        )

    def test_registered_artifact_tampering_is_detected(self):
        export_run_artifacts(self.store)
        atomic_write_text(self.store.run_dir / RESULTS_FILENAME, '{"tampered":true}\n')
        violations = self.store.verify_integrity()
        self.assertTrue(any('results_jsonl sha256 mismatch' in item for item in violations))

    def test_audit_exports_do_not_materialize_prompts(self):
        export_run_artifacts(self.store)
        for filename in ('requests.jsonl', 'run_manifest.json', 'events.jsonl'):
            text = (self.store.run_dir / filename).read_text(encoding='utf-8')
            self.assertNotIn('system secret prompt', text)
            self.assertNotIn('user secret prompt', text)

    def test_usage_outbox_replays_and_deduplicates_after_ack_gap(self):
        first = deliver_usage_outbox(self.store, game_root=self.game_root)
        self.assertEqual(first['inserted'], 2)
        self.assertEqual(first['pending_after'], 0)
        ledger = model_usage_ledger.UsageLedger(self.game_root)
        self.assertEqual(len(ledger.load()['records']), 2)

        with self.store._tx() as conn:
            conn.execute(
                'UPDATE usage_outbox SET delivered_at = NULL WHERE run_id = ?',
                (self.store.run_id,),
            )
        replay = deliver_usage_outbox(self.store, game_root=self.game_root)
        self.assertEqual(replay['inserted'], 0)
        self.assertEqual(replay['duplicates'], 2)
        self.assertEqual(replay['pending_after'], 0)
        self.assertEqual(len(ledger.load()['records']), 2)
        self.assertEqual(
            {record['dedupe_key'] for record in ledger.load()['records']},
            {
                f'usage:{attempt["attempt_id"]}'
                for attempt in self.store.list_attempts()
            },
        )


if __name__ == '__main__':
    unittest.main()
