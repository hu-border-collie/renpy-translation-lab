# -*- coding: utf-8 -*-
"""Service and production backend seam tests for issue #347 P4."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from durable_sync_executor import ProviderFailure, ProviderOutcome
from sync_retry_policy import ExecutorPolicy
from sync_run_contracts import (
    ErrorCategory,
    ErrorCode,
    LineageKind,
    RunStatus,
    SyncRunError,
    build_run_id,
)
from sync_run_service import (
    ProductionSyncBackendAdapter,
    SyncRunService,
    build_production_sync_run_service,
    find_latest_run,
)
from sync_run_store import SyncRunStore


def plan_build():
    plan = {
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
    request = {
        'request_id': 'req-1',
        'plan_id': plan['plan_id'],
        'chunk_id': 'chunk-1',
        'system_instruction': 'system',
        'user_prompt': 'user',
        'response_schema': {},
        'expected_ids': ['item-1'],
        'capability_requirements': {},
        'generation_config': {},
        'transport_metadata': {},
        'context_assembly': {},
        'prompt_fingerprint': 'prompt-1',
        'request_fingerprint': 'request-1',
    }
    return {'plan': plan, 'requests': [request]}


def child_builder(parent, target_ids, suffix, lineage_kind):
    child = dict(parent)
    child['request_id'] = parent['request_id'] + suffix
    child['chunk_id'] = parent['chunk_id'] + suffix
    child['expected_ids'] = list(target_ids)
    child['prompt_fingerprint'] += suffix
    child['request_fingerprint'] += suffix
    child['transport_metadata'] = {
        'retry_parent_request_id': parent['request_id'],
        'retry_parent_chunk_id': parent['chunk_id'],
        'retry_lineage_kind': lineage_kind.value,
        'retry_item_ids': list(target_ids),
    }
    return child


class Backend:
    def __init__(self):
        self.calls = 0

    def send(self, request, *, attempt, timeout_seconds):
        self.calls += 1
        return ProviderOutcome(
            accepted_items={'item-1': {'translation': '一'}},
            usage_metadata={'total_tokens': 1},
        )

    def cancel(self, *, attempt):
        return True


class ReservationFactory:
    def preflight(self, _request):
        return {'estimated_cost': 0.1}

    def __call__(self, _store):
        return lambda _request: {'estimated_cost': 0.1}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'sync_runs'
        self.game = Path(self.tmp.name) / 'game'
        self.game.mkdir()
        self.backend = Backend()
        self.fresh = {
            'resume_allowed': True,
            'source': 'fresh',
            'profile': 'fresh',
            'config': 'fresh',
            'reasons': [],
        }
        self.service = SyncRunService(
            self.root,
            backend_factory=lambda _store: self.backend,
            derived_builder_factory=lambda _store: child_builder,
            freshness_reporter=lambda _store: dict(self.fresh),
            game_root=self.game,
            reuse_validator=lambda _item_id, translation: bool(translation),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_materializes_terminal_artifacts_and_status(self):
        snapshot = self.service.start(plan_build())
        self.assertEqual(snapshot['run_status'], RunStatus.COMPLETED.value)
        self.assertTrue(snapshot['changed'])
        self.assertEqual(self.backend.calls, 1)
        self.assertTrue(Path(snapshot['artifacts']['results_jsonl']).is_file())
        status = self.service.status(snapshot['run_id'])
        self.assertFalse(status['changed'])
        self.assertEqual(status['next_action'], 'check')
        self.assertEqual(status['freshness']['source'], 'fresh')

    def test_client_token_reopens_without_second_provider_call(self):
        first = self.service.start(plan_build(), client_token='stable-token')
        second = self.service.start(plan_build(), client_token='stable-token')
        self.assertEqual(first['run_id'], second['run_id'])
        self.assertEqual(self.backend.calls, 1)
        self.assertFalse(second['changed'])

    def test_missing_token_always_creates_new_run(self):
        first = self.service.start(plan_build())
        second = self.service.start(plan_build(), client_token='   ')
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertEqual(self.backend.calls, 2)

    def test_terminal_resume_only_repairs_artifacts(self):
        started = self.service.start(plan_build())
        resumed = self.service.resume(started['run_id'])
        self.assertEqual(resumed['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual(self.backend.calls, 1)
        self.assertFalse(resumed['changed'])

    def test_derive_reuses_strictly_equal_validated_results_without_model_call(self):
        source = self.service.start(plan_build())
        derived = self.service.derive(source['run_id'], plan_build())
        self.assertNotEqual(derived['run_id'], source['run_id'])
        self.assertEqual(derived['run_status'], RunStatus.COMPLETED.value)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(derived['progress']['attempts']['total'], 0)
        manifest = Path(derived['artifacts']['run_manifest_json']).read_text(
            encoding='utf-8'
        )
        self.assertIn(source['run_id'], manifest)

    def test_derive_lineage_refusal_never_redispatches_reusable_items(self):
        current = plan_build()
        current['requests'][0]['expected_ids'] = ['item-1', 'item-2']
        source, _created = SyncRunStore.bootstrap(
            self.root,
            build_run_id(),
            plan=current['plan'],
            requests=current['requests'],
        )
        owner = 'source-builder'
        source.acquire_lease(owner_token=owner)
        child = child_builder(
            current['requests'][0],
            ['item-2'],
            '--M-000000000000',
            LineageKind.MISSING_IDS,
        )
        attempt_id = source.prepare_attempt(request_id='req-1', owner_token=owner)
        source.dispatch_attempt(attempt_id=attempt_id, owner_token=owner)
        source.record_success(
            attempt_id=attempt_id,
            owner_token=owner,
            accepted_items={'item-1': {'translation': '一'}},
            derived_requests=[child],
        )
        child_attempt_id = source.prepare_attempt(
            request_id=child['request_id'],
            owner_token=owner,
        )
        source.dispatch_attempt(attempt_id=child_attempt_id, owner_token=owner)
        source.record_failure(
            attempt_id=child_attempt_id,
            owner_token=owner,
            error_category=ErrorCategory.AUTHENTICATION,
            error_reason_code='bad_credential',
        )
        source.finalize_run(owner_token=owner)
        source.release_lease(owner_token=owner)

        with (
            mock.patch.object(
                SyncRunStore,
                'lineage_budget_reason',
                return_value='root.max_derived_requests_exhausted',
            ),
            self.assertRaises(SyncRunError) as raised,
        ):
            self.service.derive(source.run_id, current)

        self.assertEqual(
            raised.exception.code,
            ErrorCode.SYNC_RUN_BUDGET_EXHAUSTED,
        )
        self.assertEqual(self.backend.calls, 0)
        target = SyncRunStore(self.root, raised.exception.safe_details['run_id'])
        self.assertEqual(target.get_run()['status'], RunStatus.FAILED.value)
        self.assertEqual(target.list_attempts(), [])

    def test_exclude_unknown_conflict_is_structured(self):
        current = plan_build()
        source, _created = SyncRunStore.bootstrap(
            self.root,
            build_run_id(),
            plan=current['plan'],
            requests=current['requests'],
        )
        owner = 'unknown-builder'
        source.acquire_lease(owner_token=owner)
        attempt_id = source.prepare_attempt(request_id='req-1', owner_token=owner)
        source.dispatch_attempt(attempt_id=attempt_id, owner_token=owner)
        source.mark_outcome_unknown(attempt_id=attempt_id, owner_token=owner)
        source.finalize_run(owner_token=owner)
        source.release_lease(owner_token=owner)

        with self.assertRaises(SyncRunError) as raised:
            self.service.derive(
                source.run_id,
                current,
                exclude_unknown=True,
            )

        self.assertEqual(
            raised.exception.code,
            ErrorCode.SYNC_RUN_OUTCOME_UNKNOWN,
        )
        self.assertEqual(raised.exception.safe_details['unknown_count'], 1)
        self.assertEqual(self.backend.calls, 0)

    def test_latest_ignores_legacy_preview_directory(self):
        started = self.service.start(plan_build())
        legacy = self.root / '20260826T120000.000000Z'
        legacy.mkdir(parents=True)
        (legacy / 'preview.json').write_text('{}', encoding='utf-8')
        self.assertEqual(find_latest_run(self.root), started['run_id'])
        latest = self.service.status(latest=True)
        self.assertEqual(latest['run_id'], started['run_id'])

    def test_freshness_refusal_happens_before_provider_call(self):
        self.fresh.update({
            'resume_allowed': False,
            'source': 'stale',
            'reasons': ['source_snapshot_changed'],
        })
        with self.assertRaises(SyncRunError) as ctx:
            self.service.start(plan_build())
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_FRESHNESS_MISMATCH)
        self.assertEqual(self.backend.calls, 0)

    def test_hard_cost_cap_requires_preflight_pricing(self):
        with self.assertRaises(SyncRunError) as ctx:
            self.service.start(
                plan_build(),
                policy=ExecutorPolicy.from_mapping({'max_estimated_cost': 1.0}),
            )
        self.assertEqual(ctx.exception.code, ErrorCode.SYNC_RUN_BUDGET_EXHAUSTED)
        self.assertEqual(list(self.root.glob('sync-run-v1-*')), [])

    def test_cost_preflight_factory_allows_run(self):
        service = SyncRunService(
            self.root,
            backend_factory=lambda _store: self.backend,
            derived_builder_factory=lambda _store: child_builder,
            reservation_factory=ReservationFactory(),
            freshness_reporter=lambda _store: dict(self.fresh),
        )
        result = service.start(
            plan_build(),
            policy=ExecutorPolicy.from_mapping({'max_estimated_cost': 1.0}),
        )
        self.assertEqual(result['run_status'], RunStatus.COMPLETED.value)


class ProductionBackendAdapterTests(unittest.TestCase):
    def test_one_call_result_is_validated_into_provider_outcome(self):
        calls = []

        def generate(request, timeout):
            calls.append((request['request_id'], timeout))
            return {
                'response_payload': {'id': 'response-1'},
                'parsed': {'translations': [
                    {'id': 'item-1', 'translation': '一'},
                    {'id': 'item-2', 'translation': '二'},
                ]},
                'usage_metadata': {'total_tokens': 3},
            }

        items = {
            'item-1': {'id': 'item-1', 'text': 'one'},
            'item-2': {'id': 'item-2', 'text': 'two'},
        }
        adapter = ProductionSyncBackendAdapter(
            generate,
            lambda _request, ids: [items[item_id] for item_id in ids],
        )
        request = plan_build()['requests'][0] | {'expected_ids': ['item-1', 'item-2']}
        result = adapter.send(request, attempt={}, timeout_seconds=30)
        self.assertEqual(calls, [('req-1', 30)])
        self.assertEqual(set(result.accepted_items), {'item-1', 'item-2'})
        self.assertEqual(result.normalized_payload['translations'][0]['id'], 'item-1')

    def test_empty_translation_is_never_accepted_as_success(self):
        adapter = ProductionSyncBackendAdapter(
            lambda _request, _timeout: {
                'parsed': {
                    'translations': [
                        {'id': 'item-1', 'translation': '   '}
                    ]
                }
            },
            lambda _request, _ids: [{'id': 'item-1', 'text': 'one'}],
            translation_validator=lambda _item, _translation: (True, 'OK'),
        )
        request = plan_build()['requests'][0]

        with self.assertRaises(ProviderFailure) as raised:
            adapter.send(request, attempt={}, timeout_seconds=30)

        self.assertEqual(
            raised.exception.category,
            ErrorCategory.INVALID_STRUCTURED_RESPONSE,
        )
        self.assertEqual(
            raised.exception.reason_code,
            'result_empty_translation',
        )

    def test_production_service_freezes_offline_targets_before_dispatch(self):
        class Context:
            plan_build = plan_build()
            routing_plan = object()
            route = object()

            @staticmethod
            def item_resolver(_request, ids):
                return [{'id': item_id, 'text': 'one'} for item_id in ids]

            @staticmethod
            def context_resolver(_request):
                return {}

            @staticmethod
            def validate_translation(_item, _translation):
                return True, 'OK'

            @staticmethod
            def validate_reused_translation(_item_id, payload):
                return bool(payload)

            @staticmethod
            def durable_targets_payload(*, run_id=''):
                return {
                    'schema_version': 1,
                    'run_id': run_id,
                    'plan_id': 'plan-1111111111111111',
                    'plan_fingerprint': 'b' * 16,
                    'files': {},
                    'chunks': [],
                }

        with tempfile.TemporaryDirectory() as tmp:
            service = build_production_sync_run_service(
                Path(tmp) / 'runs',
                Context(),
            )
            response = {
                'parsed': {
                    'translations': [{'id': 'item-1', 'translation': '一'}]
                },
                'usage_metadata': {'total_tokens': 1},
            }
            with mock.patch(
                'gemini_translate_batch.run_sync_request',
                return_value=response,
            ) as generate:
                snapshot = service.start(Context.plan_build)

            generate.assert_called_once()
            self.assertEqual(generate.call_args.kwargs['retry_attempts'], 1)
            self.assertFalse(
                generate.call_args.kwargs['allow_credential_rotation']
            )
            target_path = Path(snapshot['artifacts']['targets_json'])
            self.assertTrue(target_path.is_file())
            self.assertIn(snapshot['run_id'], target_path.read_text(encoding='utf-8'))
            self.assertEqual(snapshot['freshness']['source'], 'fresh')


if __name__ == '__main__':
    unittest.main()
