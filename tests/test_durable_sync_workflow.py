# -*- coding: utf-8 -*-
"""End-to-end offline check/preview/apply coverage for issue #347 P5."""

from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from durable_sync_executor import ProviderOutcome, TranslationPlanDerivedRequestBuilder
import gemini_translate_batch as batch
from sync_run_contracts import canonical_json
from sync_run_service import SyncRunService
from sync_run_store import SyncRunStore
import translator_runtime as runtime


class _Backend:
    def send(self, request, *, attempt, timeout_seconds):
        return ProviderOutcome(
            accepted_items={
                item_id: {'translation': '你好'}
                for item_id in request.get('expected_ids') or []
            },
            normalized_payload={
                'translations': [
                    {'id': item_id, 'translation': '你好'}
                    for item_id in request.get('expected_ids') or []
                ]
            },
            usage_metadata={'total_tokens': 2},
        )

    def cancel(self, *, attempt):
        return False


class DurableSyncWorkflowTests(unittest.TestCase):
    def test_completed_run_requires_check_then_bound_preview_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            target = tl_dir / 'script.rpy'
            target.write_text('    "Hello"\n', encoding='utf-8')
            context_file = tl_dir / 'context.rpy'
            context_file.write_text('# context v1\n', encoding='utf-8')
            log_dir = root / 'logs'
            run_root = log_dir / 'sync_runs'
            patches = (
                mock.patch.object(runtime, 'BASE_DIR', str(root)),
                mock.patch.object(runtime, 'TL_DIR', str(tl_dir)),
                mock.patch.object(runtime, 'TL_SUBDIR', 'game/tl/schinese'),
                mock.patch.object(runtime, 'LOG_DIR', str(log_dir)),
                mock.patch.object(runtime, 'PROGRESS_LOG', str(log_dir / 'progress.json')),
                mock.patch.object(runtime, 'SYNC_BACKEND', 'litellm'),
                mock.patch.object(runtime, 'MODELS', ['openai/test-model']),
                mock.patch.object(runtime, 'CURRENT_MODEL_INDEX', 0),
                mock.patch.object(runtime, 'PREP_ENABLED', False),
                mock.patch.object(runtime, 'INCLUDE_FILES', []),
                mock.patch.object(runtime, 'INCLUDE_PREFIXES', []),
                mock.patch.object(runtime, 'SYNC_RAG_ENABLED', False),
                mock.patch.object(runtime, 'SYNC_STORY_MEMORY_ENABLED', False),
                mock.patch.object(runtime, 'load_config'),
                mock.patch.object(runtime, 'load_translator_settings'),
                mock.patch.object(runtime, 'load_glossary'),
                mock.patch.object(runtime, 'load_progress', return_value={}),
                mock.patch.object(runtime, 'maybe_update_sync_rag_store'),
            )
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                context = runtime.prepare_sync_translation_execution_context(
                    require_provider=False
                )

                def artifacts(store):
                    payload = context.durable_targets_payload(run_id=store.run_id)
                    return [{
                        'kind': 'targets_json',
                        'relative_path': 'targets.json',
                        'content': canonical_json(payload) + '\n',
                        'schema_version': 1,
                    }]

                service = SyncRunService(
                    run_root,
                    backend_factory=lambda _store: _Backend(),
                    derived_builder_factory=lambda _store: (
                        TranslationPlanDerivedRequestBuilder(
                            context.item_resolver,
                            context_resolver=context.context_resolver,
                        )
                    ),
                    freshness_reporter=lambda _store: {
                        'resume_allowed': True,
                        'source': 'fresh',
                        'profile': 'fresh',
                        'config': 'fresh',
                        'reasons': [],
                    },
                    reuse_validator=context.validate_reused_translation,
                    run_artifact_provider=artifacts,
                    run_artifact_kinds=('targets_json',),
                )
                started = service.start(context.plan_build)
                store = SyncRunStore(run_root, started['run_id'])

                with self.assertRaisesRegex(
                    batch.cli_contract.MachineContractError,
                    'preview',
                ):
                    batch.apply_durable_sync_results(store)

                checked = batch.check_durable_sync_results(store)
                self.assertEqual(
                    checked['last_check_summary']['writeback_gate']['decision'],
                    'allow',
                )
                preview_artifact = store.get_artifact(kind='preview_manifest')
                self.assertIsNotNone(preview_artifact)
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')

                blocked = dict(checked)
                blocked['last_check_summary'] = {
                    **dict(checked['last_check_summary']),
                    'writeback_gate': {'decision': 'deny'},
                }
                with mock.patch.object(batch, 'check_results', return_value=blocked):
                    batch.check_durable_sync_results(store)
                self.assertIsNone(store.get_artifact(kind='preview_manifest'))
                with self.assertRaisesRegex(
                    batch.cli_contract.MachineContractError,
                    'preview_manifest',
                ):
                    batch.apply_results(started['run_id'], force=True)
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')

                checked = batch.check_durable_sync_results(store)
                preview_artifact = store.get_artifact(kind='preview_manifest')
                self.assertIsNotNone(preview_artifact)
                preview_path = store.resolve_artifact_path(
                    preview_artifact['relative_path']
                )
                with self.assertRaisesRegex(
                    SystemExit,
                    'Durable Sync previews must be applied',
                ):
                    batch.legacy.apply_sync_translation_preview(preview_path)
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')

                check_path = Path(checked['_manifest_path'])
                with self.assertRaises(
                    batch.cli_contract.MachineContractError,
                ) as direct_apply:
                    batch.apply_results(str(check_path), force=True)
                self.assertEqual(
                    direct_apply.exception.code_name,
                    'DURABLE_SYNC_APPLY_REQUIRES_RUN',
                )
                self.assertEqual(
                    direct_apply.exception.details['run_id'],
                    started['run_id'],
                )
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')

                checked_bytes = check_path.read_bytes()
                check_path.write_bytes(checked_bytes + b'\n')
                with self.assertRaisesRegex(SystemExit, 'check_manifest changed'):
                    batch.apply_results(started['run_id'], force=True)
                check_path.write_bytes(checked_bytes)

                context_file.write_text('# context v2\n', encoding='utf-8')
                with self.assertRaisesRegex(
                    batch.cli_contract.MachineContractError,
                    'source snapshot changed',
                ) as stale:
                    batch.apply_results(started['run_id'], force=True)
                self.assertEqual(
                    stale.exception.code_name,
                    'SYNC_RUN_FRESHNESS_MISMATCH',
                )
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')
                context_file.write_text('# context v1\n', encoding='utf-8')

                applied = batch.apply_results(started['run_id'], force=True)
                self.assertEqual(applied['state'], 'applied')
                self.assertEqual(target.read_text(encoding='utf-8'), '    "你好"\n')
                applied_again = batch.apply_results(started['run_id'], force=True)
                self.assertEqual(applied_again['last_apply_result'], 'already_applied')
                self.assertEqual(target.read_text(encoding='utf-8'), '    "你好"\n')


if __name__ == '__main__':
    unittest.main()
