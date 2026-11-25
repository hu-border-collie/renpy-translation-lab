# -*- coding: utf-8 -*-
"""End-to-end offline check/preview/apply coverage for issue #347 P5."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from execution.durable_sync_executor import ProviderOutcome, TranslationPlanDerivedRequestBuilder
import gemini_translate_batch as batch
from execution.sync_run_contracts import ErrorCode, SyncRunError, canonical_json
from execution.sync_run_service import SyncRunService
from execution.sync_run_store import SyncRunStore
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


def _runtime_fixture(root, *, rag_enabled=False):
    tl_dir = root / 'game' / 'tl' / 'schinese'
    log_dir = root / 'logs'
    return mock.patch.multiple(
        runtime, BASE_DIR=str(root), TL_DIR=str(tl_dir), TL_SUBDIR='game/tl/schinese',
        LOG_DIR=str(log_dir), PROGRESS_LOG=str(log_dir / 'progress.json'),
        SYNC_BACKEND='litellm', MODELS=['openai/test-model'], CURRENT_MODEL_INDEX=0,
        PREP_ENABLED=False, INCLUDE_FILES=[], INCLUDE_PREFIXES=[], SYNC_RAG_ENABLED=rag_enabled,
        SYNC_STORY_MEMORY_ENABLED=False, SYNC_RAG_UPDATE_ON_SUCCESS=True,
        SYNC_RAG_STORE_DIR=str(root / 'rag'), SYNC_RAG_OUTPUT_DIMENSIONALITY=3,
        SYNC_RAG_SEGMENT_LINES=1, _SYNC_RAG_STORE=None, MODEL_ROUTING_CONFIG=None,
        load_config=mock.DEFAULT, load_translator_settings=mock.DEFAULT, load_glossary=mock.DEFAULT,
    )


def _start_run(root):
    run_root = root / 'logs' / 'sync_runs'
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
    return SyncRunStore(run_root, started['run_id'])


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
            with _runtime_fixture(root):
                store = _start_run(root)
                run_id = store.run_id

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
                    batch.apply_results(run_id, force=True)
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
                    run_id,
                )
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')

                checked_bytes = check_path.read_bytes()
                check_path.write_bytes(checked_bytes + b'\n')
                with self.assertRaisesRegex(SystemExit, 'check_manifest changed'):
                    batch.apply_results(run_id, force=True)
                check_path.write_bytes(checked_bytes)

                context_file.write_text('# context v2\n', encoding='utf-8')
                with self.assertRaisesRegex(
                    batch.cli_contract.MachineContractError,
                    'source snapshot changed',
                ) as stale:
                    batch.apply_results(run_id, force=True)
                self.assertEqual(
                    stale.exception.code_name,
                    'SYNC_RUN_FRESHNESS_MISMATCH',
                )
                self.assertEqual(target.read_text(encoding='utf-8'), '    "Hello"\n')
                context_file.write_text('# context v1\n', encoding='utf-8')

                preview_bytes = preview_path.read_bytes()
                applied = batch.apply_results(run_id, force=True)
                self.assertEqual(applied['state'], 'applied')
                self.assertEqual(target.read_text(encoding='utf-8'), '    "你好"\n')
                applied_again = batch.apply_results(run_id, force=True)
                self.assertEqual(applied_again['last_apply_result'], 'already_applied')
                self.assertEqual(target.read_text(encoding='utf-8'), '    "你好"\n')
                self.assertEqual(preview_path.read_bytes(), preview_bytes)
                self.assertEqual(store.get_artifact(kind='preview_manifest'), preview_artifact)
                self.assertEqual(store.verify_integrity(), [])

                with batch._durable_sync_writeback_lock(store):
                    for command in (batch.check_durable_sync_results, batch.apply_durable_sync_results):
                        with self.assertRaises(SyncRunError) as blocked_by_lock:
                            command(store)
                        self.assertEqual(blocked_by_lock.exception.code, ErrorCode.SYNC_RUN_BUSY)

    def _rag_run(self, root, files):
        tl_dir = root / 'game' / 'tl' / 'schinese'
        tl_dir.mkdir(parents=True)
        for path, length in files.items():
            (tl_dir / f'{path}.rpy').write_text(
                f'translate schinese script_{path}:\n' + ''.join(
                    f'    # e "{path} source {index}"\n    e "{path} source {index}"\n'
                    for index in range(length)
                ),
                encoding='utf-8',
            )
        with _runtime_fixture(root):
            store = _start_run(root)
            batch.check_durable_sync_results(store)
        return tl_dir, store

    def test_apply_resumes_all_files_after_process_exit_in_first_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, store = self._rag_run(root, {'a': 17, 'b': 1})
            child = subprocess.run(
                [sys.executable, '-c', """
import os
import sys
from pathlib import Path
from unittest.mock import patch

import gemini_translate_batch as batch
import translator_runtime as runtime
from execution.sync_run_store import SyncRunStore
from tests.execution.test_durable_sync_workflow import _runtime_fixture

root = Path(sys.argv[1])
store = SyncRunStore(root / 'logs' / 'sync_runs', sys.argv[2])

def embed(texts, *args, **kwargs):
    if len(texts) == 16:
        return [[1.0, 0.0, 0.0]] * len(texts)
    os._exit(72)

with _runtime_fixture(root, rag_enabled=True), patch.object(runtime, 'embed_texts', side_effect=embed):
    batch.apply_durable_sync_results(store)
""", tmp, store.run_id],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                timeout=15,
            )
            self.assertEqual(child.returncode, 72, msg=child.stdout + child.stderr)
            self.assertEqual(store.verify_integrity(), [])
            translated = [(tl_dir / name).read_bytes() for name in ('a.rpy', 'b.rpy')]
            self.assertTrue(all('你好'.encode() in script for script in translated))

            def embed(texts, *args, **kwargs):
                return [[0.0, 1.0, 0.0]] * len(texts)

            with (
                _runtime_fixture(root, rag_enabled=True),
                mock.patch.object(runtime, 'embed_texts', side_effect=embed) as provider,
                mock.patch.object(runtime.sync_translation_preview, 'atomic_write_many_lines') as write,
            ):
                result = batch.apply_durable_sync_results(store)
                again = batch.apply_durable_sync_results(store)
            self.assertEqual([len(call.args[0]) for call in provider.call_args_list], [1, 1])
            self.assertEqual(result['last_apply_result'], 'already_applied')
            self.assertEqual(result['_sync_rag_recovery'], {'pending_files': 0, 'resumed_embeddings': 16})
            self.assertNotIn('_sync_rag_recovery', again)
            self.assertEqual(again['applied_at'], result['applied_at'])
            write.assert_not_called()
            history = runtime.JsonRagStore(root / 'rag')
            self.assertEqual(history.count_history(), 18)
            self.assertEqual(len(history.history_ids_for_file('b.rpy')), 1)

    def test_apply_recovers_when_history_lock_prevents_creating_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, store = self._rag_run(root, {'a': 1})
            rag_store = runtime.JsonRagStore(root / 'rag')
            with (
                _runtime_fixture(root, rag_enabled=True),
                mock.patch.object(runtime, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as provider,
            ):
                with rag_store.write_session('blocked_apply_fixture'):
                    result = batch.apply_durable_sync_results(store)
                self.assertEqual(result['last_apply_result'], 'applied')
                self.assertEqual(result['_sync_rag_recovery'], {'pending_files': 1, 'resumed_embeddings': 0})
                provider.assert_not_called()
                self.assertEqual(rag_store.count_history(), 0)
                resumed = batch.apply_durable_sync_results(store)
            self.assertEqual(resumed['last_apply_result'], 'already_applied')
            self.assertNotIn('_sync_rag_recovery', resumed)
            provider.assert_called_once()
            self.assertEqual(runtime.JsonRagStore(root / 'rag').count_history(), 1)

    def test_apply_resume_survives_failure_to_save_final_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, store = self._rag_run(root, {'a': 17})
            atomic_write_json = runtime.sync_translation_preview.atomic_write_json

            def fail_final_receipt(path, payload, **kwargs):
                if payload.get('state') == 'applied':
                    raise OSError('fixture interrupted final receipt')
                return atomic_write_json(path, payload, **kwargs)

            with _runtime_fixture(root, rag_enabled=True):
                with (
                    mock.patch.object(runtime, 'embed_texts', side_effect=[
                        [[1.0, 0.0, 0.0]] * 16, OSError('fixture provider unavailable'),
                    ]),
                    mock.patch.object(runtime.sync_translation_preview, 'atomic_write_json',
                                      side_effect=fail_final_receipt),
                    self.assertRaisesRegex(OSError, 'interrupted final receipt'),
                ):
                    batch.apply_durable_sync_results(store)
                self.assertEqual(store.verify_integrity(), [])
                with mock.patch.object(runtime, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as provider:
                    resumed = batch.apply_durable_sync_results(store)
            self.assertEqual(resumed['last_apply_result'], 'already_applied')
            self.assertEqual(resumed['_sync_rag_recovery'], {'pending_files': 0, 'resumed_embeddings': 16})
            provider.assert_called_once()
            self.assertEqual(len(provider.call_args.args[0]), 1)
            self.assertEqual(runtime.JsonRagStore(root / 'rag').count_history(), 17)


if __name__ == '__main__':
    unittest.main()
