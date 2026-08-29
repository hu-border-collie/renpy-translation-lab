# -*- coding: utf-8 -*-
"""P4 exit tests for plan diffs, dispatch freshness, and legacy diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
import sync_translation_preview
import translation_plan
import translator_runtime as runtime
from tests.test_translation_plan import build_fixture_plan


def _batch_manifest(tmp_dir):
    build = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
    jobs = json.loads(
        (
            Path(__file__).parent
            / 'fixtures'
            / 'translation_plan_minimal'
            / 'inputs'
            / 'file_jobs.json'
        ).read_text(encoding='utf-8')
    )
    items_by_file = {job['file_rel_path']: job['tasks'] for job in jobs}
    chunks = []
    offsets = {}
    for plan_chunk, request in zip(build.plan.chunks, build.requests):
        start = offsets.get(plan_chunk.file_rel_path, 0)
        end = start + len(request.expected_ids)
        offsets[plan_chunk.file_rel_path] = end
        chunks.append({
            'key': request.chunk_id,
            'file_rel_path': plan_chunk.file_rel_path,
            'file_path': plan_chunk.file_path,
            'items': items_by_file[plan_chunk.file_rel_path][start:end],
            **request.to_dict(),
        })
    requests_path = Path(tmp_dir) / 'requests.jsonl'
    rows = [batch.build_batch_request(chunk, model='fixture-model') for chunk in chunks]
    requests_path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
        encoding='utf-8',
    )
    return {
        'batch_model': 'fixture-model',
        'input_jsonl_path': str(requests_path),
        'translation_plan': build.plan.to_dict(),
        'chunks': chunks,
    }, build


class DispatchFreshnessTests(unittest.TestCase):
    def _validate_current_batch(self, manifest, build, *, operation='submit'):
        with (
            mock.patch.object(batch, 'collect_pending_file_jobs', return_value=[]),
            mock.patch.object(
                batch,
                '_batch_plan_source_identity',
                return_value=translation_plan.SourceIdentity.from_dict(
                    build.plan.source_identity
                ),
            ),
        ):
            return batch.validate_batch_translation_plan_before_dispatch(
                manifest,
                operation=operation,
            )

    def test_sync_source_or_adapter_stale_refuses_before_dispatch(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        current = dict(build.plan.source_identity)
        current['adapter_version'] = 'changed-adapter'
        with mock.patch.object(runtime, 'current_sync_source_identity', return_value=current):
            with self.assertRaisesRegex(RuntimeError, 'adapter_version_changed'):
                runtime.validate_sync_translation_plan_before_dispatch(build)

    def test_sync_request_tamper_refuses_before_dispatch(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        build.requests[0].user_prompt += '\ntampered'
        with mock.patch.object(
            runtime,
            'current_sync_source_identity',
            return_value=dict(build.plan.source_identity),
        ):
            with self.assertRaisesRegex(RuntimeError, 'request binding is stale'):
                runtime.validate_sync_translation_plan_before_dispatch(build)

    def test_sync_generation_tamper_refuses_before_dispatch(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        build.requests[0].generation_config['temperature'] = 0.9
        with mock.patch.object(
            runtime,
            'current_sync_source_identity',
            return_value=dict(build.plan.source_identity),
        ):
            with self.assertRaisesRegex(RuntimeError, 'request binding is stale'):
                runtime.validate_sync_translation_plan_before_dispatch(build)

    def test_batch_source_or_adapter_stale_refuses_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            stale = dict(build.plan.source_identity)
            stale['source_snapshot_fingerprint'] = 'changed-source'
            with (
                mock.patch.object(batch, 'collect_pending_file_jobs', return_value=[]),
                mock.patch.object(
                    batch,
                    '_batch_plan_source_identity',
                    return_value=translation_plan.SourceIdentity.from_dict(stale),
                ),
            ):
                with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                    batch.validate_batch_translation_plan_before_dispatch(manifest)
            self.assertEqual(raised.exception.code_name, 'TRANSLATION_PLAN_SOURCE_STALE')

    def test_batch_request_tamper_refuses_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, _build = _batch_manifest(tmp_dir)
            path = Path(manifest['input_jsonl_path'])
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            rows[0]['request']['contents'][0]['parts'][0]['text'] += '\ntampered'
            path.write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                encoding='utf-8',
            )
            with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                batch.validate_batch_translation_plan_before_dispatch(manifest)
            self.assertEqual(
                raised.exception.code_name,
                'TRANSLATION_PLAN_REQUEST_BINDING_MISMATCH',
            )

    def test_batch_coordinated_chunk_and_jsonl_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            manifest['chunks'][0]['user_prompt'] += '\ntampered together'
            path = Path(manifest['input_jsonl_path'])
            rows = [
                batch.build_batch_request(chunk, model=manifest['batch_model'])
                for chunk in manifest['chunks']
            ]
            path.write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                encoding='utf-8',
            )
            with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                self._validate_current_batch(manifest, build)
            self.assertEqual(
                raised.exception.code_name,
                'TRANSLATION_PLAN_REQUEST_BINDING_MISMATCH',
            )

    def test_batch_adapter_version_stale_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            stale = dict(build.plan.source_identity)
            stale['adapter_version'] = 'changed-adapter'
            with (
                mock.patch.object(batch, 'collect_pending_file_jobs', return_value=[]),
                mock.patch.object(
                    batch,
                    '_batch_plan_source_identity',
                    return_value=translation_plan.SourceIdentity.from_dict(stale),
                ),
            ):
                with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                    batch.validate_batch_translation_plan_before_dispatch(manifest)
            self.assertEqual(raised.exception.code_name, 'TRANSLATION_PLAN_SOURCE_STALE')
            self.assertIn('adapter_version_changed', raised.exception.details['reasons'])

    def test_split_child_plan_is_sliced_signed_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            child_chunks = manifest['chunks'][:1]
            child = {
                'batch_model': manifest['batch_model'],
                'chunks': child_chunks,
                'split_from_manifest': 'parent.json',
                'input_jsonl_path': str(Path(tmp_dir) / 'split.requests.jsonl'),
            }
            batch.copy_split_context_metadata(manifest, child, child_chunks)
            Path(child['input_jsonl_path']).write_text(
                json.dumps(
                    batch.build_batch_request(
                        child_chunks[0], model=child['batch_model']
                    ),
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            diagnostics = [
                self._validate_current_batch(child, build, operation=operation)
                for operation in ('submit', 'check', 'apply')
            ]
            self.assertEqual(len(child['translation_plan']['request_summaries']), 1)
            self.assertEqual(
                child['translation_plan']['artifacts']['derivation']['kind'],
                'split',
            )
            self.assertTrue(all(item['plan'] == 'fresh' for item in diagnostics))

    def test_split_manifest_current_plan_children_pass_all_plan_gates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            manifest.update({
                'version': 2,
                'display_name': 'fixture',
                'settings': {},
            })
            manifest_path = Path(tmp_dir) / 'manifest.json'
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding='utf-8',
            )
            with mock.patch.object(
                batch,
                'LATEST_MANIFEST_FILE',
                str(Path(tmp_dir) / 'latest.txt'),
            ):
                children = batch.split_manifest(
                    str(manifest_path),
                    max_chunks=1,
                )
            self.assertGreater(len(children), 1)
            for child_path in children:
                child = batch.load_manifest(child_path)
                for operation in ('submit', 'check', 'apply'):
                    diagnostic = self._validate_current_batch(
                        child,
                        build,
                        operation=operation,
                    )
                    self.assertEqual(diagnostic['plan'], 'fresh')

    def test_split_child_cannot_resign_tampered_parent_request(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, _build = _batch_manifest(tmp_dir)
            chunk = manifest['chunks'][0]
            chunk['user_prompt'] += '\ncoordinated parent tamper'
            prompt_fingerprint, request_fingerprint = (
                translation_plan.recompute_request_fingerprints(chunk)
            )
            chunk['prompt_fingerprint'] = prompt_fingerprint
            chunk['request_fingerprint'] = request_fingerprint
            child = {
                'chunks': [chunk],
                'split_from_manifest': 'parent.json',
            }
            with self.assertRaisesRegex(ValueError, 'not bound'):
                batch.copy_split_context_metadata(manifest, child, [chunk])

    def test_retry_child_plan_binds_derived_request_and_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            retry_chunk = batch.build_retry_subchunk(
                manifest['chunks'][0], 0, 1, 1
            )
            child = {
                'batch_model': manifest['batch_model'],
                'chunks': [retry_chunk],
                'retry_of_manifest': 'parent.json',
                'input_jsonl_path': str(Path(tmp_dir) / 'retry.requests.jsonl'),
            }
            batch.copy_split_context_metadata(manifest, child, [retry_chunk])
            Path(child['input_jsonl_path']).write_text(
                json.dumps(
                    batch.build_batch_request(
                        retry_chunk, model=child['batch_model']
                    ),
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            diagnostics = [
                self._validate_current_batch(child, build, operation=operation)
                for operation in ('submit', 'check', 'apply')
            ]
            summary = child['translation_plan']['request_summaries'][0]
            self.assertEqual(summary['request_id'], retry_chunk['request_id'])
            self.assertEqual(
                child['translation_plan']['artifacts']['derivation']['kind'],
                'retry',
            )
            self.assertTrue(all(item['request_count'] == 1 for item in diagnostics))

    def test_apply_force_cannot_bypass_stale_plan_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, _build = _batch_manifest(tmp_dir)
            manifest.update({
                '_manifest_path': str(Path(tmp_dir) / 'manifest.json'),
                '_package_dir': tmp_dir,
                'applied_at': '2026-08-28T00:00:00',
                'files': {},
            })
            manifest['translation_plan']['artifacts']['tampered'] = True

            def allow_contract(_manifest, summary):
                summary['writeback_gate'] = {'decision': batch.translation_quality.GATE_ALLOW}
                return summary

            with (
                mock.patch.object(batch, 'load_manifest', return_value=manifest),
                mock.patch.object(batch, 'require_manifest_mode'),
                mock.patch.object(batch, 'require_manifest_project_match'),
                mock.patch.object(batch, 'recover_atomic_write_transaction'),
                mock.patch.object(batch, 'require_safe_check_for_apply'),
                mock.patch.object(
                    batch,
                    'collect_result_actions',
                    return_value=({}, {}, [], {}),
                ),
                mock.patch.object(batch, 'attach_check_contract', side_effect=allow_contract),
                mock.patch.object(
                    batch,
                    '_validate_adapter_writeback_plan',
                    return_value=(None, None),
                ),
            ):
                with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                    batch.apply_results('manifest.json', force=True)
            self.assertEqual(raised.exception.code_name, 'TRANSLATION_PLAN_STALE')

    def test_durable_invalid_plan_uses_structured_contract_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, _build = _batch_manifest(tmp_dir)
            manifest['durable_sync_source'] = {'run_id': 'run-1'}
            manifest['translation_plan']['artifacts']['tampered'] = True
            with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                batch.validate_batch_translation_plan_before_dispatch(manifest)
            self.assertEqual(raised.exception.code_name, 'TRANSLATION_PLAN_STALE')

    def test_validation_preserves_context_diagnostic_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest, build = _batch_manifest(tmp_dir)
            expected = translation_plan.summarize_request_diagnostics(
                manifest['translation_plan']['request_summaries']
            )
            manifest['translation_plan_diagnostics'] = {
                'code': 'TRANSLATION_PLAN_CURRENT',
                **expected,
            }
            diagnostic = self._validate_current_batch(manifest, build)
            self.assertEqual(
                diagnostic['context_truncated_requests'],
                expected['context_truncated_requests'],
            )
            self.assertEqual(
                diagnostic['context_dropped_entries'],
                expected['context_dropped_entries'],
            )


class CompatibilityDiagnosticsTests(unittest.TestCase):
    def test_legacy_batch_package_is_explicitly_downgraded(self):
        for operation in ('check', 'apply'):
            with self.subTest(operation=operation):
                diagnostic = batch.validate_batch_translation_plan_before_dispatch(
                    {},
                    operation=operation,
                )
                self.assertEqual(diagnostic['mode'], 'legacy')
                self.assertEqual(
                    diagnostic['code'],
                    'TRANSLATION_PLAN_LEGACY_FALLBACK',
                )

    def test_legacy_batch_submit_is_blocked_before_provider_setup(self):
        manifest = {'job_name': '', 'submit_disabled': False}
        with (
            mock.patch.object(batch, 'load_manifest', return_value=manifest),
            mock.patch.object(batch, '_manifest_package_dir', return_value='package'),
            mock.patch.object(
                batch.batch_submit_recovery,
                'get_uncertain_submit_state',
                return_value=None,
            ),
            mock.patch.object(batch, 'resolve_manifest_routing_plan'),
            mock.patch.object(batch, 'require_valid_routing_plan'),
            mock.patch.object(batch, 'create_batch_client') as create_client,
        ):
            with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                batch.submit_manifest('legacy-manifest.json')
        self.assertEqual(
            raised.exception.code_name,
            'TRANSLATION_PLAN_LEGACY_SUBMIT_BLOCKED',
        )
        self.assertEqual(
            raised.exception.suggested_action,
            'rebuild_batch_package',
        )
        self.assertEqual(
            raised.exception.details['compatibility']['code'],
            'TRANSLATION_PLAN_LEGACY_FALLBACK',
        )
        create_client.assert_not_called()

    def test_legacy_sync_preview_loads_with_runtime_only_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'manifest.json'
            manifest = {
                'schema': sync_translation_preview.SCHEMA,
                'version': sync_translation_preview.VERSION,
                'created_at': '2026-01-01T00:00:00+00:00',
                'project_root': tmp_dir,
                'tl_dir': tmp_dir,
                'report_path': 'preview.diff',
                'report_sha256': '',
                'summary': {},
                'files': [],
            }
            manifest['preview_fingerprint'] = sync_translation_preview._fingerprint(manifest)
            path.write_text(json.dumps(manifest), encoding='utf-8')
            loaded = sync_translation_preview.load_sync_preview(path)
            self.assertEqual(
                loaded['_translation_plan_compatibility']['code'],
                'TRANSLATION_PLAN_LEGACY_FALLBACK',
            )
            self.assertNotIn('_translation_plan_compatibility', json.loads(path.read_text()))


class RagIndependentContextTests(unittest.TestCase):
    def test_no_retrieval_keeps_macro_and_all_local_glossary_kinds(self):
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider='',
            analysis_blocks_provider='',
        )
        prompt = '\n'.join(request.user_prompt for request in build.requests)
        system = '\n'.join(request.system_instruction for request in build.requests)
        self.assertIn('A college a cappella story', system)
        self.assertIn('Existing mapping: setlist -> 曲目单', prompt)
        self.assertIn('Preserve: Dawn Chorus', prompt)
        self.assertIn('Non-translatable: B-side', prompt)
        for request in build.requests:
            project = next(
                layer
                for layer in request.context_assembly['layers']
                if layer['layer'] == translation_plan.CONTEXT_LAYER_PROJECT
            )
            self.assertTrue(project['diagnostics']['rag_independent'])


class PersistedDiagnosticsSafetyTests(unittest.TestCase):
    def test_plan_requests_and_diff_artifacts_do_not_persist_credentials(self):
        secret = 'credential-value-must-not-leak'
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            model_profile_snapshot={
                'id': 'fixture',
                'extra_headers': {'Authorization': f'Bearer {secret}'},
            },
            generation_config={'temperature': 0.2, 'api_key': secret},
            transport_metadata={'auth_token': secret},
        )
        peer = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            model_profile_snapshot={
                'id': 'fixture',
                'extra_headers': {'Authorization': f'Bearer {secret}'},
            },
            generation_config={'temperature': 0.2, 'api_key': secret},
            transport_metadata={'auth_token': secret},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / 'manifest.json').write_text(
                json.dumps(build.plan.to_dict(), ensure_ascii=False),
                encoding='utf-8',
            )
            (root / 'requests.jsonl').write_text(
                ''.join(
                    json.dumps(request.to_dict(), ensure_ascii=False) + '\n'
                    for request in build.requests
                ),
                encoding='utf-8',
            )
            (root / 'plan_diff.txt').write_text(
                translation_plan.format_plan_diff(
                    translation_plan.plan_diff(build.requests, peer.requests)
                ),
                encoding='utf-8',
            )
            persisted = '\n'.join(
                path.read_text(encoding='utf-8') for path in root.iterdir()
            )
        self.assertNotIn(secret, persisted)
        self.assertNotIn('Bearer ', persisted)
        self.assertIn(translation_plan.REDACTED_VALUE, persisted)


if __name__ == '__main__':
    unittest.main()
