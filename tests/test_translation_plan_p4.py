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


class CompatibilityDiagnosticsTests(unittest.TestCase):
    def test_legacy_batch_package_is_explicitly_downgraded(self):
        diagnostic = batch.validate_batch_translation_plan_before_dispatch({})
        self.assertEqual(diagnostic['mode'], 'legacy')
        self.assertEqual(diagnostic['code'], 'TRANSLATION_PLAN_LEGACY_FALLBACK')

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
