"""Offline MV Batch requests, canonical results and guarded JSON writeback."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import gemini_translate_batch as batch
import translator_runtime as runtime
from foundation import atomic_io
from quality_review import structure_protection
from tests.support.mv_project import make_mv_project, add_mv_display_material, review_mv_project, write_mv_scope


class MvBatchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.config_file, self.config, self.positions = make_mv_project(self.root)
        self.source, self.target = self.root / 'source', self.root / 'target'
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        old_runtime = runtime.snapshot_runtime_config()
        self.stack.callback(runtime.apply_runtime_config, old_runtime)
        old_batch = {key: value for key, value in vars(batch).items() if key.isupper()}
        self.stack.callback(lambda: [setattr(batch, key, value) for key, value in old_batch.items()])
        self.stack.enter_context(mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(self.config_file)))
        self.stack.enter_context(mock.patch.object(runtime, 'LOG_DIR', str(self.root / 'logs')))
        self.config['batch'] = {'model': 'gemini-3.1-flash-lite', 'rag': {'enabled': False},
                                'source_index': {'enabled': False}, 'story_memory': {'enabled': False},
                                'project_analysis': {'enabled': False, 'inject_published_brief': False}}
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        runtime.load_translator_settings()
        batch.load_batch_settings()
        for key, value in {
            'LOG_DIR': str(self.root / 'logs'), 'BATCH_JOBS_DIR': str(self.root / 'jobs'),
            'LATEST_MANIFEST_FILE': str(self.root / 'jobs/latest_manifest.txt'),
            'PROGRESS_LOG': str(self.root / 'logs/progress.json'),
            'RAG_ENABLED': False, 'SOURCE_INDEX_ENABLED': False, 'STORY_MEMORY_ENABLED': False,
            'PROJECT_ANALYSIS_ENABLED': False, 'PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF': False,
        }.items():
            setattr(batch, key, value)
        self.provider = self.stack.enter_context(mock.patch.object(batch, 'create_batch_client',
            side_effect=AssertionError('Real provider forbidden in MV Batch tests')))

    def build(self):
        path = batch.create_batch_package(display_name_override='original-mv-batch')
        self.assertTrue(path)
        return batch.load_manifest(path)

    def result_rows(self, manifest):
        positions = {(row['file'], tuple(row['path'])): row['translation'] for row in self.positions}
        rows = []
        for chunk in manifest['chunks']:
            translations = []
            protection = chunk['transport_metadata'][structure_protection.KEY]
            for item in chunk['items']:
                text = positions[chunk['file_rel_path'], tuple(item['mv_context']['json_path'])]
                for entry in protection['items'][item['id']]['entries']:
                    text = text.replace(entry['value'], entry['marker'], 1)
                translations.append({'id': item['id'], 'translation': text})
            rows.append({'key': chunk['key'], 'response': {'candidates': [{
                'content': {'parts': [{'text': json.dumps({'translations': translations}, ensure_ascii=False)}]},
                'finishReason': 'STOP'}]}})
        return rows

    def results(self, manifest, rows=None):
        path = Path(manifest['_package_dir']) / 'results.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
            for row in (rows if rows is not None else self.result_rows(manifest))), encoding='utf-8')
        return path

    def test_build_check_apply_keeps_source_and_event_mirrors(self):
        original = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        manifest = self.build()
        self.assertEqual(manifest['summary']['item_count'], 14)
        self.assertEqual(manifest['translation_plan']['execution_strategy'], 'gemini_batch')
        self.assertEqual(manifest['translation_plan']['source_identity']['engine'], 'rpgmaker_mv')
        self.results(manifest)
        checked = batch.check_results(manifest['_manifest_path'])
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
        applied = batch.apply_results(manifest['_manifest_path'])
        self.assertEqual(applied['apply_summary']['applied_lines'], 14)
        self.assertEqual(sum(len(row['mirrors']) for row in self.positions), 6)
        for row in self.positions:
            for address in [row['path'], *row.get('mirrors', [])]:
                value = json.loads((self.target / row['file']).read_text(encoding='utf-8'))
                for key in address:
                    value = value[key]
                self.assertEqual(value, row['translation'])
        self.assertEqual(original, {path: path.read_bytes() for path in original})
        self.provider.assert_not_called()

    def checked(self):
        manifest = self.build()
        self.results(manifest)
        return batch.check_results(manifest['_manifest_path'])

    def target_bytes(self):
        return {path: path.read_bytes() for path in self.target.rglob('*') if path.is_file()}

    def test_results_or_requests_changed_after_check_cannot_be_forced(self):
        for artifact in ('results.jsonl', 'requests.jsonl'):
            with self.subTest(artifact=artifact):
                manifest = self.checked()
                path = Path(manifest['_package_dir']) / artifact
                path.write_bytes(path.read_bytes() + b'\n')
                before = self.target_bytes()
                with self.assertRaises((ValueError, SystemExit)):
                    batch.apply_results(manifest['_manifest_path'], force=True)
                self.assertEqual(self.target_bytes(), before)

    def test_source_target_core_and_config_drift_refuse_before_dispatch_and_apply(self):
        selected_file = self.positions[0]['file']
        for path in (self.source / selected_file, self.target / selected_file,
                     self.source / 'js/rpg_core.js', self.target / 'js/rpg_core.js',
                     self.source / 'js/plugins/GALV_MessageStyles.js', self.config_file):
            with self.subTest(path=path.relative_to(self.root)):
                manifest = self.checked()
                original = path.read_bytes()
                if path == self.config_file:
                    changed = json.loads(original)
                    changed['batch']['temperature'] = 0.321
                    path.write_text(json.dumps(changed), encoding='utf-8')
                else:
                    path.write_bytes(original + b' ')
                before = self.target_bytes()
                try:
                    with self.assertRaises((ValueError, SystemExit)):
                        batch.submit_manifest(manifest['_manifest_path'])
                    with self.assertRaises((ValueError, SystemExit)):
                        batch.apply_results(manifest['_manifest_path'], force=True)
                    self.assertEqual(self.target_bytes(), before)
                    self.provider.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_apply_requires_a_current_check(self):
        manifest = self.build()
        self.results(manifest)
        before = self.target_bytes()
        with self.assertRaises(SystemExit):
            batch.apply_results(manifest['_manifest_path'], force=True)
        self.assertEqual(self.target_bytes(), before)

    def test_invalid_structure_can_only_be_repaired_by_retry_and_recheck(self):
        manifest = self.build()
        rows = self.result_rows(manifest)
        content = rows[0]['response']['candidates'][0]['content']['parts'][0]
        payload = json.loads(content['text'])
        payload['translations'][0]['translation'] = '丢失结构标记'
        content['text'] = json.dumps(payload, ensure_ascii=False)
        self.results(manifest, rows)
        before = self.target_bytes()
        checked = batch.check_results(manifest['_manifest_path'])
        self.assertGreater(checked['last_check_summary']['failure_items'], 0)
        self.assertEqual(self.target_bytes(), before)
        retry = batch.load_manifest(batch.build_retry_package(manifest['_manifest_path']))
        self.assertEqual(retry['summary']['item_count'], 1)
        self.assertEqual(retry['translation_plan']['source_identity'], manifest['translation_plan']['source_identity'])
        self.results(retry)
        batch.check_results(retry['_manifest_path'])
        batch.merge_retry_results(manifest['_manifest_path'], retry['_manifest_path'])
        with self.assertRaises(SystemExit):
            batch.apply_results(manifest['_manifest_path'], force=True)
        repaired = batch.check_results(manifest['_manifest_path'])
        self.assertEqual(repaired['last_check_summary']['failure_items'], 0)
        applied = batch.apply_results(manifest['_manifest_path'])
        self.assertEqual(applied['apply_summary']['applied_lines'], 14)

    def test_batch_preflight_and_doctor_advertise_the_reviewed_mv_route_offline(self):
        with mock.patch('providers.model_capability_probe.default_credential_loader', return_value='test-only'):
            payload = batch.run_translate_preflight(SimpleNamespace(strategy='gemini_batch', profile=''))
        self.assertEqual(payload['strategy'], 'gemini_batch')
        self.assertEqual(payload['project']['engine'], 'rpgmaker_mv')
        self.assertEqual(payload['counts']['pending_items'], 14)
        self.assertEqual(payload['profile']['model'], 'gemini-3.1-flash-lite')
        report = batch.collect_doctor_report()
        self.assertIn('gemini_batch', report['engine_status']['supported_routes'])
        self.assertNotIn('gemini_batch', report['engine_status']['unsupported_routes'])
        self.provider.assert_not_called()

    def test_unsupported_context_and_prepare_refuse_before_provider(self):
        for name in ('STORY_MEMORY_ENABLED', 'PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF'):
            with self.subTest(name=name), mock.patch.object(batch, name, True):
                with self.assertRaisesRegex(ValueError, 'does not support'):
                    self.build()
                with self.assertRaises((ValueError, SystemExit)):
                    batch.run_translate_preflight(SimpleNamespace(strategy='gemini_batch', profile=''))
        with mock.patch.object(runtime, 'PREP_ENABLED', True), mock.patch.object(runtime, 'run_prepare_steps') as prepare:
            with self.assertRaisesRegex(ValueError, 'prepare'):
                self.build()
            prepare.assert_not_called()
        self.provider.assert_not_called()

    def test_scoped_display_fields_use_typed_json_positions(self):
        self.positions += add_mv_display_material(self.root)
        selected = [row for row in self.positions if row['path'] == ['displayName']]
        self.assertTrue(selected)
        scope = self.root / 'scope.json'
        write_mv_scope(scope, self.source, selected)
        self.config['rpgmaker_mv'].update(display_fields=['map_display_name', 'system_terms'],
                                          occurrence_scope_file=str(scope))
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        review_mv_project(self.config, self.config_file)
        runtime.load_translator_settings()
        batch.load_batch_settings()
        before = self.target_bytes()
        manifest = self.checked()
        self.assertEqual(manifest['summary']['item_count'], len(selected))
        batch.apply_results(manifest['_manifest_path'])
        changed = {path for path in before if path.read_bytes() != before[path]}
        self.assertEqual(changed, {self.target / row['file'] for row in selected})

    def test_split_parts_apply_in_order_and_only_accept_committed_sibling_bytes(self):
        batch.BATCH_TARGET_SIZE = 2
        manifest = self.build()
        parts = batch.split_manifest(manifest['_manifest_path'], max_chunks=1)
        self.assertGreater(len(parts), 3)
        for path in parts:
            self.results(batch.load_manifest(path))
            batch.check_results(path)
        with self.assertRaisesRegex(ValueError, 'earlier'):
            batch.apply_results(parts[1])
        total = 0
        for path in parts:
            batch.check_results(path)
            total += batch.apply_results(path)['apply_summary']['applied_lines']
        self.assertEqual(total, 14)

    def test_committed_files_recover_progress_after_manifest_save_failure(self):
        manifest = self.checked()
        original_save = batch.save_manifest
        def save_unless_applied(current, *args, **kwargs):
            if current.get('applied_at'):
                raise OSError('original test bookkeeping interruption')
            return original_save(current, *args, **kwargs)
        with mock.patch.object(batch, 'save_manifest', side_effect=save_unless_applied):
            with self.assertRaises(OSError):
                batch.apply_results(manifest['_manifest_path'])
        after = self.target_bytes()
        with mock.patch.object(batch, 'atomic_write_many_lines', side_effect=AssertionError('duplicate writes')):
            recovered = batch.apply_results(manifest['_manifest_path'])
        self.assertEqual(recovered['apply_summary']['applied_lines'], 14)
        self.assertEqual(self.target_bytes(), after)

    def test_export_options_refuse_without_creating_another_output_tree(self):
        manifest = self.checked()
        before = self.target_bytes()
        for option in ('export_only', 'export_dir'):
            destination = self.root / option
            with self.assertRaises((ValueError, SystemExit)):
                batch.apply_results(manifest['_manifest_path'], **{option: str(destination)})
            self.assertFalse(destination.exists())
        self.assertEqual(self.target_bytes(), before)

    def test_receipt_cannot_report_applied_after_targets_were_restored_to_original(self):
        manifest = self.checked()
        original = self.target_bytes()
        batch.apply_results(manifest['_manifest_path'])
        for path, content in original.items():
            path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, 'receipt'):
            batch.apply_results(manifest['_manifest_path'], force=True)
        self.assertEqual(self.target_bytes(), original)

    def test_configured_quality_blocker_cannot_be_forced(self):
        from quality_review import translation_quality
        batch.BATCH_QUALITY_POLICY = translation_quality.normalize_policy({'rules': {'halfwidth_punctuation': 'blocker'}})
        manifest = self.build()
        rows = self.result_rows(manifest)
        content = rows[0]['response']['candidates'][0]['content']['parts'][0]
        payload = json.loads(content['text'])
        payload['translations'][0]['translation'] += '测试,标点'
        content['text'] = json.dumps(payload, ensure_ascii=False)
        self.results(manifest, rows)
        checked = batch.check_results(manifest['_manifest_path'])
        self.assertGreater(checked['last_check_summary']['quality_gate']['blocker_count'], 0)
        before = self.target_bytes()
        with self.assertRaises(SystemExit):
            batch.apply_results(manifest['_manifest_path'], force=True)
        self.assertEqual(self.target_bytes(), before)

    def test_split_receipt_does_not_allow_foreign_target_edits_or_changed_sibling_results(self):
        manifest = self.build()
        parts = batch.split_manifest(manifest['_manifest_path'], max_chunks=1)
        first, second = [batch.load_manifest(path) for path in parts[:2]]
        self.results(first)
        self.results(second)
        batch.check_results(parts[0])
        batch.apply_results(parts[0])
        path = self.target / first['chunks'][0]['file_rel_path']
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises((ValueError, SystemExit)):
            batch.check_results(parts[1])
        path.write_bytes(original)
        result = Path(first['_package_dir']) / 'results.jsonl'
        result.write_bytes(result.read_bytes() + b'\n')
        with self.assertRaises((ValueError, SystemExit)):
            batch.check_results(parts[1])

    def fake_client(self, manifest, *, upload_side_effect=None):
        job = SimpleNamespace(name='batches/original-fixture', state='JOB_STATE_SUCCEEDED',
                              dest=SimpleNamespace(file_name='files/original-results'))
        client = SimpleNamespace(
            files=SimpleNamespace(upload=mock.Mock(side_effect=upload_side_effect,
                                  return_value=SimpleNamespace(name='files/original-requests')),
                                  download=mock.Mock(return_value=(''.join(json.dumps(row) + '\n'
                                      for row in self.result_rows(manifest))).encode('utf-8'))),
            batches=SimpleNamespace(create=mock.Mock(return_value=job), get=mock.Mock(return_value=job)),
        )
        self.provider.side_effect = None
        self.provider.return_value = client
        return client

    def test_fake_remote_submit_download_check_apply_uses_existing_batch_transport(self):
        manifest = self.build()
        client = self.fake_client(manifest)
        submitted = batch.submit_manifest(manifest['_manifest_path'])
        self.assertEqual(batch.load_manifest(submitted)['job_name'], 'batches/original-fixture')
        client.files.upload.assert_called_once()
        client.batches.create.assert_called_once()
        with mock.patch.object(batch, 'fetch_batch_job_for_manifest',
                               return_value=(client, client.batches.get.return_value)):
            batch.download_results(manifest['_manifest_path'])
        client.files.download.assert_called_once()
        checked = batch.check_results(manifest['_manifest_path'])
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
        batch.apply_results(manifest['_manifest_path'])

    def test_cli_dispatch_routes_build_submit_status_download_check_apply(self):
        parser = batch.build_arg_parser()
        with mock.patch.object(runtime, 'load_config'), mock.patch.object(runtime, 'load_glossary'), \
                mock.patch.object(runtime, 'API_KEYS', ['original-test-key']):
            path = batch.dispatch_command(parser, parser.parse_args(['build', '--skip-prepare']))
            manifest = batch.load_manifest(path)
            client = self.fake_client(manifest)
            for command in ('submit', 'status', 'download', 'check', 'apply'):
                with self.subTest(command=command):
                    batch.dispatch_command(parser, parser.parse_args([command, path]))
            self.assertEqual(batch.load_manifest(path)['apply_summary']['applied_lines'], 14)
            client.batches.create.assert_called_once()
        for command in ('bootstrap-rag', 'bootstrap-source-index'):
            args = parser.parse_args([command, '--strategy', 'gemini_batch'])
            self.assertEqual(args.strategy, 'gemini_batch')

    def test_upload_time_material_drift_prevents_remote_job_creation(self):
        manifest = self.build()
        path = self.source / self.positions[0]['file']
        def change_during_upload(**kwargs):
            path.write_bytes(path.read_bytes() + b' ')
            return SimpleNamespace(name='files/original-requests')
        client = self.fake_client(manifest, upload_side_effect=change_during_upload)
        with self.assertRaises((ValueError, SystemExit)):
            batch.submit_manifest(manifest['_manifest_path'])
        client.files.upload.assert_called_once()
        client.batches.create.assert_not_called()

    def test_commit_time_source_drift_rolls_back_all_targets(self):
        manifest = self.checked()
        before = self.target_bytes()
        original_replace = atomic_io.os.replace
        changed = []
        def replace_and_change_source(source, target):
            original_replace(source, target)
            if Path(target) in before and not changed:
                changed.append(True)
                path = self.source / self.positions[0]['file']
                path.write_bytes(path.read_bytes() + b' ')
        with mock.patch.object(atomic_io.os, 'replace', side_effect=replace_and_change_source):
            with self.assertRaises(ValueError):
                batch.apply_results(manifest['_manifest_path'])
        self.assertTrue(changed)
        self.assertEqual(self.target_bytes(), before)

    def test_commit_time_results_drift_rolls_back_targets_and_receipt(self):
        manifest = self.checked()
        before = self.target_bytes()
        original_replace = atomic_io.os.replace
        changed = []
        def replace_and_change_result(source, target):
            original_replace(source, target)
            if Path(target) in before and not changed:
                changed.append(True)
                path = Path(manifest['_package_dir']) / 'results.jsonl'
                path.write_bytes(path.read_bytes() + b'\n')
        with mock.patch.object(atomic_io.os, 'replace', side_effect=replace_and_change_result):
            with self.assertRaises(ValueError):
                batch.apply_results(manifest['_manifest_path'])
        self.assertTrue(changed)
        self.assertEqual(self.target_bytes(), before)
        self.assertFalse((Path(manifest['_package_dir']) / 'mv_apply_receipt.json').exists())

    def test_interrupted_transaction_recovers_before_retrying_apply(self):
        manifest = self.checked()
        before = self.target_bytes()
        original_replace = atomic_io.os.replace
        interrupted = []
        def replace_and_interrupt(source, target):
            original_replace(source, target)
            if Path(target) in before and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt('original test interruption')
        with mock.patch.object(atomic_io.os, 'replace', side_effect=replace_and_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                batch.apply_results(manifest['_manifest_path'])
        self.assertTrue(interrupted)
        applied = batch.apply_results(manifest['_manifest_path'])
        self.assertEqual(applied['apply_summary']['applied_lines'], 14)
