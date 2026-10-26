"""Opt-in display fields through the real durable Sync gates with fixed responses."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import gemini_translate_batch as batch
import translator_runtime as runtime
from engine_adapters.rpgmaker_mv import RpgMakerMvAdapter
from execution.sync_run_service import build_production_sync_run_service
from foundation import atomic_io
from foundation.cli_contract import MachineContractError
from quality_review import structure_protection
from tests.execution import test_mv_durable_sync as narrative_tests
from tests.support.mv_project import add_mv_display_material, review_mv_project, write_mv_scope


class MvDisplaySyncTests(unittest.TestCase):
    response = narrative_tests.MvDurableSyncTests.response
    completed = narrative_tests.MvDurableSyncTests.completed
    checked = narrative_tests.MvDurableSyncTests.checked

    def setUp(self):
        narrative_tests.MvDurableSyncTests.setUp(self)
        self.positions = add_mv_display_material(self.root)
        self.config['rpgmaker_mv']['display_fields'] = ['system_terms', 'map_display_name']
        self.scope = self.root / 'display-scope.json'
        write_mv_scope(self.scope, self.source, self.positions)
        self.config['rpgmaker_mv']['occurrence_scope_file'] = str(self.scope)
        self.runs = Path(runtime.LOG_DIR) / 'sync_runs'
        self.configure()

    def configure(self):
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        review_mv_project(self.config, self.config_file)
        self.context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.service = build_production_sync_run_service(self.runs, self.context)

    def assert_output(self):
        fixture = Path(__file__).resolve().parents[1] / 'fixtures/rpgmaker_mv_display/expected/data'
        for name in ('System.json', 'Map003.json'):
            self.assertEqual((self.target / 'data' / name).read_bytes(), (fixture / name).read_bytes())
        for row in self.positions:
            if row['file'] in ('data/Map001.json', 'data/Map002.json'):
                expected = (self.source / row['file']).read_bytes().replace(
                    json.dumps(row['source']).encode(), json.dumps(row['translation'], ensure_ascii=False).encode())
                self.assertEqual((self.target / row['file']).read_bytes(), expected)

    def test_offline_cli_counts_prompt_roles_check_apply_and_repeat(self):
        self.assertEqual(len(self.context.items_by_id), 11)
        self.assertEqual(batch.collect_doctor_report()['pending_task_count'], 11)
        with mock.patch('providers.model_capability_probe.default_credential_loader', return_value='test-only'):
            preflight = batch.run_translate_preflight(SimpleNamespace(strategy='sync', profile=''))
        self.assertEqual(preflight['counts']['pending_items'], 11)
        export = self.root / 'display-coverage'
        batch.export_live_coverage_package(export)
        report = json.loads((export / 'coverage_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['catalog_provenance']['display_physical_positions'], 13)
        source_before = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        store = self.checked()
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        self.assert_output()
        self.assertEqual(batch.apply_durable_sync_results(store)['last_apply_result'], 'already_applied')
        self.assertEqual(batch.collect_doctor_report()['pending_task_count'], 0)
        self.assertEqual(source_before, {path: path.read_bytes() for path in source_before})

    def test_selection_change_rejects_dispatch_and_checked_apply(self):
        original = self.config_file.read_bytes()
        self.config['rpgmaker_mv']['display_fields'] = ['system_terms']
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            with self.assertRaisesRegex(Exception, 'freshness'):
                self.service.start(self.context.plan_build)
        provider.assert_not_called()
        self.config_file.write_bytes(original)
        store = self.checked()
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        with self.assertRaises(ValueError):
            batch.apply_results(str(store.run_dir), force=True)

    def test_apply_force_still_requires_latest_allow_check(self):
        store = self.completed()
        with self.assertRaises(MachineContractError):
            batch.apply_results(str(store.run_dir), force=True)
        batch.check_durable_sync_results(store)
        path = self.source / 'data/System.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            batch.check_durable_sync_results(store)
        path.write_bytes(original)
        with self.assertRaises(MachineContractError):
            batch.apply_results(str(store.run_dir), force=True)

    def test_same_scope_filters_can_select_system_file_and_reject_conflicts(self):
        self.config['include_files'] = ['data/System.json']
        write_mv_scope(self.scope, self.source, [row for row in self.positions if row['file'] == 'data/System.json'])
        self.configure()
        self.assertEqual(len(self.context.items_by_id), 8)
        with mock.patch('execution.sync_request.run_sync_request', side_effect=self.response):
            store = self.checked()
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        self.assertEqual((self.target / 'data/Map001.json').read_bytes(), (self.source / 'data/Map001.json').read_bytes())

    def test_numbered_parameters_are_masked_and_damage_is_not_accepted(self):
        entries = [mapping for request in self.context.plan_build.requests
                   for mapping in request.transport_metadata[structure_protection.KEY]['items'].values()]
        numbered = [token for mapping in entries for token in mapping['entries'] if token['value'].startswith('%')]
        self.assertEqual(len(numbered), 6)
        self.assertTrue(all(token['kind'] == 'variable' for token in numbered))
        original = self.response

        def damaged(request, *args, **kwargs):
            result = original(request, *args, **kwargs)
            for row in result['parsed']['translations']:
                row['translation'] = structure_protection.MARKER.sub('', row['translation'])
            return result

        with mock.patch('execution.sync_request.run_sync_request', side_effect=damaged):
            result = self.service.start(self.context.plan_build)
        self.assertGreater(result['progress']['items']['unresolved'], 0)
        self.assertLess(result['progress']['items']['accepted'], 11)
        self.assertEqual((self.target / 'data/System.json').read_bytes(), (self.source / 'data/System.json').read_bytes())

    def test_crash_across_display_files_recovers_and_reapplies(self):
        store = self.checked()
        real_replace = atomic_io.os.replace
        replaced = 0

        def crash(source, destination):
            nonlocal replaced
            if str(source).endswith('.txn.tmp') and Path(destination).is_relative_to(self.target):
                replaced += 1
                if replaced == 2:
                    raise SystemExit('original display test crash')
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, 'replace', side_effect=crash):
            with self.assertRaisesRegex(SystemExit, 'display test crash'):
                batch.apply_durable_sync_results(store)
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        self.assert_output()

    def test_old_adapter_approval_is_readable_but_cannot_apply_or_recheck(self):
        with mock.patch.object(RpgMakerMvAdapter, 'adapter_version', '0.2.0'):
            self.configure()
            store = self.checked()
        self.assertEqual(store.get_plan()['plan']['source_identity']['adapter_version'], '0.2.0')
        with self.assertRaises(ValueError):
            batch.apply_durable_sync_results(store)
        with self.assertRaises(ValueError):
            batch.check_durable_sync_results(store)
        self.assertIsNone(store.get_artifact(kind='preview_manifest'))

    def test_template_source_target_and_scope_drift_refuse_checked_apply(self):
        store = self.checked()
        for path in (self.source / 'data/System.json', self.target / 'data/Map003.json', self.scope):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):
                batch.apply_durable_sync_results(store)
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
