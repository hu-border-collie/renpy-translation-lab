"""Original fixed-response MV product workflow and fail-closed regressions."""
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
from execution.sync_run_service import build_production_sync_run_service
from execution.sync_run_store import SyncRunStore
from execution.sync_run_contracts import build_run_id
from engine_adapters.coverage import build_review_template, default_coverage_review_path
from foundation import atomic_io
from foundation import cli_contract
from quality_review import structure_protection, translation_plan
from tests.support.mv_project import make_mv_project, write_mv_scope


class MvDurableSyncTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.config_file, self.config, self.positions = make_mv_project(self.root)
        self.source, self.target = self.root / 'source', self.root / 'target'
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        saved = runtime.snapshot_runtime_config()
        self.stack.callback(runtime.apply_runtime_config, saved)
        self.stack.enter_context(mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(self.config_file)))
        self.stack.enter_context(mock.patch.object(runtime, 'load_config'))
        self.stack.enter_context(mock.patch.object(runtime, 'LOG_DIR', str(self.root / 'logs')))
        self.stack.enter_context(mock.patch.object(runtime, 'PROGRESS_LOG', str(self.root / 'logs/progress.json')))
        self.stack.enter_context(mock.patch.object(runtime, 'maybe_update_sync_rag_store'))
        runtime.load_translator_settings()
        self.context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.runs = self.root / 'runs'
        self.service = build_production_sync_run_service(self.runs, self.context)

    def response(self, request, *_args, **_kwargs):
        if 'contents' in request:
            request = next(row.to_dict() for row in self.context.plan_build.requests
                           if row.user_prompt == request['contents'])
        by_position = {(row['file'], tuple(row['path'])): row['translation'] for row in self.positions}
        translations = []
        protection = request['transport_metadata'][structure_protection.KEY]
        for item in self.context.item_resolver(request, request['expected_ids']):
            address = item['mv_context']
            text = by_position[item['file_rel_path'], tuple(address['json_path'])]
            for token in protection['items'][item['id']]['entries']:
                text = text.replace(token['value'], token['marker'], 1)
            translations.append({'id': item['id'], 'translation': text})
        return {'parsed': {'translations': translations}, 'usage_metadata': {'total_tokens': 1}}

    def completed(self):
        with mock.patch('execution.sync_request.run_sync_request', side_effect=self.response):
            started = self.service.start(self.context.plan_build)
        self.assertEqual(started['run_status'], 'completed', started)
        return SyncRunStore(self.runs, started['run_id'])

    def checked(self):
        store = self.completed()
        batch.check_durable_sync_results(store)
        return store

    def test_original_fixture_through_production_plan_check_apply(self):
        self.assertEqual(runtime.BASE_DIR, str(self.target))
        self.assertEqual(runtime.TL_DIR, str(self.target))
        self.assertEqual(len(self.context.items_by_id), 14)
        request = self.context.plan_build.requests[0]
        self.assertIn('RPG Maker MV', request.system_instruction)
        self.assertIn('mv_context', request.user_prompt)
        before = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        store = self.completed()
        checked = batch.check_durable_sync_results(store)
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
        applied = batch.apply_durable_sync_results(store)
        self.assertEqual(applied['state'], 'applied')
        fixture = Path(__file__).resolve().parents[1] / 'fixtures/rpgmaker_mv_narrative/expected'
        for path in fixture.rglob('*.json'):
            self.assertEqual((self.target / path.relative_to(fixture)).read_bytes(), path.read_bytes())
        self.assertEqual(before, {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob('*') if path.is_file()})
        self.assertEqual(batch.apply_durable_sync_results(store)['last_apply_result'], 'already_applied')

    def test_mv_batch_plan_is_explicitly_refused(self):
        with self.assertRaisesRegex(ValueError, 'durable Sync'):
            translation_plan.build_translation_plan(self.context.pending_jobs,
                execution_strategy='gemini_batch', source_identity=self.context.plan_build.plan.source_identity)

    def test_omitted_engine_uses_the_same_review_and_durable_writeback_contract(self):
        self.config.pop('engine')
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        runtime.load_translator_settings(persist_corrected_game_root=False)
        self.context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.service = build_production_sync_run_service(self.runs, self.context)
        self.assertEqual(runtime.current_project_engine(), 'rpgmaker_mv')
        self.assertEqual(runtime.BASE_DIR, str(self.target))
        self.assertEqual(len(self.context.items_by_id), 14)
        before = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        store = self.checked()
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def configure_occurrence_scope(self):
        self.scope_file = self.root / 'pilot-scope.json'
        self.scope_rows = self.positions[:3]
        write_mv_scope(self.scope_file, self.source, self.scope_rows)
        self.config['rpgmaker_mv']['occurrence_scope_file'] = str(self.scope_file)
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        self.context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.service = build_production_sync_run_service(self.runs, self.context)

    def test_exact_scope_check_apply_changes_only_selected_positions_and_mirrors(self):
        self.configure_occurrence_scope()
        self.assertEqual(len(self.context.items_by_id), 3)
        self.assertEqual(batch.collect_doctor_report()['pending_task_count'], 3)
        before = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        store = self.checked()
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        selected = {(row['file'], tuple(row['path'])) for row in self.scope_rows}
        mirrors = {(row['file'], tuple(path)): row['translation'] for row in self.scope_rows for path in row['mirrors']}
        for row in self.positions:
            actual = json.loads((self.target / row['file']).read_text(encoding='utf-8'))
            for part in row['path']:
                actual = actual[part]
            self.assertEqual(actual, row['translation'] if (row['file'], tuple(row['path'])) in selected else row['source'])
        for (relative, path), expected in mirrors.items():
            actual = json.loads((self.target / relative).read_text(encoding='utf-8'))
            for part in path:
                actual = actual[part]
            self.assertEqual(actual, expected)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(batch.collect_doctor_report()['pending_task_count'], 0)

    def test_scope_change_blocks_dispatch_and_already_checked_apply(self):
        self.configure_occurrence_scope()
        original = self.scope_file.read_bytes()
        self.scope_file.write_bytes(original + b' ')
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            with self.assertRaisesRegex(Exception, 'freshness'):
                self.service.start(self.context.plan_build)
        provider.assert_not_called()
        self.scope_file.write_bytes(original)
        store = self.checked()
        self.scope_file.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            batch.apply_durable_sync_results(store)

    def test_mv_bracketed_message_survives_renpy_filtering_through_apply(self):
        self.assertTrue(runtime.is_non_translatable('[Continue]'))
        position = next(row for row in self.positions if row['source'] == 'The door is open.')
        position.update(source='[Continue]', translation='[继续]')
        for root in (self.source, self.target):
            path = root / position['file']
            content = path.read_bytes()
            self.assertEqual(content.count(b'"The door is open."'), 1)
            path.write_bytes(content.replace(b'"The door is open."', b'"[Continue]"'))
        snapshot = runtime.current_mv_sync_snapshot(require_review=False)
        review_file = Path(default_coverage_review_path(self.target))
        previous_review = json.loads(review_file.read_text(encoding='utf-8'))
        review = build_review_template(snapshot.report)
        review.update({key: previous_review[key] for key in ('status', 'confirmed_at', 'reviewer')})
        review_file.write_text(json.dumps(review), encoding='utf-8')

        self.context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.service = build_production_sync_run_service(self.runs, self.context)
        self.assertEqual(self.context.adapter_snapshot.pending_task_count, 14)
        self.assertEqual(batch.collect_doctor_report()['pending_task_count'], 14)
        self.assertEqual(len(self.context.items_by_id), 14)
        self.assertEqual(sum(item['text'] == '[Continue]' for item in self.context.items_by_id.values()), 1)
        with mock.patch('providers.model_capability_probe.default_credential_loader', return_value='test-only'):
            preflight = batch.run_translate_preflight(SimpleNamespace(strategy='sync', profile=''))
        self.assertEqual(preflight['counts']['pending_items'], 14)

        before = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        store = self.completed()
        checked = batch.check_durable_sync_results(store)
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        translated = json.loads((self.target / position['file']).read_text(encoding='utf-8'))
        for component in position['path']:
            translated = translated[component]
        self.assertEqual(translated, '[继续]')
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(batch.collect_doctor_report()['pending_task_count'], 0)

    def test_all_material_changes_block_apply_including_untouched_data_and_review(self):
        store = self.checked()
        candidates = [self.source / 'data/CommonEvents.json', self.target / 'data/Actors.json',
                      self.source / 'js/rpg_core.js', self.target / 'js/plugins.js',
                      self.config_file, self.root / 'approved-window.json',
                      Path(default_coverage_review_path(self.target))]
        original_targets = {path: path.read_bytes() for path in self.target.rglob('*.json')}
        for path in candidates:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path == self.config_file:
                    changed = json.loads(original)
                    changed['sync']['chunk_size'] = 5
                    path.write_text(json.dumps(changed), encoding='utf-8')
                elif path.name == 'coverage_review.json':
                    changed = json.loads(original)
                    changed['reviewer']['id'] = 'another-test-review'
                    path.write_text(json.dumps(changed), encoding='utf-8')
                elif path.name == 'approved-window.json':
                    changed = json.loads(original)
                    changed['message_styles'] = False
                    path.write_text(json.dumps(changed), encoding='utf-8')
                else:
                    path.write_bytes(original + b' ')
                with self.assertRaises((ValueError, SystemExit)):
                    batch.apply_durable_sync_results(store)
                path.write_bytes(original)
        self.assertEqual(original_targets, {path: path.read_bytes() for path in original_targets})

    def test_source_change_after_plan_blocks_first_provider_dispatch(self):
        path = self.source / 'data/Map001.json'
        path.write_bytes(path.read_bytes() + b' ')
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            with self.assertRaisesRegex(Exception, 'freshness'):
                self.service.start(self.context.plan_build)
        provider.assert_not_called()

    def test_late_source_change_rolls_back_all_target_replacements(self):
        store = self.checked()
        before = {path: path.read_bytes() for path in self.target.rglob('*.json')}
        original_replace = atomic_io.os.replace
        touched = False

        def late_change(source, destination):
            nonlocal touched
            result = original_replace(source, destination)
            if not touched and str(source).endswith('.txn.tmp') and Path(destination).is_relative_to(self.target):
                touched = True
                path = self.source / 'data/Actors.json'
                path.write_bytes(path.read_bytes() + b' ')
            return result

        with mock.patch.object(atomic_io.os, 'replace', side_effect=late_change):
            with self.assertRaises((ValueError, SystemExit)):
                batch.apply_durable_sync_results(store)
        self.assertTrue(touched)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_provider_control_damage_is_not_accepted(self):
        original = self.response

        def damaged(request, *args, **kwargs):
            response = original(request, *args, **kwargs)
            for item in response['parsed']['translations']:
                item['translation'] = structure_protection.MARKER.sub('', item['translation'])
            return response

        with mock.patch('execution.sync_request.run_sync_request', side_effect=damaged):
            result = self.service.start(self.context.plan_build)
        self.assertLess(result['progress']['items']['accepted'], 14)
        self.assertGreater(result['progress']['items']['unresolved'], 0)
        self.assertEqual((self.target / 'data/Map001.json').read_bytes(),
                         (self.source / 'data/Map001.json').read_bytes())

    def test_missing_review_blocks_plan_and_doctor_preserves_unknown_vs_zero(self):
        Path(default_coverage_review_path(self.target)).unlink()
        with self.assertRaisesRegex(ValueError, 'review'):
            runtime.prepare_sync_translation_execution_context(require_provider=False)
        report = batch.collect_doctor_report()
        self.assertEqual(report['engine_status']['status'], 'blocked')
        self.assertEqual(report['pending_task_count'], 14)
        self.assertEqual(report['workflow_state'], '')
        (self.source / 'data/CommonEvents.json').unlink()
        report = batch.collect_doctor_report()
        self.assertNotIn('pending_task_count', report)
        self.assertFalse(report['coverage_evidence_available'])

    def test_completed_resume_and_derive_reuse_without_another_provider_call(self):
        store = self.completed()
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            resumed = self.service.resume(store.run_id)
            derived = self.service.derive(store.run_id, self.context.plan_build)
        provider.assert_not_called()
        self.assertEqual(resumed['run_status'], 'completed')
        self.assertEqual(derived['run_status'], 'completed')
        self.assertNotEqual(derived['run_id'], store.run_id)

    def test_new_denied_check_revokes_preview_even_with_force(self):
        store = self.checked()
        changed = self.source / 'data/Actors.json'
        original = changed.read_bytes()
        changed.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            batch.check_durable_sync_results(store)
        changed.write_bytes(original)
        self.assertIsNone(store.get_artifact(kind='preview_manifest'))
        with self.assertRaises(cli_contract.MachineContractError):
            batch.apply_durable_sync_results(store)

    def test_preflight_and_coverage_export_are_offline_and_routes_fail_before_prepare(self):
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            output = self.root / 'coverage-export'
            batch.export_live_coverage_package(output)
            self.assertTrue((output / 'coverage_review_template.json').is_file())
            with mock.patch('providers.model_capability_probe.default_credential_loader', return_value='test-only'):
                payload = batch.run_translate_preflight(SimpleNamespace(strategy='sync', profile=''))
            self.assertEqual(payload['project']['engine'], 'rpgmaker_mv')
            self.assertEqual(payload['counts']['pending_items'], 14)
            with self.assertRaises(cli_contract.MachineContractError):
                batch.run_translate_preflight(SimpleNamespace(strategy='gemini_batch', profile=''))
            with mock.patch.object(runtime, 'run_prepare_steps') as prepare:
                with self.assertRaisesRegex(ValueError, 'durable Sync'):
                    batch.create_batch_package()
            prepare.assert_not_called()
        provider.assert_not_called()

    def test_crash_mid_apply_recovers_before_retry(self):
        store = self.checked()
        real_replace = atomic_io.os.replace
        replaced = 0

        def crash(source, destination):
            nonlocal replaced
            if str(source).endswith('.txn.tmp') and Path(destination).is_relative_to(self.target):
                replaced += 1
                if replaced == 2:
                    raise SystemExit('Original MV product test crash')
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, 'replace', side_effect=crash):
            with self.assertRaisesRegex(SystemExit, 'test crash'):
                batch.apply_durable_sync_results(store)
        self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        expected = Path(__file__).resolve().parents[1] / 'fixtures/rpgmaker_mv_narrative/expected'
        for path in expected.rglob('*.json'):
            self.assertEqual((self.target / path.relative_to(expected)).read_bytes(), path.read_bytes())

    def test_unknown_dispatched_attempt_is_not_automatically_replayed(self):
        plan = self.context.plan_build
        store, _created = SyncRunStore.bootstrap(self.runs, build_run_id(),
            plan=plan.plan.to_dict(), requests=[request.to_dict() for request in plan.requests])
        owner = 'original-mv-crash-fixture'
        store.acquire_lease(owner_token=owner)
        first = plan.requests[0]
        attempt = store.prepare_attempt(request_id=first.request_id, owner_token=owner)
        store.dispatch_attempt(attempt_id=attempt, owner_token=owner)
        store.mark_outcome_unknown(attempt_id=attempt, owner_token=owner)
        store.release_lease(owner_token=owner)
        with mock.patch('execution.sync_request.run_sync_request', side_effect=self.response) as provider:
            result = self.service.resume(store.run_id)
        self.assertNotIn(first.user_prompt, [call.args[0]['contents'] for call in provider.call_args_list])
        self.assertGreater(result['progress']['items']['unresolved'], 0)
        self.assertEqual(result['progress']['requests']['outcome_unknown'], 1)


if __name__ == '__main__':
    unittest.main()
