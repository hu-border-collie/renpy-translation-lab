"""Original MV preprocessing and frozen retrieval without real providers."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemini_translate_batch as batch
import translator_runtime as runtime
from execution import mv_preprocessing as mv
from tests.support.mv_project import make_mv_project, review_mv_project, write_mv_scope


class MvPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.file, self.config, self.positions = make_mv_project(self.root)
        self.source, self.target = self.root / 'source', self.root / 'target'
        saved = runtime.snapshot_runtime_config()
        self.stack.callback(runtime.apply_runtime_config, saved)
        batch_saved = {key: value for key, value in vars(batch).items() if key.upper() == key}
        self.stack.callback(vars(batch).update, batch_saved)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(self.file)))
        self.stack.enter_context(mock.patch.object(runtime, 'load_config'))
        self.stack.enter_context(mock.patch.object(runtime, 'LOG_DIR', str(self.root / 'logs')))
        self.stack.enter_context(mock.patch.object(batch, 'SYNC_RUNS_DIR', str(self.root / 'sync-runs')))
        self.config['sync']['rag'] = {'enabled': True, 'embedding_backend': 'openai_compatible',
            'embedding_provider': 'openai', 'embedding_model': 'original-test-embedding',
            'output_dimensionality': 3, 'store_dir': str(self.root / 'history'), 'min_similarity': 0.1}
        self.config['sync']['source_index'] = {'enabled': True, 'store_dir': str(self.root / 'source-index'),
                                               'min_similarity': 0.1}
        self.history = self.root / 'reviewed.jsonl'
        self.config['rpgmaker_mv']['reviewed_history_file'] = str(self.history)
        self.configure()
        self.write_history()
        self.configure()
        self.embed = self.stack.enter_context(mock.patch.object(runtime, 'embed_texts',
            side_effect=lambda texts, *_args: [[1.0, 0.5, 0.25] for _ in texts]))

    def configure(self):
        self.file.write_text(json.dumps(self.config), encoding='utf-8')
        runtime.load_translator_settings()
        batch.load_batch_settings()
        self.snapshot = review_mv_project(self.config, self.file)

    def write_history(self, *, status='approved', occurrence=None, path=None):
        row = occurrence or self.snapshot.selected_occurrences[0]
        position = next(item for item in self.positions if item['source'] == row.unit.source_text)
        payload = {'schema_version': 1, 'engine': 'rpgmaker_mv',
            'adapter_version': self.snapshot.project.adapter_version,
            'support_window_digest': self.snapshot.project.support_window_digest,
            'file_rel_path': row.unit.file_rel_path, 'json_path': list(row.locator.locator['json_path']),
            'source': row.unit.source_text, 'translation': position['translation'],
            'source_file_sha256': self.snapshot.project.document_by_path()[row.unit.file_rel_path].sha256,
            'review': {'status': status, 'reviewed_at': '2026-10-26T00:00:00Z',
                       'reviewer': {'type': 'agent', 'id': 'original-unittest-review'}}}
        (path or self.history).write_text(json.dumps(payload, ensure_ascii=False) + '\n', encoding='utf-8')
        return payload

    def prepare_stores(self):
        source = batch.bootstrap_source_index(skip_prepare=True)
        history = batch.bootstrap_rag_store(skip_prepare=True)
        self.assertTrue(source['prepared'], source)
        self.assertTrue(history['prepared'], history)
        return source, history

    def test_source_bootstrap_reports_waiting_before_provider_and_ready_after_persistence(self):
        output = io.StringIO()

        def embed(texts, *_args):
            self.assertIn('MV context preparation: 14 segments, 0 reused, 14 pending.', output.getvalue())
            self.assertIn('MV context embedding request: 1-14/14.', output.getvalue())
            self.assertNotIn('MV context preparation complete:', output.getvalue())
            data = self.root / 'source-index/source_segments.jsonl'
            self.assertFalse(data.exists() and data.read_text(encoding='utf-8').strip())
            return [[1.0, 0.5, 0.25] for _ in texts]

        self.embed.side_effect = embed
        with redirect_stdout(output):
            result = batch.bootstrap_source_index(skip_prepare=True)
        self.assertTrue(result['prepared'], result)
        progress = [line for line in output.getvalue().splitlines() if line.startswith('MV context')]
        self.assertEqual(progress[-3:], [
            'MV context validation started.', 'MV context persistence started.',
            'MV context preparation complete: 14 segments.',
        ])
        metadata = json.loads((self.root / 'source-index/source_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['mv_preparation']['status'], 'ready')

    def test_source_bootstrap_provider_failure_never_reports_complete(self):
        output = io.StringIO()
        self.embed.side_effect = TimeoutError('original fixture timeout')
        with redirect_stdout(output):
            result = batch.bootstrap_source_index(skip_prepare=True)
        self.assertFalse(result['prepared'], result)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('MV context embedding request: 1-14/14.', output.getvalue())
        self.assertNotIn('MV context persistence started.', output.getvalue())
        self.assertNotIn('MV context preparation complete:', output.getvalue())

    def test_actual_source_occurrences_and_reviewed_history_are_separate_and_reusable(self):
        source, history = self.prepare_stores()
        self.assertEqual(source['scanned'], 14)
        self.assertEqual(history['scanned'], 1)
        source_rows = [json.loads(line) for line in (self.root / 'source-index/source_segments.jsonl').read_text(encoding='utf-8').splitlines()]
        history_rows = [json.loads(line) for line in (self.root / 'history/history.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all('translated_text' not in row for row in source_rows))
        self.assertEqual(len({row['source_id'] for row in source_rows}), 14)
        duplicate = [row for row in source_rows if row['source_text'] == 'Same words.']
        self.assertEqual(len(duplicate), 2)
        self.assertNotEqual(duplicate[0]['json_path'], duplicate[1]['json_path'])
        self.assertTrue(all('line_start' not in row for row in source_rows))
        self.assertEqual(history_rows[0]['quality_state'], 'reviewed')
        self.embed.reset_mock()
        source, history = self.prepare_stores()
        self.assertEqual(source['reused_embeddings'], 14)
        self.embed.assert_not_called()

    def test_v1_sync_embedding_and_keyword_stage_are_used_instead_of_retained_batch_fields(self):
        import copy
        from providers.model_routing_migration import preview_migration
        routing = preview_migration(self.config).config['model_routing']
        primary = routing['legacy_entrypoints']['sync_profile_id']
        embedding = routing['profiles'][primary]['embedding_profile_id']
        routing['profiles'][embedding]['model'] = 'original-v1-sync-embedding'
        routing['profiles']['mv-keywords'] = copy.deepcopy(routing['profiles'][primary])
        routing['profiles']['mv-keywords']['label'] = 'Original MV keyword fixture'
        routing['profiles']['mv-context-embedding'] = copy.deepcopy(routing['profiles'][embedding])
        routing['profiles']['mv-context-embedding']['model'] = 'original-v1-routed-embedding'
        routing['profiles']['mv-translation'] = copy.deepcopy(routing['profiles'][primary])
        routing['profiles']['mv-translation']['embedding_profile_id'] = 'mv-context-embedding'
        routing['defaults']['execution_strategy'] = 'sync'
        routing['routes']['keyword'] = {'profile_id': 'mv-keywords', 'strategy': 'sync'}
        routing['routes']['translation'] = {'profile_id': 'mv-translation', 'strategy': 'sync'}
        self.config['model_routing'] = routing
        self.configure()
        config_before = self.file.read_bytes()
        self.prepare_stores()
        self.assertEqual(self.file.read_bytes(), config_before)
        self.assertEqual(runtime.current_sync_embedding_settings().model, 'original-v1-routed-embedding')
        status = mv.context_status(self.config, self.file)
        self.assertTrue(status['source_index']['prepared'], status)
        self.assertTrue(status['history']['prepared'], status)
        payload = {'candidates': [], 'chunk_summary': '', 'summary_evidence_item_ids': []}
        with mock.patch.object(batch, 'run_sync_request', return_value={
                'response_payload': {'parsed': payload}, 'response_text': json.dumps(payload), 'finish_reason': 'STOP'}):
            result = batch.sync_keyword_candidates(limit=1)
        manifest = batch.load_manifest(result['manifest_path'])
        self.assertEqual(manifest['model_routing']['routes']['keyword']['profile_id'], 'mv-keywords')

    def test_pending_history_never_becomes_adopted_and_target_is_not_scanned(self):
        path = self.target / 'data/Map001.json'
        path.write_bytes(path.read_bytes().replace(b'The door is open.', '门已打开。'.encode()))
        self.write_history(status='pending')
        self.configure()
        result = batch.bootstrap_rag_store(skip_prepare=True)
        self.assertFalse(result['prepared'])
        self.assertEqual(result['reason'], 'no_reviewed_history')
        self.assertFalse((self.root / 'history').exists())
        self.embed.assert_not_called()

    def test_history_selection_changes_are_stale_before_query(self):
        for selection in ('empty_file', 'same_bytes_file', 'unset'):
            with self.subTest(selection=selection):
                self.config['rpgmaker_mv']['reviewed_history_file'] = str(self.history)
                self.configure()
                self.prepare_stores()
                source_bytes = (self.root / 'source-index/source_segments.jsonl').read_bytes()
                replacement = self.root / 'replacement-history.jsonl'
                replacement.write_bytes(self.history.read_bytes() if selection == 'same_bytes_file' else b'')
                self.config['rpgmaker_mv']['reviewed_history_file'] = str(replacement) if selection != 'unset' else ''
                self.configure()
                self.embed.reset_mock()
                status = mv.context_status(self.config, self.file)
                self.assertFalse(status['history']['prepared'], status)
                self.assertEqual(status['history']['status'], 'stale')
                self.assertEqual(status['history']['reason'], 'reviewed_history_selection_changed')
                self.assertIsNone(status['history']['record_count'])
                self.assertTrue(status['source_index']['prepared'])
                hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Shared help'}], 'history')
                self.assertFalse(hits)
                self.assertEqual(diagnostics['reason'], 'reviewed_history_selection_changed')
                self.embed.assert_not_called()
                self.assertEqual((self.root / 'source-index/source_segments.jsonl').read_bytes(), source_bytes)

    def test_new_requests_omit_previous_history_after_corpus_switch(self):
        self.config['sync']['source_index']['enabled'] = False
        self.configure()
        batch.bootstrap_rag_store(skip_prepare=True)
        old_translation = json.loads(self.history.read_text(encoding='utf-8'))['translation']
        before = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.assertIn(old_translation, '\n'.join(request.user_prompt for request in before.plan_build.requests))
        replacement = self.root / 'empty-history.jsonl'
        replacement.write_text('', encoding='utf-8')
        self.config['rpgmaker_mv']['reviewed_history_file'] = str(replacement)
        self.configure()
        self.embed.reset_mock()
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        self.assertTrue(context.plan_build.requests)
        self.assertFalse(old_translation in '\n'.join(request.user_prompt for request in context.plan_build.requests),
                         'Old corpus translation leaked into a new frozen request')
        self.embed.assert_not_called()
        provider.assert_not_called()

    def test_explicit_rebuild_after_history_switch_prepares_known_empty_corpus(self):
        self.prepare_stores()
        replacement = self.root / 'empty-history.jsonl'
        replacement.write_text('', encoding='utf-8')
        self.config['rpgmaker_mv']['reviewed_history_file'] = str(replacement)
        self.configure()
        self.embed.reset_mock()
        summary = batch.bootstrap_rag_store(skip_prepare=True)
        self.assertTrue(summary['prepared'])
        self.assertEqual(summary['pruned'], 1)
        self.assertEqual(summary['current_records'], 0)
        status = mv.context_status(self.config, self.file)
        self.assertTrue(status['history']['prepared'])
        self.assertEqual(status['history']['record_count'], 0)
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Shared help'}], 'history')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['reason'], 'prepared_empty_store')
        self.embed.assert_not_called()

    def test_seed_only_history_remains_valid_and_tracks_seed_changes(self):
        self.config['rpgmaker_mv']['reviewed_history_file'] = ''
        self.configure()
        summary = batch.bootstrap_rag_store(skip_prepare=True, seed_jsonl_paths=[str(self.history)])
        self.assertTrue(summary['prepared'])
        status = mv.context_status(self.config, self.file)
        self.assertTrue(status['history']['prepared'], status)
        self.assertEqual(status['history']['record_count'], 1)
        hits, _diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Shared help'}], 'history')
        self.assertEqual(len(hits), 1)
        self.history.write_bytes(self.history.read_bytes() + b'\n')
        self.embed.reset_mock()
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Shared help'}], 'history')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['status'], 'stale')
        self.assertEqual(diagnostics['reason'], 'reviewed_history_changed')
        self.embed.assert_not_called()

    def test_configured_and_seed_history_are_both_bound_to_the_selection(self):
        seed = self.root / 'extra-reviewed.jsonl'
        self.write_history(occurrence=self.snapshot.selected_occurrences[1], path=seed)
        summary = batch.bootstrap_rag_store(skip_prepare=True, seed_jsonl_paths=[str(seed)])
        self.assertTrue(summary['prepared'])
        self.assertEqual(summary['current_records'], 2)
        metadata = json.loads((self.root / 'history/metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(set(metadata['mv_preparation']['history_inputs']), {str(self.history), str(seed)})
        self.assertTrue(mv.context_status(self.config, self.file)['history']['prepared'])
        self.config['rpgmaker_mv']['reviewed_history_file'] = ''
        self.configure()
        self.embed.reset_mock()
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Shared help'}], 'history')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['reason'], 'reviewed_history_selection_changed')
        self.embed.assert_not_called()

    def test_history_without_selection_metadata_requires_explicit_rebuild(self):
        for selection in (str(self.history), ''):
            with self.subTest(selection=selection):
                self.config['rpgmaker_mv']['reviewed_history_file'] = selection
                self.configure()
                summary = batch.bootstrap_rag_store(skip_prepare=True, seed_jsonl_paths=[str(self.history)])
                self.assertTrue(summary['prepared'])
                path = self.root / 'history/metadata.json'
                metadata = json.loads(path.read_text(encoding='utf-8'))
                metadata['mv_preparation'].pop('history_selection', None)
                path.write_text(json.dumps(metadata), encoding='utf-8')
                self.embed.reset_mock()
                hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Shared help'}], 'history')
                self.assertFalse(hits)
                self.assertFalse(diagnostics['prepared'])
                self.assertEqual(diagnostics['status'], 'stale')
                self.assertEqual(diagnostics['reason'], 'reviewed_history_selection_changed')
                self.embed.assert_not_called()

    def test_equivalent_relative_history_path_keeps_preparation_current(self):
        self.prepare_stores()
        metadata = (self.root / 'history/metadata.json').read_bytes()
        self.config['rpgmaker_mv']['reviewed_history_file'] = self.history.name
        self.configure()
        self.embed.reset_mock()
        status = mv.context_status(self.config, self.file)
        self.assertTrue(status['history']['prepared'], status)
        self.assertEqual(status['history']['record_count'], 1)
        self.assertEqual((self.root / 'history/metadata.json').read_bytes(), metadata)
        self.embed.assert_not_called()

    def test_zero_hits_failed_store_and_disabled_are_distinct(self):
        self.prepare_stores()
        self.config['sync']['source_index']['min_similarity'] = 1.01
        self.configure()
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'No close result.'}], 'source')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['reason'], 'zero_hits')
        self.assertTrue(diagnostics['prepared'])
        self.assertGreater(self.embed.call_count, 0)
        self.embed.reset_mock()
        path = self.root / 'source-index/source_segments.jsonl'
        path.write_bytes(path.read_bytes() + b' ')
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Same words.'}], 'source')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['status'], 'failed')
        self.assertEqual(diagnostics['reason'], 'store_records_changed')
        self.embed.assert_not_called()
        self.config['sync']['source_index']['enabled'] = False
        self.configure()
        status = mv.context_status(self.config, self.file, snapshot=self.snapshot)
        self.assertEqual(status['source_index']['status'], 'disabled')
        self.assertFalse(status['source_index']['prepared'])

    def test_no_prune_cannot_retrieve_old_scope_records(self):
        self.prepare_stores()
        scope = self.root / 'narrow.json'
        write_mv_scope(scope, self.source, self.positions[:1])
        self.config['rpgmaker_mv']['occurrence_scope_file'] = str(scope)
        self.configure()
        summary = batch.bootstrap_source_index(skip_prepare=True, prune=False)
        self.assertEqual(summary['stale_count'], 13)
        self.assertEqual(summary['pruned'], 0)
        self.assertEqual(summary['current_records'], 1)
        self.assertEqual(summary['history_records_after'], 14)
        hits, stats = mv.retrieve(runtime, self.snapshot, [{'text': 'Same words.'}], 'source')
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['source_id'], self.snapshot.selected_occurrences[0].unit.id)
        self.assertTrue(stats['prepared'])

    def test_approved_history_is_bound_and_bad_rows_fail_before_embedding(self):
        original = self.write_history()
        import copy
        for key, value in (('source_file_sha256', '0' * 64), ('json_path', ['events', True]),
                           ('support_window_digest', 'changed'), ('adapter_version', '0.2.0')):
            payload = copy.deepcopy(original)
            payload[key] = value
            self.history.write_text(json.dumps(payload), encoding='utf-8')
            self.configure()
            with self.subTest(key=key), self.assertRaises(ValueError):
                batch.bootstrap_rag_store(skip_prepare=True)
        self.embed.assert_not_called()
        self.assertFalse((self.root / 'history').exists())

    def test_frozen_context_run_still_checks_applies_and_does_not_auto_adopt_history(self):
        self.prepare_stores()
        from tests.execution import test_mv_durable_sync as narrative
        from execution.sync_run_service import build_production_sync_run_service
        from execution.sync_run_store import SyncRunStore
        self.context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        runs = self.root / 'logs/sync_runs'
        service = build_production_sync_run_service(runs, self.context)
        history_before = (self.root / 'history/history.jsonl').read_bytes()
        with mock.patch('execution.sync_request.run_sync_request',
                side_effect=lambda *args, **kwargs: narrative.MvDurableSyncTests.response(self, *args, **kwargs)):
            started = service.start(self.context.plan_build)
        self.assertEqual(started['run_status'], 'completed', started)
        store = SyncRunStore(runs, started['run_id'])
        checked = batch.check_durable_sync_results(store)
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
        with mock.patch.object(runtime, 'collect_sync_rag_records_for_file', side_effect=AssertionError('RenPy scan forbidden')):
            self.assertEqual(batch.apply_durable_sync_results(store)['state'], 'applied')
        self.assertEqual((self.root / 'history/history.jsonl').read_bytes(), history_before)

    def test_frozen_requests_contain_typed_source_and_bilingual_context(self):
        self.prepare_stores()
        context = runtime.prepare_sync_translation_execution_context(require_provider=False)
        prompt = '\n'.join(request.user_prompt for request in context.plan_build.requests)
        self.assertIn('RELATED PROJECT CONTEXT', prompt)
        self.assertIn('RETRIEVED MEMORY', prompt)
        self.assertIn('Source excerpt:', prompt)
        self.assertIn('Translation:', prompt)
        self.assertIn('data/Map001.json::[', prompt)
        self.assertNotIn('data/Map001.json:0-0', prompt)
        self.assertGreater(len(context.plan_build.requests), 0)

    def test_preflight_is_offline_with_retrieval_enabled(self):
        self.prepare_stores()
        self.embed.reset_mock()
        with mock.patch('execution.sync_request.run_sync_request') as provider:
            context = runtime.prepare_sync_translation_execution_context(require_provider=False, preflight=True)
        self.embed.assert_not_called()
        provider.assert_not_called()
        self.assertTrue(all(not capture['retrieval_blocks_text'] for capture in context.captures))
        from types import SimpleNamespace
        with mock.patch('providers.model_capability_probe.default_credential_loader', return_value=''), \
                mock.patch('execution.sync_request.run_sync_request') as provider:
            result = batch.run_translate_preflight(SimpleNamespace(strategy='sync', output='json', profile=''))
        self.assertTrue(result['context_preparation']['source_index']['prepared'])
        self.assertTrue(result['context_preparation']['history']['prepared'])
        provider.assert_not_called()
        self.embed.assert_not_called()

    def test_empty_history_is_known_zero_and_failure_is_not_prepared(self):
        self.history.write_text('', encoding='utf-8')
        self.configure()
        result = batch.bootstrap_rag_store(skip_prepare=True)
        self.assertTrue(result['prepared'])
        self.assertEqual(result['scanned'], 0)
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Same words.'}], 'history')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['record_count'], 0)
        self.assertEqual(diagnostics['reason'], 'prepared_empty_store')
        self.embed.assert_not_called()
        self.embed.side_effect = RuntimeError('original failure detail must stay private')
        result = batch.bootstrap_source_index(skip_prepare=True)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['prepared'])
        self.assertNotIn('original failure detail', json.dumps(result))
        state = mv.context_status(self.config, self.file)
        self.assertEqual(state['source_index']['status'], 'failed')
        self.assertIsNone(state['source_index']['record_count'])

    def test_preparation_holds_existing_store_lock_and_queries_downgrade_during_build(self):
        from context_engine.rag_memory import JsonSourceIndexStore, JsonSourceIndexStoreLockError
        def embedding(texts, *_args):
            competing = JsonSourceIndexStore(self.root / 'source-index')
            with self.assertRaises(JsonSourceIndexStoreLockError):
                competing.set_metadata(original_fixture='must not interleave')
            status = mv.context_status(self.config, self.file)
            self.assertFalse(status['source_index']['prepared'])
            return [[1.0, 0.5, 0.25] for _ in texts]
        self.embed.side_effect = embedding
        self.assertTrue(batch.bootstrap_source_index(skip_prepare=True)['prepared'])
        self.assertFalse((self.root / 'source-index/.source_index.lock').exists())

    def test_reviewed_keyword_evidence_keeps_duplicate_json_occurrences(self):
        from quality_review import keyword_history
        payloads = []
        for row in self.snapshot.selected_occurrences:
            if row.unit.source_text != 'Same words.':
                continue
            payload = self.write_history()
            payload.update(file_rel_path=row.unit.file_rel_path, json_path=row.locator.locator['json_path'],
                source=row.unit.source_text, translation='第一种说法。' if not payloads else '第二种说法。',
                source_file_sha256=self.snapshot.project.document_by_path()[row.unit.file_rel_path].sha256)
            payloads.append(payload)
        self.history.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in payloads), encoding='utf-8')
        self.configure()
        from execution.mv_sync import material_identity
        corpus = batch.collect_keyword_history_corpus({'mv_preprocessing_identity': material_identity(self.snapshot)})
        self.assertEqual(len(corpus['items']), 2, corpus)
        evidence = keyword_history.build_keyword_history_evidence(
            {'source': 'Same words.', 'suggested_target': '说法'}, corpus['items'])
        self.assertEqual(evidence['occurrence_count'], 2)
        self.assertTrue(evidence['review_required'])
        self.assertEqual(len({json.dumps(row['locator'], sort_keys=True) for row in evidence['occurrences']}), 2)
        self.assertNotIn(':L0', batch.format_keyword_history_markdown(evidence))

    def test_mv_literal_brackets_are_not_renpy_interpolation_history(self):
        from quality_review import keyword_history
        from execution.mv_sync import material_identity
        corpus = batch.collect_keyword_history_corpus({'mv_preprocessing_identity': material_identity(self.snapshot)})
        row = {**corpus['items'][0], 'source': '[Continue] \\v[1]', 'current_translation': '[继续] \\v[1]'}
        evidence = keyword_history.build_keyword_history_evidence(
            {'source': '[Continue]', 'suggested_target': '[继续]'}, [row])
        self.assertEqual(evidence['occurrence_count'], 1)
        self.assertNotIn('renpy_interpolation_only', evidence['conflict_codes'])

    def test_bootstrap_machine_output_does_not_turn_missing_into_ready(self):
        self.history.unlink()
        self.configure()
        result = batch.bootstrap_rag_store(skip_prepare=True)
        envelope = batch.build_machine_success_envelope('bootstrap-rag', result, None)
        self.assertEqual(envelope['status'], 'missing')
        self.assertFalse(envelope['result']['prepared'])
        self.assertEqual(envelope['result']['reason'], 'reviewed_history_missing')

    def test_disabled_bootstrap_needs_no_embedding_binding_or_credentials(self):
        self.config['sync']['rag']['enabled'] = False
        self.config['sync']['source_index']['enabled'] = False
        self.configure()
        with mock.patch.object(runtime, 'current_sync_embedding_settings', side_effect=AssertionError('disabled embedding')), \
                mock.patch.object(runtime, '_require_gemini_api_key', side_effect=AssertionError('disabled credentials')):
            for command in ('bootstrap-rag', 'bootstrap-source-index'):
                output = io.StringIO()
                with redirect_stdout(output):
                    batch.main([command, '--output', 'json'])
                envelope = json.loads(output.getvalue())
                self.assertEqual(envelope['status'], 'disabled')
                self.assertFalse(envelope['result']['prepared'])
        self.embed.assert_not_called()

    def test_identity_drift_and_source_change_are_diagnosed_without_embedding(self):
        self.prepare_stores()
        settings = runtime.current_sync_embedding_settings()
        from dataclasses import replace
        status = mv.inspect_store(self.snapshot, self.root / 'source-index', replace(settings, model='another-model'), 'source')
        self.assertEqual(status['status'], 'incompatible')
        self.assertEqual(mv.inspect_store(self.snapshot, self.root / 'missing', settings, 'source')['status'], 'missing')
        self.embed.reset_mock()
        for root in (self.source, self.target):
            path = root / 'data/Map002.json'
            path.write_bytes(path.read_bytes() + b' ')
        self.configure()
        hits, diagnostics = mv.retrieve(runtime, self.snapshot, [{'text': 'Same words.'}], 'source')
        self.assertFalse(hits)
        self.assertEqual(diagnostics['status'], 'stale')
        self.embed.assert_not_called()

    def test_mv_keyword_jobs_and_report_are_real_source_positions(self):
        def response(request, *_args, **_kwargs):
            text = request['contents'][0]['parts'][0]['text']
            items = json.loads(text.split('TARGET LINES:\n', 1)[1].split('\n\n', 1)[0])
            self.assertTrue(all('line' not in item for item in items))
            self.assertTrue(all(item['source_locator']['engine'] == 'rpgmaker_mv' for item in items))
            payload = {'candidates': [{'source': 'Shared help', 'suggested_target': '共用提示', 'category': 'term',
                'confidence': 0.9, 'evidence': 'original fixture', 'source_item_ids': [items[0]['id']]}],
                'chunk_summary': '原创场景概要。', 'summary_evidence_item_ids': [items[0]['id']]}
            return {'response_payload': {'parsed': payload}, 'response_text': json.dumps(payload), 'finish_reason': 'STOP'}
        with mock.patch.object(batch, 'run_sync_request', side_effect=response):
            result = batch.sync_keyword_candidates(limit=1)
        manifest = batch.load_manifest(result['manifest_path'])
        self.assertEqual(manifest['mode'], 'keyword_extraction')
        self.assertIn('mv_preprocessing_identity', manifest)
        self.assertTrue(all('source_locator' in item for item in manifest['chunks'][0]['items']))
        candidates = [json.loads(line) for line in Path(result['jsonl_path']).read_text(encoding='utf-8').splitlines()]
        self.assertTrue(candidates, result)
        self.assertTrue(candidates[0]['source_locations'])
        self.assertTrue(candidates[0]['history_evidence']['review_required'])
        self.assertIn(manifest['chunks'][0]['file_rel_path'] + '::[', Path(result['markdown_path']).read_text(encoding='utf-8'))
        self.assertNotIn(':L0', Path(result['markdown_path']).read_text(encoding='utf-8'))
        self.assertFalse((self.target / 'glossary.json').exists())
        for operation in (batch.check_results, batch.apply_results):
            with self.assertRaises(SystemExit):
                operation(result['manifest_path'])


if __name__ == '__main__':
    unittest.main()
