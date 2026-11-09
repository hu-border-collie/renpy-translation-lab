"""Offline stop/restart of the actual sync keyword execution and export."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemini_translate_batch as batch
from execution.keyword_checkpoint import KeywordCheckpoint, KeywordCheckpointError, KeywordRequestUnknownError
from foundation.atomic_io import exclusive_file_lock, file_sha256


class SyncKeywordResumeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.source = self.root / 'script.rpy'
        self.source.write_text('translate schinese strings:\n    old "Fixture"\n    new ""\n', encoding='utf-8')
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(batch.legacy, 'current_project_engine', return_value='renpy'))
        self.stack.enter_context(mock.patch.multiple(batch.legacy, BASE_DIR=str(self.root), TL_DIR=str(self.root)))
        self.stack.enter_context(mock.patch.object(batch, 'import_manifest_usage_best_effort'))
        self.plan = batch.model_profile.resolve_routing_plan(
            {'sync': {'backend': 'gemini', 'model': 'fixture-keyword-model'}})
        self.chunks = [{
            'key': f'kw-{index}', 'file_path': str(self.source), 'file_rel_path': 'script.rpy',
            'chunk_index': index + 1, 'context_past': [], 'context_future': [],
            'items': [{'id': f'item-{index}', 'text': 'Fixture', 'file_rel_path': 'script.rpy',
                       'line_number': 2, 'line': 1, 'start': 8, 'end': 15, 'prefix': '', 'quote': '"'}],
        } for index in range(2)]
        self.rows = [{'key': chunk['key'], 'request': {
            'contents': [{'parts': [{'text': chunk['items'][0]['id']}]}],
            'generationConfig': {'maxOutputTokens': 256},
        }} for chunk in self.chunks]
        self.package = self.root / 'package'
        self.package.mkdir()
        self.manifest = batch.make_sync_manifest(
            package_dir=str(self.package), mode=batch.MANIFEST_MODE_KEYWORD_EXTRACTION,
            display_name='Original keyword restart fixture', chunks=self.chunks,
            request_rows=self.rows, settings={}, routing_plan=self.plan,
            extra_fields={'keyword_checkpoint_version': 1,
                          'keyword_source_files': {str(self.source): file_sha256(self.source)}})

    def response(self, request, *_args, **_kwargs):
        item_id = request['contents'][0]['parts'][0]['text']
        text = json.dumps({'candidates': [], 'chunk_summary': 'An original scene.',
                           'summary_evidence_item_ids': [item_id]})
        return {'response_text': text, 'response_payload': {
            'candidates': [{'content': {'parts': [{'text': text}]}}]},
            'finish_reason': 'STOP', 'usage_metadata': {'totalTokenCount': 7}}

    def stop_after_first_saved_chunk(self):
        complete = KeywordCheckpoint.complete

        def stop(checkpoint, key, result):
            complete(checkpoint, key, result)
            raise KeyboardInterrupt

        with (mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch,
              mock.patch.object(KeywordCheckpoint, 'complete', stop)):
            with self.assertRaises(KeyboardInterrupt):
                batch.execute_sync_request_rows(self.manifest, self.rows)
        self.assertEqual(dispatch.call_count, 1)

    def test_stop_then_resume_only_dispatches_missing_chunk_and_exports(self):
        self.stop_after_first_saved_chunk()
        with mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch:
            exported = batch.sync_keyword_candidates(resume=self.manifest)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(dispatch.call_args.args[0], self.rows[1]['request'])
        manifest = batch.load_manifest(exported['manifest_path'])
        self.assertEqual(manifest['sync_summary']['resumed_request_count'], 1)
        self.assertEqual(manifest['sync_summary']['reused_response_count'], 1)
        rows = [json.loads(line) for line in Path(manifest['result_jsonl_path']).read_text().splitlines()]
        self.assertEqual([row['key'] for row in rows], ['kw-0', 'kw-1'])
        self.assertEqual(exported['summary']['chunk_summary_count'], 2)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()

    def test_interrupt_during_second_call_requires_ack_but_preserves_first(self):
        def interrupt_second(request, *args, **kwargs):
            if request == self.rows[1]['request']:
                raise KeyboardInterrupt
            return self.response(request)
        with mock.patch.object(batch, 'run_sync_request', side_effect=interrupt_second):
            with self.assertRaises(KeyboardInterrupt):
                batch.execute_sync_request_rows(self.manifest, self.rows)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaises(KeywordRequestUnknownError):
                batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
        with mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest, retry_unknown=True)
        self.assertEqual(dispatch.call_count, 1)

    def test_partial_service_failure_resume_reuses_successful_response(self):
        def fail_second(request, *args, **kwargs):
            if request == self.rows[1]['request']:
                raise RuntimeError('service temporarily unavailable')
            return self.response(request)

        with mock.patch.object(batch, 'run_sync_request', side_effect=fail_second):
            batch.execute_sync_request_rows(self.manifest, self.rows)
        manifest = batch.load_manifest(self.manifest)
        self.assertEqual(manifest['job_state'], 'SYNC_PARTIAL')
        self.assertEqual(manifest['sync_summary']['failed_request_count'], 1)

        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaises(KeywordRequestUnknownError):
                batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()

        output = io.StringIO()
        with (mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch,
              redirect_stdout(output)):
            exported = batch.sync_keyword_candidates(resume=self.manifest, retry_unknown=True)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(dispatch.call_args.args[0], self.rows[1]['request'])
        self.assertEqual(exported['summary']['chunk_summary_count'], 2)
        manifest = batch.load_manifest(self.manifest)
        self.assertEqual(manifest['sync_summary']['failed_request_count'], 0)
        self.assertEqual(manifest['sync_summary']['reused_response_count'], 1)
        self.assertIn('total=2; reused=1; pending=1; failed=1; incomplete=0.', output.getvalue())
        self.assertIn('[1/1] kw-1', output.getvalue())
        self.assertNotIn('[1/2]', output.getvalue())
        self.assertIn('Sync keyword progress: 1/1 pending blocks processed.', output.getvalue())
        self.assertEqual(manifest['sync_summary']['pending_request_count'], 1)

    def test_completed_results_are_restored_without_contract_replay_or_checkpoint_rewrite(self):
        with mock.patch.object(batch, 'run_sync_request', side_effect=self.response):
            batch.execute_sync_request_rows(self.manifest, self.rows)
        before = batch.load_manifest(self.manifest)['sync_summary']
        checkpoint_files = list((self.package/'keyword_checkpoint/completed').glob('*.json'))
        preimages = {p: p.read_bytes() for p in checkpoint_files}
        with (mock.patch.object(batch, 'run_sync_request') as dispatch,
              mock.patch.object(batch, '_contract_from_sync_result', side_effect=AssertionError('replay')),
              mock.patch.object(KeywordCheckpoint, 'complete', side_effect=AssertionError('rewrite'))):
            batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
        for p, content in preimages.items():
            self.assertEqual(p.read_bytes(), content)
        after = batch.load_manifest(self.manifest)['sync_summary']
        for field in ('successful_request_count', 'failed_request_count',
                      'contract_first_pass_complete_chunks', 'contract_final_complete_chunks',
                      'completion_tokens', 'completion_tokens_known_requests', 'reason_counts'):
            self.assertEqual(before[field], after[field], field)
        self.assertEqual(after['restored_request_count'], 2)
        self.assertEqual(after['pending_request_count'], 0)

    def create_incomplete_result(self):
        malformed = {'response_text': '{}', 'response_payload': {}, 'finish_reason': 'STOP'}
        with mock.patch.object(batch, 'run_sync_request', side_effect=[
                malformed, malformed, self.response(self.rows[1]['request'])]):
            batch.execute_sync_request_rows(self.manifest, self.rows)
        self.assertEqual(batch.load_manifest(self.manifest)['sync_summary']['contract_partial_requests'], 1)

    def test_incomplete_result_gets_one_fresh_repair_and_keeps_original_responses(self):
        self.create_incomplete_result()
        attempts = self.package/'keyword_checkpoint/attempts'
        originals = {p: p.read_bytes() for p in attempts.glob('*.json')}
        with mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(dispatch.call_args.args[0], self.rows[0]['request'])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        manifest = batch.load_manifest(self.manifest)
        self.assertEqual(manifest['job_state'], 'SYNC_COMPLETED')
        self.assertEqual(manifest['sync_summary']['contract_final_complete_chunks'], 2)
        self.assertEqual(manifest['sync_summary']['pending_request_count'], 1)
        rows = [json.loads(s) for s in Path(manifest['result_jsonl_path']).read_text().splitlines()]
        self.assertEqual([r['key'] for r in rows], ['kw-0', 'kw-1'])
        self.assertEqual(rows[0]['keyword_checkpoint_generation'], 1)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()

    def test_repair_response_saved_before_interruption_is_not_resent(self):
        self.create_incomplete_result()
        with (mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch,
              mock.patch.object(batch, '_contract_from_sync_result', side_effect=KeyboardInterrupt)):
            with self.assertRaises(KeyboardInterrupt):
                batch.sync_keyword_candidates(resume=self.manifest)
        self.assertEqual(dispatch.call_count, 1)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
        self.assertEqual(batch.load_manifest(self.manifest)['job_state'], 'SYNC_COMPLETED')

    def test_report_write_failure_rebuilds_from_repaired_results_without_dispatch(self):
        self.create_incomplete_result()
        with (mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch,
              mock.patch.object(batch, 'atomic_write_jsonl', side_effect=OSError('write interrupted'))):
            with self.assertRaises(OSError):
                batch.sync_keyword_candidates(resume=self.manifest)
        self.assertEqual(dispatch.call_count, 1)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            exported = batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
        self.assertEqual(exported['summary']['processed_chunks'], 2)

    def test_restored_targeted_retry_retains_full_result_summary(self):
        malformed = {'response_text': '{', 'response_payload': {}, 'finish_reason': 'STOP'}
        with mock.patch.object(batch, 'run_sync_request', side_effect=[
                malformed, self.response(self.rows[0]['request']), self.response(self.rows[1]['request'])]):
            batch.execute_sync_request_rows(self.manifest, self.rows)
        before = batch.load_manifest(self.manifest)['sync_summary']
        self.assertEqual(before['contract_final_complete_chunks'], 2)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
        after = batch.load_manifest(self.manifest)['sync_summary']
        for field in ('successful_request_count', 'contract_final_complete_chunks',
                      'contract_first_pass_complete_chunks', 'targeted_retry_requests',
                      'targeted_retry_items', 'reason_counts', 'diagnostic_counts'):
            self.assertEqual(after[field], before[field], field)

    def test_failed_repair_keeps_unknown_gate_and_resumes_same_round(self):
        self.create_incomplete_result()
        with mock.patch.object(batch, 'run_sync_request', side_effect=RuntimeError('unavailable')):
            batch.sync_keyword_candidates(resume=self.manifest)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaises(KeywordRequestUnknownError):
                batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
        with mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest, retry_unknown=True)
        self.assertEqual(dispatch.call_count, 1)
        manifest = batch.load_manifest(self.manifest)
        rows = [json.loads(s) for s in Path(manifest['result_jsonl_path']).read_text().splitlines()]
        self.assertEqual(rows[0]['keyword_checkpoint_generation'], 1)

    def test_incomplete_repair_is_bounded_to_one_round_per_explicit_resume(self):
        self.create_incomplete_result()
        malformed = {'response_text': '{}', 'response_payload': {}, 'finish_reason': 'STOP'}
        for generation in (1, 2):
            with mock.patch.object(batch, 'run_sync_request', return_value=malformed) as dispatch:
                batch.sync_keyword_candidates(resume=self.manifest)
            self.assertEqual(dispatch.call_count, 2)  # one pass + one targeted retry
            manifest = batch.load_manifest(self.manifest)
            self.assertEqual(manifest['job_state'], 'SYNC_PARTIAL')
            rows = [json.loads(s) for s in Path(manifest['result_jsonl_path']).read_text().splitlines()]
            self.assertEqual(rows[0]['keyword_checkpoint_generation'], generation)

    def test_first_response_is_saved_before_targeted_retry(self):
        malformed = {'response_text': '{}', 'response_payload': {}, 'finish_reason': 'STOP'}
        with mock.patch.object(batch, 'run_sync_request', side_effect=[malformed, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                batch.execute_sync_request_rows(self.manifest, self.rows)
        with mock.patch.object(batch, 'run_sync_request', side_effect=self.response) as dispatch:
            batch.sync_keyword_candidates(resume=self.manifest, retry_unknown=True)
        self.assertEqual(dispatch.call_count, 2)  # retry first chunk, then second chunk
        self.assertEqual(dispatch.call_args_list[0].args[0], self.rows[0]['request'])

    def test_source_drift_blocks_resume_before_any_provider_call(self):
        self.stop_after_first_saved_chunk()
        self.source.write_text('Changed original fixture', encoding='utf-8')
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaisesRegex(ValueError, 'source changed'):
                batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()

    def test_request_drift_blocks_reuse(self):
        self.stop_after_first_saved_chunk()
        changed = [*self.rows]
        changed[1] = {'key': 'kw-1', 'request': {'contents': []}}
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaises(KeywordCheckpointError):
                batch.execute_sync_request_rows(self.manifest, changed)
        dispatch.assert_not_called()

    def test_concurrent_resume_is_locked_before_dispatch(self):
        with (exclusive_file_lock(self.package / 'sync_keywords.lock'),
              mock.patch.object(batch, 'run_sync_request') as dispatch):
            with self.assertRaises(TimeoutError):
                batch.execute_sync_request_rows(self.manifest, self.rows)
        dispatch.assert_not_called()

    def test_resume_rejects_new_range_and_pre_checkpoint_runs(self):
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaises(ValueError):
                batch.sync_keyword_candidates(resume=self.manifest, limit=1)
            manifest = batch.load_manifest(self.manifest)
            manifest.pop('keyword_checkpoint_version')
            batch.save_manifest(manifest, update_latest=False)
            with self.assertRaisesRegex(ValueError, 'old runs'):
                batch.sync_keyword_candidates(resume=self.manifest)
        dispatch.assert_not_called()
