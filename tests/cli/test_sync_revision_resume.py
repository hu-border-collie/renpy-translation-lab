"""Recovery and publication through the Ren'Py synchronous revision entry point."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
from execution.sync_response_checkpoint import SyncRequestUnknownError, SyncResponseCheckpointError
from providers import model_usage_ledger


@contextmanager
def revision_runtime(root):
    with (
        mock.patch.multiple(batch.legacy, BASE_DIR=str(root), TL_DIR=str(root / 'tl'),
                            INCLUDE_FILES=set(), INCLUDE_PREFIXES=set()),
        mock.patch.object(batch.legacy, 'current_project_engine', return_value='renpy'),
        mock.patch.multiple(batch, SYNC_RUNS_DIR=str(root / 'sync_runs'), LOG_DIR=str(root / 'logs'),
                            BATCH_JOBS_DIR=str(root / 'batch_jobs'), REPAIR_RUNS_DIR=str(root / 'repair_runs'),
                            LATEST_MANIFEST_FILE=str(root / 'latest_manifest.txt'), RAG_ENABLED=False,
                            STORY_MEMORY_ENABLED=False, PROJECT_ANALYSIS_ENABLED=False),
    ):
        yield


def request_targets(request):
    prompt = request['contents'][0]['parts'][0]['text']
    return json.JSONDecoder().raw_decode(prompt.split('TARGET:\n', 1)[1])[0]


def response(request, *_args, **_kwargs):
    targets = request_targets(request)
    text = json.dumps({'revisions': [
        {'id': item['id'], 'should_update': True,
         'revised_translation': item['current_translation'] + '修改', 'reason': '措辞'}
        for item in targets
    ]})
    return {
        'response_text': text, 'finish_reason': 'STOP',
        'response_payload': {'responseId': targets[0]['id']},
        'usage_metadata': {'totalTokenCount': 7},
    }


class SyncRevisionResumeTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.source = self.root / 'tl' / 'script.rpy'
        self.source.parent.mkdir()
        self.source.write_text(
            'translate schinese strings:\n'
            '    old "First line"\n    new "第一行"\n'
            '    old "Second line"\n    new "第二行"\n', encoding='utf-8',
        )
        self.original = self.source.read_bytes()
        stack.enter_context(revision_runtime(self.root))
        stack.enter_context(redirect_stdout(io.StringIO()))

    def manifest_path(self):
        [path] = (self.root / 'sync_runs').glob('*/manifest.json')
        return path

    def test_process_exit_requires_acknowledgement_and_only_resends_unknown_request(self):
        child = subprocess.run(
            [sys.executable, '-B', '-c', """
import os
import sys
from pathlib import Path
from unittest import mock
from tests.cli.test_sync_revision_resume import batch, request_targets, response, revision_runtime

def dispatch(request, *args, **kwargs):
    if request_targets(request)[0]['source'] == 'Second line':
        os._exit(76)
    return response(request)

with revision_runtime(Path(sys.argv[1])), mock.patch.object(batch, 'run_sync_request', side_effect=dispatch):
    batch.sync_revisions(skip_prepare=True, chunk_size=1)
""", str(self.root)], cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=30,
        )
        self.assertEqual(child.returncode, 76, child.stderr.decode(errors='replace'))
        path = self.manifest_path()
        self.assertFalse((path.parent / 'results.jsonl').exists())
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaises(SyncRequestUnknownError):
                batch.sync_revisions(resume=str(path))
        dispatch.assert_not_called()
        with mock.patch.object(batch, 'run_sync_request', side_effect=response) as dispatch:
            manifest = batch.sync_revisions(resume=str(path), retry_unknown=True)
        dispatch.assert_called_once()
        self.assertEqual(request_targets(dispatch.call_args.args[0])[0]['source'], 'Second line')
        self.assertEqual(manifest['sync_summary']['resumed_request_count'], 1)
        self.assertEqual(manifest['sync_summary']['reused_response_count'], 1)
        self.assertEqual(manifest['sync_summary']['checkpointed_request_count'], 2)
        self.assertEqual(manifest['last_revision_preview']['summary']['valid_items'], 2)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse((self.root / 'latest_manifest.txt').exists())

    def test_saved_response_before_validation_replays_and_preview_rebuild_is_offline(self):
        with (
            mock.patch.object(batch, 'run_sync_request', side_effect=response) as dispatch,
            mock.patch.object(batch, '_contract_from_sync_result', side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            batch.sync_revisions(skip_prepare=True, chunk_size=1)
        dispatch.assert_called_once()
        path = self.manifest_path()
        with mock.patch.object(batch, 'run_sync_request', side_effect=response) as dispatch:
            manifest = batch.sync_revisions(resume=str(path))
        dispatch.assert_called_once()
        self.assertEqual(manifest['sync_summary']['reused_response_count'], 1)

        preview_path = Path(manifest['last_revision_preview']['jsonl_path'])
        expected = preview_path.read_bytes()
        preview_path.unlink()
        Path(manifest['result_jsonl_path']).unlink()
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            recovered = batch.sync_revisions(resume=str(path))
        dispatch.assert_not_called()
        self.assertEqual(recovered['sync_summary']['reused_response_count'], 2)
        self.assertEqual(preview_path.read_bytes(), expected)
        usage = model_usage_ledger.query_usage(str(self.root), task='revision')
        self.assertEqual(usage['totals']['calls'], 2)
        self.assertEqual(usage['totals']['total_tokens'], 14)

    def test_targeted_retry_keeps_original_prompt_and_parameters_on_resume(self):
        def partial(request, *_args, **_kwargs):
            targets = request_targets(request)
            if len(targets) == 1:
                raise KeyboardInterrupt
            complete = response(request)
            payload = json.loads(complete['response_text'])
            payload['revisions'].pop()
            return {**complete, 'response_text': json.dumps(payload)}

        with mock.patch.object(batch, 'run_sync_request', side_effect=partial) as dispatch:
            with self.assertRaises(KeyboardInterrupt):
                batch.sync_revisions(skip_prepare=True, chunk_size=2)
        self.assertEqual(dispatch.call_count, 2)
        frozen_retry = dispatch.call_args.args[0]
        self.assertEqual([item['source'] for item in request_targets(frozen_retry)], ['Second line'])
        with (
            mock.patch.object(batch, 'run_sync_request', side_effect=response) as dispatch,
            mock.patch.object(batch, 'BATCH_TEMPERATURE', 0.777),
            mock.patch.object(batch, 'load_injectable_project_context_for_prompts', side_effect=AssertionError),
        ):
            manifest = batch.sync_revisions(resume=str(self.manifest_path()), retry_unknown=True)
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.args[0], frozen_retry)
        self.assertEqual(manifest['sync_summary']['contract_final_valid_items'], 2)

    def test_refuses_source_drift_and_missing_terminal_checkpoint_before_dispatch(self):
        with mock.patch.object(batch, 'run_sync_request', side_effect=response):
            batch.sync_revisions(skip_prepare=True, chunk_size=1)
        path = self.manifest_path()
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            self.source.write_bytes(self.original + b'\n')
            with self.assertRaisesRegex(SyncResponseCheckpointError, 'source changed'):
                batch.sync_revisions(resume=str(path))
            self.source.write_bytes(self.original)
            next((path.parent / 'revision_checkpoint' / 'attempts').glob('*.json')).unlink()
            with self.assertRaisesRegex(SyncResponseCheckpointError, 'missing a saved attempt'):
                batch.sync_revisions(resume=str(path))
            with self.assertRaisesRegex(ValueError, 'cannot change.*or write back'):
                batch.sync_revisions(resume=str(path), apply=True)
        dispatch.assert_not_called()

    def test_missing_response_file_in_interrupted_run_is_not_redispatched(self):
        with (
            mock.patch.object(batch, 'run_sync_request', side_effect=response),
            mock.patch.object(batch, '_contract_from_sync_result', side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            batch.sync_revisions(skip_prepare=True, chunk_size=1)
        path = self.manifest_path()
        next((path.parent / 'revision_checkpoint' / 'attempts').glob('*.json')).unlink()
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaisesRegex(SyncResponseCheckpointError, 'missing a saved attempt'):
                batch.sync_revisions(resume=str(path), retry_unknown=True)
        dispatch.assert_not_called()
