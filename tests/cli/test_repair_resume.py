"""Fault injection through the Ren'Py repair entry point."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import gemini_translate_batch as batch
from execution.sync_response_checkpoint import SyncResponseCheckpointError, read_record
from foundation import atomic_io
from providers import model_usage_ledger


@contextmanager
def repair_runtime(root):
    model = 'gemini-3.1-flash-lite'
    with (
        mock.patch.multiple(
            batch.legacy, BASE_DIR=str(root), TL_DIR=str(root / 'tl'),
            TRANSLATOR_CONFIG=str(root / 'translator_config.json'),
            API_KEYS=['test-only-credential'], MODEL_ROUTING_CONFIG=None,
            CUSTOM_LITELLM_PROVIDERS={}, MODELS=[model],
            USE_TRANSLATION_MEMORY=False, NORMALIZE_TRANSLATION_MAP={}, PRESERVE_TERMS=[],
        ),
        mock.patch.multiple(
            batch, REPAIR_RUNS_DIR=str(root / 'repair_runs'),
            SYNC_BACKEND='gemini', SYNC_MODEL=model, SYNC_TIMEOUT_SECONDS=30,
            STORY_MEMORY_ENABLED=False, RAG_ENABLED=False, PROJECT_ANALYSIS_ENABLED=False,
            BATCH_MACRO_SETTING='', BATCH_TEMPERATURE=0.2, BATCH_MAX_OUTPUT_TOKENS=4096,
            BATCH_THINKING_LEVEL='', BATCH_SAFETY_SETTINGS=[],
        ),
    ):
        yield


def targets(request):
    prompt = request['contents'][0]['parts'][0]['text']
    return json.JSONDecoder().raw_decode(prompt.split('TARGET:\n', 1)[1])[0]


def repair_response(request, *_args, **_kwargs):
    items = targets(request)
    return {
        'response_text': json.dumps({'translations': [
            {'id': item['id'], 'translation': '译文'} for item in items
        ]}),
        'finish_reason': 'STOP',
        'response_payload': {'responseId': items[0]['id']},
        'usage_metadata': {'totalTokenCount': 7},
    }


class RepairResumeTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.report = self.root / 'remaining.jsonl'
        (self.root / 'translator_config.json').write_text('{}', encoding='utf-8')
        tl_dir = self.root / 'tl'
        tl_dir.mkdir()
        self.sources = [tl_dir / 'first.rpy', tl_dir / 'second.rpy']
        rows = []
        for path, text in zip(self.sources, ('First line', 'Second line')):
            path.write_bytes(f'    "{text}"\n'.encode())
            rows.append({'file': str(path), 'line': 1, 'source': text})
        self.original = [path.read_bytes() for path in self.sources]
        atomic_io.atomic_write_jsonl(self.report, rows)
        stack.enter_context(repair_runtime(self.root))
        stack.enter_context(redirect_stdout(io.StringIO()))

    def _interrupt(self, code):
        child = subprocess.run(
            [sys.executable, '-B', '-c', """
import os
from pathlib import Path
import sys
from unittest import mock
from tests.cli.test_repair_resume import atomic_io, batch, repair_response, repair_runtime, targets
root = Path(sys.argv[1])
""" + code, str(self.root)],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=35,
        )
        self.assertEqual(child.returncode, 76, child.stderr.decode(errors='replace'))
        [run_dir] = (self.root / 'repair_runs').iterdir()
        self.assertIn(str(run_dir), child.stdout.decode(errors='replace'))
        return run_dir

    def test_process_exit_after_dispatch_requires_confirmation_and_replays_first_response(self):
        run_dir = self._interrupt("""
def dispatch(request, *args, **kwargs):
    if targets(request)[0]['text'] == 'Second line':
        os._exit(76)
    return repair_response(request)
with repair_runtime(root), mock.patch.object(batch, 'run_sync_request', side_effect=dispatch):
    batch.repair_remaining_items(root / 'remaining.jsonl')
""")
        self.assertFalse((run_dir / 'repair_summary.json').exists())
        self.assertEqual([path.read_bytes() for path in self.sources], self.original)
        attempts = [read_record(path) for path in (run_dir / 'repair_checkpoint' / 'attempts').glob('*.json')]
        returned, = (attempt for attempt in attempts if attempt['state'] == 'returned')
        self.assertEqual(returned['response']['usage_metadata']['totalTokenCount'], 7)
        with mock.patch.object(batch, 'run_sync_request') as dispatch:
            with self.assertRaisesRegex(SyncResponseCheckpointError, 'without a saved response'):
                batch.repair_remaining_items(self.report, resume=run_dir)
            with self.assertRaisesRegex(SyncResponseCheckpointError, 'options'):
                batch.repair_remaining_items(self.report, limit=1, resume=run_dir, retry_unknown=True)
            self.sources[0].write_bytes(self.original[0] + b'# externally changed\n')
            with self.assertRaisesRegex(SyncResponseCheckpointError, 'target changed'):
                batch.repair_remaining_items(self.report, resume=run_dir, retry_unknown=True)
            self.sources[0].write_bytes(self.original[0])
        dispatch.assert_not_called()

        with mock.patch.object(batch, 'run_sync_request', side_effect=repair_response) as dispatch:
            summary = batch.repair_remaining_items(self.report, resume=run_dir, retry_unknown=True)
        dispatch.assert_called_once()
        self.assertEqual(targets(dispatch.call_args.args[0])[0]['text'], 'Second line')
        self.assertEqual(summary['saved_requests'], 2)
        self.assertEqual(summary['reused_requests'], 1)
        self.assertEqual(summary['applied_items'], 2)
        self.assertEqual(len(model_usage_ledger.UsageLedger(self.root).load()['records']), 2)

    def test_process_exit_during_publication_recovers_all_files_without_dispatch(self):
        run_dir = self._interrupt("""
replace = atomic_io.os.replace
def interrupt_first_target(source, target):
    replace(source, target)
    if Path(target) == root / 'tl' / 'first.rpy':
        os._exit(76)
with (repair_runtime(root), mock.patch.object(batch, 'run_sync_request', side_effect=repair_response),
      mock.patch.object(atomic_io.os, 'replace', side_effect=interrupt_first_target)):
    batch.repair_remaining_items(root / 'remaining.jsonl')
""")
        self.assertNotEqual(self.sources[0].read_bytes(), self.original[0])
        self.assertEqual(self.sources[1].read_bytes(), self.original[1])
        with (
            mock.patch.object(batch, 'run_sync_request') as dispatch,
            mock.patch.object(batch.legacy, '_require_gemini_api_key', side_effect=AssertionError('credential check')),
        ):
            summary = batch.repair_remaining_items(self.report, resume=run_dir)
            self.assertEqual(batch.repair_remaining_items(self.report, resume=run_dir), summary)
        dispatch.assert_not_called()
        self.assertEqual(summary['reused_requests'], 2)
        self.assertEqual(summary['applied_files'], 2)
        self.assertTrue(all('译文' in path.read_text(encoding='utf-8') for path in self.sources))
        self.assertFalse((run_dir / '.repair_writeback_transaction.json').exists())
        self.assertEqual(len(model_usage_ledger.UsageLedger(self.root).load()['records']), 2)

    def test_invalid_response_saved_before_validation_replays_without_confirmation(self):
        run_dir = self._interrupt("""
def dispatch(request, *args, **kwargs):
    return {**repair_response(request), 'response_text': 'invalid model output'}
with (repair_runtime(root), mock.patch.object(batch, 'run_sync_request', side_effect=dispatch),
      mock.patch.object(batch, 'parse_json_payload', side_effect=lambda *_args: os._exit(76))):
    batch.repair_remaining_items(root / 'remaining.jsonl')
""")
        with mock.patch.object(batch, 'run_sync_request', side_effect=repair_response) as dispatch:
            summary = batch.repair_remaining_items(self.report, resume=run_dir)
        dispatch.assert_called_once()
        self.assertEqual(targets(dispatch.call_args.args[0])[0]['text'], 'Second line')
        self.assertEqual(summary['reused_requests'], 1)
        self.assertEqual(summary['parse_errors'], 1)
        self.assertEqual(summary['applied_items'], 1)
        self.assertEqual(self.sources[0].read_bytes(), self.original[0])
        self.assertIn('译文', self.sources[1].read_text(encoding='utf-8'))
