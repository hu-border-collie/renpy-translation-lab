"""Interrupted keyword attempts never silently resend a saved response."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine_adapters.coverage import digest_json
from execution.keyword_checkpoint import (
    KeywordCheckpoint, KeywordCheckpointError, KeywordRequestUnknownError,
    unknown_keyword_requests,
)


class KeywordCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = {'contents': [{'parts': [{'text': 'An original fixture.'}]}]}
        self.response = {'response_text': '{"candidates":[]}', 'finish_reason': 'STOP',
                         'usage_metadata': {'totalTokenCount': 12}}

    def test_response_survives_reopening_before_chunk_completion(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        resumed = KeywordCheckpoint(self.root, 'binding')
        dispatch = mock.Mock(side_effect=AssertionError('saved response was resent'))
        self.assertEqual(resumed.call('one', 'first_pass', self.request, dispatch), self.response)
        self.assertEqual(resumed.reused_attempts, 1)
        dispatch.assert_not_called()

    def test_unknown_dispatch_requires_explicit_acknowledgement(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        with self.assertRaises(KeyboardInterrupt):
            checkpoint.call('one', 'first_pass', self.request,
                            mock.Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(unknown_keyword_requests(self.root), ['one'])
        with self.assertRaises(KeywordRequestUnknownError):
            KeywordCheckpoint(self.root, 'binding')
        resumed = KeywordCheckpoint(self.root, 'binding', retry_unknown=True)
        resumed.call('one', 'first_pass', self.request, lambda: self.response)
        self.assertEqual(unknown_keyword_requests(self.root), [])

    def test_failed_inflight_save_prevents_dispatch(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        dispatch = mock.Mock()
        with mock.patch('execution.keyword_checkpoint.atomic_write_json', side_effect=OSError):
            with self.assertRaises(KeywordCheckpointError):
                checkpoint.call('one', 'first_pass', self.request, dispatch)
        dispatch.assert_not_called()

    def test_failed_response_save_leaves_unknown_instead_of_resending(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        from foundation.atomic_io import atomic_write_json

        def fail_response(path, payload, **kwargs):
            if payload['payload'].get('state') == 'returned':
                raise OSError('fixture disk failure')
            return atomic_write_json(path, payload, **kwargs)

        with mock.patch('execution.keyword_checkpoint.atomic_write_json', side_effect=fail_response):
            with self.assertRaises(KeywordCheckpointError):
                checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        with self.assertRaises(KeywordRequestUnknownError):
            KeywordCheckpoint(self.root, 'binding')

    def test_changed_binding_or_request_refuses_reuse(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        with self.assertRaises(KeywordCheckpointError):
            KeywordCheckpoint(self.root, 'other-binding')
        with self.assertRaises(KeywordCheckpointError):
            checkpoint.call('one', 'first_pass', {'contents': []}, mock.Mock())

    def test_corrupt_saved_response_is_rejected_before_dispatch(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        path = next((self.root / 'keyword_checkpoint/attempts').glob('*.json'))
        envelope = json.loads(path.read_text(encoding='utf-8'))
        envelope['payload']['response']['response_text'] = 'tampered'
        path.write_text(json.dumps(envelope), encoding='utf-8')
        with self.assertRaises(KeywordCheckpointError):
            KeywordCheckpoint(self.root, 'binding')

    def test_material_validation_failure_does_not_mark_dispatch_unknown(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        dispatch = mock.Mock()
        with self.assertRaises(ValueError):
            checkpoint.call('one', 'first_pass', self.request, dispatch,
                            before_dispatch=mock.Mock(side_effect=ValueError('changed material')))
        dispatch.assert_not_called()
        self.assertEqual(unknown_keyword_requests(self.root), [])

    def test_saved_chunk_count_survives_restart(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        checkpoint.complete('one', {'key': 'one'})
        resumed = KeywordCheckpoint(self.root, 'binding')
        self.assertEqual(resumed.resumed_chunks, 1)
        self.assertEqual(resumed.saved_keys, {'one'})
        path = self.root / 'keyword_checkpoint/completed' / f'{digest_json("one")}.json'
        self.assertTrue(path.exists())

    def test_missing_saved_response_is_not_silently_resent(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        checkpoint.complete('one', {'key': 'one'})
        next((self.root / 'keyword_checkpoint/attempts').glob('*.json')).unlink()
        with self.assertRaisesRegex(KeywordCheckpointError, 'missing'):
            KeywordCheckpoint(self.root, 'binding')

    def test_missing_repair_identity_refuses_recovery_before_resending(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        checkpoint.complete('one', {'key': 'one', 'contract_diagnostics': {'complete': False}})
        checkpoint.begin_repair('one')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        next((self.root/'keyword_checkpoint/repairs').glob('*.json')).unlink()
        with self.assertRaisesRegex(KeywordCheckpointError, 'repair attempt identity'):
            KeywordCheckpoint(self.root, 'binding')

    def test_failed_repair_round_save_prevents_another_dispatch(self):
        checkpoint = KeywordCheckpoint(self.root, 'binding')
        checkpoint.call('one', 'first_pass', self.request, lambda: self.response)
        checkpoint.complete('one', {'key': 'one', 'contract_diagnostics': {'complete': False}})
        with mock.patch('execution.keyword_checkpoint.atomic_write_json', side_effect=OSError):
            with self.assertRaises(KeywordCheckpointError):
                checkpoint.begin_repair('one')
        self.assertEqual(checkpoint.generations, {})
