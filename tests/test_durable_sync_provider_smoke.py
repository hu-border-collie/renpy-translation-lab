# -*- coding: utf-8 -*-

import io
import json
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import run_durable_sync_provider_smoke as smoke
from sync_run_store import SyncRunStore


class DurableProviderSmokeTests(unittest.TestCase):
    def test_billable_acknowledgement_is_mandatory(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), mock.patch.object(smoke.subprocess, 'run') as run:
            code = smoke.main(['--provider-class', 'gemini'])
        self.assertEqual(code, 2)
        run.assert_not_called()
        self.assertIn(smoke.ACK_TEXT, stderr.getvalue())

    def test_missing_secret_is_safe_skip(self):
        stdout = io.StringIO()
        with (
            mock.patch.object(smoke.contract_smoke, 'api_key_for', return_value=''),
            mock.patch.object(smoke.subprocess, 'run') as run,
            redirect_stdout(stdout),
        ):
            code = smoke.main([
                '--provider-class', 'litellm',
                '--litellm-provider', 'deepseek',
                '--acknowledge-billable-request', smoke.ACK_TEXT,
            ])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn('SKIP provider=deepseek', stdout.getvalue())

    def test_parent_resume_never_duplicates_interrupted_request(self):
        def abrupt_child(command, **_kwargs):
            values = {
                command[index]: command[index + 1]
                for index in range(len(command) - 1)
                if str(command[index]).startswith('--_worker-')
            }
            store = SyncRunStore(values['--_worker-root'], values['--_worker-run'])
            owner = 'mock-abrupt-child'
            store.acquire_lease(owner_token=owner, ttl_seconds=1)
            attempt_id = store.prepare_attempt(request_id='req-1', owner_token=owner)
            store.dispatch_attempt(attempt_id=attempt_id, owner_token=owner)
            with store._tx() as conn:
                conn.execute(
                    'UPDATE leases SET expires_at = ? WHERE run_id = ?',
                    ('2000-01-01T00:00:00.000000Z', store.run_id),
                )
            Path(values['--_worker-marker']).write_text(
                json.dumps({
                    'status': 'provider_succeeded_before_t3',
                    'attempt_id': attempt_id,
                    'provider': 'gemini',
                }),
                encoding='utf-8',
            )
            return subprocess.CompletedProcess(command, smoke.CHILD_SUCCESS_EXIT, '', '')

        stdout = io.StringIO()
        with (
            mock.patch.object(smoke.contract_smoke, 'api_key_for', return_value='secret'),
            mock.patch.object(smoke.subprocess, 'run', side_effect=abrupt_child),
            mock.patch.object(smoke.time, 'sleep'),
            redirect_stdout(stdout),
        ):
            code = smoke.main([
                '--provider-class', 'gemini',
                '--acknowledge-billable-request', smoke.ACK_TEXT,
            ])
        self.assertEqual(code, 0)
        self.assertIn('duplicate_requests=0', stdout.getvalue())
        self.assertIn('outcome=unknown', stdout.getvalue())

    def test_custom_class_requires_endpoint_and_model(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = smoke.main([
                '--provider-class', 'custom',
                '--acknowledge-billable-request', smoke.ACK_TEXT,
            ])
        self.assertEqual(code, 2)
        self.assertIn('--custom-base-url', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
