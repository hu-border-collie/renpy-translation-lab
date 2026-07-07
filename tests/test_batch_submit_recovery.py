import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import batch_submit_recovery
import gemini_translate_batch as batch_mod


class BatchSubmitRecoveryModuleTests(unittest.TestCase):
    def test_compute_request_checksum_hashes_jsonl_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, 'requests.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as handle:
                handle.write('{"key":"a"}\n')
            manifest = {'input_jsonl_path': jsonl_path}
            checksum = batch_submit_recovery.compute_request_checksum(manifest)
            self.assertEqual(len(checksum), 64)

    def test_get_uncertain_submit_state_detects_upload_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest = {
                'job_name': '',
                'submit_state': batch_submit_recovery.SUBMIT_STATE_UPLOADED,
                'uploaded_file_name': 'files/uploaded-1',
                'submit_attempt_id': 'attempt-1',
                'request_checksum': 'abc',
            }
            batch_submit_recovery.append_submit_journal_entry(
                package_dir,
                {
                    'event': batch_submit_recovery.EVENT_UPLOAD_COMPLETED,
                    'submit_attempt_id': 'attempt-1',
                    'uploaded_file_name': 'files/uploaded-1',
                },
            )
            state = batch_submit_recovery.get_uncertain_submit_state(
                manifest,
                package_dir=str(package_dir),
            )
            self.assertEqual(state['kind'], 'upload_pending_job_create')

    def test_get_uncertain_submit_state_detects_uncommitted_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest = {'job_name': ''}
            batch_submit_recovery.append_submit_journal_entry(
                package_dir,
                {
                    'event': batch_submit_recovery.EVENT_JOB_CREATED,
                    'submit_attempt_id': 'attempt-2',
                    'job_name': 'batches/job-2',
                    'job_state': 'JOB_STATE_PENDING',
                    'uploaded_file_name': 'files/uploaded-2',
                    'display_name': 'demo',
                    'request_checksum': 'deadbeef',
                },
            )
            state = batch_submit_recovery.get_uncertain_submit_state(
                manifest,
                package_dir=str(package_dir),
            )
            self.assertEqual(state['kind'], 'job_created_uncommitted')
            self.assertEqual(state['job_name'], 'batches/job-2')


class BatchSubmitRecoveryFlowTests(unittest.TestCase):
    def _make_package(self, tmp: str) -> tuple[Path, Path, Path]:
        root = Path(tmp)
        package_dir = root / 'package'
        package_dir.mkdir()
        input_path = package_dir / 'requests.jsonl'
        manifest_path = package_dir / 'manifest.json'
        latest_path = root / 'latest_manifest.txt'
        input_path.write_text('{}\n', encoding='utf-8')
        manifest_path.write_text(
            json.dumps(
                {
                    'display_name': 'demo package',
                    'batch_model': 'gemini-test',
                    'input_jsonl_path': str(input_path),
                    'job_name': '',
                },
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        return package_dir, manifest_path, latest_path

    def test_submit_blocks_after_upload_without_job_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir, manifest_path, latest_path = self._make_package(tmp)
            checksum = batch_submit_recovery.compute_request_checksum(
                {'input_jsonl_path': str(package_dir / 'requests.jsonl')}
            )
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest.update(
                {
                    'submit_state': batch_submit_recovery.SUBMIT_STATE_UPLOADED,
                    'uploaded_file_name': 'files/uploaded-blocked',
                    'submit_attempt_id': 'attempt-blocked',
                    'request_checksum': checksum,
                }
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
            batch_submit_recovery.append_submit_journal_entry(
                str(package_dir),
                {
                    'event': batch_submit_recovery.EVENT_UPLOAD_COMPLETED,
                    'submit_attempt_id': 'attempt-blocked',
                    'uploaded_file_name': 'files/uploaded-blocked',
                    'request_checksum': checksum,
                },
            )
            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)):
                with self.assertRaises(SystemExit) as ctx:
                    batch_mod.submit_manifest(str(manifest_path))
            self.assertIn('--resume', str(ctx.exception))

    def test_submit_blocks_when_journal_has_uncommitted_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir, manifest_path, latest_path = self._make_package(tmp)
            batch_submit_recovery.append_submit_journal_entry(
                str(package_dir),
                {
                    'event': batch_submit_recovery.EVENT_JOB_CREATED,
                    'submit_attempt_id': 'attempt-3',
                    'job_name': 'batches/job-3',
                    'job_state': 'JOB_STATE_PENDING',
                    'uploaded_file_name': 'files/uploaded-3',
                    'display_name': 'demo package',
                    'request_checksum': 'abc',
                },
            )
            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)):
                with self.assertRaises(SystemExit) as ctx:
                    batch_mod.submit_manifest(str(manifest_path))
            self.assertIn(batch_submit_recovery.BLOCKED_MESSAGE_PREFIX, str(ctx.exception))
            self.assertIn('recover-submit', str(ctx.exception))

    def test_recover_submit_restores_job_name_from_journal(self):
        class BatchJob:
            name = 'batches/job-3'
            state = 'JOB_STATE_PENDING'

        class FakeBatches:
            def get(self, **_kwargs):
                return BatchJob()

        class FakeClient:
            batches = FakeBatches()

        with tempfile.TemporaryDirectory() as tmp:
            package_dir, manifest_path, latest_path = self._make_package(tmp)
            batch_submit_recovery.append_submit_journal_entry(
                str(package_dir),
                {
                    'event': batch_submit_recovery.EVENT_JOB_CREATED,
                    'submit_attempt_id': 'attempt-3',
                    'job_name': 'batches/job-3',
                    'job_state': 'JOB_STATE_PENDING',
                    'uploaded_file_name': 'files/uploaded-3',
                    'display_name': 'demo package',
                    'request_checksum': 'abc',
                },
            )
            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)), \
                 mock.patch.object(batch_mod, 'create_batch_client', return_value=FakeClient()):
                result = batch_mod.recover_submit_manifest(str(manifest_path))

            saved = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(result, str(manifest_path))
            self.assertEqual(saved['job_name'], 'batches/job-3')
            self.assertEqual(saved['submit_state'], batch_submit_recovery.SUBMIT_STATE_COMMITTED)

    def test_submit_resume_reuses_uploaded_file_without_reupload(self):
        class UploadedFile:
            name = 'files/uploaded-resume'

        class BatchJob:
            name = 'batches/job-resume'
            state = 'JOB_STATE_PENDING'

        class FakeFiles:
            def upload(self, **_kwargs):
                raise AssertionError('upload should not be called when --resume is used')

        class FakeBatches:
            def create(self, **_kwargs):
                self.last_src = _kwargs.get('src')
                return BatchJob()

        class FakeClient:
            files = FakeFiles()

            def __init__(self):
                self.batches = FakeBatches()

        with tempfile.TemporaryDirectory() as tmp:
            package_dir, manifest_path, latest_path = self._make_package(tmp)
            checksum = batch_submit_recovery.compute_request_checksum(
                {'input_jsonl_path': str(package_dir / 'requests.jsonl')}
            )
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest.update(
                {
                    'submit_state': batch_submit_recovery.SUBMIT_STATE_UPLOADED,
                    'uploaded_file_name': 'files/uploaded-resume',
                    'submit_attempt_id': 'attempt-resume',
                    'request_checksum': checksum,
                }
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
            batch_submit_recovery.append_submit_journal_entry(
                str(package_dir),
                {
                    'event': batch_submit_recovery.EVENT_UPLOAD_COMPLETED,
                    'submit_attempt_id': 'attempt-resume',
                    'uploaded_file_name': 'files/uploaded-resume',
                    'request_checksum': checksum,
                },
            )
            fake_types = mock.Mock(UploadFileConfig=lambda **kwargs: kwargs)
            stdout = io.StringIO()

            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)), \
                 mock.patch.object(batch_mod.legacy, 'API_KEYS', ['key']), \
                 mock.patch.object(batch_mod, 'genai_types', fake_types), \
                 mock.patch.object(batch_mod, 'create_batch_client', return_value=FakeClient()), \
                 mock.patch('sys.stdout', stdout):
                result = batch_mod.submit_manifest(str(manifest_path), resume_upload=True)

            saved = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(result, str(manifest_path))
            self.assertEqual(saved['job_name'], 'batches/job-resume')
            self.assertIn('Reusing uploaded JSONL', stdout.getvalue())

    def test_submit_records_journal_before_manifest_commit_on_success(self):
        class UploadedFile:
            name = 'files/uploaded-ok'

        class BatchJob:
            name = 'batches/job-ok'
            state = 'JOB_STATE_PENDING'

        class FakeFiles:
            def upload(self, **_kwargs):
                return UploadedFile()

        class FakeBatches:
            def create(self, **_kwargs):
                return BatchJob()

        class FakeClient:
            files = FakeFiles()
            batches = FakeBatches()

        with tempfile.TemporaryDirectory() as tmp:
            package_dir, manifest_path, latest_path = self._make_package(tmp)
            fake_types = mock.Mock(UploadFileConfig=lambda **kwargs: kwargs)

            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)), \
                 mock.patch.object(batch_mod.legacy, 'API_KEYS', ['key']), \
                 mock.patch.object(batch_mod, 'genai_types', fake_types), \
                 mock.patch.object(batch_mod, 'create_batch_client', return_value=FakeClient()):
                batch_mod.submit_manifest(str(manifest_path))

            journal_entries = batch_submit_recovery.read_submit_journal_entries(str(package_dir))
            events = [entry.get('event') for entry in journal_entries]
            self.assertIn(batch_submit_recovery.EVENT_JOB_CREATED, events)
            self.assertIn(batch_submit_recovery.EVENT_MANIFEST_COMMITTED, events)
            job_created_index = events.index(batch_submit_recovery.EVENT_JOB_CREATED)
            committed_index = events.index(batch_submit_recovery.EVENT_MANIFEST_COMMITTED)
            self.assertLess(job_created_index, committed_index)

            saved = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['job_name'], 'batches/job-ok')
            self.assertEqual(saved['submit_state'], batch_submit_recovery.SUBMIT_STATE_COMMITTED)
            self.assertTrue(saved.get('request_checksum'))


if __name__ == '__main__':
    unittest.main()