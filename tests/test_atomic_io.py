import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import atomic_io
import gemini_translate_batch as batch_mod
import translator_runtime as runtime


class AtomicIoHelperTests(unittest.TestCase):
    def test_atomic_write_text_replaces_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'note.txt'
            path.write_text('old\n', encoding='utf-8')
            atomic_io.atomic_write_text(path, 'new\n')
            self.assertEqual(path.read_text(encoding='utf-8'), 'new\n')
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_atomic_write_preserves_original_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            path.write_text('{"ok": true}\n', encoding='utf-8')
            with mock.patch.object(atomic_io.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    atomic_io.atomic_write_json(path, {'ok': False})
            self.assertEqual(path.read_text(encoding='utf-8'), '{"ok": true}\n')
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_atomic_write_preserves_original_when_writer_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'script.rpy'
            path.write_text('old line\n', encoding='utf-8')

            def boom(_handle):
                raise RuntimeError('disk full')

            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                atomic_io.atomic_write(path, boom)
            self.assertEqual(path.read_text(encoding='utf-8'), 'old line\n')
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_result_artifact_is_complete_requires_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.jsonl'
            path.write_text('', encoding='utf-8')
            self.assertFalse(atomic_io.result_artifact_is_complete(path))

            path.write_text('{"key": "a"}\nnot-json\n', encoding='utf-8')
            self.assertFalse(atomic_io.result_artifact_is_complete(path))

            path.write_text('{"key": "a"}\n{"key": "b"}\n', encoding='utf-8')
            self.assertTrue(atomic_io.result_artifact_is_complete(path))

    def test_result_artifact_is_complete_checks_expected_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.jsonl'
            content = '{"key": "a"}\n'
            atomic_io.atomic_write_text(path, content)
            digest = atomic_io.sha256_text(content)
            self.assertEqual(atomic_io.file_sha256(path), digest)
            self.assertTrue(atomic_io.result_artifact_is_complete(path, digest))
            self.assertFalse(atomic_io.result_artifact_is_complete(path, '0' * 64))

    @unittest.skipIf(os.name == 'nt', 'POSIX mode bits are not available on Windows')
    def test_atomic_write_preserves_existing_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'script.rpy'
            path.write_text('old\n', encoding='utf-8')
            path.chmod(0o644)

            atomic_io.atomic_write_text(path, 'new\n')

            self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    def test_atomic_write_many_lines_commits_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / 'first.rpy'
            second = root / 'second.rpy'
            journal = root / 'writeback-transaction.json'
            first.write_text('first old\n', encoding='utf-8')
            second.write_text('second old\n', encoding='utf-8')

            atomic_io.atomic_write_many_lines(
                [
                    (first, ['first new\n']),
                    (second, ['second new\n']),
                ],
                journal_path=journal,
            )

            self.assertEqual(first.read_text(encoding='utf-8'), 'first new\n')
            self.assertEqual(second.read_text(encoding='utf-8'), 'second new\n')
            self.assertFalse(journal.exists())
            self.assertEqual(list(root.glob('*.txn.*')), [])

    def test_atomic_write_many_lines_rolls_back_prior_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / 'first.rpy'
            second = root / 'second.rpy'
            journal = root / 'writeback-transaction.json'
            first.write_text('first old\n', encoding='utf-8')
            second.write_text('second old\n', encoding='utf-8')
            real_replace = atomic_io.os.replace
            failed = False

            def fail_second_staged_replace(source, destination):
                nonlocal failed
                if (
                    not failed
                    and os.path.abspath(os.fspath(destination)) == os.path.abspath(second)
                    and str(source).endswith('.txn.tmp')
                ):
                    failed = True
                    raise OSError('second replace failed')
                return real_replace(source, destination)

            with mock.patch.object(
                atomic_io.os,
                'replace',
                side_effect=fail_second_staged_replace,
            ):
                with self.assertRaisesRegex(OSError, 'second replace failed'):
                    atomic_io.atomic_write_many_lines(
                        [
                            (first, ['first new\n']),
                            (second, ['second new\n']),
                        ],
                        journal_path=journal,
                    )

            self.assertEqual(first.read_text(encoding='utf-8'), 'first old\n')
            self.assertEqual(second.read_text(encoding='utf-8'), 'second old\n')
            self.assertFalse(journal.exists())
            self.assertEqual(list(root.glob('*.txn.*')), [])

    def test_recover_prepared_transaction_rolls_back_consumed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'script.rpy'
            backup = root / '.script.rpy.demo.txn.bak'
            staged = root / '.script.rpy.demo.txn.tmp'
            journal = root / 'writeback-transaction.json'
            target.write_text('new\n', encoding='utf-8')
            backup.write_text('old\n', encoding='utf-8')
            journal.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'state': 'prepared',
                        'entries': [
                            {
                                'target': str(target),
                                'staged_path': str(staged),
                                'backup_path': str(backup),
                                'existed': True,
                            }
                        ],
                    }
                ),
                encoding='utf-8',
            )

            recovered = atomic_io.recover_atomic_write_transaction(journal)

            self.assertTrue(recovered)
            self.assertEqual(target.read_text(encoding='utf-8'), 'old\n')
            self.assertFalse(backup.exists())
            self.assertFalse(journal.exists())

    def test_recover_rejects_malformed_journal_before_rollback(self):
        malformed_payloads = [
            [],
            {'version': 2, 'state': 'prepared', 'entries': []},
            {'version': 1, 'state': 'prepared', 'entries': ['invalid']},
            {
                'version': 1,
                'state': 'prepared',
                'entries': [
                    {
                        'target': 'target.rpy',
                        'staged_path': 'staged.tmp',
                        'backup_path': 'backup.bak',
                        'existed': 1,
                    }
                ],
            },
        ]

        for payload in malformed_payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                journal = Path(tmp) / 'writeback-transaction.json'
                journal.write_text(json.dumps(payload), encoding='utf-8')

                with self.assertRaises(atomic_io.AtomicWriteTransactionError):
                    atomic_io.recover_atomic_write_transaction(journal)

                self.assertTrue(journal.exists())

    def test_recover_validates_all_entries_before_mutating_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'created.rpy'
            journal = root / 'writeback-transaction.json'
            target.write_text('keep\n', encoding='utf-8')
            journal.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'state': 'prepared',
                        'entries': [
                            {
                                'target': 123,
                                'staged_path': 'invalid.tmp',
                                'backup_path': '',
                                'existed': False,
                            },
                            {
                                'target': str(target),
                                'staged_path': str(root / 'consumed.tmp'),
                                'backup_path': '',
                                'existed': False,
                            },
                        ],
                    }
                ),
                encoding='utf-8',
            )

            with self.assertRaises(atomic_io.AtomicWriteTransactionError):
                atomic_io.recover_atomic_write_transaction(journal)

            self.assertEqual(target.read_text(encoding='utf-8'), 'keep\n')
            self.assertTrue(journal.exists())

    def test_recover_prepared_transaction_is_retryable_after_partial_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / 'first.rpy'
            second = root / 'second.rpy'
            first_backup = root / '.first.rpy.demo.txn.bak'
            second_backup = root / '.second.rpy.demo.txn.bak'
            first_staged = root / '.first.rpy.demo.txn.tmp'
            second_staged = root / '.second.rpy.demo.txn.tmp'
            journal = root / 'writeback-transaction.json'
            first.write_text('first new\n', encoding='utf-8')
            second.write_text('second new\n', encoding='utf-8')
            first_backup.write_text('first old\n', encoding='utf-8')
            second_backup.write_text('second old\n', encoding='utf-8')
            journal.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'state': 'prepared',
                        'entries': [
                            {
                                'target': str(second),
                                'staged_path': str(second_staged),
                                'backup_path': str(second_backup),
                                'existed': True,
                            },
                            {
                                'target': str(first),
                                'staged_path': str(first_staged),
                                'backup_path': str(first_backup),
                                'existed': True,
                            },
                        ],
                    }
                ),
                encoding='utf-8',
            )
            real_replace = atomic_io.os.replace

            def fail_second_restore(source, destination):
                if os.path.abspath(os.fspath(destination)) == os.path.abspath(second):
                    raise OSError('second restore failed')
                return real_replace(source, destination)

            with mock.patch.object(
                atomic_io.os,
                'replace',
                side_effect=fail_second_restore,
            ):
                with self.assertRaisesRegex(OSError, 'second restore failed'):
                    atomic_io.recover_atomic_write_transaction(journal)

            self.assertEqual(first.read_text(encoding='utf-8'), 'first old\n')
            self.assertEqual(second.read_text(encoding='utf-8'), 'second new\n')
            self.assertTrue(first_backup.exists())
            self.assertTrue(second_backup.exists())
            self.assertTrue(journal.exists())

            self.assertTrue(atomic_io.recover_atomic_write_transaction(journal))
            self.assertEqual(first.read_text(encoding='utf-8'), 'first old\n')
            self.assertEqual(second.read_text(encoding='utf-8'), 'second old\n')
            self.assertFalse(first_backup.exists())
            self.assertFalse(second_backup.exists())
            self.assertFalse(journal.exists())


class CommitReplacementsAtomicTests(unittest.TestCase):
    def test_render_replacement_lines_skips_inverted_range(self):
        lines = ['    "Hello"\n']
        replacements = {0: [(11, 4, '你好', '', '"')]}

        self.assertEqual(runtime.render_replacement_lines(lines, replacements), lines)

    def test_commit_replacements_writes_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'script.rpy'
            original = '    "Hello"\n'
            path.write_text(original, encoding='utf-8')
            lines = [original]
            replacements = {0: [(4, 11, '你好', '', '"')]}

            with mock.patch.object(atomic_io.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    runtime.commit_replacements(str(path), list(lines), replacements)

            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_commit_replacements_updates_file_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'script.rpy'
            lines = ['    "Hello"\n']
            path.write_text(lines[0], encoding='utf-8')
            replacements = {0: [(4, 11, '你好', '', '"')]}
            runtime.commit_replacements(str(path), lines, replacements)
            self.assertEqual(path.read_text(encoding='utf-8'), '    "你好"\n')


class BatchArtifactAtomicTests(unittest.TestCase):
    def test_save_manifest_preserves_original_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            manifest_path = package / 'manifest.json'
            manifest_path.write_text('{"version": 1}\n', encoding='utf-8')
            manifest = {
                '_manifest_path': str(manifest_path),
                '_package_dir': str(package),
                'version': 2,
                'display_name': 'demo',
            }
            with mock.patch.object(atomic_io.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    batch_mod.save_manifest(manifest, update_latest=False)
            self.assertEqual(manifest_path.read_text(encoding='utf-8'), '{"version": 1}\n')

    def test_download_results_redownloads_incomplete_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            result_path = package / 'results.jsonl'
            result_path.write_text('{"truncated": true\n', encoding='utf-8')  # invalid JSONL
            manifest_path = package / 'manifest.json'
            manifest = {
                '_manifest_path': str(manifest_path),
                '_package_dir': str(package),
                'job_state': 'JOB_STATE_SUCCEEDED',
                'result_file_name': 'files/demo-result',
                'result_jsonl_path': str(result_path),
                'execution': 'sync',
            }
            atomic_io.atomic_write_json(manifest_path, {
                'job_state': 'JOB_STATE_SUCCEEDED',
                'result_file_name': 'files/demo-result',
                'result_jsonl_path': str(result_path),
                'execution': 'sync',
            })

            class FakeFiles:
                def download(self, file):
                    return b'{"key": "ok"}\n'

            class FakeClient:
                files = FakeFiles()

            with (
                mock.patch.object(batch_mod, 'load_manifest', return_value=manifest),
                mock.patch.object(batch_mod, 'refresh_manifest_status', side_effect=lambda m: m),
                mock.patch.object(batch_mod, 'resolve_manifest_result_path', return_value=str(result_path)),
                mock.patch.object(batch_mod, 'create_batch_client', return_value=FakeClient()),
                mock.patch.object(batch_mod, 'save_manifest') as save_mock,
            ):
                returned = batch_mod.download_results(force=False)

            self.assertEqual(returned, str(result_path))
            self.assertEqual(result_path.read_text(encoding='utf-8'), '{"key": "ok"}\n')
            self.assertTrue((package / 'results.jsonl.sha256').is_file())
            self.assertEqual(manifest.get('result_jsonl_sha256'), atomic_io.file_sha256(result_path))
            save_mock.assert_called_once()

    def test_download_results_skips_complete_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            result_path = package / 'results.jsonl'
            content = '{"key": "ok"}\n'
            atomic_io.atomic_write_text(result_path, content)
            digest = atomic_io.sha256_text(content)
            manifest = {
                '_manifest_path': str(package / 'manifest.json'),
                '_package_dir': str(package),
                'job_state': 'JOB_STATE_SUCCEEDED',
                'result_file_name': 'files/demo-result',
                'result_jsonl_sha256': digest,
            }

            with (
                mock.patch.object(batch_mod, 'load_manifest', return_value=manifest),
                mock.patch.object(batch_mod, 'refresh_manifest_status', side_effect=lambda m: m),
                mock.patch.object(batch_mod, 'resolve_manifest_result_path', return_value=str(result_path)),
                mock.patch.object(batch_mod, 'create_batch_client') as client_mock,
            ):
                returned = batch_mod.download_results(force=False)

            self.assertEqual(returned, str(result_path))
            client_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
