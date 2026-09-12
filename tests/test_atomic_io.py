import errno
import hashlib
import json
import os
import tempfile
import threading
import time
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

    def test_exclusive_file_lock_serializes_and_keeps_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / 'demo.lock'
            with atomic_io.exclusive_file_lock(lock_path, timeout=1.0):
                self.assertTrue(lock_path.is_file())
                with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                    with atomic_io.exclusive_file_lock(lock_path, timeout=0.05):
                        pass
            # Kernel lock files persist by design (#474): deleting one on
            # release could strand waiters that already opened the same inode.
            self.assertTrue(lock_path.is_file())
            with atomic_io.exclusive_file_lock(lock_path, timeout=1.0):
                pass

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



class AtomicWriteStrictRecoveryGuardTests(unittest.TestCase):
    def test_prepared_recovery_blocks_uncommitted_external_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'script.rpy'
            backup = root / '.script.rpy.demo.txn.bak'
            staged = root / '.script.rpy.demo.txn.tmp'
            journal = root / 'writeback-transaction.json'
            target.write_text('out-of-transaction\n', encoding='utf-8')
            backup.write_text('preimage\n', encoding='utf-8')
            staged.write_text('transaction-new\n', encoding='utf-8')
            journal.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'transaction_kind': 'apply',
                        'state': 'prepared',
                        'entries': [
                            {
                                'target': str(target),
                                'staged_path': str(staged),
                                'backup_path': str(backup),
                                'existed': True,
                                'staged_sha256': atomic_io.sha256_text('transaction-new\n'),
                                'target_preimage_sha256': atomic_io.sha256_text('preimage\n'),
                            }
                        ],
                    }
                ),
                encoding='utf-8',
            )

            with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind='apply',
                    verify_targets=True,
                )
            self.assertEqual(target.read_text(encoding='utf-8'), 'out-of-transaction\n')
            self.assertTrue(journal.exists())
            self.assertTrue(staged.exists())

    def test_committed_recovery_never_rolls_back_external_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'script.rpy'
            backup = root / '.script.rpy.demo.txn.bak'
            journal = root / 'writeback-transaction.json'
            target.write_text('external change after commit\n', encoding='utf-8')
            backup.write_text('preimage\n', encoding='utf-8')
            journal.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'transaction_kind': 'apply',
                        'state': 'committed',
                        'entries': [
                            {
                                'target': str(target),
                                'staged_path': str(root / 'consumed.tmp'),
                                'backup_path': str(backup),
                                'existed': True,
                                'staged_sha256': atomic_io.sha256_text('committed\n'),
                                'target_preimage_sha256': atomic_io.sha256_text('preimage\n'),
                            }
                        ],
                    }
                ),
                encoding='utf-8',
            )

            self.assertTrue(
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind='apply',
                    verify_targets=True,
                )
            )
            self.assertEqual(
                target.read_text(encoding='utf-8'),
                'external change after commit\n',
            )
            self.assertFalse(journal.exists())
            self.assertFalse(backup.exists())



class AtomicWriteRollbackPhaseTests(unittest.TestCase):
    def _strict_journal(self, root: Path, targets):
        entries = []
        for target, preimage in targets:
            backup = root / f".{target.name}.txn.bak"
            backup.write_bytes(preimage)
            entries.append(
                {
                    "target": str(target),
                    "staged_path": str(root / f".{target.name}.txn.tmp"),
                    "backup_path": str(backup),
                    "existed": True,
                    "staged_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "target_preimage_sha256": hashlib.sha256(preimage).hexdigest(),
                }
            )
        journal = root / "writeback-transaction.json"
        journal.write_text(
            json.dumps(
                {
                    "version": 1,
                    "transaction_kind": "apply",
                    "state": "prepared",
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )
        return journal

    def test_strict_rollback_resumes_after_partial_restore_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.rpy"
            second = root / "second.rpy"
            first.write_bytes(b"first new\n")
            second.write_bytes(b"second new\n")
            journal = self._strict_journal(
                root,
                ((first, b"first old\n"), (second, b"second old\n")),
            )
            real_restore = atomic_io._restore_backup_copy
            calls = {"count": 0}

            def fail_second_restore(backup, target):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("injected rollback interruption")
                return real_restore(backup, target)

            with mock.patch.object(
                atomic_io,
                "_restore_backup_copy",
                side_effect=fail_second_restore,
            ):
                with self.assertRaisesRegex(OSError, "rollback interruption"):
                    atomic_io.recover_atomic_write_transaction(
                        journal,
                        expected_transaction_kind="apply",
                        verify_targets=True,
                    )

            payload = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "rolling_back")
            # Recovery walks entries in reverse; the second target was restored
            # before the injected failure interrupted the first target.
            self.assertEqual(first.read_bytes(), b"first new\n")
            self.assertEqual(second.read_bytes(), b"second old\n")
            self.assertTrue(
                any(
                    entry.get("rollback_state") == "rolled_back"
                    for entry in payload["entries"]
                )
            )

            # The already-restored target must be recognized as a legal
            # rollback phase; retry converges without treating it as external.
            self.assertTrue(
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind="apply",
                    verify_targets=True,
                )
            )
            self.assertEqual(first.read_bytes(), b"first old\n")
            self.assertEqual(second.read_bytes(), b"second old\n")
            self.assertFalse(journal.exists())

    def test_strict_rollback_phase_still_fails_closed_on_external_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.rpy"
            second = root / "second.rpy"
            first.write_bytes(b"first new\n")
            second.write_bytes(b"second new\n")
            journal = self._strict_journal(
                root,
                ((first, b"first old\n"), (second, b"second old\n")),
            )
            real_restore = atomic_io._restore_backup_copy
            calls = {"count": 0}

            def fail_second_restore(backup, target):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("injected rollback interruption")
                return real_restore(backup, target)

            with mock.patch.object(
                atomic_io,
                "_restore_backup_copy",
                side_effect=fail_second_restore,
            ):
                with self.assertRaises(OSError):
                    atomic_io.recover_atomic_write_transaction(
                        journal,
                        expected_transaction_kind="apply",
                        verify_targets=True,
                    )

            first.write_bytes(b"external change\n")
            with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind="apply",
                    verify_targets=True,
                )
            self.assertEqual(first.read_bytes(), b"external change\n")
            self.assertTrue(journal.exists())

    def test_committed_cleanup_failure_is_recoverable_without_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "script.rpy"
            journal = root / "writeback-transaction.json"
            real_remove = atomic_io._remove_if_present

            def fail_journal_removal(path):
                if os.path.abspath(os.fspath(path)) == os.path.abspath(journal):
                    raise OSError("journal cleanup failed")
                return real_remove(path)

            with (
                mock.patch.object(
                    atomic_io,
                    "_cleanup_transaction_entries",
                    side_effect=OSError("cleanup failed"),
                ),
                mock.patch.object(
                    atomic_io,
                    "_remove_if_present",
                    side_effect=fail_journal_removal,
                ),
            ):
                atomic_io.atomic_write_many_bytes(
                    [(target, b"new\n")],
                    journal_path=journal,
                    transaction_kind="apply",
                )
            # The commit is durable even when cleanup fails; the leftover
            # committed journal is cleaned on the next recovery pass without
            # touching the committed target.
            self.assertEqual(target.read_bytes(), b"new\n")
            self.assertTrue(journal.exists())
            self.assertTrue(
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind="apply",
                    verify_targets=True,
                )
            )
            self.assertEqual(target.read_bytes(), b"new\n")
            self.assertFalse(journal.exists())



class AtomicWriteLegacyJournalCompatibilityTests(unittest.TestCase):
    def test_legacy_prepared_journal_keeps_historical_rollback_boundary(self):
        """Legacy journals without content digests keep the old rollback behavior.

        Strict fail-closed recovery only applies to new journals carrying
        staged/preimage digests.  This test pins the documented compatibility
        boundary; it does not claim legacy journals are safe against external
        modifications.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "script.rpy"
            backup = root / ".script.rpy.demo.txn.bak"
            journal = root / "writeback-transaction.json"
            target.write_bytes(b"external change\n")
            backup.write_bytes(b"preimage\n")
            journal.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "transaction_kind": "apply",
                        "state": "prepared",
                        "entries": [
                            {
                                "target": str(target),
                                "staged_path": str(root / "missing.txn.tmp"),
                                "backup_path": str(backup),
                                "existed": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind="apply",
                    verify_targets=True,
                )
            )
            self.assertEqual(target.read_bytes(), b"preimage\n")
            self.assertFalse(journal.exists())

    def test_committed_cleanup_failure_with_journal_removed_leaves_no_journal(self):
        """FE-08a: entry cleanup fails but journal deletion succeeds."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "script.rpy"
            journal = root / "writeback-transaction.json"
            with mock.patch.object(
                atomic_io,
                "_cleanup_transaction_entries",
                side_effect=OSError("entry cleanup failed"),
            ):
                atomic_io.atomic_write_many_bytes(
                    [(target, b"new\n")],
                    journal_path=journal,
                    transaction_kind="apply",
                )

            self.assertEqual(target.read_bytes(), b"new\n")
            self.assertFalse(journal.exists())



class AtomicProcessFileLockTests(unittest.TestCase):
    """Kernel-backed lock behavior for non-latest writers (#474)."""

    def test_held_lock_cannot_be_stolen_by_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "writer.lock"
            with atomic_io.exclusive_file_lock(lock, timeout=1.0):
                old = time.time() - 86400.0
                os.utime(lock, (old, old))
                with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                    with atomic_io.exclusive_file_lock(lock, timeout=0.25):
                        pass
            # Kernel lock files persist by design.
            self.assertTrue(lock.exists())

    def test_crashed_holder_is_released_by_the_kernel(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "writer.lock"
            marker = root / "held"
            script = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "import atomic_io\n"
                "lock, marker = sys.argv[1:3]\n"
                "with atomic_io.exclusive_file_lock(lock, timeout=5.0):\n"
                "    Path(marker).write_text('held')\n"
                "    time.sleep(30.0)\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(lock), str(marker)],
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline and not marker.exists():
                    if process.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(marker.exists())
                with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                    with atomic_io.exclusive_file_lock(lock, timeout=0.3):
                        pass
                process.kill()
                process.wait(timeout=10.0)
                started = time.monotonic()
                with atomic_io.exclusive_file_lock(lock, timeout=2.0) as owner:
                    self.assertEqual(owner["pid"], os.getpid())
                    self.assertEqual(owner["lock_protocol"], "os_lock_v1")
                self.assertLess(time.monotonic() - started, 1.0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10.0)

    def test_leftover_dead_owner_record_does_not_block_writers(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "writer.lock"
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait(timeout=5.0)
            lock.write_text(
                json.dumps(
                    {
                        "pid": child.pid,
                        "token": "dead-owner",
                        "created_at": time.time() - 86400.0,
                    }
                ),
                encoding="utf-8",
            )
            old = time.time() - 86400.0
            os.utime(lock, (old, old))
            started = time.monotonic()
            with atomic_io.exclusive_file_lock(lock, timeout=1.0) as owner:
                self.assertEqual(owner["pid"], os.getpid())
                self.assertEqual(owner["lock_protocol"], "os_lock_v1")
            self.assertLess(time.monotonic() - started, 1.0)

    def test_symlink_lock_path_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / "precious.txt"
            victim.write_text("do not touch\n", encoding="utf-8")
            lock = root / "writer.lock"
            try:
                lock.symlink_to(victim)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(OSError, "not a regular file"):
                with atomic_io.exclusive_file_lock(lock, timeout=0.5):
                    pass
            self.assertEqual(victim.read_text(encoding="utf-8"), "do not touch\n")
            self.assertTrue(lock.is_symlink())

    def test_foreign_regular_file_is_used_but_never_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "writer.lock"
            original = "user data that must survive\n"
            lock.write_text(original, encoding="utf-8")
            with atomic_io.exclusive_file_lock(lock, timeout=1.0):
                self.assertEqual(lock.read_text(encoding="utf-8"), original)
                with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                    with atomic_io.exclusive_file_lock(lock, timeout=0.2):
                        pass
            self.assertEqual(lock.read_text(encoding="utf-8"), original)

    def test_two_writers_never_enter_critical_section_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "writer.lock"
            state = {"inside": 0, "overlap": False, "entries": 0}
            state_guard = threading.Lock()
            first_entered = threading.Event()
            errors: list[Exception] = []

            def critical_section(label):
                with state_guard:
                    state["inside"] += 1
                    state["entries"] += 1
                    if state["inside"] > 1:
                        state["overlap"] = True
                if label == "a":
                    first_entered.set()
                    time.sleep(0.4)
                else:
                    first_entered.wait(timeout=2.0)
                    time.sleep(0.1)
                with state_guard:
                    state["inside"] -= 1

            def worker(label):
                try:
                    with atomic_io.exclusive_file_lock(lock, timeout=3.0):
                        critical_section(label)
                except Exception as exc:
                    errors.append(exc)

            first = threading.Thread(
                target=worker, args=("a",), name="writer-a", daemon=True
            )
            second = threading.Thread(
                target=worker, args=("b",), name="writer-b", daemon=True
            )
            first.start()
            self.assertTrue(first_entered.wait(timeout=2.0))
            second.start()
            first.join(timeout=10.0)
            second.join(timeout=10.0)

            self.assertFalse(first.is_alive(), "first writer did not finish")
            self.assertFalse(second.is_alive(), "second writer did not finish")
            self.assertEqual([repr(exc) for exc in errors], [])
            self.assertEqual(state["entries"], 2)
            self.assertFalse(
                state["overlap"],
                "two writers entered the critical section concurrently",
            )
            self.assertTrue(lock.exists())

    def test_unsupported_filesystem_lock_is_classified_as_lock_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "writer.lock"
            with mock.patch.object(
                atomic_io,
                "_acquire_kernel_lock",
                side_effect=OSError(errno.ENOTSUP, "locking not supported"),
            ):
                with self.assertRaises(atomic_io.AtomicFileLockUnavailableError):
                    with atomic_io.exclusive_file_lock(lock, timeout=0.2):
                        pass
        # Callers that map AtomicFileLockTimeoutError keep their retryable
        # "busy" classification instead of leaking a bare OSError.
        self.assertTrue(
            issubclass(
                atomic_io.AtomicFileLockUnavailableError,
                atomic_io.AtomicFileLockTimeoutError,
            )
        )

    def test_kernel_lock_primitive_is_acquired_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "writer.lock"
            with (
                mock.patch.object(
                    atomic_io,
                    "_acquire_kernel_lock",
                    wraps=atomic_io._acquire_kernel_lock,
                ) as acquire,
                mock.patch.object(
                    atomic_io,
                    "_release_kernel_lock",
                    wraps=atomic_io._release_kernel_lock,
                ) as release,
            ):
                with atomic_io.exclusive_file_lock(lock, timeout=1.0):
                    pass
            self.assertEqual(acquire.call_count, 1)
            self.assertEqual(release.call_count, 1)


class LatestManifestFileLockTests(unittest.TestCase):
    """The latest cursor keeps its non-preempting manual-recovery lock (#422)."""

    def test_held_lock_is_not_stolen_by_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "latest.lock"
            with atomic_io._latest_manifest_file_lock(lock, timeout=1.0):
                old = time.time() - 86400.0
                os.utime(lock, (old, old))
                with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                    with atomic_io._latest_manifest_file_lock(lock, timeout=0.25):
                        pass
            self.assertFalse(lock.exists())

    def test_dead_owner_record_is_not_preempted(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "latest.lock"
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait(timeout=5.0)
            payload = {
                "pid": child.pid,
                "token": "dead-owner",
                "created_at": time.time() - 86400.0,
            }
            lock.write_text(json.dumps(payload), encoding="utf-8")
            old = time.time() - 86400.0
            os.utime(lock, (old, old))
            with self.assertRaises(atomic_io.AtomicFileLockTimeoutError):
                with atomic_io._latest_manifest_file_lock(lock, timeout=0.3):
                    pass
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8")), payload)

    def test_release_keeps_replacement_owner_and_tolerates_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "latest.lock"
            replacement = '{"pid": 12345, "token": "different-owner"}'
            with atomic_io._latest_manifest_file_lock(lock, timeout=1.0):
                lock.write_text(replacement, encoding="utf-8")
            self.assertEqual(lock.read_text(encoding="utf-8"), replacement)
            lock.unlink()
            with atomic_io._latest_manifest_file_lock(lock, timeout=1.0):
                lock.unlink()
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
