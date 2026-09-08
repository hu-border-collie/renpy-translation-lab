import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import batch_export


class BatchExportOnlyTests(unittest.TestCase):
    def _setup_paths(self, root: Path):
        game_root = root / "project"
        package_dir = root / "package"
        game_root.mkdir()
        package_dir.mkdir()
        source = game_root / "game" / "custom" / "locale" / "dialogue.rpy"
        source.parent.mkdir(parents=True)
        return game_root, package_dir, source

    def _export_root(self, root: Path, game_root: Path, package_dir: Path):
        target = root / "exports"
        return batch_export.validate_export_root(
            str(target),
            game_root=str(game_root),
            package_dir=str(package_dir),
        )

    def _payload(self, source: Path, source_bytes: bytes, output_bytes: bytes):
        return {
            "source_path": str(source),
            "source_bytes": source_bytes,
            "content": output_bytes,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }

    def test_exports_complete_bytes_using_real_custom_source_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"\xef\xbb\xbflabel start:\r\n    e \"Hello\"\r\n"
            output_bytes = b"\xef\xbb\xbflabel start:\r\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\r\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)

            summary = batch_export.export_only(
                export_root,
                game_root=str(game_root),
                package_dir=str(package_dir),
                payloads=[self._payload(source, source_bytes, output_bytes)],
                request_payload={"check_fingerprint": "check-1", "plan": "plan-1"},
                journal_path=str(package_dir / ".export_only_transaction.json"),
            )

            exported = root / "exports" / "game" / "custom" / "locale" / "dialogue.rpy"
            self.assertEqual(exported.read_bytes(), output_bytes)
            self.assertEqual(summary["status"], "exported")
            self.assertEqual(summary["exported_files"], 1)
            self.assertEqual(summary["applied_files"], 0)
            self.assertEqual(summary["files"][0]["relative_path"], "game/custom/locale/dialogue.rpy")
            self.assertTrue(Path(summary["record_path"]).is_file())
            self.assertFalse((package_dir / ".export_only_transaction.json").exists())

    def test_same_request_is_idempotent_and_noop_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n    e \"Hello\"\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)
            journal = str(package_dir / ".export_only_transaction.json")
            request = {"check_fingerprint": "check-1", "plan": "plan-1"}

            first = batch_export.export_only(
                export_root,
                game_root=str(game_root),
                package_dir=str(package_dir),
                payloads=[],
                request_payload=request,
                journal_path=journal,
            )
            second = batch_export.export_only(
                export_root,
                game_root=str(game_root),
                package_dir=str(package_dir),
                payloads=[],
                request_payload=request,
                journal_path=journal,
            )

            self.assertEqual(first["status"], "no-op")
            self.assertEqual(first["payload_files"], 0)
            self.assertEqual(second["status"], "no-op")
            self.assertTrue(second["idempotent"])
            self.assertEqual(second["applied_files"], 0)

    def test_byte_identical_payload_is_filtered_to_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n    e \"Hello\"\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)

            summary = batch_export.export_only(
                export_root,
                game_root=str(game_root),
                package_dir=str(package_dir),
                payloads=[self._payload(source, source_bytes, source_bytes)],
                request_payload={"check_fingerprint": "check-no-op"},
                journal_path=str(package_dir / ".export_only_transaction.json"),
            )

            self.assertEqual(summary["status"], "no-op")
            self.assertEqual(summary["payload_files"], 0)
            self.assertEqual(list((root / "exports").rglob("*")), [])

    def test_root_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, _source = self._setup_paths(root)

            with self.assertRaises(batch_export.ExportOnlyError) as game_overlap:
                batch_export.validate_export_root(
                    str(game_root),
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                )
            self.assertEqual(
                game_overlap.exception.reason_code,
                "export_only.path_overlaps_game_root",
            )

            with self.assertRaises(batch_export.ExportOnlyError) as package_overlap:
                batch_export.validate_export_root(
                    str(package_dir),
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                )
            self.assertEqual(
                package_overlap.exception.reason_code,
                "export_only.path_overlaps_package_root",
            )

    def test_conflict_and_path_escape_are_rejected_without_merging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)
            (root / "exports" / "unrelated.txt").parent.mkdir(parents=True)
            (root / "exports" / "unrelated.txt").write_bytes(b"keep")

            with self.assertRaises(batch_export.ExportOnlyError) as conflict:
                batch_export.export_only(
                    export_root,
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                    payloads=[],
                    request_payload={"check_fingerprint": "check-1"},
                    journal_path=str(package_dir / ".export_only_transaction.json"),
                )
            self.assertEqual(conflict.exception.reason_code, "export_only.destination_conflict")
            self.assertEqual((root / "exports" / "unrelated.txt").read_bytes(), b"keep")

            outside = root / "outside.rpy"
            outside.write_bytes(source_bytes)
            with self.assertRaises(batch_export.ExportOnlyError) as escaped:
                batch_export.export_only(
                    export_root,
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                    payloads=[self._payload(outside, source_bytes, b"changed")],
                    request_payload={"check_fingerprint": "check-2"},
                    journal_path=str(package_dir / ".export_only_transaction.json"),
                )
            self.assertEqual(escaped.exception.reason_code, "export_only.source_path_escape")

    def test_request_or_tree_conflict_is_rejected_after_first_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n    e \"Hello\"\n"
            output_bytes = b"label start:\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)
            journal = str(package_dir / ".export_only_transaction.json")
            payload = self._payload(source, source_bytes, output_bytes)

            batch_export.export_only(
                export_root,
                game_root=str(game_root),
                package_dir=str(package_dir),
                payloads=[payload],
                request_payload={"check_fingerprint": "check-1"},
                journal_path=journal,
            )

            with self.assertRaises(batch_export.ExportOnlyError) as request_conflict:
                batch_export.export_only(
                    export_root,
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                    payloads=[payload],
                    request_payload={"check_fingerprint": "check-2"},
                    journal_path=journal,
                )
            self.assertEqual(
                request_conflict.exception.reason_code,
                "export_only.record_conflict",
            )

            exported = root / "exports" / "game" / "custom" / "locale" / "dialogue.rpy"
            exported.write_bytes(b"tampered")
            with self.assertRaises(batch_export.ExportOnlyError) as tree_conflict:
                batch_export.export_only(
                    export_root,
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                    payloads=[payload],
                    request_payload={"check_fingerprint": "check-1"},
                    journal_path=journal,
                )
            self.assertEqual(
                tree_conflict.exception.reason_code,
                "export_only.tree_conflict",
            )

    def test_commit_failure_is_structured_and_removes_new_empty_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n    e \"Hello\"\n"
            output_bytes = b"label start:\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)

            with mock.patch.object(
                batch_export,
                "atomic_write_many_bytes",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(batch_export.ExportOnlyError) as failed:
                    batch_export.export_only(
                        export_root,
                        game_root=str(game_root),
                        package_dir=str(package_dir),
                        payloads=[self._payload(source, source_bytes, output_bytes)],
                        request_payload={"check_fingerprint": "check-fail"},
                        journal_path=str(package_dir / ".export_only_transaction.json"),
                    )

            self.assertEqual(failed.exception.reason_code, "export_only.commit_failed")
            self.assertFalse((root / "exports").exists())

    def test_cleanup_failure_does_not_replace_recovery_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n    e \"Hello\"\n"
            output_bytes = b"label start:\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)
            journal = package_dir / ".export_only_transaction.json"

            def fail_commit(*_args, **kwargs):
                Path(kwargs["journal_path"]).write_text("pending", encoding="utf-8")
                raise OSError("disk full")

            with (
                mock.patch.object(
                    batch_export,
                    "atomic_write_many_bytes",
                    side_effect=fail_commit,
                ),
                mock.patch.object(
                    batch_export,
                    "_prune_empty_recovery_directories",
                    side_effect=batch_export.ExportOnlyError(
                        "export_only.path_unreadable",
                        "cleanup failed",
                    ),
                ),
                self.assertRaises(batch_export.ExportOnlyError) as failed,
            ):
                batch_export.export_only(
                    export_root,
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                    payloads=[self._payload(source, source_bytes, output_bytes)],
                    request_payload={"check_fingerprint": "check-recovery"},
                    journal_path=str(journal),
                )

            self.assertEqual(
                failed.exception.reason_code,
                "export_only.recovery_required",
            )
            self.assertIn("disk full", str(failed.exception))
            self.assertEqual(
                failed.exception.details["recovery_state"],
                "recovery_required",
            )

    def test_interrupted_export_journal_recovers_before_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"label start:\n    e \"Hello\"\n"
            output_bytes = b"label start:\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\n"
            source.write_bytes(source_bytes)
            export_root = self._export_root(root, game_root, package_dir)
            destination = root / "exports" / "game" / "custom" / "locale" / "dialogue.rpy"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(output_bytes)
            journal = package_dir / ".export_only_transaction.json"
            journal.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "transaction_kind": batch_export.EXPORT_TRANSACTION_KIND,
                        "state": "prepared",
                        "entries": [
                            {
                                "target": str(destination),
                                "staged_path": str(destination.parent / "missing.txn.tmp"),
                                "backup_path": "",
                                "existed": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                batch_export.recover_export_only_transaction(
                    export_root,
                    str(journal),
                )
            )
            self.assertFalse(destination.exists())

            summary = batch_export.export_only(
                export_root,
                game_root=str(game_root),
                package_dir=str(package_dir),
                payloads=[self._payload(source, source_bytes, output_bytes)],
                request_payload={"check_fingerprint": "check-1"},
                journal_path=str(journal),
                recovery_state="recovered",
            )
            self.assertEqual(summary["status"], "exported")
            self.assertEqual(destination.read_bytes(), output_bytes)


if __name__ == "__main__":
    unittest.main()
