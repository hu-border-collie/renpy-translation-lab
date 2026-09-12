import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import atomic_io
import batch_export


def _same_path(first, second) -> bool:
    """Case/short-name tolerant path comparison for Windows CI."""

    def canonical(value):
        return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(value))))

    return canonical(first) == canonical(second)


class BatchApplyExportTests(unittest.TestCase):
    def _setup_paths(self, root: Path):
        game_root = root / "project"
        package_dir = root / "package"
        game_root.mkdir()
        package_dir.mkdir()
        source = game_root / "game" / "tl" / "schinese" / "chapter.rpy"
        source.parent.mkdir(parents=True)
        return game_root, package_dir, source

    def _export_root(self, root: Path, game_root: Path, package_dir: Path):
        return batch_export.validate_apply_export_root(
            str(root / "exports"),
            game_root=str(game_root),
            package_dir=str(package_dir),
        )

    def _workspace_payload(self, source: Path, source_bytes: bytes, output_bytes: bytes):
        return {
            "target_path": str(source),
            "source_path": str(source),
            "source_bytes": source_bytes,
            "content": output_bytes,
            "file_key": "chapter.rpy",
        }

    def _export_payload(self, source: Path, source_bytes: bytes, output_bytes: bytes):
        return {
            "source_path": str(source),
            "source_bytes": source_bytes,
            "content": output_bytes,
        }

    def _plan(self, package_dir: Path, *, applied_files=1, applied_lines=1):
        return {
            "manifest_path": str(package_dir / "manifest.json"),
            "apply_summary": {
                "applied_files": applied_files,
                "applied_lines": applied_lines,
            },
            "progress": [
                {"file_key": "chapter.rpy", "line_numbers": list(range(applied_lines))}
            ],
            "rag_jobs": [],
            "quality_state": "batch_applied",
            "latest_target": str(package_dir / "manifest.json"),
            "should_update_latest": True,
            "next_split_manifest_path": "",
            "applied_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }

    def _run(
        self,
        root,
        game_root,
        package_dir,
        source,
        journal=None,
        *,
        source_bytes=None,
        output_bytes=None,
        prepare_source=True,
        **overrides,
    ):
        if source_bytes is None:
            source_bytes = b"label start:\n    e \"Hello\"\n"
        if output_bytes is None:
            output_bytes = b"label start:\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\n"
        if prepare_source:
            source.write_bytes(source_bytes)
        export_root = self._export_root(root, game_root, package_dir)
        values = {
            "game_root": str(game_root),
            "workspace_root": str(source.parent),
            "package_dir": str(package_dir),
            "workspace_payloads": [
                self._workspace_payload(source, source_bytes, output_bytes)
            ],
            "export_payloads": [self._export_payload(source, source_bytes, output_bytes)],
            "request_payload": {"check_fingerprint": "check-1"},
            "apply_identity": "a" * 64,
            "state_advancement": self._plan(package_dir),
            "journal_path": str(journal or package_dir / ".apply_export_transaction.json"),
        }
        values.update(overrides)
        return batch_export.apply_and_export(export_root, **values)

    def test_commits_both_sides_and_writes_pending_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            summary = self._run(root, game_root, package_dir, source)

            output_bytes = b"label start:\n    e \"\xe4\xbd\xa0\xe5\xa5\xbd\"\n"
            exported = (
                root / "exports" / "game" / "tl" / "schinese" / "chapter.rpy"
            )
            self.assertEqual(source.read_bytes(), output_bytes)
            self.assertEqual(exported.read_bytes(), output_bytes)
            self.assertEqual(summary["status"], "applied_and_exported")
            self.assertEqual(summary["applied_files"], 1)
            self.assertEqual(summary["actual_applied_files"], 1)
            self.assertEqual(summary["exported_files"], 1)
            self.assertFalse((package_dir / ".apply_export_transaction.json").exists())
            record = json.loads(
                (package_dir / batch_export.APPLY_EXPORT_RECORD_FILE).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                record["exports"][0]["state_advancement"]["status"],
                "pending",
            )
            self.assertEqual(record["exports"][0]["mode"], "apply-export")

    def test_idempotent_replay_uses_receipt_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            first = self._run(root, game_root, package_dir, source)
            source_mtime = source.stat().st_mtime_ns
            exported = (
                root / "exports" / "game" / "tl" / "schinese" / "chapter.rpy"
            )
            exported_mtime = exported.stat().st_mtime_ns

            second = self._run(
                root,
                game_root,
                package_dir,
                source,
                prepare_source=False,
            )

            self.assertTrue(second["idempotent"])
            self.assertEqual(second["recovery_state"], "already_committed")
            self.assertEqual(second["status"], "applied_and_exported")
            self.assertEqual(second["actual_applied_files"], 0)
            self.assertEqual(source.stat().st_mtime_ns, source_mtime)
            self.assertEqual(exported.stat().st_mtime_ns, exported_mtime)
            self.assertEqual(first["request_fingerprint"], second["request_fingerprint"])

    def test_sides_mismatch_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"old\n"
            source.write_bytes(source_bytes)
            export_payload = self._export_payload(
                source, source_bytes, b"export bytes\n"
            )
            with self.assertRaises(batch_export.ApplyExportError) as raised:
                self._run(
                    root,
                    game_root,
                    package_dir,
                    source,
                    source_bytes=source_bytes,
                    output_bytes=b"workspace bytes\n",
                    prepare_source=False,
                    export_payloads=[export_payload],
                    workspace_payloads=[
                        self._workspace_payload(
                            source,
                            source_bytes,
                            b"workspace bytes\n",
                        )
                    ],
                )
            self.assertEqual(raised.exception.reason_code, "apply_export.sides_mismatch")
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertFalse((root / "exports").exists())

    def test_non_empty_export_root_is_rejected_before_workspace_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"old\n"
            source.write_bytes(source_bytes)
            export_root = root / "exports"
            export_root.mkdir()
            (export_root / "unrelated.txt").write_bytes(b"keep")
            with self.assertRaises(batch_export.ApplyExportError) as raised:
                self._run(
                    root,
                    game_root,
                    package_dir,
                    source,
                    source_bytes=source_bytes,
                    output_bytes=b"new\n",
                    prepare_source=False,
                )
            self.assertEqual(
                raised.exception.reason_code,
                "apply_export.destination_conflict",
            )
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual((export_root / "unrelated.txt").read_bytes(), b"keep")

    def test_replace_failure_rolls_back_both_sides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, source = self._setup_paths(root)
            source_bytes = b"old\n"
            source.write_bytes(source_bytes)
            output_bytes = b"new\n"
            export_root = self._export_root(root, game_root, package_dir)
            destination = (
                root / "exports" / "game" / "tl" / "schinese" / "chapter.rpy"
            )
            real_replace = atomic_io.os.replace
            failed = False

            def fail_export_replace(source_path, destination_path):
                nonlocal failed
                if not failed and _same_path(destination_path, destination):
                    failed = True
                    raise OSError("disk full")
                return real_replace(source_path, destination_path)

            with mock.patch.object(
                atomic_io.os,
                "replace",
                side_effect=fail_export_replace,
            ):
                with self.assertRaises(batch_export.ApplyExportError) as raised:
                    batch_export.apply_and_export(
                        export_root,
                        game_root=str(game_root),
                        workspace_root=str(source.parent),
                        package_dir=str(package_dir),
                        workspace_payloads=[
                            self._workspace_payload(
                                source, source_bytes, output_bytes
                            )
                        ],
                        export_payloads=[
                            self._export_payload(source, source_bytes, output_bytes)
                        ],
                        request_payload={"check_fingerprint": "check-1"},
                        apply_identity="b" * 64,
                        state_advancement=self._plan(package_dir),
                        journal_path=str(
                            package_dir / ".apply_export_transaction.json"
                        ),
                    )
            self.assertIn(
                raised.exception.reason_code,
                {"apply_export.commit_failed", "apply_export.recovery_required"},
            )
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertFalse(destination.exists())
            self.assertFalse((root / "exports").exists())
            self.assertFalse(
                (package_dir / batch_export.APPLY_EXPORT_RECORD_FILE).exists()
            )
            self.assertFalse(
                (package_dir / ".apply_export_transaction.json").exists()
            )

    def test_strict_recovery_refuses_to_overwrite_out_of_transaction_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "chapter.rpy"
            backup = root / ".chapter.rpy.txn.bak"
            staged = root / ".chapter.rpy.txn.tmp"  # missing: treated as committed
            journal = root / "transaction.json"
            target.write_text("transaction-new\n", encoding="utf-8")
            backup.write_text("preimage\n", encoding="utf-8")
            journal.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "transaction_kind": "apply",
                        "state": "prepared",
                        "entries": [
                            {
                                "target": str(target),
                                "staged_path": str(staged),
                                "backup_path": str(backup),
                                "existed": True,
                                "staged_sha256": atomic_io.sha256_text(
                                    "transaction-new\n"
                                ),
                                "target_preimage_sha256": atomic_io.sha256_text(
                                    "preimage\n"
                                ),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            target.write_text("out-of-transaction\n", encoding="utf-8")

            with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
                atomic_io.recover_atomic_write_transaction(
                    journal,
                    expected_transaction_kind="apply",
                    verify_targets=True,
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "out-of-transaction\n")
            self.assertTrue(journal.exists())
            self.assertTrue(backup.exists())

    def test_post_commit_validator_failure_rolls_back_all_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            journal = root / "transaction.json"
            first.write_bytes(b"first old")
            second.write_bytes(b"second old")

            def reject_commit():
                raise RuntimeError("verification failed")

            with self.assertRaises(RuntimeError):
                atomic_io.atomic_write_many_bytes(
                    [(first, b"first new"), (second, b"second new")],
                    journal_path=journal,
                    transaction_kind="apply",
                    post_commit_validator=reject_commit,
                )
            self.assertEqual(first.read_bytes(), b"first old")
            self.assertEqual(second.read_bytes(), b"second old")
            self.assertFalse(journal.exists())

    def test_expected_preimage_blocks_external_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "script.rpy"
            journal = root / "transaction.json"
            target.write_bytes(b"external change")
            with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
                atomic_io.atomic_write_many_bytes(
                    [(target, b"new")],
                    journal_path=journal,
                    transaction_kind="apply",
                    expected_preimages={
                        target: atomic_io.sha256_text("validated source")
                    },
                )
            self.assertEqual(target.read_bytes(), b"external change")
            self.assertFalse(journal.exists())


if __name__ == "__main__":
    unittest.main()


class BatchApplyExportFaultInjectionTests(unittest.TestCase):
    def _setup_multi(self, root: Path):
        game_root = root / "project"
        package_dir = root / "package"
        game_root.mkdir()
        package_dir.mkdir()
        sources = []
        for relative in (
            "game/tl/schinese/a_first.rpy",
            "game/tl/schinese/b_middle.rpy",
            "game/tl/schinese/c_last.rpy",
        ):
            source = game_root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"old {relative}\n".encode("utf-8"))
            sources.append(source)
        export_root = batch_export.validate_apply_export_root(
            str(root / "exports"),
            game_root=str(game_root),
            package_dir=str(package_dir),
        )
        return game_root, package_dir, sources, export_root

    def _payloads(self, sources):
        workspace = []
        exports = []
        for source in sources:
            source_bytes = source.read_bytes()
            output_bytes = source_bytes.replace(b"old ", b"new ")
            workspace.append(
                {
                    "target_path": str(source),
                    "source_path": str(source),
                    "source_bytes": source_bytes,
                    "content": output_bytes,
                    "file_key": source.name,
                }
            )
            exports.append(
                {
                    "source_path": str(source),
                    "source_bytes": source_bytes,
                    "content": output_bytes,
                }
            )
        return workspace, exports

    def _plan(self, package_dir: Path, count: int):
        return {
            "manifest_path": str(package_dir / "manifest.json"),
            "apply_summary": {
                "applied_files": count,
                "applied_lines": count,
            },
            "progress": [],
            "rag_jobs": [],
            "quality_state": "batch_applied",
            "latest_target": "",
            "should_update_latest": False,
            "next_split_manifest_path": "",
            "applied_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }

    def test_replace_failure_at_each_workspace_and_export_position_rolls_back(self):
        positions = (
            "game/tl/schinese/a_first.rpy",
            "game/tl/schinese/b_middle.rpy",
            "game/tl/schinese/c_last.rpy",
        )
        for relative in positions:
            for side in ("workspace", "exports"):
                with self.subTest(relative=relative, side=side), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    game_root, package_dir, sources, export_root = self._setup_multi(root)
                    workspace_payloads, export_payloads = self._payloads(sources)
                    originals = {source: source.read_bytes() for source in sources}
                    if side == "workspace":
                        target = str(game_root / relative)
                    else:
                        target = str(root / "exports" / relative)
                    real_replace = atomic_io.os.replace
                    failed = False

                    def fail_once(source_path, destination_path):
                        nonlocal failed
                        if not failed and _same_path(destination_path, target):
                            failed = True
                            raise OSError("injected replace failure")
                        return real_replace(source_path, destination_path)

                    with mock.patch.object(
                        atomic_io.os,
                        "replace",
                        side_effect=fail_once,
                    ):
                        with self.assertRaises(batch_export.ApplyExportError) as raised:
                            batch_export.apply_and_export(
                                export_root,
                                game_root=str(game_root),
                                workspace_root=str(sources[0].parent),
                                package_dir=str(package_dir),
                                workspace_payloads=workspace_payloads,
                                export_payloads=export_payloads,
                                request_payload={"check": "fault"},
                                apply_identity="c" * 64,
                                state_advancement=self._plan(package_dir, len(sources)),
                                journal_path=str(
                                    package_dir / ".apply_export_transaction.json"
                                ),
                            )
                    self.assertIn(
                        raised.exception.reason_code,
                        {
                            "apply_export.commit_failed",
                            "apply_export.recovery_required",
                        },
                    )
                    for source, content in originals.items():
                        self.assertEqual(source.read_bytes(), content)
                    self.assertFalse(
                        (package_dir / batch_export.APPLY_EXPORT_RECORD_FILE).exists()
                    )
                    self.assertFalse(
                        (package_dir / ".apply_export_transaction.json").exists()
                    )

    def test_stage_failure_leaves_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, sources, export_root = self._setup_multi(root)
            workspace_payloads, export_payloads = self._payloads(sources)
            originals = {source: source.read_bytes() for source in sources}
            real_stage = atomic_io._stage_bytes

            exports_prefix = os.path.normcase(
                os.path.realpath(os.path.abspath(str(root / "exports")))
            )

            def fail_export_stage(target, content):
                target_text = os.path.normcase(
                    os.path.realpath(os.path.abspath(str(target)))
                )
                if target_text == exports_prefix or target_text.startswith(
                    exports_prefix + os.sep
                ):
                    raise OSError("stage failed")
                return real_stage(target, content)

            with mock.patch.object(
                atomic_io,
                "_stage_bytes",
                side_effect=fail_export_stage,
            ):
                with self.assertRaises(batch_export.ApplyExportError) as raised:
                    batch_export.apply_and_export(
                        export_root,
                        game_root=str(game_root),
                        workspace_root=str(sources[0].parent),
                        package_dir=str(package_dir),
                        workspace_payloads=workspace_payloads,
                        export_payloads=export_payloads,
                        request_payload={"check": "fault"},
                        apply_identity="d" * 64,
                        state_advancement=self._plan(package_dir, len(sources)),
                        journal_path=str(package_dir / ".apply_export_transaction.json"),
                    )
            self.assertEqual(raised.exception.reason_code, "apply_export.commit_failed")
            for source, content in originals.items():
                self.assertEqual(source.read_bytes(), content)
            self.assertFalse((root / "exports").exists())

    def test_committed_journal_write_failure_rolls_back_all_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root, package_dir, sources, export_root = self._setup_multi(root)
            workspace_payloads, export_payloads = self._payloads(sources)
            originals = {source: source.read_bytes() for source in sources}
            real_write_json = atomic_io.atomic_write_json

            def fail_committed(journal, payload, **kwargs):
                if isinstance(payload, dict) and payload.get("state") == "committed":
                    raise OSError("cannot persist committed journal")
                return real_write_json(journal, payload, **kwargs)

            with mock.patch.object(
                atomic_io,
                "atomic_write_json",
                side_effect=fail_committed,
            ):
                with self.assertRaises(batch_export.ApplyExportError) as raised:
                    batch_export.apply_and_export(
                        export_root,
                        game_root=str(game_root),
                        workspace_root=str(sources[0].parent),
                        package_dir=str(package_dir),
                        workspace_payloads=workspace_payloads,
                        export_payloads=export_payloads,
                        request_payload={"check": "fault"},
                        apply_identity="e" * 64,
                        state_advancement=self._plan(package_dir, len(sources)),
                        journal_path=str(package_dir / ".apply_export_transaction.json"),
                    )
            self.assertIn(
                raised.exception.reason_code,
                {"apply_export.commit_failed", "apply_export.recovery_required"},
            )
            for source, content in originals.items():
                self.assertEqual(source.read_bytes(), content)
            self.assertFalse(
                (package_dir / batch_export.APPLY_EXPORT_RECORD_FILE).exists()
            )
            self.assertFalse(
                (package_dir / ".apply_export_transaction.json").exists()
            )

    def test_committed_journal_cleanup_preserves_committed_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root = root / "project"
            package_dir = root / "package"
            game_root.mkdir()
            package_dir.mkdir()
            target = game_root / "game" / "tl" / "schinese" / "chapter.rpy"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"committed new\n")
            backup = target.parent / ".chapter.rpy.txn.bak"
            backup.write_bytes(b"old\n")
            staged = target.parent / ".chapter.rpy.txn.tmp"  # consumed
            journal = package_dir / ".apply_export_transaction.json"
            journal.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "transaction_kind": batch_export.APPLY_EXPORT_TRANSACTION_KIND,
                        "state": "committed",
                        "metadata": {
                            "mode": "apply-export",
                            "export_root": str(root / "exports"),
                            "workspace_root": str(target.parent),
                            "game_root": str(game_root),
                            "package_dir": str(package_dir),
                        },
                        "entries": [
                            {
                                "target": str(target),
                                "staged_path": str(staged),
                                "backup_path": str(backup),
                                "existed": True,
                                "staged_sha256": hashlib.sha256(
                                    b"committed new\n"
                                ).hexdigest(),
                                "target_preimage_sha256": hashlib.sha256(
                                    b"old\n"
                                ).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            export_root = batch_export.validate_apply_export_root(
                str(root / "exports"),
                game_root=str(game_root),
                package_dir=str(package_dir),
            )
            self.assertTrue(
                batch_export.recover_apply_export_transaction(
                    export_root,
                    str(journal),
                    workspace_root=str(target.parent),
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                )
            )
            self.assertEqual(target.read_bytes(), b"committed new\n")
            self.assertFalse(journal.exists())
            self.assertFalse(backup.exists())


class BatchApplyExportRecoveryGuardTests(unittest.TestCase):
    def _setup(self, root: Path):
        game_root = root / "project"
        package_dir = root / "package"
        game_root.mkdir()
        package_dir.mkdir()
        source = game_root / "game" / "tl" / "schinese" / "chapter.rpy"
        source.parent.mkdir(parents=True)
        source_bytes = b"old\n"
        source.write_bytes(source_bytes)
        export_root = batch_export.validate_apply_export_root(
            str(root / "exports"),
            game_root=str(game_root),
            package_dir=str(package_dir),
        )
        workspace_payloads = [
            {
                "target_path": str(source),
                "source_path": str(source),
                "source_bytes": source_bytes,
                "content": b"new\n",
                "file_key": "chapter.rpy",
            }
        ]
        export_payloads = [
            {
                "source_path": str(source),
                "source_bytes": source_bytes,
                "content": b"new\n",
            }
        ]
        state_advancement = {
            "manifest_path": str(package_dir / "manifest.json"),
            "apply_summary": {"applied_files": 1, "applied_lines": 1},
            "progress": [{"file_key": "chapter.rpy", "line_numbers": [0]}],
            "rag_jobs": [],
            "quality_state": "batch_applied",
            "latest_target": "",
            "should_update_latest": False,
            "next_split_manifest_path": "",
            "applied_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        return {
            "game_root": game_root,
            "package_dir": package_dir,
            "source": source,
            "source_bytes": source_bytes,
            "export_root": export_root,
            "workspace_payloads": workspace_payloads,
            "export_payloads": export_payloads,
            "state_advancement": state_advancement,
            "journal_path": str(package_dir / ".apply_export_transaction.json"),
        }

    def test_receipt_replace_failure_rolls_back_both_sides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup = self._setup(root)
            source = setup["source"]
            record_path = setup["export_root"].record_path
            real_replace = atomic_io.os.replace
            failed = False

            def fail_receipt(source_path, destination_path):
                nonlocal failed
                if not failed and _same_path(destination_path, record_path):
                    failed = True
                    raise OSError("receipt replace failed")
                return real_replace(source_path, destination_path)

            with mock.patch.object(
                atomic_io.os,
                "replace",
                side_effect=fail_receipt,
            ):
                with self.assertRaises(batch_export.ApplyExportError) as raised:
                    batch_export.apply_and_export(
                        setup["export_root"],
                        game_root=str(setup["game_root"]),
                        workspace_root=str(source.parent),
                        package_dir=str(setup["package_dir"]),
                        workspace_payloads=setup["workspace_payloads"],
                        export_payloads=setup["export_payloads"],
                        request_payload={"check": "receipt"},
                        apply_identity="f" * 64,
                        state_advancement=setup["state_advancement"],
                        journal_path=setup["journal_path"],
                    )
            self.assertIn(
                raised.exception.reason_code,
                {"apply_export.commit_failed", "apply_export.recovery_required"},
            )
            self.assertEqual(source.read_bytes(), setup["source_bytes"])
            self.assertFalse(Path(record_path).exists())
            self.assertFalse((root / "exports").exists())

    def test_prepared_journal_write_failure_leaves_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup = self._setup(root)
            source = setup["source"]
            with mock.patch.object(
                atomic_io,
                "atomic_write_json",
                side_effect=OSError("journal unavailable"),
            ):
                with self.assertRaises(batch_export.ApplyExportError) as raised:
                    batch_export.apply_and_export(
                        setup["export_root"],
                        game_root=str(setup["game_root"]),
                        workspace_root=str(source.parent),
                        package_dir=str(setup["package_dir"]),
                        workspace_payloads=setup["workspace_payloads"],
                        export_payloads=setup["export_payloads"],
                        request_payload={"check": "journal"},
                        apply_identity="1" * 64,
                        state_advancement=setup["state_advancement"],
                        journal_path=setup["journal_path"],
                    )
            self.assertEqual(raised.exception.reason_code, "apply_export.commit_failed")
            self.assertEqual(source.read_bytes(), setup["source_bytes"])
            self.assertFalse(Path(setup["export_root"].record_path).exists())
            self.assertFalse(Path(setup["journal_path"]).exists())

    def test_workspace_tamper_after_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup = self._setup(root)
            source = setup["source"]
            batch_export.apply_and_export(
                setup["export_root"],
                game_root=str(setup["game_root"]),
                workspace_root=str(source.parent),
                package_dir=str(setup["package_dir"]),
                workspace_payloads=setup["workspace_payloads"],
                export_payloads=setup["export_payloads"],
                request_payload={"check": "tamper"},
                apply_identity="2" * 64,
                state_advancement=setup["state_advancement"],
                journal_path=setup["journal_path"],
            )
            source.write_bytes(b"out-of-transaction\n")
            with self.assertRaises(batch_export.ApplyExportError) as raised:
                batch_export.apply_and_export(
                    setup["export_root"],
                    game_root=str(setup["game_root"]),
                    workspace_root=str(source.parent),
                    package_dir=str(setup["package_dir"]),
                    workspace_payloads=setup["workspace_payloads"],
                    export_payloads=setup["export_payloads"],
                    request_payload={"check": "tamper"},
                    apply_identity="2" * 64,
                    state_advancement=setup["state_advancement"],
                    journal_path=setup["journal_path"],
                )
            self.assertEqual(
                raised.exception.reason_code,
                "apply_export.workspace_conflict",
            )
            self.assertEqual(source.read_bytes(), b"out-of-transaction\n")

    def test_export_tree_tamper_after_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup = self._setup(root)
            source = setup["source"]
            batch_export.apply_and_export(
                setup["export_root"],
                game_root=str(setup["game_root"]),
                workspace_root=str(source.parent),
                package_dir=str(setup["package_dir"]),
                workspace_payloads=setup["workspace_payloads"],
                export_payloads=setup["export_payloads"],
                request_payload={"check": "tamper-export"},
                apply_identity="3" * 64,
                state_advancement=setup["state_advancement"],
                journal_path=setup["journal_path"],
            )
            exported = (
                root / "exports" / "game" / "tl" / "schinese" / "chapter.rpy"
            )
            exported.write_bytes(b"tampered\n")
            with self.assertRaises(batch_export.ApplyExportError) as raised:
                batch_export.apply_and_export(
                    setup["export_root"],
                    game_root=str(setup["game_root"]),
                    workspace_root=str(source.parent),
                    package_dir=str(setup["package_dir"]),
                    workspace_payloads=setup["workspace_payloads"],
                    export_payloads=setup["export_payloads"],
                    request_payload={"check": "tamper-export"},
                    apply_identity="3" * 64,
                    state_advancement=setup["state_advancement"],
                    journal_path=setup["journal_path"],
                )
            self.assertEqual(raised.exception.reason_code, "apply_export.tree_conflict")
            self.assertEqual(exported.read_bytes(), b"tampered\n")


class BatchApplyExportRollbackPhaseTests(unittest.TestCase):
    def test_apply_export_rollback_phase_resumes_after_partial_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_root = root / "project"
            package_dir = root / "package"
            game_root.mkdir()
            package_dir.mkdir()
            source = game_root / "game" / "tl" / "schinese" / "chapter.rpy"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"transaction new\n")
            backup = source.parent / ".chapter.rpy.txn.bak"
            backup.write_bytes(b"old\n")
            exported = root / "exports" / "game" / "tl" / "schinese" / "chapter.rpy"
            exported.parent.mkdir(parents=True)
            exported.write_bytes(b"transaction new\n")
            journal = package_dir / ".apply_export_transaction.json"
            journal.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "transaction_kind": batch_export.APPLY_EXPORT_TRANSACTION_KIND,
                        "state": "prepared",
                        "entries": [
                            {
                                "target": str(source),
                                "staged_path": str(source.parent / "missing.txn.tmp"),
                                "backup_path": str(backup),
                                "existed": True,
                                "staged_sha256": hashlib.sha256(
                                    b"transaction new\n"
                                ).hexdigest(),
                                "target_preimage_sha256": hashlib.sha256(
                                    b"old\n"
                                ).hexdigest(),
                            },
                            {
                                "target": str(exported),
                                "staged_path": str(exported.parent / "missing.txn.tmp"),
                                "backup_path": "",
                                "existed": False,
                                "staged_sha256": hashlib.sha256(
                                    b"transaction new\n"
                                ).hexdigest(),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            export_root = batch_export.validate_apply_export_root(
                str(root / "exports"),
                game_root=str(game_root),
                package_dir=str(package_dir),
            )

            with mock.patch.object(
                atomic_io,
                "_restore_backup_copy",
                side_effect=OSError("injected rollback interruption"),
            ):
                with self.assertRaises(batch_export.ApplyExportError):
                    batch_export.recover_apply_export_transaction(
                        export_root,
                        str(journal),
                        workspace_root=str(source.parent),
                        game_root=str(game_root),
                        package_dir=str(package_dir),
                    )

            payload = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "rolling_back")
            self.assertFalse(exported.exists())
            self.assertEqual(source.read_bytes(), b"transaction new\n")

            # The deleted export target and the still-new workspace target are
            # both legal rollback-phase states; retry converges.
            self.assertTrue(
                batch_export.recover_apply_export_transaction(
                    export_root,
                    str(journal),
                    workspace_root=str(source.parent),
                    game_root=str(game_root),
                    package_dir=str(package_dir),
                )
            )
            self.assertEqual(source.read_bytes(), b"old\n")
            self.assertFalse(exported.exists())
            self.assertFalse(journal.exists())


class BatchApplyExportReparseFormTests(unittest.TestCase):
    def test_reparse_attribute_without_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "game" / "tl" / "schinese"
            parent.mkdir(parents=True)
            target = parent / "chapter.rpy"
            target.write_bytes(b"bytes\n")
            fake_dir_stat = SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o755,
                st_file_attributes=0x0400,
            )
            real_stat = os.stat

            def fake_stat(path, *args, **kwargs):
                if os.path.abspath(str(path)) == os.path.abspath(str(parent)):
                    return fake_dir_stat
                return real_stat(path, *args, **kwargs)

            with (
                mock.patch.object(
                    batch_export.os,
                    "stat",
                    side_effect=fake_stat,
                ),
                mock.patch.object(
                    batch_export.os.path,
                    "islink",
                    return_value=False,
                ),
            ):
                self.assertTrue(batch_export._is_link_or_reparse(str(parent)))
                with self.assertRaises(batch_export.ExportOnlyError):
                    batch_export._assert_no_link_components(
                        str(target),
                        "reparse parent",
                        allow_final_file=True,
                    )


if __name__ == "__main__":
    unittest.main()
