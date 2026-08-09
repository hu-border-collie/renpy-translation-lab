import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemini_translate_batch as batch
import translation_core
from engine_adapters.contracts import (
    Occurrence,
    OpaqueLocator,
    ProjectDiscoveryRequest,
)
from engine_adapters.coverage import digest_json
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
import engine_adapters.versioning as versioning


class TestEngineAdapterP3(unittest.TestCase):
    def _occurrence(self, version, spec, project_fingerprint):
        key = spec["key"]
        file_rel_path = spec.get("file", "chapter.rpy")
        source = spec["source"]
        line_number = int(spec.get("line", 1))
        locator_key = spec.get("locator", f"{version}-{key}")
        content_key = spec.get("content", f"{version}-{key}")
        unit = translation_core.TranslationUnit(
            id=f"unit-{version}-{key}",
            mode=translation_core.MODE_TRANSLATION,
            text=source,
            source=source,
            current_translation=f"translated-{version}-{key}",
            file_rel_path=file_rel_path,
            line=max(0, line_number - 1),
            line_number=line_number,
            start=4,
            end=4 + len(source),
            speaker_id=spec.get("speaker", ""),
            speaker_name=spec.get("speaker_name", ""),
        )
        return Occurrence(
            occurrence_id=f"occ-{version}-{key}",
            engine="renpy",
            project_snapshot_fingerprint=project_fingerprint,
            content_fingerprint=digest_json({"content": content_key}),
            candidate_id=f"candidate-{version}-{key}",
            locator=OpaqueLocator(
                engine="renpy",
                locator_schema_version=1,
                locator={"stable_key": locator_key},
            ),
            unit=unit,
        )

    def _snapshot(
        self,
        version,
        specs,
        *,
        review_digest=None,
        generated_at="2026-08-09T00:00:00+00:00",
    ):
        project_fingerprint = f"project-{version}"
        source_fingerprint = f"source-{version}"
        occurrences = [
            self._occurrence(version, spec, project_fingerprint) for spec in specs
        ]
        files = sorted({spec.get("file", "chapter.rpy") for spec in specs})
        coverage = versioning.CoverageBinding.create(
            coverage_digest=f"coverage-{version}",
            coverage_status="ready",
            coverage_schema_version=1,
            inventory_digest=f"inventory-{version}",
            source_fingerprint=source_fingerprint,
            candidate_count=len(specs),
            classification_counts={"translatable": len(specs)},
            review_digest=review_digest or f"review-{version}",
            review_status="human_reviewed",
            review_policy="agent_or_human",
            review_policy_satisfied=True,
            unresolved_findings=0,
        )
        lineage = {
            f"occ-{version}-{spec['key']}": spec["lineage"]
            for spec in specs
            if spec.get("lineage")
        }
        return versioning.create_project_snapshot(
            versioning.GameVersion(version_id=version, label=f"Version {version}"),
            engine="renpy",
            adapter_version="test-adapter",
            localization_mode="hybrid",
            target_language="schinese",
            source_fingerprint=source_fingerprint,
            project_snapshot_fingerprint=project_fingerprint,
            source_files=[
                {
                    "file_rel_path": path,
                    "size": 10,
                    "sha256": digest_json({"version": version, "path": path}),
                }
                for path in files
            ],
            coverage=coverage,
            occurrences=occurrences,
            lineage_by_occurrence=lineage,
            generated_at=generated_at,
        )

    def test_snapshot_export_load_is_deterministic_and_source_only(self):
        specs = [
            {"key": "a", "source": "Hello", "file": "a.rpy", "line": 3},
            {"key": "b", "source": "World", "file": "a.rpy", "line": 5},
        ]
        first = self._snapshot(
            "1.0",
            specs,
            generated_at="2026-08-09T00:00:00+00:00",
        )
        second = self._snapshot(
            "1.0",
            specs,
            generated_at="2026-08-10T00:00:00+00:00",
        )
        self.assertEqual(first.snapshot_digest, second.snapshot_digest)

        with tempfile.TemporaryDirectory() as tmp:
            first_paths = versioning.export_project_snapshot(first, Path(tmp) / "one")
            second_paths = versioning.export_project_snapshot(second, Path(tmp) / "two")
            self.assertEqual(
                Path(first_paths.occurrences_path).read_bytes(),
                Path(second_paths.occurrences_path).read_bytes(),
            )
            self.assertNotIn(
                "current_translation",
                Path(first_paths.occurrences_path).read_text(encoding="utf-8"),
            )
            loaded = versioning.load_project_snapshot(first_paths.snapshot_path)
            self.assertEqual(loaded.snapshot_digest, first.snapshot_digest)
            self.assertEqual(len(loaded.occurrences), 2)
            self.assertEqual(loaded.occurrences[0].context_after, "World")

    def test_snapshot_loader_rejects_tampered_occurrence(self):
        snapshot = self._snapshot("1.0", [{"key": "a", "source": "Hello"}])
        with tempfile.TemporaryDirectory() as tmp:
            paths = versioning.export_project_snapshot(snapshot, tmp)
            occurrence_path = Path(paths.occurrences_path)
            row = json.loads(occurrence_path.read_text(encoding="utf-8"))
            row["translation_unit"]["source_text"] = "Tampered"
            occurrence_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                versioning.VersioningArtifactError,
                "Occurrence digest",
            ):
                versioning.load_project_snapshot(paths.snapshot_path)

    def test_occurrence_context_uses_normalized_relative_paths(self):
        occurrences = [
            self._occurrence(
                "1.0",
                {
                    "key": "first",
                    "source": "First",
                    "file": "./chapter.rpy",
                    "line": 1,
                },
                "project-1.0",
            ),
            self._occurrence(
                "1.0",
                {
                    "key": "second",
                    "source": "Second",
                    "file": "chapter.rpy",
                    "line": 2,
                },
                "project-1.0",
            ),
        ]

        records = versioning.build_unit_occurrence_records(occurrences)

        self.assertEqual(records[0].file_rel_path, "chapter.rpy")
        self.assertEqual(records[0].context_after, "Second")
        self.assertEqual(records[1].context_before, "First")

    def test_snapshot_rejects_absolute_source_paths(self):
        for path in ("C:/game/chapter.rpy", "/game/chapter.rpy", "//server/game.rpy"):
            with self.subTest(path=path), self.assertRaisesRegex(
                versioning.VersioningArtifactError,
                "Invalid relative path",
            ):
                self._snapshot(
                    "1.0",
                    [{"key": "absolute", "source": "Hello", "file": path}],
                )

    def test_reconciliation_covers_match_classes_and_version_deltas(self):
        base = self._snapshot(
            "1.0",
            [
                {
                    "key": "lineage",
                    "source": "Lineage text",
                    "lineage": "lineage-1",
                },
                {
                    "key": "locator",
                    "source": "Hello world",
                    "locator": "stable-locator",
                },
                {
                    "key": "moved",
                    "source": "Move me",
                    "file": "old/chapter.rpy",
                    "line": 4,
                    "content": "old-move-context",
                },
                {
                    "key": "alice-repeat",
                    "source": "Repeat",
                    "speaker": "alice",
                    "content": "old-alice-context",
                },
                {
                    "key": "bob-repeat",
                    "source": "Repeat",
                    "speaker": "bob",
                    "content": "old-bob-context",
                },
                {
                    "key": "modified",
                    "source": "The gate is closed.",
                    "speaker": "guard",
                },
                {
                    "key": "speaker-change",
                    "source": "The speaker changed",
                    "speaker": "alice",
                },
                {"key": "deleted", "source": "Removed in the update"},
            ],
        )
        target = self._snapshot(
            "2.0",
            [
                {
                    "key": "lineage-new",
                    "source": "Lineage text edited",
                    "lineage": "lineage-1",
                },
                {
                    "key": "locator-new",
                    "source": "Hello brave world",
                    "locator": "stable-locator",
                },
                {
                    "key": "moved-new",
                    "source": "Move me",
                    "file": "new/chapter.rpy",
                    "line": 40,
                    "content": "new-move-context",
                },
                {
                    "key": "bob-repeat-new",
                    "source": "Repeat",
                    "speaker": "bob",
                    "content": "new-bob-context",
                },
                {
                    "key": "alice-repeat-new",
                    "source": "Repeat",
                    "speaker": "alice",
                    "content": "new-alice-context",
                },
                {
                    "key": "modified-new",
                    "source": "The gate is almost closed.",
                    "speaker": "guard",
                },
                {
                    "key": "speaker-change-new",
                    "source": "The speaker changed",
                    "speaker": "bob",
                },
                {"key": "added", "source": "New in the update"},
            ],
        )

        report = versioning.reconcile_project_snapshots(
            base,
            target,
            generated_at="2026-08-09T00:00:00+00:00",
        )

        self.assertEqual(report.summary["matched"], 7)
        self.assertEqual(report.summary["confirmed_lineage"], 1)
        self.assertEqual(report.summary["locator_exact"], 1)
        self.assertEqual(report.summary["moved_exact"], 2)
        self.assertEqual(report.summary["context_high_confidence"], 2)
        self.assertEqual(report.summary["source_modified"], 1)
        self.assertEqual(report.summary["added"], 1)
        self.assertEqual(report.summary["deleted"], 1)
        self.assertEqual(report.summary["ambiguous"], 0)
        self.assertTrue(report.coverage_changes["coverage_digest_changed"])
        self.assertEqual(report.coverage_changes["candidate_count_delta"], 0)
        locator_item = next(
            item for item in report.items if item.match_kind == "locator_exact"
        )
        self.assertFalse(locator_item.evidence["source_equal"])
        moved_item = next(
            item
            for item in report.items
            if item.match_kind == "moved_exact" and item.evidence["file_moved"]
        )
        self.assertTrue(moved_item.evidence["line_changed"])

    def test_duplicate_source_without_evidence_is_explicitly_ambiguous(self):
        base = self._snapshot(
            "1.0",
            [
                {"key": "one", "source": "Same", "content": "shared"},
                {"key": "two", "source": "Same", "content": "shared"},
            ],
        )
        target = self._snapshot(
            "2.0",
            [
                {"key": "three", "source": "Same", "content": "shared"},
                {"key": "four", "source": "Same", "content": "shared"},
            ],
        )
        report = versioning.reconcile_project_snapshots(base, target)
        self.assertEqual(report.summary["matched"], 0)
        self.assertEqual(report.summary["ambiguous"], 2)
        self.assertEqual(report.summary["ambiguous_target_count"], 2)
        self.assertEqual(report.summary["added"], 0)
        self.assertEqual(report.summary["deleted"], 0)
        ambiguous = [item for item in report.items if item.disposition == "ambiguous"]
        self.assertTrue(
            all(len(item.candidate_target_occurrence_ids) == 2 for item in ambiguous)
        )

    def test_large_ambiguity_group_lists_every_target_explicitly(self):
        base = self._snapshot(
            "1.0",
            [{"key": "base", "source": "Same", "content": "shared"}],
        )
        target = self._snapshot(
            "2.0",
            [
                {
                    "key": f"target-{index}",
                    "source": "Same",
                    "content": "shared",
                }
                for index in range(12)
            ],
        )

        report = versioning.reconcile_project_snapshots(base, target)

        self.assertEqual(report.summary["ambiguous"], 1)
        self.assertEqual(report.summary["ambiguous_target_count"], 12)
        self.assertEqual(report.summary["added"], 0)
        ambiguous = next(
            item for item in report.items if item.disposition == "ambiguous"
        )
        self.assertEqual(len(ambiguous.candidate_target_occurrence_ids), 8)
        self.assertTrue(
            ambiguous.evidence["candidate_target_occurrence_ids_truncated"]
        )
        target_items = [
            item
            for item in report.items
            if item.disposition == "ambiguous_target"
        ]
        self.assertEqual(
            {item.target_occurrence_id for item in target_items},
            {item.occurrence_id for item in target.occurrences},
        )
        base_group_ids = set(ambiguous.evidence["ambiguity_group_ids"])
        self.assertTrue(
            all(
                base_group_ids & set(item.evidence["ambiguity_group_ids"])
                for item in target_items
            )
        )

    def test_duplicate_source_is_not_resolved_only_by_elimination(self):
        base = self._snapshot(
            "1.0",
            [
                {"key": "alice", "source": "Repeat", "speaker": "alice"},
                {"key": "bob", "source": "Repeat", "speaker": "bob"},
            ],
        )
        target = self._snapshot(
            "2.0",
            [
                {"key": "alice-new", "source": "Repeat", "speaker": "alice"},
                {"key": "unknown", "source": "Repeat", "speaker": "charlie"},
            ],
        )
        report = versioning.reconcile_project_snapshots(base, target)
        self.assertEqual(report.summary["matched"], 1)
        self.assertEqual(report.summary["context_high_confidence"], 1)
        self.assertEqual(report.summary["ambiguous"], 1)
        self.assertEqual(report.summary["ambiguous_target_count"], 1)
        self.assertEqual(report.summary["added"], 0)
        self.assertEqual(report.summary["deleted"], 0)

    def test_candidate_generation_is_indexed_not_cartesian(self):
        base_specs = [
            {
                "key": f"line-{index}",
                "source": f"Unique dialogue line {index:04d} with stable content.",
                "line": index + 1,
            }
            for index in range(300)
        ]
        target_specs = [dict(spec) for spec in base_specs]
        base = self._snapshot("1.0", base_specs)
        target = self._snapshot("2.0", target_specs)
        with mock.patch.object(
            versioning,
            "_pair_evidence",
            wraps=versioning._pair_evidence,
        ) as pair_evidence:
            report = versioning.reconcile_project_snapshots(base, target)
        self.assertEqual(report.summary["matched"], 300)
        self.assertLess(pair_evidence.call_count, 1000)

    def test_saved_report_round_trip_and_tamper_detection(self):
        base = self._snapshot("1.0", [{"key": "a", "source": "Hello"}])
        target = self._snapshot("2.0", [{"key": "a", "source": "Hello"}])
        report = versioning.reconcile_project_snapshots(base, target)
        with tempfile.TemporaryDirectory() as tmp:
            paths = versioning.export_reconciliation_report(report, tmp)
            loaded = versioning.load_reconciliation_report(paths.report_path)
            self.assertEqual(loaded.reconciliation_digest, report.reconciliation_digest)
            items_path = Path(paths.items_path)
            row = json.loads(items_path.read_text(encoding="utf-8"))
            row["confidence"] = 0.01
            items_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                versioning.VersioningArtifactError,
                "item digest",
            ):
                versioning.load_reconciliation_report(paths.report_path)

    def test_reconciliation_item_rejects_non_numeric_confidence(self):
        item = versioning.ReconciliationItem.create(
            disposition="deleted",
            match_kind="",
            base_occurrence_id="occ-base",
        )
        payload = item.to_dict()
        payload["confidence"] = "not-a-number"
        with self.assertRaisesRegex(
            versioning.VersioningArtifactError,
            "confidence must be a finite number",
        ):
            versioning.ReconciliationItem.from_dict(payload)
        with self.assertRaisesRegex(
            versioning.VersioningArtifactError,
            "confidence must be a finite number",
        ):
            versioning.ReconciliationItem.create(
                disposition="deleted",
                match_kind="",
                base_occurrence_id="occ-base",
                confidence=float("nan"),
            )

    def test_coverage_dependency_change_marks_old_report_stale(self):
        specs = [{"key": "a", "source": "Hello"}]
        base = self._snapshot("1.0", specs)
        target = self._snapshot("2.0", specs)
        report = versioning.reconcile_project_snapshots(base, target)
        changed_target = self._snapshot(
            "2.0",
            specs,
            review_digest="review-2.0-replaced",
        )
        freshness = versioning.validate_reconciliation_freshness(
            report,
            base,
            changed_target,
        )
        self.assertEqual(freshness.effective_status, "stale")
        self.assertIn("target_snapshot_digest", freshness.stale_reasons)
        self.assertIn(
            "target_coverage_dependency_digest",
            freshness.stale_reasons,
        )

    def test_live_adapter_snapshot_freezes_pending_review_without_writing_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / "game" / "tl" / "schinese"
            tl_dir.mkdir(parents=True)
            script = tl_dir / "script.rpy"
            original = (
                "translate schinese chapter:\n"
                '    # e "Hello there"\n'
                '    e "Hello there"\n'
            )
            script.write_text(original, encoding="utf-8")
            adapter_snapshot = build_translation_snapshot(
                RenPyAdapter(),
                ProjectDiscoveryRequest(
                    project_root=str(root),
                    localization_root=str(tl_dir),
                    target_language="schinese",
                ),
            )
            snapshot = versioning.build_project_snapshot(
                adapter_snapshot,
                versioning.GameVersion(version_id="build-1"),
                generated_at="2026-08-09T00:00:00+00:00",
            )
            self.assertEqual(snapshot.coverage.review_status, "pending")
            self.assertFalse(snapshot.coverage.review_policy_satisfied)
            self.assertTrue(snapshot.coverage.review_digest)
            self.assertEqual(script.read_text(encoding="utf-8"), original)

            with mock.patch.object(
                batch,
                "build_translation_snapshot",
                return_value=adapter_snapshot,
            ), mock.patch.object(batch.legacy, "BASE_DIR", str(root)):
                exported = batch.run_project_snapshot_export(
                    version_id="build-1",
                    output_dir=str(root / "snapshot-out"),
                )
            persisted = versioning.load_project_snapshot(
                exported["paths"]["snapshot"]
            )
            self.assertEqual(persisted.game_version.version_id, "build-1")
            self.assertEqual(script.read_text(encoding="utf-8"), original)

    def test_cli_registers_machine_commands_and_diagnostics_references(self):
        parser = batch.build_arg_parser()
        snapshot_args = parser.parse_args(
            [
                "export-project-snapshot",
                "--version-id",
                "1.0",
                "--output",
                "json",
            ]
        )
        self.assertEqual(snapshot_args.version_id, "1.0")
        reconcile_args = parser.parse_args(
            [
                "reconcile-project-snapshots",
                "old/project_snapshot.json",
                "new/project_snapshot.json",
                "--output",
                "json",
            ]
        )
        self.assertEqual(reconcile_args.base, "old/project_snapshot.json")

        stdout = io.StringIO()
        with mock.patch.object(
            batch,
            "dispatch_command",
            return_value={
                "version_id": "1.0",
                "engine": "renpy",
                "snapshot_digest": "snapshot-digest",
                "occurrence_count": 4,
                "coverage": {
                    "coverage_status": "attention",
                    "review_status": "pending",
                    "review_policy_satisfied": False,
                },
                "paths": {
                    "snapshot": "C:/snap/project_snapshot.json",
                    "occurrences": "C:/snap/unit_occurrences.jsonl",
                },
            },
        ), redirect_stdout(stdout):
            exit_code = batch.main(
                [
                    "export-project-snapshot",
                    "--version-id",
                    "1.0",
                    "--output",
                    "json",
                ]
            )
        self.assertEqual(exit_code, 0)
        envelope = json.loads(stdout.getvalue())
        self.assertEqual(envelope["command"], "export-project-snapshot")
        self.assertEqual(envelope["result"]["occurrence_count"], 4)
        self.assertEqual(
            envelope["artifacts"]["project_snapshot"],
            "C:/snap/project_snapshot.json",
        )

        from gui_qt.diagnostics_context import build_cli_commands

        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path="manifest.json",
            manifest={"mode": "translation"},
        )
        by_label = {item.label: item.command for item in commands}
        self.assertIn("export-project-snapshot", by_label["版本资产·导出项目快照"])
        self.assertIn("<GAME_VERSION>", by_label["版本资产·导出项目快照"])
        self.assertIn(
            "reconcile-project-snapshots",
            by_label["版本资产·比较两个快照"],
        )

    def test_machine_output_file_cannot_overwrite_versioning_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "snapshot"
            protected = output_dir / versioning.DEFAULT_SNAPSHOT_FILENAME
            stdout = io.StringIO()
            with mock.patch.object(batch, "dispatch_command") as dispatch, redirect_stdout(
                stdout
            ):
                exit_code = batch.main(
                    [
                        "export-project-snapshot",
                        "--version-id",
                        "1.0",
                        "--output-dir",
                        str(output_dir),
                        "--output",
                        "json",
                        "--output-file",
                        str(protected),
                    ]
                )
            envelope = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
            self.assertEqual(
                envelope["error"]["code"],
                "OUTPUT_FILE_PATH_CONFLICT",
            )
            dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
