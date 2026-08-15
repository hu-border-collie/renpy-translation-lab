import json
import tempfile
import unittest
from pathlib import Path

import translation_core
from engine_adapters.contracts import Occurrence, OpaqueLocator
from engine_adapters.coverage import digest_json
from engine_adapters.versioning import reconcile_project_snapshots
import engine_adapters.reuse as reuse


class TestEngineAdapterP4(unittest.TestCase):
    def _occurrence(self, version, spec, project_fingerprint):
        key = spec["key"]
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
            file_rel_path=spec.get("file", "chapter.rpy"),
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

    def _snapshot(self, version, specs, *, coverage_status="ready"):
        project_fingerprint = f"project-{version}"
        source_fingerprint = f"source-{version}"
        occurrences = [
            self._occurrence(version, spec, project_fingerprint) for spec in specs
        ]
        files = sorted({spec.get("file", "chapter.rpy") for spec in specs})
        coverage = reuse_versioning.CoverageBinding.create(
            coverage_digest=f"coverage-{version}",
            coverage_status=coverage_status,
            coverage_schema_version=1,
            inventory_digest=f"inventory-{version}",
            source_fingerprint=source_fingerprint,
            candidate_count=len(specs),
            classification_counts={"translatable": len(specs)},
            review_digest=f"review-{version}",
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
        return reuse_versioning.create_project_snapshot(
            reuse_versioning.GameVersion(version_id=version),
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
            generated_at="2026-08-15T00:00:00+00:00",
        )

    def _records(self, snapshot, translations):
        inputs = [
            reuse.TranslationInput(
                unit_id=f"unit-{snapshot.game_version.version_id}-{key}",
                translation_text=text,
                source_text=next(
                    spec["source"]
                    for spec in translations
                    if spec["key"] == key
                ),
            )
            for key, text in [
                (spec["key"], spec["translation"]) for spec in translations
            ]
        ]
        return reuse.build_translation_records(snapshot, inputs)

    def _scenario(self):
        base_specs = [
            {
                "key": "stable",
                "source": "Stable line",
                "locator": "stable-locator",
                "content": "stable-content",
            },
            {
                "key": "moved",
                "source": "Moved line",
                "file": "old.rpy",
                "line": 2,
                "content": "old-move-context",
            },
            {
                "key": "same1",
                "source": "Same words",
                "content": "same-content",
            },
            {
                "key": "same2",
                "source": "Same words",
                "content": "same-content",
            },
            {"key": "edited", "source": "The gate is closed."},
            {"key": "removed", "source": "Deleted later"},
        ]
        target_specs = [
            {
                "key": "stable",
                "source": "Stable line",
                "locator": "stable-locator",
                "content": "stable-content",
            },
            {
                "key": "moved",
                "source": "Moved line",
                "file": "new.rpy",
                "line": 30,
                "content": "new-move-context",
            },
            {
                "key": "same-new",
                "source": "Same words",
                "content": "same-content",
            },
            {"key": "edited", "source": "The gate is almost closed."},
            {"key": "extra", "source": "Added line"},
        ]
        base = self._snapshot("1.0", base_specs)
        target = self._snapshot("2.0", target_specs)
        records = self._records(
            base,
            [
                {"key": "stable", "source": "Stable line", "translation": "稳定"},
                {
                    "key": "moved",
                    "source": "Moved line",
                    "translation": "移动",
                },
                {"key": "same1", "source": "Same words", "translation": "相同一"},
                {"key": "same2", "source": "Same words", "translation": "相同二"},
                {"key": "edited", "source": "The gate is closed.", "translation": "旧文"},
                {"key": "removed", "source": "Deleted later", "translation": "删除"},
            ],
        )
        report = reconcile_project_snapshots(base, target)
        return base, target, records, report

    def setUp(self):
        import engine_adapters.versioning as versioning_module

        self.versioning = versioning_module

    def test_records_export_load_roundtrip_and_tamper_rejection(self):
        base, _target, records, _report = self._scenario()
        self.assertEqual(records.version_id, "1.0")
        self.assertEqual(len(records.records), 6)
        with tempfile.TemporaryDirectory() as tmp:
            paths = reuse.export_translation_records(records, tmp)
            loaded = reuse.load_translation_records(paths.manifest_path)
            self.assertEqual(
                loaded.record_set_digest,
                records.record_set_digest,
            )
            row_path = Path(paths.records_path)
            rows = [
                json.loads(line)
                for line in row_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            row = rows[0]
            row["translation_text"] = "篡改"
            row_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.versioning.VersioningArtifactError,
                "digest does not match",
            ):
                reuse.load_translation_records(paths.manifest_path)

    def test_candidates_classify_reuse_moved_reference_and_ambiguous(self):
        _base, _target, _records, report = self._scenario()
        base, target, records = _base, _target, _records
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        classes = {}
        for candidate in candidate_set.candidates:
            classes.setdefault(candidate.reuse_class, []).append(candidate)
        self.assertEqual(len(classes.get("exact_reuse", [])), 1)
        self.assertEqual(len(classes.get("moved_reuse", [])), 1)
        self.assertEqual(len(classes.get("source_modified_reference", [])), 1)
        self.assertEqual(len(classes.get("ambiguous", [])), 2)
        modified = classes["source_modified_reference"][0]
        self.assertTrue(modified.reference_only)
        self.assertEqual(modified.reference_translation, "旧文")
        self.assertEqual(candidate_set.summary["status_pending"], 5)
        self.assertEqual(
            candidate_set.summary["reconciliation_orphaned_records"],
            1,
        )

    def test_source_changed_locator_match_becomes_reference_only(self):
        base_specs = [
            {"key": "same", "source": "Hello world", "locator": "same-locator"}
        ]
        target_specs = [
            {"key": "same", "source": "Hello brave world", "locator": "same-locator"}
        ]
        base = self._snapshot("1.0", base_specs)
        target = self._snapshot("2.0", target_specs)
        records = self._records(
            base,
            [{"key": "same", "source": "Hello world", "translation": "原译"}],
        )
        report = reconcile_project_snapshots(base, target)
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        candidate = candidate_set.candidates[0]
        self.assertEqual(candidate.reuse_class, "source_modified_reference")
        self.assertTrue(candidate.reference_only)

    def test_decisions_require_provenance_and_apply_audits(self):
        base, target, records, report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        exact = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "exact_reuse"
            and candidate.reference_translation == "稳定"
        )
        ambiguous_items = [
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "ambiguous"
        ]
        ambiguous = ambiguous_items[0]
        target_choice = ambiguous.candidate_target_occurrence_ids[0]
        decisions = [
            reuse.ReuseDecision(
                candidate_id=exact.candidate_id,
                action="accept",
                reviewer_type="human",
                reviewer_name="reviewer",
            ),
            reuse.ReuseDecision(
                candidate_id=ambiguous.candidate_id,
                action="accept",
                reviewer_type="human",
                reviewer_name="reviewer",
                target_occurrence_id=target_choice,
            ),
        ]
        updated = reuse.apply_reuse_decisions(
            candidate_set,
            decisions,
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        self.assertEqual(updated.summary["status_accepted"], 2)
        accepted = [
            candidate
            for candidate in updated.candidates
            if candidate.status == "accepted"
        ]
        self.assertTrue(
            all(candidate.audit for candidate in accepted)
        )
        self.assertTrue(
            all(
                candidate.decision.get("reviewer_type") == "human"
                for candidate in accepted
            )
        )

        # Duplicate decisions for one candidate are rejected.
        with self.assertRaisesRegex(
            self.versioning.VersioningArtifactError,
            "Only pending",
        ):
            reuse.apply_reuse_decisions(
                updated,
                decisions,
                reconciliation=report,
                base_snapshot=base,
                target_snapshot=target,
                base_records=records,
            )

    def test_duplicate_resolved_ambiguous_targets_fail_at_decision_import(self):
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
        records = self._records(
            base,
            [
                {"key": "one", "source": "Same", "translation": "相同一"},
                {"key": "two", "source": "Same", "translation": "相同二"},
            ],
        )
        report = reconcile_project_snapshots(base, target)
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        ambiguous_items = [
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "ambiguous"
        ]
        self.assertEqual(len(ambiguous_items), 2)
        shared_target = ambiguous_items[0].candidate_target_occurrence_ids[0]
        decisions = [
            reuse.ReuseDecision(
                candidate_id=candidate.candidate_id,
                action="accept",
                reviewer_type="human",
                reviewer_name="reviewer",
                target_occurrence_id=shared_target,
            )
            for candidate in ambiguous_items
        ]
        # The duplicate resolved target must fail at decision import time,
        # not be deferred to the later prefill gate.
        with self.assertRaisesRegex(
            self.versioning.VersioningArtifactError,
            "share one target occurrence",
        ):
            reuse.apply_reuse_decisions(
                candidate_set,
                decisions,
                reconciliation=report,
                base_snapshot=base,
                target_snapshot=target,
                base_records=records,
            )

    def test_freshness_and_prefill_gates_block_stale_and_reference_only(self):
        base, target, records, report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        ambiguous_items = [
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "ambiguous"
        ]
        decisions = [
            reuse.ReuseDecision(
                candidate_id=ambiguous_items[0].candidate_id,
                action="accept",
                reviewer_type="human",
                reviewer_name="reviewer",
                target_occurrence_id=(
                    ambiguous_items[0].candidate_target_occurrence_ids[0]
                ),
            )
        ]
        decisions.extend(
            reuse.ReuseDecision(
                candidate_id=candidate.candidate_id,
                action="accept" if candidate.reuse_class != "ambiguous" else "reject",
                reviewer_type="human",
                reviewer_name="reviewer",
            )
            for candidate in candidate_set.candidates
            if candidate.candidate_id != ambiguous_items[0].candidate_id
        )
        updated = reuse.apply_reuse_decisions(
            candidate_set,
            decisions,
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        prefill = reuse.collect_reuse_prefill(
            updated,
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        prefill_classes = {entry.reuse_class for entry in prefill}
        self.assertNotIn("source_modified_reference", prefill_classes)
        self.assertEqual(len(prefill), 3)

        tampered_records = reuse.build_translation_records(
            base,
            [
                reuse.TranslationInput(
                    unit_id=record.unit_id,
                    translation_text=record.translation_text + "!",
                    source_text=record.source_text,
                )
                for record in records.records
            ],
        )
        freshness = reuse.validate_reuse_freshness(
            updated,
            report,
            base,
            target,
            tampered_records,
        )
        self.assertEqual(freshness.effective_status, "stale")
        with self.assertRaisesRegex(
            self.versioning.VersioningArtifactError,
            "stale",
        ):
            reuse.collect_reuse_prefill(
                updated,
                reconciliation=report,
                base_snapshot=base,
                target_snapshot=target,
                base_records=tampered_records,
            )

    def test_reuse_package_export_load_roundtrip(self):
        base, target, records, report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = reuse.export_reuse_candidates(
                candidate_set,
                tmp,
                target_snapshot=target,
            )
            loaded = reuse.load_reuse_candidates(paths.report_path)
            self.assertEqual(
                loaded.candidate_set_digest,
                candidate_set.candidate_set_digest,
            )
            review_text = Path(paths.review_path).read_text(encoding="utf-8")
            self.assertIn("Translation Reuse Review", review_text)
            self.assertIn("source_modified_reference", review_text)
            # Pipes in source/translation must not break the review table.
            self.assertEqual(
                reuse._review_excerpt("a | b | c"),
                "a \\| b \\| c",
            )
            template_rows = [
                json.loads(line)
                for line in Path(
                    paths.decisions_template_path
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(template_rows), 5)

    def test_loader_rejects_tampered_record_and_candidate_ids(self):
        base, target, records, report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            record_paths = reuse.export_translation_records(records, tmp)
            rows = [
                json.loads(line)
                for line in Path(record_paths.records_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            rows[0]["record_id"] = rows[0]["record_id"][:-4] + "ffff"
            Path(record_paths.records_path).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.versioning.VersioningArtifactError,
                "record id does not match",
            ):
                reuse.load_translation_records(record_paths.manifest_path)

        with tempfile.TemporaryDirectory() as tmp:
            reuse_paths = reuse.export_reuse_candidates(
                candidate_set,
                tmp,
                target_snapshot=target,
            )
            candidate_rows = [
                json.loads(line)
                for line in Path(reuse_paths.candidates_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            candidate_rows[0]["candidate_id"] = (
                candidate_rows[0]["candidate_id"][:-4] + "ffff"
            )
            Path(reuse_paths.candidates_path).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in candidate_rows
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.versioning.VersioningArtifactError,
                "candidate id does not match",
            ):
                reuse.load_reuse_candidates(reuse_paths.report_path)

    def test_override_after_accept_attributed_to_latest_reviewer(self):
        base, target, records, report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        exact = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "exact_reuse"
            and candidate.reference_translation == "稳定"
        )
        updated = reuse.apply_reuse_decisions(
            candidate_set,
            [
                reuse.ReuseDecision(
                    candidate_id=exact.candidate_id,
                    action="accept",
                    reviewer_type="human",
                    reviewer_name="alice",
                ),
                reuse.ReuseDecision(
                    candidate_id=exact.candidate_id,
                    action="override_translation",
                    reviewer_type="human",
                    reviewer_name="bob",
                    translation_text="稳定（改）",
                ),
            ],
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        prefill = reuse.collect_reuse_prefill(
            updated,
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        entry = next(
            item for item in prefill if item.candidate_id == exact.candidate_id
        )
        self.assertEqual(entry.translation_text, "稳定（改）")
        self.assertEqual(
            entry.provenance["reviewer_name"],
            "bob",
        )
        self.assertEqual(entry.provenance["reviewer_type"], "human")
        self.assertEqual(entry.provenance["override_reviewers"], ["bob"])

    def test_accept_after_override_keeps_override_reviewer_attribution(self):
        base, target, records, report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        exact = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "exact_reuse"
            and candidate.reference_translation == "稳定"
        )
        updated = reuse.apply_reuse_decisions(
            candidate_set,
            [
                reuse.ReuseDecision(
                    candidate_id=exact.candidate_id,
                    action="override_translation",
                    reviewer_type="human",
                    reviewer_name="bob",
                    translation_text="稳定（改）",
                ),
                reuse.ReuseDecision(
                    candidate_id=exact.candidate_id,
                    action="accept",
                    reviewer_type="human",
                    reviewer_name="alice",
                ),
            ],
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        prefill = reuse.collect_reuse_prefill(
            updated,
            reconciliation=report,
            base_snapshot=base,
            target_snapshot=target,
            base_records=records,
        )
        entry = next(
            item for item in prefill if item.candidate_id == exact.candidate_id
        )
        self.assertEqual(entry.translation_text, "稳定（改）")
        self.assertEqual(entry.provenance["reviewer_name"], "bob")
        self.assertEqual(entry.provenance["override_reviewers"], ["bob"])

    def test_revision_history_accumulates_deterministically(self):
        base, _target, records, _report = self._scenario()
        first = reuse.build_translation_records(
            base,
            [
                reuse.TranslationInput(
                    unit_id=record.unit_id,
                    translation_text=record.translation_text,
                    source_text=record.source_text,
                    origin=record.origin,
                )
                for record in records.records
            ],
            generated_at="2026-08-15T00:00:00+00:00",
        )
        stable_input = next(
            item for item in first.records if item.translation_text == "稳定"
        )

        # Same translation: history stays empty; changed translation: exactly
        # one deterministic entry pointing at the previous record.
        unchanged_history = reuse.derive_revision_history(
            stable_input,
            new_translation="稳定",
            new_origin="model_initial",
        )
        self.assertEqual(unchanged_history, ())
        changed_history = reuse.derive_revision_history(
            stable_input,
            new_translation="稳定（修）",
            new_origin="revision_applied",
        )
        self.assertEqual(len(changed_history), 1)
        self.assertEqual(
            changed_history[0]["previous_record_id"],
            stable_input.record_id,
        )
        self.assertEqual(
            changed_history[0]["previous_record_digest"],
            stable_input.record_digest,
        )
        self.assertEqual(changed_history[0]["translation_text"], "稳定")

        # An origin-only change (for example human confirmation without a
        # text edit) must preserve the previous attribution too.
        origin_history = reuse.derive_revision_history(
            stable_input,
            new_translation="稳定",
            new_origin="human_confirmed",
        )
        self.assertEqual(len(origin_history), 1)
        self.assertEqual(origin_history[0]["origin"], "model_initial")
        self.assertEqual(origin_history[0]["translation_text"], "稳定")
        self.assertEqual(
            origin_history[0]["superseded_by_origin"],
            "human_confirmed",
        )

        rebuilt = reuse.build_translation_records(
            base,
            [
                reuse.TranslationInput(
                    unit_id=stable_input.unit_id,
                    translation_text="稳定（修）",
                    source_text=stable_input.source_text,
                    origin="revision_applied",
                    revision_history=changed_history,
                )
            ],
            generated_at="2026-08-16T00:00:00+00:00",
        )
        self.assertEqual(
            rebuilt.records[0].revision_history,
            changed_history,
        )
        again = reuse.build_translation_records(
            base,
            [
                reuse.TranslationInput(
                    unit_id=stable_input.unit_id,
                    translation_text="稳定（修）",
                    source_text=stable_input.source_text,
                    origin="revision_applied",
                    revision_history=changed_history,
                )
            ],
            generated_at="2026-08-17T00:00:00+00:00",
        )
        self.assertEqual(
            rebuilt.record_set_digest,
            again.record_set_digest,
        )

    def test_loader_rejects_duplicate_jsonl_rows(self):
        from engine_adapters.coverage import digest_json as coverage_digest_json

        base, _target, records, _report = self._scenario()
        candidate_set = reuse.build_reuse_candidates(
            _report,
            base,
            _target,
            records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            record_paths = reuse.export_translation_records(records, tmp)
            rows = [
                json.loads(line)
                for line in Path(record_paths.records_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            duplicated = rows + [rows[0]]
            Path(record_paths.records_path).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in duplicated
                ),
                encoding="utf-8",
            )
            manifest_path = Path(record_paths.manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["record_count"] = len(duplicated)
            manifest["record_digests"] = [
                row["record_digest"] for row in duplicated
            ]
            manifest["record_set_digest"] = coverage_digest_json(
                {
                    "translation_record_set_schema_version": 1,
                    "version_id": manifest["version_id"],
                    "snapshot_digest": manifest["snapshot_digest"],
                    "target_language": manifest["target_language"],
                    "record_count": len(duplicated),
                    "record_digests": manifest["record_digests"],
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.versioning.VersioningArtifactError,
                "Duplicate translation record id",
            ):
                reuse.load_translation_records(manifest_path)

        with tempfile.TemporaryDirectory() as tmp:
            reuse_paths = reuse.export_reuse_candidates(
                candidate_set,
                tmp,
                target_snapshot=_target,
            )
            candidate_rows = [
                json.loads(line)
                for line in Path(reuse_paths.candidates_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            duplicated = candidate_rows + [candidate_rows[0]]
            Path(reuse_paths.candidates_path).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in duplicated
                ),
                encoding="utf-8",
            )
            report_path = Path(reuse_paths.report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["candidate_count"] = len(duplicated)
            report["candidate_digests"] = [
                row["candidate_digest"] for row in duplicated
            ]
            inputs = report["inputs"]
            report["candidate_set_digest"] = coverage_digest_json(
                {
                    "reuse_candidate_set_schema_version": 1,
                    "inputs": inputs,
                    "status": report["status"],
                    "stale_reasons": report["stale_reasons"],
                    "summary": report["summary"],
                    "lineage_decisions": report["lineage_decisions"],
                    "candidate_digests": report["candidate_digests"],
                }
            )
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.versioning.VersioningArtifactError,
                "Duplicate reuse candidate id",
            ):
                reuse.load_reuse_candidates(report_path)

    def test_ambiguous_accept_rejects_changed_source_target_at_import(self):
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
                # Same content fingerprint keeps the group ambiguous, but the
                # target text itself differs from the base source.
                {"key": "three", "source": "Same edited", "content": "shared"},
                {"key": "four", "source": "Same", "content": "shared"},
            ],
        )
        records = self._records(
            base,
            [
                {"key": "one", "source": "Same", "translation": "相同一"},
                {"key": "two", "source": "Same", "translation": "相同二"},
            ],
        )
        report = reconcile_project_snapshots(base, target)
        candidate_set = reuse.build_reuse_candidates(
            report,
            base,
            target,
            records,
        )
        ambiguous = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.reuse_class == "ambiguous"
        )
        target_by_source = {
            occurrence.source_text: occurrence.occurrence_id
            for occurrence in target.occurrences
        }
        changed_target = target_by_source["Same edited"]
        self.assertIn(changed_target, ambiguous.candidate_target_occurrence_ids)
        with self.assertRaisesRegex(
            self.versioning.VersioningArtifactError,
            "target source differs",
        ):
            reuse.apply_reuse_decisions(
                candidate_set,
                [
                    reuse.ReuseDecision(
                        candidate_id=ambiguous.candidate_id,
                        action="accept",
                        reviewer_type="human",
                        reviewer_name="reviewer",
                        target_occurrence_id=changed_target,
                    )
                ],
                reconciliation=report,
                base_snapshot=base,
                target_snapshot=target,
                base_records=records,
            )


import engine_adapters.versioning as reuse_versioning  # noqa: E402


if __name__ == "__main__":
    unittest.main()
