# -*- coding: utf-8 -*-
"""P1 contract, equivalence, coverage, and workflow tests."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from engine_adapters.contracts import (
    CoverageReportDraft,
    EngineAdapter,
    InventoryPolicy,
    ProjectDiscoveryRequest,
)
from engine_adapters.coverage import (
    CLASSIFICATIONS,
    build_coverage_report,
    build_review_template,
    export_coverage_package,
    load_review_record,
    review_input_digest,
    validate_coverage_report_freshness,
    validate_review_record,
)
from engine_adapters.renpy import (
    RenPyAdapter,
    build_translation_snapshot,
)
import translation_core
import translator_runtime as runtime
import sync_translation_preview as preview


class TestRenPyAdapterP1(unittest.TestCase):
    def make_project(self, files):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        for rel_path, text in files.items():
            target = tl_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return root, tl_dir

    @staticmethod
    def request(root, tl_dir, **kwargs):
        return ProjectDiscoveryRequest(
            project_root=str(root),
            localization_root=str(tl_dir),
            target_language="schinese",
            **kwargs,
        )

    def test_protocol_and_capabilities_expose_p2_contract(self):
        adapter = RenPyAdapter(legacy_module=runtime)
        self.assertIsInstance(adapter, EngineAdapter)
        capabilities = adapter.capabilities()
        self.assertEqual(capabilities.engine, "renpy")
        self.assertEqual(capabilities.selected_localization_mode.value, "hybrid")
        self.assertTrue(capabilities.source_inventory)
        self.assertTrue(capabilities.native_catalog)
        self.assertTrue(capabilities.relocation)
        self.assertEqual(capabilities.declarative_writeback, ("text_span_replace",))
        self.assertTrue(capabilities.native_catalog_required_for_writeback)
        self.assertEqual(capabilities.adapter_version, "1.1.0")
        self.assertNotEqual(adapter.behavior_digest(), "")


    def test_adapter_matches_legacy_units_ids_speakers_sources_and_spans(self):
        lines = [
            'define e = Character(_("Eileen"))\n',
            "translate schinese chapter:\n",
            '    # e "Hello {player}!"\n',
            '    e "你好，[player]！"\n',
            '    # "Choose wisely"\n',
            '    "Choose wisely"\n',
            "translate schinese strings:\n",
            '    old "Start game"\n',
            '    new "开始游戏"\n',
            "translate schinese chapter:\n",
            '    # e "Hello {player}!"\n',
            '    e "Hello {player}!"\n',
        ]
        root, tl_dir = self.make_project({"script.rpy": "".join(lines)})
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )

        legacy_tasks, legacy_progress = runtime.collect_tasks_with_progress(lines)
        expected_tasks = []
        for task in legacy_tasks:
            item = dict(task)
            item["file_rel_path"] = "script.rpy"
            item["file_path"] = str(tl_dir / "script.rpy")
            item["id"] = translation_core.build_identity_v2(
                "script.rpy",
                item["block_name"],
                item["block_index"],
                item["source_for_id"],
                block_occurrence=item["block_occurrence"],
            )
            expected_tasks.append(item)

        actual_tasks = list(snapshot.pending_tasks_by_file["script.rpy"])
        comparable_fields = (
            "id",
            "text",
            "line",
            "start",
            "end",
            "quote",
            "prefix",
            "block_name",
            "block_index",
            "block_occurrence",
            "source_for_id",
            "speaker_id",
            "speaker_name",
        )
        self.assertEqual(
            [tuple(item.get(field, "") for field in comparable_fields) for item in actual_tasks],
            [tuple(item.get(field, "") for field in comparable_fields) for item in expected_tasks],
        )
        self.assertEqual(
            snapshot.progress_by_file["script.rpy"],
            legacy_progress,
        )

        legacy_mapping = runtime.scan_all_translation_units(lines, "script.rpy")
        occurrence_by_id = {occurrence.unit.id: occurrence for occurrence in snapshot.occurrences}
        self.assertEqual(set(occurrence_by_id), set(legacy_mapping))
        for item_id, (line, start, end, text) in legacy_mapping.items():
            unit = occurrence_by_id[item_id].unit
            self.assertEqual(
                (unit.line, unit.start, unit.end, unit.text),
                (line, start, end, text),
            )

        first_id = translation_core.build_identity_v2(
            "script.rpy",
            "chapter",
            1,
            "Hello {player}!",
        )
        first = occurrence_by_id[first_id].unit
        self.assertEqual(first.source, "Hello {player}!")
        self.assertEqual(first.current_translation, "你好，[player]！")
        self.assertEqual((first.speaker_id, first.speaker_name), ("e", "Eileen"))
        self.assertEqual(snapshot.pending_task_count, 2)
        self.assertEqual(snapshot.recognized_unit_count, 4)

    def test_candidate_inventory_exposes_parse_and_unsupported_regions(self):
        source = """translate schinese start:
    # "Dangling source"
    old "Old without new"
    text f"Dynamic {name}"
    "unterminated
"""
        root, tl_dir = self.make_project({"broken.rpy": source})
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )

        candidates = snapshot.inventory.candidates
        self.assertEqual(
            len({candidate.candidate_id for candidate in candidates}),
            len(candidates),
        )
        self.assertTrue(
            all(candidate.classification in CLASSIFICATIONS for candidate in candidates)
        )
        parse_errors = [
            candidate for candidate in candidates if candidate.classification == "parse_error"
        ]
        self.assertGreaterEqual(len(parse_errors), 3)
        self.assertTrue(
            any(
                "renpy.source_marker_unpaired" in candidate.reason_codes
                for candidate in parse_errors
            )
        )
        self.assertTrue(
            any("renpy.tokenize_error" in candidate.reason_codes for candidate in parse_errors)
        )
        self.assertTrue(
            any(
                candidate.classification == "unsupported"
                and "renpy.dynamic_string_expression" in candidate.reason_codes
                for candidate in candidates
            )
        )
        self.assertTrue(
            all(int(candidate.locator.locator["line_hint"]) >= 1 for candidate in parse_errors)
        )
        self.assertEqual(snapshot.report.coverage_status, "block")
        self.assertGreater(
            snapshot.report.classification_counts["parse_error"],
            0,
        )

    def test_valid_multiline_character_definition_is_not_a_parse_error(self):
        source = """define e = Character(
    _("Eileen"),
    color="#ffffff",
)
translate schinese start:
    # e "Hello there."
    e "Hello there."
"""
        root, tl_dir = self.make_project({"script.rpy": source})
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        self.assertFalse(
            any(
                candidate.classification == "parse_error"
                for candidate in snapshot.inventory.candidates
            )
        )
        self.assertTrue(
            any(
                candidate.structure_kind == "character_display_definition"
                and candidate.classification == "explicitly_excluded"
                for candidate in snapshot.inventory.candidates
            )
        )
        task = snapshot.pending_tasks_by_file["script.rpy"][0]
        self.assertEqual((task["speaker_id"], task["speaker_name"]), ("e", "Eileen"))

    def test_nonstandard_old_marker_is_visible_as_unsupported(self):
        source = """translate schinese strings:
    old 'Single quoted source'
    new 'Single quoted target'
"""
        root, tl_dir = self.make_project({"strings.rpy": source})
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        self.assertTrue(
            any(
                candidate.classification == "unsupported"
                and candidate.structure_kind == "nonstandard_old_source_marker"
                for candidate in snapshot.inventory.candidates
            )
        )
        self.assertEqual(snapshot.pending_task_count, 1)
        self.assertEqual(snapshot.report.coverage_status, "attention")

    def test_nonstandard_source_comment_is_visible_as_unsupported(self):
        source = """translate schinese start:
    # 'Single quoted source'
    "Target"
"""
        root, tl_dir = self.make_project({"comments.rpy": source})
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        comments = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.structure_kind == "nonstandard_source_comment"
        ]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].classification, "unsupported")
        self.assertIn("renpy.custom_statement_unsupported", comments[0].reason_codes)
        self.assertEqual(comments[0].locator.locator["start_col_hint"], 6)

    def test_unrecognized_string_structure_is_unknown_not_silently_excluded(self):
        root, tl_dir = self.make_project(
            {"custom.rpy": ('translate schinese start:\n    custom_statement "xy"\n')}
        )
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        unknown = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.structure_kind == "unknown_string_structure"
        ]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0].classification, "unknown")
        self.assertIn("renpy.visibility_unknown", unknown[0].reason_codes)
        self.assertEqual(snapshot.report.coverage_status, "block")

    def test_no_pending_is_distinct_from_no_recognized_units(self):
        translated_root, translated_tl = self.make_project(
            {
                "translated.rpy": (
                    'translate schinese start:\n    # e "Hello there."\n    e "你好。"\n'
                )
            }
        )
        translated = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(translated_root, translated_tl),
        )
        self.assertEqual(translated.pending_task_count, 0)
        self.assertEqual(translated.recognized_unit_count, 1)
        self.assertGreater(translated.report.candidate_count, 0)

        empty_root, empty_tl = self.make_project({"empty.rpy": "# no text\n"})
        empty = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(empty_root, empty_tl),
        )
        self.assertEqual(empty.pending_task_count, 0)
        self.assertEqual(empty.recognized_unit_count, 0)
        self.assertEqual(empty.report.candidate_count, 0)

    def test_coverage_digest_freshness_and_review_provenance(self):
        root, tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        adapter = RenPyAdapter(legacy_module=runtime)
        first = build_translation_snapshot(
            adapter,
            self.request(root, tl_dir),
        )
        second = build_translation_snapshot(
            adapter,
            self.request(root, tl_dir),
        )
        self.assertEqual(
            first.report.coverage_digest,
            second.report.coverage_digest,
        )
        self.assertNotEqual(first.report.generated_at, "")
        other_root, other_tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        same_source_elsewhere = build_translation_snapshot(
            adapter,
            self.request(other_root, other_tl_dir),
        )
        self.assertEqual(
            first.report.coverage_digest,
            same_source_elsewhere.report.coverage_digest,
        )

        template = build_review_template(first.report)
        review = copy.deepcopy(template)
        review["reviewer"] = {
            "type": "agent",
            "id": "coverage-agent",
            "tool": "offline-test",
            "model": "",
            "session": "test-session",
        }
        review["status"] = "agent_reviewed"
        review["confirmed_at"] = "2026-07-30T00:00:00Z"
        validation = validate_review_record(
            review,
            first.report,
            first.inventory,
        )
        self.assertTrue(validation.policy_satisfied)
        self.assertEqual(validation.effective_status, "agent_reviewed")

        fake_human = copy.deepcopy(review)
        fake_human["status"] = "human_reviewed"
        with self.assertRaisesRegex(ValueError, "reviewer.type=human"):
            validate_review_record(
                fake_human,
                first.report,
                first.inventory,
            )

        invalid_pending_reviewer = copy.deepcopy(template)
        invalid_pending_reviewer["reviewer"]["type"] = "robot"
        with self.assertRaisesRegex(ValueError, "Unsupported reviewer.type"):
            validate_review_record(
                invalid_pending_reviewer,
                first.report,
                first.inventory,
            )

        invalid_finding = copy.deepcopy(review)
        invalid_finding["findings"] = [
            {
                "code": "review.false_positive",
                "candidate_id": first.inventory.candidates[0].candidate_id,
                "resolved": "false",
            }
        ]
        with self.assertRaisesRegex(ValueError, "resolved must be a boolean"):
            validate_review_record(
                invalid_finding,
                first.report,
                first.inventory,
            )

        class BumpedAdapter(RenPyAdapter):
            adapter_version = "1.0.1"

        bumped_adapter = BumpedAdapter(legacy_module=runtime)
        bumped_project = bumped_adapter.discover_project(self.request(root, tl_dir))
        freshness = validate_coverage_report_freshness(
            first.report,
            bumped_project,
            adapter_behavior_digest=bumped_adapter.behavior_digest(),
        )
        self.assertEqual(freshness.effective_status, "stale")
        self.assertIn("adapter_version", freshness.stale_reasons)

        (tl_dir / "script.rpy").write_text(
            'translate schinese start:\n    "Changed source."\n',
            encoding="utf-8",
        )
        changed = build_translation_snapshot(
            adapter,
            self.request(root, tl_dir),
        )
        self.assertNotEqual(
            first.report.coverage_digest,
            changed.report.coverage_digest,
        )
        old_review_against_new = validate_review_record(
            review,
            changed.report,
            changed.inventory,
        )
        self.assertEqual(old_review_against_new.effective_status, "stale")
        self.assertFalse(old_review_against_new.policy_satisfied)

        # human_required rejects agent_reviewed even when digests match.
        human_required = copy.deepcopy(review)
        human_required["review_policy"] = "human_required"
        human_required["review_input_digest"] = review_input_digest(
            first.report,
            review_policy="human_required",
            sampling_plan=human_required.get("sampling_plan"),
        )
        human_policy = validate_review_record(
            human_required,
            first.report,
            first.inventory,
        )
        self.assertEqual(human_policy.effective_status, "agent_reviewed")
        self.assertFalse(human_policy.policy_satisfied)

        package_dir = root / "coverage_export_guard"
        with self.assertRaisesRegex(ValueError, "does not match candidate inventory"):
            export_coverage_package(
                package_dir / "mismatch",
                first.project,
                first.inventory,
                replace(first.report, inventory_digest="not-the-live-inventory"),
            )
        with self.assertRaisesRegex(ValueError, "Coverage report is stale"):
            export_coverage_package(
                package_dir / "stale",
                first.project,
                first.inventory,
                replace(first.report, adapter_version="0.0.0-stale"),
            )

    def test_source_change_during_scan_blocks_automatic_report(self):
        root, tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        adapter = RenPyAdapter(legacy_module=runtime)
        project = adapter.discover_project(self.request(root, tl_dir))
        (tl_dir / "script.rpy").write_text(
            'translate schinese start:\n    "Changed after discovery."\n',
            encoding="utf-8",
        )
        inventory = adapter.inventory_candidates(
            project,
            InventoryPolicy(),
        )
        draft = adapter.audit_extraction(project, inventory)
        report = build_coverage_report(
            project,
            inventory,
            draft,
            adapter_behavior_digest=adapter.behavior_digest(),
        )
        self.assertTrue(draft.source_changed_during_scan)
        self.assertEqual(report.coverage_status, "block")
        self.assertEqual(
            report.reason_counts["coverage.source_changed_during_scan"],
            1,
        )
        stable_report = build_coverage_report(
            project,
            inventory,
            CoverageReportDraft(
                source_fingerprint=draft.source_fingerprint,
                reason_codes=draft.reason_codes,
                catalog_provenance=draft.catalog_provenance,
                catalog_freshness=draft.catalog_freshness,
            ),
            adapter_behavior_digest=adapter.behavior_digest(),
        )
        self.assertNotEqual(report.coverage_digest, stable_report.coverage_digest)
        self.assertTrue(report.source_changed_during_scan)

    def test_duplicate_candidate_position_blocks_report(self):
        root, tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        adapter = RenPyAdapter(legacy_module=runtime)
        snapshot = build_translation_snapshot(
            adapter,
            self.request(root, tl_dir),
        )
        candidate = snapshot.inventory.candidates[0]
        duplicate = replace(candidate, candidate_id="cand1:" + ("0" * 64))
        duplicate_inventory = replace(
            snapshot.inventory,
            candidates=(candidate, duplicate),
        )
        report = build_coverage_report(
            snapshot.project,
            duplicate_inventory,
            adapter.audit_extraction(snapshot.project, duplicate_inventory),
            adapter_behavior_digest=adapter.behavior_digest(),
        )
        self.assertEqual(report.coverage_status, "block")
        self.assertEqual(
            report.reason_counts["coverage.inventory.duplicate_candidate"],
            1,
        )

    def test_coverage_export_and_review_import_do_not_modify_sources(self):
        root, tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        source_path = tl_dir / "script.rpy"
        before = hashlib.sha256(source_path.read_bytes()).hexdigest()
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )

        package_dir = root / "coverage"
        paths = export_coverage_package(
            package_dir,
            snapshot.project,
            snapshot.inventory,
            snapshot.report,
        )
        for path in (
            paths.candidates_path,
            paths.report_path,
            paths.review_markdown_path,
            paths.review_template_path,
        ):
            self.assertTrue(Path(path).is_file())
        candidate_lines = Path(paths.candidates_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            len(candidate_lines),
            snapshot.report.candidate_count,
        )
        self.assertEqual(
            json.loads(Path(paths.report_path).read_text(encoding="utf-8"))["coverage_digest"],
            snapshot.report.coverage_digest,
        )
        loaded_review = load_review_record(paths.review_template_path)
        self.assertEqual(loaded_review["status"], "pending")
        self.assertEqual(
            hashlib.sha256(source_path.read_bytes()).hexdigest(),
            before,
        )

    def test_discovery_filters_are_deterministic_and_root_bounded(self):
        root, tl_dir = self.make_project(
            {
                "b.rpy": '    "B line"\n',
                "chapter/a.rpy": '    "A line"\n',
                "chapter/ignore.txt": "ignored\n",
            }
        )
        adapter = RenPyAdapter(legacy_module=runtime)
        request = self.request(
            root,
            tl_dir,
            include_prefixes=("chapter/",),
        )
        first = adapter.discover_project(request)
        second = adapter.discover_project(request)
        self.assertEqual(
            [item.file_rel_path for item in first.source_documents],
            ["chapter/a.rpy"],
        )
        self.assertEqual(
            first.project_snapshot_fingerprint,
            second.project_snapshot_fingerprint,
        )

        outside = root / "outside.rpy"
        outside.write_text('    "Outside"\n', encoding="utf-8")
        linked = tl_dir / "linked.rpy"
        try:
            linked.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are unavailable on this platform.")
        with self.assertRaisesRegex(ValueError, "outside localization root"):
            adapter.discover_project(self.request(root, tl_dir))

    def test_importing_adapter_does_not_import_gui_or_model_provider(self):
        repository_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; import engine_adapters.renpy; "
            "blocked = [name for name in sys.modules "
            "if name.startswith(('PySide6', 'google.genai'))]; "
            "raise SystemExit(1 if blocked else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_batch_translation_jobs_carry_the_same_adapter_snapshot(self):
        import gemini_translate_batch as batch

        root, tl_dir = self.make_project(
            {"script.rpy": ('translate schinese start:\n    # "Line one"\n    "Line one"\n')}
        )
        with (
            mock.patch.object(batch.legacy, "BASE_DIR", str(root)),
            mock.patch.object(batch.legacy, "TL_DIR", str(tl_dir)),
            mock.patch.object(batch.legacy, "INCLUDE_FILES", set()),
            mock.patch.object(batch.legacy, "INCLUDE_PREFIXES", set()),
        ):
            jobs = batch.collect_pending_file_jobs()

        self.assertIsInstance(jobs, batch.TranslationFileJobs)
        self.assertIsNotNone(jobs.coverage_snapshot)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            jobs[0]["tasks"],
            list(jobs.coverage_snapshot.pending_tasks_by_file["script.rpy"]),
        )

    def test_batch_package_exports_coverage_without_changing_manifest_v2(self):
        import gemini_translate_batch as batch

        root, tl_dir = self.make_project(
            {"script.rpy": ('translate schinese start:\n    "Hello there."\n')}
        )
        jobs_dir = root / "logs" / "batch_jobs"
        with (
            mock.patch.object(batch.legacy, "BASE_DIR", str(root)),
            mock.patch.object(batch.legacy, "TL_DIR", str(tl_dir)),
            mock.patch.object(batch.legacy, "INCLUDE_FILES", set()),
            mock.patch.object(batch.legacy, "INCLUDE_PREFIXES", set()),
            mock.patch.object(batch, "BATCH_JOBS_DIR", str(jobs_dir)),
            mock.patch.object(
                batch,
                "LATEST_MANIFEST_FILE",
                str(jobs_dir / "latest_manifest.txt"),
            ),
            mock.patch.object(batch, "RAG_ENABLED", False),
            mock.patch.object(batch, "SOURCE_INDEX_ENABLED", False),
            mock.patch.object(batch, "STORY_MEMORY_ENABLED", False),
        ):
            manifest_path = batch.create_batch_package(skip_prepare=True)

        self.assertIsNotNone(manifest_path)
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(manifest["manifest_version"], 2)
        self.assertEqual(manifest["core_schema_version"], 2)
        self.assertNotIn("coverage", manifest)
        self.assertEqual(
            {path.name for path in (manifest_file.parent / "coverage").iterdir()},
            {
                "coverage_candidates.jsonl",
                "coverage_report.json",
                "coverage_review.md",
                "coverage_review_template.json",
            },
        )

    def test_freshness_marks_localization_mode_mismatch_stale(self):
        root, tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        payload = snapshot.report.to_dict()
        payload["localization_mode"] = "source_extraction"
        freshness = validate_coverage_report_freshness(
            payload,
            snapshot.project,
            adapter_behavior_digest=snapshot.report.adapter_behavior_digest,
        )
        self.assertEqual(freshness.effective_status, "stale")
        self.assertIn("localization_mode", freshness.stale_reasons)

    def test_review_digest_ignores_nested_finding_display_message(self):
        root, tl_dir = self.make_project(
            {"script.rpy": 'translate schinese start:\n    "Hello there."\n'}
        )
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        template = build_review_template(snapshot.report)
        base = copy.deepcopy(template)
        base["reviewer"] = {
            "type": "agent",
            "id": "coverage-agent",
            "tool": "offline-test",
            "model": "",
            "session": "test-session",
        }
        base["status"] = "agent_reviewed"
        base["confirmed_at"] = "2026-07-30T00:00:00Z"
        base["findings"] = [
            {
                "code": "review.false_positive",
                "candidate_id": snapshot.inventory.candidates[0].candidate_id,
                "resolved": True,
                "display_message": "first wording",
            }
        ]
        first = validate_review_record(base, snapshot.report, snapshot.inventory)
        second_record = copy.deepcopy(base)
        second_record["findings"][0]["display_message"] = "different wording"
        second = validate_review_record(
            second_record,
            snapshot.report,
            snapshot.inventory,
        )
        self.assertEqual(first.coverage_review_digest, second.coverage_review_digest)

    def test_unpaired_source_marker_preserves_prior_reason_codes(self):
        source = """translate schinese start:
    # "Dangling source"
    old "Old without new"
"""
        root, tl_dir = self.make_project({"script.rpy": source})
        snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )
        unpaired = [
            candidate
            for candidate in snapshot.inventory.candidates
            if "renpy.source_marker_unpaired" in candidate.reason_codes
        ]
        self.assertTrue(unpaired)
        for candidate in unpaired:
            self.assertEqual(
                len(candidate.reason_codes),
                len(set(candidate.reason_codes)),
            )
            # Mutation path must keep earlier classification reasons when present.
            if candidate.structure_kind == "old_source_marker":
                self.assertIn("renpy.source_marker_unpaired", candidate.reason_codes)
                self.assertGreaterEqual(len(candidate.reason_codes), 1)

    def test_progress_key_upgrade_normalizes_windows_separators(self):
        progress = {
            r"nested\script.rpy": ["id:1"],
            "top.rpy": ["id:2"],
        }
        upgraded = runtime._upgrade_legacy_progress_keys(
            progress,
            [
                str(Path("nested") / "script.rpy"),
                "top.rpy",
            ],
        )
        self.assertIn("nested/script.rpy", upgraded)
        self.assertEqual(upgraded["nested/script.rpy"], ["id:1"])
        self.assertEqual(upgraded["top.rpy"], ["id:2"])
        self.assertNotIn(r"nested\script.rpy", upgraded)

    def test_sync_preview_export_failure_does_not_block_manifest(self):
        root, tl_dir = self.make_project(
            {
                "script.rpy": (
                    'translate schinese start:\n'
                    '    # "Hello"\n'
                    '    "Hello"\n'
                )
            }
        )

        def translate_batch(batch, replacements, usage_run_id="", **_kwargs):
            task = batch[0]
            replacements.setdefault(task["line"], []).append(
                (task["start"], task["end"], "你好", task.get("prefix") or "", task["quote"])
            )
            return [task.get("progress_entry") or f"id:{task['line']}"]

        with (
            mock.patch.object(runtime, "BASE_DIR", str(root)),
            mock.patch.object(runtime, "TL_DIR", str(tl_dir)),
            mock.patch.object(runtime, "LOG_DIR", str(root / "logs")),
            mock.patch.object(runtime, "SYNC_BACKEND", "litellm"),
            mock.patch.object(runtime, "PREP_ENABLED", False),
            mock.patch.object(runtime, "INCLUDE_FILES", set()),
            mock.patch.object(runtime, "INCLUDE_PREFIXES", set()),
            mock.patch.object(runtime, "load_config"),
            mock.patch.object(runtime, "load_translator_settings"),
            mock.patch.object(runtime, "load_glossary"),
            mock.patch.object(runtime, "load_progress", return_value={}),
            mock.patch.object(runtime, "process_batch_with_retry", side_effect=translate_batch),
            mock.patch.object(
                runtime,
                "export_coverage_package",
                side_effect=ValueError("stale report for test"),
            ),
        ):
            manifest_path = runtime.run_translation()

        self.assertTrue(Path(manifest_path).is_file())
        self.assertFalse((Path(manifest_path).parent / "coverage").exists())

    def test_batch_package_export_failure_still_writes_manifest(self):
        import gemini_translate_batch as batch

        root, tl_dir = self.make_project(
            {
                "script.rpy": (
                    'translate schinese start:\n'
                    '    # "Hello there."\n'
                    '    "Hello there."\n'
                )
            }
        )
        jobs_dir = root / "logs" / "batch_jobs"
        with (
            mock.patch.object(batch.legacy, "BASE_DIR", str(root)),
            mock.patch.object(batch.legacy, "TL_DIR", str(tl_dir)),
            mock.patch.object(batch.legacy, "INCLUDE_FILES", set()),
            mock.patch.object(batch.legacy, "INCLUDE_PREFIXES", set()),
            mock.patch.object(batch, "BATCH_JOBS_DIR", str(jobs_dir)),
            mock.patch.object(
                batch,
                "LATEST_MANIFEST_FILE",
                str(jobs_dir / "latest_manifest.txt"),
            ),
            mock.patch.object(batch, "RAG_ENABLED", False),
            mock.patch.object(batch, "SOURCE_INDEX_ENABLED", False),
            mock.patch.object(batch, "STORY_MEMORY_ENABLED", False),
            mock.patch.object(
                batch,
                "export_coverage_package",
                side_effect=ValueError("stale report for test"),
            ),
        ):
            manifest_path = batch.create_batch_package(skip_prepare=True)

        self.assertIsNotNone(manifest_path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(manifest["manifest_version"], 2)
        self.assertTrue(
            any("Coverage export skipped" in item for item in manifest.get("build_warnings") or [])
        )
        self.assertTrue((Path(manifest_path).parent / "requests.jsonl").is_file())

    def test_sync_preview_preserves_utf8_bom_and_crlf_through_apply(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        target = tl_dir / "script.rpy"
        body = 'translate schinese start:\r\n    # "Hello"\r\n    "Hello"\r\n'
        target.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

        def translate_batch(batch, replacements, usage_run_id="", **_kwargs):
            task = batch[0]
            replacements.setdefault(task["line"], []).append(
                (task["start"], task["end"], "你好", task.get("prefix") or "", task["quote"])
            )
            return [task.get("progress_entry") or f"id:{task['line']}"]

        with (
            mock.patch.object(runtime, "BASE_DIR", str(root)),
            mock.patch.object(runtime, "TL_DIR", str(tl_dir)),
            mock.patch.object(runtime, "LOG_DIR", str(root / "logs")),
            mock.patch.object(runtime, "SYNC_BACKEND", "litellm"),
            mock.patch.object(runtime, "PREP_ENABLED", False),
            mock.patch.object(runtime, "INCLUDE_FILES", set()),
            mock.patch.object(runtime, "INCLUDE_PREFIXES", set()),
            mock.patch.object(runtime, "load_config"),
            mock.patch.object(runtime, "load_translator_settings"),
            mock.patch.object(runtime, "load_glossary"),
            mock.patch.object(runtime, "load_progress", return_value={}),
            mock.patch.object(runtime, "process_batch_with_retry", side_effect=translate_batch),
            mock.patch.object(runtime, "export_coverage_package"),
        ):
            manifest_path = runtime.run_translation()

        package = Path(manifest_path).parent
        source_snapshot = next((package / "source").rglob("*.rpy"))
        preview_snapshot = next((package / "preview").rglob("*.rpy"))
        self.assertTrue(source_snapshot.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertTrue(preview_snapshot.read_bytes().startswith(b"\xef\xbb\xbf"))
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["files"][0]["source_sha256"],
            hashlib.sha256(target.read_bytes()).hexdigest(),
        )
        self.assertIn("writeback_plan", manifest["files"][0])

        preview.apply_sync_preview(
            manifest_path,
            active_project_root=root,
            active_tl_dir=tl_dir,
        )
        self.assertEqual(
            target.read_bytes(),
            b"\xef\xbb\xbf" + body.replace('    "Hello"', '    "你好"').encode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
