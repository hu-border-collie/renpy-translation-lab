# -*- coding: utf-8 -*-
"""P2 Ren'Py relocation, validation, and declarative writeback tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from engine_adapters.contracts import (
    OpaqueLocator,
    ProjectDiscoveryRequest,
    ValidatedTranslation,
)
from engine_adapters.coverage import digest_json
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
from engine_adapters.writeback import WritebackPlanError, render_writeback_plan
import translator_runtime as runtime


class TestRenPyAdapterP2(unittest.TestCase):
    def make_project(self, source: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        (tl_dir / "script.rpy").write_text(source, encoding="utf-8")
        return root, tl_dir

    @staticmethod
    def request(root: Path, tl_dir: Path) -> ProjectDiscoveryRequest:
        return ProjectDiscoveryRequest(
            project_root=str(root),
            localization_root=str(tl_dir),
            target_language="schinese",
        )

    @staticmethod
    def occurrence_for(snapshot, source_text: str):
        return next(
            occurrence
            for occurrence in snapshot.occurrences
            if occurrence.unit.source_text == source_text
        )

    def snapshot(self, source: str):
        root, tl_dir = self.make_project(source)
        adapter = RenPyAdapter(legacy_module=runtime)
        snapshot = build_translation_snapshot(
            adapter,
            self.request(root, tl_dir),
        )
        return root, tl_dir, adapter, snapshot

    def test_relocation_handles_inserted_lines_and_preserves_identity_v2(self):
        source = (
            'translate schinese chapter:\n'
            '    # e "Hello {player}!"\n'
            '    e "Hello {player}!"\n'
        )
        root, tl_dir, adapter, snapshot = self.snapshot(source)
        original = self.occurrence_for(snapshot, "Hello {player}!")
        original_id = original.unit.id
        (tl_dir / "script.rpy").write_text(
            'translate schinese chapter:\n'
            '    # e "Inserted"\n'
            '    e "Inserted"\n'
            '    # e "Hello {player}!"\n'
            '    e "Hello {player}!"\n',
            encoding="utf-8",
        )
        live = adapter.discover_project(self.request(root, tl_dir))

        result = adapter.relocate_occurrences(
            snapshot.project,
            (original,),
            live.source_documents,
        )

        self.assertEqual(result.unresolved_occurrence_ids, ())
        self.assertEqual(len(result.occurrences), 1)
        relocated = result.occurrences[0]
        self.assertEqual(relocated.unit.id, original_id)
        self.assertGreater(relocated.unit.line, original.unit.line)
        self.assertTrue(
            any(item.get("match") == "content_evidence" for item in result.diagnostics)
        )

    def test_relocation_rejects_ambiguous_content_fallback(self):
        source = (
            'translate schinese first:\n'
            '    # e "Same text"\n'
            '    e "Same text"\n'
            'translate schinese second:\n'
            '    # e "Same text"\n'
            '    e "Same text"\n'
        )
        _root, _tl_dir, adapter, snapshot = self.snapshot(source)
        original = self.occurrence_for(snapshot, "Same text")
        stale = replace(
            original,
            content_fingerprint="",
            locator=OpaqueLocator(
                engine="renpy",
                locator_schema_version=1,
                locator={
                    "file_rel_path": "script.rpy",
                    "translate_block": "stale",
                    "block_occurrence": 1,
                    "ordinal": 99,
                    "line_hint": 1,
                    "start_col_hint": 0,
                    "end_col_hint": 0,
                    "source_marker_kind": "direct_source",
                    "candidate_ordinal": 99,
                },
            ),
            unit=replace(original.unit, id="script.rpy:stale:99:deadbeef"),
        )
        result = adapter.relocate_occurrences(
            snapshot.project, (stale,), snapshot.project.source_documents
        )
        self.assertEqual(result.occurrences, ())
        self.assertEqual(result.unresolved_occurrence_ids, (stale.occurrence_id,))
        self.assertEqual(result.diagnostics[0]["status"], "ambiguous")

    def test_relocation_reports_source_change_as_unresolved(self):
        source = (
            'translate schinese chapter:\n'
            '    # e "Hello {player}!"\n'
            '    e "Hello {player}!"\n'
        )
        root, tl_dir, adapter, snapshot = self.snapshot(source)
        original = self.occurrence_for(snapshot, "Hello {player}!")
        (tl_dir / "script.rpy").write_text(
            'translate schinese chapter:\n'
            '    # e "Goodbye {player}!"\n'
            '    e "Goodbye {player}!"\n',
            encoding="utf-8",
        )
        live = adapter.discover_project(self.request(root, tl_dir))

        result = adapter.relocate_occurrences(
            snapshot.project,
            (original,),
            live.source_documents,
        )

        self.assertEqual(result.occurrences, ())
        self.assertEqual(result.unresolved_occurrence_ids, (original.occurrence_id,))
        self.assertEqual(result.diagnostics[0]["reason_code"], "common.locator.unresolved")

    def test_validation_maps_renpy_token_differences_to_stable_reason_codes(self):
        _, _, adapter, snapshot = self.snapshot('e "Hello {player} [count] %d"\n')
        occurrence = self.occurrence_for(snapshot, "Hello {player} [count] %d")

        valid = adapter.validate_translation(
            occurrence,
            "你好 {player} [count] %d",
        )
        invalid = adapter.validate_translation(occurrence, "你好")

        self.assertEqual(valid.status, "pass")
        self.assertEqual(invalid.status, "block")
        self.assertIn("renpy.tag.changed", invalid.reason_codes)
        self.assertIn("renpy.field.changed", invalid.reason_codes)
        self.assertIn("renpy.percent_token.changed", invalid.reason_codes)

    def test_writeback_plan_is_declarative_and_uses_live_source_hashes(self):
        source = 'e "Hello {player}!"\n'
        _root, tl_dir, adapter, snapshot = self.snapshot(source)
        occurrence = self.occurrence_for(snapshot, "Hello {player}!")
        validation = adapter.validate_translation(occurrence, "你好 {player}!")
        self.assertEqual(validation.status, "pass")

        plan = adapter.build_writeback_plan(
            snapshot.project,
            (ValidatedTranslation(occurrence, "你好 {player}!", validation),),
            snapshot.project.source_documents,
        )

        self.assertEqual(plan.engine, "renpy")
        self.assertEqual(len(plan.operations), 1)
        operation = plan.operations[0]
        self.assertEqual(operation.kind, "text_span_replace")
        self.assertEqual(operation.target_root, "localization_catalog")
        self.assertEqual(operation.target_rel_path, "script.rpy")
        self.assertNotIn("\\", operation.target_rel_path)
        self.assertEqual(operation.expected_file_sha256, snapshot.project.source_documents[0].sha256)
        self.assertEqual(operation.replacement_fragment, '\"你好 {player}!\"')
        self.assertNotIn(str(tl_dir), json.dumps(plan.to_dict(), ensure_ascii=False))
        self.assertTrue(plan.plan_digest)
        self.assertEqual((tl_dir / "script.rpy").read_text(encoding="utf-8"), source)

    def test_common_plan_consumer_rechecks_snapshot_and_only_renders_memory_lines(self):
        source = 'e "Hello {player}!"\r\n'
        root, tl_dir, adapter, snapshot = self.snapshot(source)
        (tl_dir / "script.rpy").write_bytes(source.encode("utf-8"))
        snapshot = build_translation_snapshot(adapter, self.request(root, tl_dir))
        occurrence = self.occurrence_for(snapshot, "Hello {player}!")
        validation = adapter.validate_translation(occurrence, "你好 {player}!")
        plan = adapter.build_writeback_plan(
            snapshot.project,
            (ValidatedTranslation(occurrence, "你好 {player}!", validation),),
            snapshot.project.source_documents,
        )

        rendered = render_writeback_plan(plan, snapshot.project.source_documents)

        self.assertEqual(rendered["script.rpy"], ['e "你好 {player}!"\r\n'])
        self.assertEqual((tl_dir / "script.rpy").read_bytes(), source.encode("utf-8"))

        (tl_dir / "script.rpy").write_text('e "Changed {player}!"\n', encoding="utf-8")
        live = adapter.discover_project(self.request(root, tl_dir))
        with self.assertRaises(WritebackPlanError) as context:
            render_writeback_plan(plan, live.source_documents)
        self.assertEqual(context.exception.reason_code, "common.writeback.source_snapshot_mismatch")

    def test_common_plan_consumer_rejects_path_escape(self):
        _root, _tl_dir, adapter, snapshot = self.snapshot('e "Hello {player}!"\n')
        occurrence = self.occurrence_for(snapshot, "Hello {player}!")
        validation = adapter.validate_translation(occurrence, "你好 {player}!")
        plan = adapter.build_writeback_plan(
            snapshot.project,
            (ValidatedTranslation(occurrence, "你好 {player}!", validation),),
            snapshot.project.source_documents,
        )

        operation = replace(plan.operations[0], target_rel_path="../outside.rpy")
        operation_payload = operation.to_dict()
        operation_payload.pop("operation_id")
        operation = replace(
            operation,
            operation_id="op1:" + digest_json(operation_payload),
        )
        escaped_plan = replace(plan, operations=(operation,))
        plan_payload = escaped_plan.to_dict()
        plan_payload.pop("plan_digest")
        escaped_plan = replace(escaped_plan, plan_digest=digest_json(plan_payload))

        with self.assertRaises(WritebackPlanError) as context:
            render_writeback_plan(escaped_plan, snapshot.project.source_documents)
        self.assertEqual(context.exception.reason_code, "common.writeback.path_escape")

    def test_writeback_plan_rejects_stale_live_span(self):
        source = 'e "Hello {player}!"\n'
        root, tl_dir, adapter, snapshot = self.snapshot(source)
        occurrence = self.occurrence_for(snapshot, "Hello {player}!")
        validation = adapter.validate_translation(occurrence, "你好 {player}!")
        (tl_dir / "script.rpy").write_text('e "Changed {player}!"\n', encoding="utf-8")
        live = adapter.discover_project(self.request(root, tl_dir))

        with self.assertRaisesRegex(ValueError, "span/source mismatch"):
            adapter.build_writeback_plan(
                snapshot.project,
                (ValidatedTranslation(occurrence, "你好 {player}!", validation),),
                live.source_documents,
            )

    def test_writeback_plan_rejects_non_pass_validation(self):
        _, _, adapter, snapshot = self.snapshot('e "Hello {player}!"\n')
        occurrence = self.occurrence_for(snapshot, "Hello {player}!")
        validation = adapter.validate_translation(occurrence, "你好")

        with self.assertRaisesRegex(ValueError, "non-pass validation"):
            adapter.build_writeback_plan(
                snapshot.project,
                (ValidatedTranslation(occurrence, "你好", validation),),
                snapshot.project.source_documents,
            )


if __name__ == "__main__":
    unittest.main()
