"""Targeted tests for Ren'Py multi-line string inventory (issue #460)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot


class RenPyMultilineStringTests(unittest.TestCase):
    def make_project(self, filename: str, text: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        target = tl_dir / filename
        target.write_text(text, encoding="utf-8")
        return root, tl_dir

    @staticmethod
    def request(root: Path, tl_dir: Path) -> ProjectDiscoveryRequest:
        return ProjectDiscoveryRequest(
            project_root=str(root),
            localization_root=str(tl_dir),
            target_language="schinese",
        )

    def snapshot_for(self, text: str, filename: str = "script.rpy"):
        root, tl_dir = self.make_project(filename, text)
        return build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request(root, tl_dir),
        )

    def test_source_triple_quoted_dialogue_is_extractable(self):
        snapshot = self.snapshot_for('e """Hello\nworld."""\n')

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(parse_errors, [])
        candidates = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None and candidate.unit.text == "Hello\nworld."
        ]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.classification, "translatable")
        self.assertEqual(candidate.structure_kind, "dialogue_string")
        self.assertIn("renpy.dialogue_string", candidate.reason_codes)
        self.assertEqual(candidate.locator.locator.get("line_hint"), 1)
        self.assertEqual(candidate.locator.locator.get("end_line_hint"), 2)
        self.assertTrue(candidate.locator.locator.get("multiline"))
        self.assertEqual(snapshot.recognized_unit_count, 1)
        # #460 fixes inventory/extraction; legacy per-line task collection and
        # multi-line writeback spans remain separate follow-up work.
        self.assertEqual(snapshot.pending_task_count, 0)
        self.assertNotEqual(snapshot.report.coverage_status, "block")

    def test_translated_triple_quoted_dialogue_pairs_source_marker(self):
        source = (
            "translate schinese chapter:\n"
            '    # e "Hello there."\n'
            '    e """你好，\n'
            '世界。"""\n'
        )
        snapshot = self.snapshot_for(source, "translated.rpy")

        self.assertEqual(snapshot.report.classification_counts["parse_error"], 0)
        candidates = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None and candidate.unit.text == "你好，\n世界。"
        ]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.classification, "already_translated")
        self.assertIn("renpy.catalog.translation_present", candidate.reason_codes)
        self.assertIn("renpy.translate_comment_pair", candidate.reason_codes)
        self.assertEqual(candidate.locator.locator.get("end_line_hint"), 4)
        self.assertEqual(snapshot.pending_task_count, 0)
        self.assertEqual(snapshot.recognized_unit_count, 1)

    def test_backslash_continued_dialogue_keeps_python_string_semantics(self):
        snapshot = self.snapshot_for('e "Hello \\\nworld."\n')

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(parse_errors, [])
        candidates = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None and candidate.unit.text == "Hello world."
        ]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.classification, "translatable")
        self.assertEqual(candidate.locator.locator.get("line_hint"), 1)
        self.assertEqual(candidate.locator.locator.get("end_line_hint"), 2)

    def test_unterminated_single_line_string_stays_parse_error(self):
        snapshot = self.snapshot_for('e "unterminated\n')

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(len(parse_errors), 1)
        self.assertIn("renpy.tokenize_error", parse_errors[0].reason_codes)
        self.assertEqual(snapshot.report.coverage_status, "block")

    def test_comment_with_open_quote_is_not_joined(self):
        snapshot = self.snapshot_for('# "comment only\ne "Hello."\n')

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(parse_errors, [])
        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None and candidate.unit.text == "Hello."
        )
        self.assertEqual(candidate.classification, "translatable")


if __name__ == "__main__":
    unittest.main()
