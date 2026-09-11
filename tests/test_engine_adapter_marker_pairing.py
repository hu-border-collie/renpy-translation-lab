"""Targeted tests for Ren'Py source marker pairing (issue #461)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot


class RenPySourceMarkerPairingTests(unittest.TestCase):
    def make_project(self, text: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        (tl_dir / "script.rpy").write_text(text, encoding="utf-8")
        return root, tl_dir

    def snapshot_for(self, text: str):
        root, tl_dir = self.make_project(text)
        return build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            ),
        )

    def test_unsupported_dynamic_string_keeps_source_marker_paired(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    # e "Dynamic greeting."\n'
            '    e f"动态问候 {player_name}"\n'
        )

        self.assertEqual(snapshot.report.classification_counts["parse_error"], 0)
        self.assertNotIn("renpy.source_marker_unpaired", snapshot.report.reason_counts)
        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "unsupported"
        )
        self.assertEqual(candidate.structure_kind, "dynamic_string_expression")
        self.assertIn("renpy.dynamic_string_expression", candidate.reason_codes)

    def test_old_marker_does_not_pair_with_non_new_statement(self):
        snapshot = self.snapshot_for(
            "translate schinese start:\n"
            '    # "Dangling source"\n'
            '    old "Old without new"\n'
            '    text f"Dynamic {name}"\n'
            '    "unterminated\n'
        )

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertGreaterEqual(len(parse_errors), 3)
        self.assertTrue(
            any(
                "renpy.source_marker_unpaired" in candidate.reason_codes
                for candidate in parse_errors
            )
        )
        self.assertTrue(
            any(
                "renpy.tokenize_error" in candidate.reason_codes
                for candidate in parse_errors
            )
        )

    def test_trailing_quoted_comment_without_target_stays_parse_error(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    # e "Hello."\n'
            '    e "你好。"\n'
            '    # TODO "internal-only-note"\n'
        )

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(len(parse_errors), 1)
        self.assertEqual(parse_errors[0].structure_kind, "source_comment")
        self.assertIn("renpy.source_marker_unpaired", parse_errors[0].reason_codes)

    def test_voice_statement_does_not_consume_comment_marker(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    # e "Hello."\n'
            '    voice "voice/sample.ogg"\n'
            '    e "你好。"\n'
        )

        self.assertEqual(snapshot.report.classification_counts["parse_error"], 0)
        dialogue = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None and candidate.unit.text == "你好。"
        )
        self.assertEqual(dialogue.classification, "already_translated")
        self.assertIn("renpy.translate_comment_pair", dialogue.reason_codes)


if __name__ == "__main__":
    unittest.main()
