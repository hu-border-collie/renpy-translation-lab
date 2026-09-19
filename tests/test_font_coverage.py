"""Offline tests for the read-only font coverage spike (#487)."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import font_coverage as fc


def format_12_subtable(groups: list[tuple[int, int]]) -> bytes:
    """Build one cmap format 12 subtable."""

    group_bytes = b"".join(
        struct.pack(">III", start, start, glyph_id)
        for start, glyph_id in groups
    )
    return struct.pack(
        ">HHIII",
        12,
        0,
        16 + len(group_bytes),
        0,
        len(groups),
    ) + group_bytes


def cmap_format_12_bytes(groups: list[tuple[int, int]]) -> bytes:
    """Build a minimal cmap with one format 12 subtable."""

    header = struct.pack(">HH", 0, 1) + struct.pack(">HHI", 3, 10, 12)
    return header + format_12_subtable(groups)


def format_4_subtable(pairs: list[tuple[int, int]]) -> bytes:
    """Build one cmap format 4 subtable with one segment per pair."""

    segments = [(codepoint, codepoint, glyph_id) for codepoint, glyph_id in pairs]
    segments.append((0xFFFF, 0xFFFF, 1))
    seg_count = len(segments)
    end_codes = b"".join(struct.pack(">H", end) for _start, end, _gid in segments)
    start_codes = b"".join(
        struct.pack(">H", start) for start, _end, _gid in segments
    )
    deltas = b"".join(
        struct.pack(">H", (glyph_id - start) & 0xFFFF)
        for start, _end, glyph_id in segments
    )
    range_offsets = b"\x00\x00" * seg_count
    length = 16 + 8 * seg_count
    return (
        struct.pack(">HHHHHHH", 4, length, 0, seg_count * 2, 0, 0, 0)
        + end_codes
        + b"\x00\x00"
        + start_codes
        + deltas
        + range_offsets
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "font_coverage_minimal"
GAME_ROOT = FIXTURES / "game"
TEXTS = FIXTURES / "texts"
TL_DIR = FIXTURES / "tl"
CJK_FONT = GAME_ROOT / "fonts" / "test_cjk_subset.ttf"
LATIN_FONT = GAME_ROOT / "fonts" / "test_latin_subset.ttf"
CORRUPT_FONT = GAME_ROOT / "fonts" / "corrupt.ttf"
SIDE_EFFECT = REPO_ROOT / "FONT_COVERAGE_SIDE_EFFECT.txt"


class FontFaceTests(unittest.TestCase):
    def test_format_12_cmap_supports_supplementary_codepoints(self) -> None:
        data = cmap_format_12_bytes([(0x20000, 7), (0x1F600, 8)])

        index = fc._parse_cmap(data, 0, len(data))

        self.assertEqual(index.format, 12)
        self.assertEqual(index.glyph_id(0x20000), 7)
        self.assertEqual(index.glyph_id(0x1F600), 8)
        self.assertIsNone(index.glyph_id(0x41))

    def test_truncated_format_0_cmap_is_rejected(self) -> None:
        header = struct.pack(">HH", 0, 1) + struct.pack(">HHI", 3, 1, 12)
        subtable = struct.pack(">HHH", 0, 262, 0) + b"\x00" * 20
        data = header + subtable

        with self.assertRaises(fc.FontCoverageError) as captured:
            fc._parse_cmap(data, 0, len(data))

        self.assertEqual(
            captured.exception.code,
            fc.REASON_FONT_CMAP_UNSUPPORTED,
        )

    def test_multiple_cmap_subtables_are_merged(self) -> None:
        format_4 = format_4_subtable([(0x41, 3)])
        format_12 = format_12_subtable([(0x20000, 7)])
        header = struct.pack(">HH", 0, 2)
        records = (
            struct.pack(">HHI", 3, 1, 20)
            + struct.pack(">HHI", 3, 10, 20 + len(format_4))
        )
        data = header + records + format_4 + format_12

        index = fc._parse_cmap(data, 0, len(data))

        self.assertEqual(index.subtable_count, 2)
        self.assertEqual(index.glyph_id(0x41), 3)
        self.assertEqual(index.glyph_id(0x20000), 7)

    def test_cjk_subset_covers_chinese_sample(self) -> None:
        face = fc.load_font_face(CJK_FONT)

        self.assertEqual(face.cmap_format, 4)
        self.assertGreater(face.glyph_count, 0)
        for character in "开始游戏你好继续，。—":
            with self.subTest(character=character):
                self.assertTrue(face.covers(character))

    def test_latin_subset_is_missing_cjk_glyphs(self) -> None:
        face = fc.load_font_face(LATIN_FONT)

        self.assertTrue(face.covers("S"))
        self.assertFalse(face.covers("你"))
        self.assertFalse(face.covers("好"))

    def test_missing_font_reports_stable_code(self) -> None:
        with self.assertRaises(fc.FontCoverageError) as captured:
            fc.load_font_face(GAME_ROOT / "fonts" / "not_here.ttf")

        self.assertEqual(captured.exception.code, fc.REASON_FONT_FILE_MISSING)

    def test_corrupt_font_reports_stable_code(self) -> None:
        with self.assertRaises(fc.FontCoverageError) as captured:
            fc.load_font_face(CORRUPT_FONT)

        self.assertEqual(captured.exception.code, fc.REASON_FONT_INVALID)


class ReferenceScanTests(unittest.TestCase):
    def setUp(self) -> None:
        if SIDE_EFFECT.exists():
            SIDE_EFFECT.unlink()

    def tearDown(self) -> None:
        if SIDE_EFFECT.exists():
            SIDE_EFFECT.unlink()

    def test_scan_never_executes_game_python(self) -> None:
        references = fc.scan_font_references(GAME_ROOT)

        self.assertTrue(references)
        self.assertFalse(SIDE_EFFECT.exists())

    def test_scan_classifies_static_dynamic_and_group_references(self) -> None:
        by_style = {item.style: item for item in fc.scan_font_references(GAME_ROOT)}

        self.assertEqual(by_style["default"].kind, "static")
        self.assertEqual(by_style["default"].path, "fonts/test_cjk_subset.ttf")
        self.assertEqual(by_style["inherited_font"].kind, "static")
        self.assertEqual(
            by_style["inherited_font"].path,
            "fonts/test_cjk_subset.ttf",
        )
        self.assertEqual(by_style["missing_file"].kind, "static")
        self.assertEqual(by_style["dynamic_font"].kind, "dynamic")
        self.assertEqual(
            by_style["dynamic_font"].reason,
            fc.REASON_FONT_DYNAMIC_EXPRESSION,
        )
        self.assertEqual(by_style["grouped_font"].kind, "group")
        self.assertEqual(
            by_style["grouped_font"].reason,
            fc.REASON_FONT_GROUP_UNSUPPORTED,
        )


class AnalyzeTests(unittest.TestCase):
    def test_mixed_fixture_reports_every_acceptance_case(self) -> None:
        report = fc.analyze_font_coverage(
            GAME_ROOT,
            text_file=TEXTS / "cjk_covered.txt",
            tl_dir=TL_DIR,
        )
        by_style = {item["style"]: item for item in report["fonts"]}

        self.assertEqual(report["status"], fc.STATUS_MISSING)
        self.assertEqual(by_style["default"]["status"], fc.STATUS_CHECKED)
        self.assertEqual(
            by_style["latin_only"]["status"],
            fc.STATUS_MISSING,
        )
        self.assertEqual(
            by_style["latin_only"]["reason"],
            fc.REASON_FONT_GLYPH_MISSING,
        )
        self.assertGreater(by_style["latin_only"]["missing_char_count"], 0)
        self.assertIn("缺少目标字符", by_style["latin_only"]["reason_text"])
        self.assertEqual(
            by_style["missing_file"]["reason"],
            fc.REASON_FONT_FILE_MISSING,
        )
        self.assertEqual(
            by_style["corrupt_font"]["reason"],
            fc.REASON_FONT_INVALID,
        )
        self.assertEqual(
            by_style["dynamic_font"]["reason"],
            fc.REASON_FONT_DYNAMIC_EXPRESSION,
        )
        self.assertEqual(
            by_style["grouped_font"]["reason"],
            fc.REASON_FONT_GROUP_UNSUPPORTED,
        )
        self.assertFalse(SIDE_EFFECT.exists())

    def test_positive_fixture_reaches_checked_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_root = Path(tmp) / "game"
            scripts = game_root / "scripts"
            fonts = game_root / "fonts"
            scripts.mkdir(parents=True)
            fonts.mkdir(parents=True)
            shutil.copyfile(CJK_FONT, fonts / "font.ttf")
            (scripts / "styles.rpy").write_text(
                'style default:\n    font "fonts/font.ttf"\n',
                encoding="utf-8",
            )

            report = fc.analyze_font_coverage(
                game_root,
                text_values=["开始游戏，你好。继续"],
            )

        self.assertEqual(report["status"], fc.STATUS_CHECKED)
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["unknown"], [])
        self.assertEqual(report["fonts"][0]["status"], fc.STATUS_CHECKED)

    def test_no_text_samples_is_unknown(self) -> None:
        report = fc.analyze_font_coverage(GAME_ROOT)

        self.assertEqual(report["status"], fc.STATUS_UNKNOWN)
        self.assertTrue(
            any(
                item["reason"] == fc.REASON_TEXT_NO_SAMPLES
                for item in report["fonts"]
            )
        )

    def test_no_font_declaration_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_root = Path(tmp) / "game"
            game_root.mkdir()
            (game_root / "script.rpy").write_text(
                'label start:\n    "Hello"\n',
                encoding="utf-8",
            )

            report = fc.analyze_font_coverage(game_root, text_values=["Hello"])

        self.assertEqual(report["status"], fc.STATUS_UNKNOWN)
        self.assertEqual(
            report["fonts"][0]["reason"],
            fc.REASON_FONT_NOT_DECLARED,
        )

    def test_font_reference_outside_game_root_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_root = Path(tmp) / "game"
            game_root.mkdir()
            outside = Path(tmp) / "outside.ttf"
            shutil.copyfile(CJK_FONT, outside)
            (game_root / "styles.rpy").write_text(
                'style default:\n    font "../outside.ttf"\n',
                encoding="utf-8",
            )

            report = fc.analyze_font_coverage(game_root, text_values=["Hello"])

        self.assertEqual(report["status"], fc.STATUS_UNKNOWN)
        self.assertEqual(
            report["fonts"][0]["reason"],
            fc.REASON_FONT_PATH_OUTSIDE_GAME,
        )

    def test_dynamic_text_samples_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_root = Path(tmp) / "game"
            game_root.mkdir()
            (game_root / "styles.rpy").write_text(
                'style default:\n    font "fonts/font.ttf"\n',
                encoding="utf-8",
            )
            fonts = game_root / "fonts"
            fonts.mkdir()
            shutil.copyfile(CJK_FONT, fonts / "font.ttf")

            report = fc.analyze_font_coverage(
                game_root,
                text_values=["{b}Hello{/b} [name]"],
            )

        self.assertEqual(report["text"]["dynamic_sample_count"], 1)


class TextExtractionTests(unittest.TestCase):
    def test_tl_extraction_keeps_target_strings_only(self) -> None:
        values = fc.extract_translation_strings(TL_DIR)

        self.assertIn("开始游戏", values)
        self.assertIn("继续", values)
        self.assertNotIn("Start Game", values)
        self.assertNotIn("Continue", values)

    def test_renpy_tags_are_removed_and_substitution_is_flagged(self) -> None:
        normalized, dynamic = fc.normalize_renpy_text("{b}你好{/b} [name]")

        self.assertEqual(normalized, "你好 ")
        self.assertTrue(dynamic)

    def test_report_json_and_markdown_are_rendered(self) -> None:
        report = fc.analyze_font_coverage(
            GAME_ROOT,
            text_file=TEXTS / "cjk_covered.txt",
        )
        document = json.loads(fc.report_to_json(report))
        markdown = fc.format_report_markdown(report)

        self.assertEqual(document["spike"], "font_coverage")
        self.assertIn("Ren'Py Font Coverage Spike", markdown)
        self.assertIn("font.glyph_missing", markdown)


class CliTests(unittest.TestCase):
    def test_cli_emits_json_without_touching_project(self) -> None:
        if SIDE_EFFECT.exists():
            SIDE_EFFECT.unlink()
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "font_coverage_report.py"),
                "--game-root",
                str(GAME_ROOT),
                "--text-file",
                str(TEXTS / "cjk_covered.txt"),
                "--tl-dir",
                str(TL_DIR),
                "--output",
                "json",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["spike"], "font_coverage")
        self.assertFalse(SIDE_EFFECT.exists())


if __name__ == "__main__":
    unittest.main()
