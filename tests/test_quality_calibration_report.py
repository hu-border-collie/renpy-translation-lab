"""Tests for the issue #364 offline quality calibration report helper."""
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import translation_quality as quality
from scripts import quality_calibration_report as report


def write_findings(path: Path) -> Path:
    """Write a deterministic synthetic findings report and return its path."""
    rows = [
        {
            "reason_code": quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE,
            "file": "chapter01/dialogue.rpy",
            "line": 2,
            "source": "She said yes.",
            "translation": "TA也说过",
        },
        {
            "reason_code": quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE,
            "file": "chapter01/dialogue.rpy",
            "line": 3,
            "source": "Use your HP.",
            "translation": "当前HP为100",
        },
        {
            "reason_code": quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE,
            "file": "chapter02/events.rpy",
            "line": 7,
            "source": "Welcome back, Sir.",
            "translation": "欢迎回来，Sir。",
        },
        {
            "reason_code": quality.REASON_SPEAKER_LABEL_UNTRANSLATED,
            "file": "chapter02/events.rpy",
            "line": 9,
            "source": "Welcome.",
            "translation": "欢迎。",
        },
        {
            "reason_code": quality.REASON_CJK_LATIN_SPACING,
            "file": "chapter01/dialogue.rpy",
            "line": 2,
            "source": "An iPhone.",
            "translation": "这是iPhone手机",
        },
    ]
    quality.write_findings(path, rows)
    return path


class QualityCalibrationReportTests(unittest.TestCase):
    def test_summary_counts_and_file_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            markdown = report.render_report(
                report.load_report_findings(path),
                source_path=str(path),
                generated_at="2026-08-22T00:00:00+00:00",
            )

        self.assertIn("- Findings: 5", markdown)
        self.assertIn("- Reason codes: 3", markdown)
        self.assertIn("- Files: 2", markdown)
        self.assertIn(
            "| `quality.language.suspicious_english_residue` "
            "| `suspicious_english_residue` | 3 | 2 |",
            markdown,
        )
        self.assertIn("| chapter01/dialogue.rpy | 2 |", markdown)
        self.assertIn("| chapter02/events.rpy | 1 |", markdown)

    def test_reason_sections_are_ordered_by_finding_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            markdown = report.render_report(
                report.load_report_findings(path),
                source_path=str(path),
                generated_at="2026-08-22T00:00:00+00:00",
            )

        residue = markdown.index(
            "## quality.language.suspicious_english_residue"
        )
        speaker = markdown.index("## quality.speaker.label_untranslated")
        spacing = markdown.index("## quality.typography.cjk_latin_spacing")

        self.assertLess(residue, speaker)
        self.assertLess(speaker, spacing)

    def test_sample_limit_bounds_each_reason_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            markdown = report.render_report(
                report.load_report_findings(path),
                source_path=str(path),
                sample_limit=1,
                generated_at="2026-08-22T00:00:00+00:00",
            )

        self.assertIn(
            "| chapter01/dialogue.rpy | 2 | She said yes. | TA也说过 |",
            markdown,
        )
        self.assertNotIn(
            "| chapter01/dialogue.rpy | 3 | Use your HP. |",
            markdown,
        )
        self.assertNotIn(
            "| chapter02/events.rpy | 7 | Welcome back, Sir. |",
            markdown,
        )

    def test_markdown_cells_escape_pipes_backslashes_and_newlines(self) -> None:
        self.assertEqual(
            report._md_cell("a|b\\c\nd"),
            "a\\|b\\\\c<br>d",
        )

    def test_main_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            output = Path(tmp) / "calibration.md"

            exit_code = report.main([str(path), "-o", str(output)])

            self.assertEqual(exit_code, 0)
            self.assertIn(
                "# Quality calibration report",
                output.read_text(encoding="utf-8"),
            )

    def test_main_missing_input_returns_error_exit_code(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = report.main(["/nonexistent/quality_findings.jsonl"])

        self.assertEqual(exit_code, 2)
        self.assertIn("file not found", stderr.getvalue())

    def test_main_unwritable_output_returns_error_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            missing_dir_output = Path(tmp) / "missing-dir" / "calibration.md"

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = report.main([str(path), "-o", str(missing_dir_output)])

        self.assertEqual(exit_code, 2)
        self.assertIn("could not write calibration report", stderr.getvalue())

    def test_output_file_uses_lf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            output = Path(tmp) / "calibration.md"

            exit_code = report.main([str(path), "-o", str(output)])
            data = output.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertNotIn(b"\r\n", data)
        self.assertNotIn(b"\r", data)
        self.assertTrue(data.endswith(b"\n"))

    def test_pinned_generated_at_makes_reruns_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_findings(Path(tmp) / "quality_findings.jsonl")
            first = Path(tmp) / "first.md"
            second = Path(tmp) / "second.md"
            argv = [str(path), "--generated-at", "2026-01-01T00:00:00+00:00"]

            self.assertEqual(report.main([*argv, "-o", str(first)]), 0)
            self.assertEqual(report.main([*argv, "-o", str(second)]), 0)

            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertIn(b"- Generated: 2026-01-01T00:00:00+00:00", first_bytes)
        self.assertEqual(first_bytes, second_bytes)

    def test_load_report_findings_rejects_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_findings.jsonl"
            path.write_text('{"reason_code": broken\n', encoding="utf-8")

            with self.assertRaises(report.CalibrationReportError):
                report.load_report_findings(path)


if __name__ == "__main__":
    unittest.main()
