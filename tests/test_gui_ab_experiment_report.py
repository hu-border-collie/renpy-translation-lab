import os
import tempfile
import unittest

from gui_qt.ab_experiment_report import (
    ab_experiment_summary_to_diagnostics_context,
    build_compare_variants_cli_args,
    parse_compare_variants_output,
    summarize_compare_variants_output,
    translation_ab_experiment_ready,
)
from gui_qt.diagnostics_context import DiagnosticsContext


COMPARE_VARIANTS_OUTPUT_OK = """
Translation A/B experiment:
- output_dir: C:\\logs\\experiments\\20260707_ab
- chunks: 1
- variants: 2
- dry_run: True
- report: C:\\logs\\experiments\\20260707_ab\\ab_report.md
- results: C:\\logs\\experiments\\20260707_ab\\ab_results.jsonl
"""


class GuiAbExperimentReportTests(unittest.TestCase):
    def test_build_compare_variants_cli_args_includes_dry_run(self):
        args = build_compare_variants_cli_args(
            r"C:\pkg\manifest.json",
            r"C:\pkg\variants.json",
            dry_run=True,
        )
        self.assertEqual(args[:4], ["compare-variants", r"C:\pkg\manifest.json", "--variants-file", r"C:\pkg\variants.json"])
        self.assertIn("--dry-run", args)

    def test_parse_compare_variants_output_extracts_paths_and_counts(self):
        parsed = parse_compare_variants_output(COMPARE_VARIANTS_OUTPUT_OK)
        self.assertEqual(parsed["chunks"], 1)
        self.assertEqual(parsed["variants"], 2)
        self.assertTrue(parsed["dry_run"])
        self.assertTrue(str(parsed["report"]).endswith("ab_report.md"))

    def test_summarize_compare_variants_output_marks_existing_report_as_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ab_report.md")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("# report\n")
            output = COMPARE_VARIANTS_OUTPUT_OK.replace(
                r"C:\logs\experiments\20260707_ab\ab_report.md",
                report_path,
            )
            summary = summarize_compare_variants_output(
                output,
                0,
                manifest_path=r"C:\pkg\manifest.json",
                variants_file=r"C:\pkg\variants.json",
            )
        self.assertEqual(summary.status, "ok")
        self.assertEqual(summary.chunk_count, 1)
        self.assertEqual(summary.variant_count, 2)
        self.assertTrue(summary.dry_run)

    def test_summarize_compare_variants_output_nonzero_exit_is_failed(self):
        summary = summarize_compare_variants_output(COMPARE_VARIANTS_OUTPUT_OK, 1)
        self.assertEqual(summary.status, "failed")

    def test_translation_ab_experiment_ready_rejects_non_translation_manifest(self):
        ready, message = translation_ab_experiment_ready(
            r"C:\pkg\manifest.json",
            {"mode": "revision", "chunks": [{"key": "chunk-1"}]},
        )
        self.assertFalse(ready)
        self.assertIn("批量翻译", message)

    def test_translation_ab_experiment_ready_requires_chunks(self):
        ready, message = translation_ab_experiment_ready(
            r"C:\pkg\manifest.json",
            {"mode": "translation", "chunks": []},
        )
        self.assertFalse(ready)
        self.assertIn("翻译块", message)

    def test_translation_ab_experiment_ready_accepts_translation_manifest(self):
        ready, message = translation_ab_experiment_ready(
            r"C:\pkg\manifest.json",
            {"mode": "translation", "chunks": [{"key": "chunk-1"}]},
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_ab_experiment_summary_to_diagnostics_context_adds_report_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ab_report.md")
            results_path = os.path.join(tmp, "ab_results.jsonl")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("# report\n")
            with open(results_path, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            output = COMPARE_VARIANTS_OUTPUT_OK.replace(
                r"C:\logs\experiments\20260707_ab\ab_report.md",
                report_path,
            ).replace(
                r"C:\logs\experiments\20260707_ab\ab_results.jsonl",
                results_path,
            )
            base = DiagnosticsContext(
                status="ready",
                heading="诊断上下文",
                message="base",
                facts=["fact-a"],
                paths=[],
                commands=[],
                manifest_json_preview="",
            )
            summary = summarize_compare_variants_output(
                output,
                0,
                manifest_path=r"C:\pkg\manifest.json",
            )
            context = ab_experiment_summary_to_diagnostics_context(summary, base)
            labels = [entry.label for entry in context.paths]
            self.assertIn("A/B 报告", labels)
            self.assertIn("A/B 结果", labels)


if __name__ == "__main__":
    unittest.main()