import json
import os
import tempfile
import unittest

from gui_qt.ab_experiment_report import (
    AB_VARIANT_CHOICE_FORCE_OFF,
    AB_VARIANT_CHOICE_FORCE_ON,
    AB_VARIANT_CHOICE_SKIP,
    ab_experiment_summary_to_diagnostics_context,
    build_compare_variants_cli_args,
    build_variants_from_gui_selection,
    collect_ab_experiment_issues,
    manifest_chunk_count,
    parse_compare_variants_output,
    read_ab_dimension_enabled_states,
    summarize_compare_variants_output,
    translation_ab_experiment_ready,
    validate_ab_experiment_variants,
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
                variant_names="baseline, story_memory_on",
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

    def test_manifest_chunk_count_uses_summary_when_chunks_omitted(self):
        manifest = {
            "mode": "translation",
            "summary": {"chunk_count": 42},
        }
        self.assertEqual(manifest_chunk_count(manifest), 42)

    def test_translation_ab_experiment_ready_accepts_lite_manifest_with_summary(self):
        ready, message = translation_ab_experiment_ready(
            r"C:\pkg\manifest.json",
            {
                "mode": "translation",
                "summary": {"chunk_count": 120},
            },
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_translation_ab_experiment_ready_accepts_translation_manifest(self):
        ready, message = translation_ab_experiment_ready(
            r"C:\pkg\manifest.json",
            {"mode": "translation", "chunks": [{"key": "chunk-1"}]},
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_read_ab_dimension_enabled_states_reads_batch_config(self):
        states = read_ab_dimension_enabled_states(
            {
                "batch": {
                    "story_memory": {"enabled": True},
                    "rag": {"enabled": False},
                    "source_index": {"enabled": True},
                },
            },
        )
        self.assertTrue(states["story_memory"])
        self.assertFalse(states["rag"])
        self.assertTrue(states["source_index"])

    def test_build_variants_from_gui_selection_includes_baseline_and_selected(self):
        variants = build_variants_from_gui_selection(
            {
                "story_memory": AB_VARIANT_CHOICE_FORCE_ON,
                "rag": AB_VARIANT_CHOICE_SKIP,
                "source_index": AB_VARIANT_CHOICE_FORCE_OFF,
            },
        )
        self.assertEqual(variants[0]["name"], "baseline")
        names = [entry["name"] for entry in variants]
        self.assertEqual(names, ["baseline", "story_memory_on", "source_index_off"])
        valid, message = validate_ab_experiment_variants(variants)
        self.assertTrue(valid)
        self.assertEqual(message, "")

    def test_validate_ab_experiment_variants_requires_second_variant(self):
        valid, message = validate_ab_experiment_variants(
            build_variants_from_gui_selection(
                {
                    "story_memory": AB_VARIANT_CHOICE_SKIP,
                    "rag": AB_VARIANT_CHOICE_SKIP,
                    "source_index": AB_VARIANT_CHOICE_SKIP,
                },
            ),
        )
        self.assertFalse(valid)
        self.assertIn("至少选择", message)

    def test_collect_ab_experiment_issues_detects_variant_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "ab_results.jsonl")
            with open(results_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "chunk_key": "chunk-1",
                            "variants": [
                                {"name": "baseline", "error": ""},
                                {"name": "rag_off", "error": "API timeout"},
                            ],
                        },
                    )
                    + "\n",
                )
            severity, findings = collect_ab_experiment_issues(
                results_path=results_path,
                expected_variant_count=2,
            )
        self.assertEqual(severity, "warn")
        self.assertTrue(any("部分变体出错" in finding for finding in findings))

    def test_summarize_compare_variants_output_marks_partial_variant_errors_as_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ab_report.md")
            results_path = os.path.join(tmp, "ab_results.jsonl")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("# report\n")
            with open(results_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "chunk_key": "chunk-1",
                            "variants": [
                                {"name": "baseline"},
                                {"name": "story_memory_on", "error": "quota exceeded"},
                            ],
                        },
                    )
                    + "\n",
                )
            output = COMPARE_VARIANTS_OUTPUT_OK.replace(
                r"C:\logs\experiments\20260707_ab\ab_report.md",
                report_path,
            ).replace(
                r"C:\logs\experiments\20260707_ab\ab_results.jsonl",
                results_path,
            )
            summary = summarize_compare_variants_output(
                output,
                0,
                manifest_path=r"C:\pkg\manifest.json",
                variant_names="baseline, story_memory_on",
            )
        self.assertEqual(summary.status, "warn")
        self.assertTrue(any("部分变体出错" in finding for finding in summary.findings))

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