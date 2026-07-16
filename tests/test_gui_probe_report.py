import unittest

from gui_qt.probe_report import (
    build_probe_cli_args,
    parse_probe_output,
    summarize_probe_output,
    translation_probe_ready,
)


PROBE_OUTPUT_OK = """
[1/3] chunk-001
  finish_reason: STOP
  parsed_items: 4/4
  parse_ok: True
Probe summary:
- sample_count: 3
- parse_ok: 3
- full_item_match: 3
- max_tokens: 0
- missing_text: 0
- request_errors: 0
- summary_file: C:\\pkg\\probe_summary.json
- results_file: C:\\pkg\\probe_results.jsonl
"""

PROBE_OUTPUT_PARTIAL = """
Probe summary:
- sample_count: 3
- parse_ok: 2
- full_item_match: 1
- max_tokens: 1
- missing_text: 0
- request_errors: 1
- summary_file: C:\\pkg\\probe_summary.json
- results_file: C:\\pkg\\probe_results.jsonl
"""


class GuiProbeReportTests(unittest.TestCase):
    def test_build_probe_cli_args_uses_defaults(self):
        args = build_probe_cli_args(r"C:\pkg\manifest.json")
        self.assertEqual(
            args,
            ["probe", r"C:\pkg\manifest.json", "--limit", "3", "--offset", "0"],
        )

    def test_build_probe_cli_args_includes_api_key_index(self):
        args = build_probe_cli_args(
            r"C:\pkg\manifest.json",
            limit=5,
            offset=2,
            api_key_index=1,
        )
        self.assertEqual(args[-2:], ["--api-key-index", "1"])
        self.assertIn("--limit", args)
        self.assertIn("5", args)

    def test_parse_probe_output_extracts_counts_and_files(self):
        parsed = parse_probe_output(PROBE_OUTPUT_OK)
        self.assertEqual(parsed["sample_count"], 3)
        self.assertEqual(parsed["parse_ok"], 3)
        self.assertEqual(parsed["request_errors"], 0)
        self.assertEqual(parsed["summary_file"], r"C:\pkg\probe_summary.json")

    def test_summarize_probe_output_marks_full_success_as_ok(self):
        summary = summarize_probe_output(
            PROBE_OUTPUT_OK,
            0,
            manifest_path=r"C:\pkg\manifest.json",
        )
        self.assertEqual(summary.status, "ok")
        self.assertEqual(summary.sample_count, 3)
        self.assertEqual(summary.parse_ok, 3)

    def test_summarize_probe_output_marks_partial_success_as_warn(self):
        summary = summarize_probe_output(
            PROBE_OUTPUT_PARTIAL,
            0,
            manifest_path=r"C:\pkg\manifest.json",
        )
        self.assertEqual(summary.status, "warn")
        self.assertTrue(any("请求失败" in finding for finding in summary.findings))

    def test_summarize_probe_output_nonzero_exit_is_failed(self):
        summary = summarize_probe_output(PROBE_OUTPUT_OK, 1)
        self.assertEqual(summary.status, "failed")

    def test_translation_probe_ready_rejects_non_translation_manifest(self):
        ready, message = translation_probe_ready(
            r"C:\pkg\manifest.json",
            {"mode": "revision", "version": 1, "input_jsonl_path": r"C:\pkg\requests.jsonl"},
        )
        self.assertFalse(ready)
        self.assertIn("批量翻译", message)

    def test_translation_probe_ready_accepts_translation_manifest(self):
        ready, message = translation_probe_ready(
            r"C:\pkg\manifest.json",
            {
                "mode": "translation",
                "version": 1,
                "input_jsonl_path": r"C:\pkg\requests.jsonl",
            },
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_translation_probe_ready_accepts_v2_translation_manifest(self):
        ready, message = translation_probe_ready(
            r"C:\pkg\manifest.json",
            {
                "mode": "translation",
                "version": 2,
                "manifest_version": 2,
                "input_jsonl_path": r"C:\pkg\requests.jsonl",
            },
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")


if __name__ == "__main__":
    unittest.main()
