import unittest

from gui_qt.split_report import (
    build_split_cli_args,
    parse_split_output,
    summarize_split_output,
    translation_split_ready,
)


SPLIT_OUTPUT_OK = """
Created split package: C:\\pkg\\split_parts\\part01_of_02
Chunks: 400
Items: 1200
Created split package: C:\\pkg\\split_parts\\part02_of_02
Chunks: 350
Items: 980
Source manifest updated: C:\\pkg\\manifest.json
Latest manifest set to first split package: C:\\pkg\\split_parts\\part01_of_02\\manifest.json
"""

SPLIT_OUTPUT_UNCHANGED = "Split not needed; current package already fits the requested limits."


class GuiSplitReportTests(unittest.TestCase):
    def test_build_split_cli_args_uses_default_max_chunks(self):
        args = build_split_cli_args(r"C:\pkg\manifest.json")
        self.assertEqual(args, ["split", r"C:\pkg\manifest.json", "--max-chunks", "600"])

    def test_build_split_cli_args_includes_optional_limits(self):
        args = build_split_cli_args(
            r"C:\pkg\manifest.json",
            max_chunks=400,
            max_items=1000,
            display_name_prefix="demo-game",
        )
        self.assertIn("--max-items", args)
        self.assertIn("1000", args)
        self.assertIn("--display-name-prefix", args)
        self.assertIn("demo-game", args)

    def test_parse_split_output_collects_child_manifest_paths(self):
        parsed = parse_split_output(SPLIT_OUTPUT_OK)
        self.assertEqual(len(parsed["child_manifest_paths"]), 2)
        self.assertTrue(
            str(parsed["child_manifest_paths"][0]).endswith("manifest.json")
        )
        self.assertEqual(
            parsed["latest_manifest_path"],
            r"C:\pkg\split_parts\part01_of_02\manifest.json",
        )

    def test_summarize_split_output_marks_success(self):
        summary = summarize_split_output(
            SPLIT_OUTPUT_OK,
            0,
            manifest_path=r"C:\pkg\manifest.json",
        )
        self.assertEqual(summary.status, "ok")
        self.assertEqual(len(summary.child_manifest_paths or []), 2)
        self.assertTrue(any("RAG" in finding for finding in summary.findings))

    def test_summarize_split_output_marks_unchanged(self):
        summary = summarize_split_output(SPLIT_OUTPUT_UNCHANGED, 0)
        self.assertEqual(summary.status, "unchanged")

    def test_summarize_split_output_nonzero_exit_is_failed(self):
        summary = summarize_split_output(SPLIT_OUTPUT_OK, 1)
        self.assertEqual(summary.status, "failed")

    def test_translation_split_ready_requires_chunks(self):
        ready, message = translation_split_ready(
            r"C:\pkg\manifest.json",
            {
                "mode": "translation",
                "version": 1,
                "input_jsonl_path": r"C:\pkg\requests.jsonl",
                "chunks": [],
            },
        )
        self.assertFalse(ready)
        self.assertIn("块", message)

    def test_translation_split_ready_accepts_translation_manifest(self):
        ready, message = translation_split_ready(
            r"C:\pkg\manifest.json",
            {
                "mode": "translation",
                "version": 1,
                "input_jsonl_path": r"C:\pkg\requests.jsonl",
                "chunks": [{"key": "a"}],
            },
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_translation_split_ready_accepts_lite_v2_translation_manifest(self):
        ready, message = translation_split_ready(
            r"C:\pkg\manifest.json",
            {
                "mode": "translation",
                "version": 2,
                "manifest_version": 2,
                "input_jsonl_path": r"C:\pkg\requests.jsonl",
                "summary": {"chunk_count": 4},
            },
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")


if __name__ == "__main__":
    unittest.main()
