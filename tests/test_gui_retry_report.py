import unittest

from gui_qt.retry_report import (
    assess_retry_eligibility,
    build_retry_cli_args,
    parse_build_retry_output,
    retry_followup_allowed,
    summarize_retry_manifest,
)

BUILD_RETRY_OUTPUT_OK = """
Created retry package: C:\\pkg\\retry_parts\\20260625_retry
Retry chunks: 2
Retry items: 5
Failure items considered: 5
Manifest: C:\\pkg\\retry_parts\\20260625_retry\\manifest.json
"""

RETRY_MANIFEST = {
    "_manifest_path": r"C:\pkg\retry_parts\20260625_retry\manifest.json",
    "retry_of_manifest": r"C:\pkg\manifest.json",
    "summary": {
        "file_count": 2,
        "chunk_count": 2,
        "item_count": 5,
    },
    "retry_reason_counts": {
        "response_missing_item_id": 2,
        "partial_result_items": 1,
    },
    "files": {
        "game/tl/schinese/script.rpy": {},
        "game/tl/schinese/common.rpy": {},
    },
}


class GuiRetryReportTests(unittest.TestCase):
    def test_assess_retry_eligibility_accepts_retry_reasons(self):
        eligibility = assess_retry_eligibility(
            {
                "last_check_summary": {
                    "safety_level": "warn",
                    "safety_reasons": {
                        "warn": {"response_missing_item_id": 2},
                    },
                },
            }
        )

        self.assertTrue(eligibility.eligible)

    def test_assess_retry_eligibility_rejects_manual_only_reasons(self):
        eligibility = assess_retry_eligibility(
            {
                "last_check_summary": {
                    "safety_level": "warn",
                    "safety_reasons": {
                        "block": {"source_text_mismatch": 1},
                    },
                },
            }
        )

        self.assertFalse(eligibility.eligible)

    def test_assess_retry_eligibility_requires_warn(self):
        eligibility = assess_retry_eligibility(
            {
                "last_check_summary": {
                    "safety_level": "safe",
                    "safety_reasons": {
                        "warn": {"response_missing_item_id": 1},
                    },
                },
            }
        )

        self.assertFalse(eligibility.eligible)

    def test_parse_build_retry_output_extracts_manifest_path(self):
        result = parse_build_retry_output(BUILD_RETRY_OUTPUT_OK, exit_code=0)

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.retry_manifest_path.endswith("manifest.json"))
        self.assertIn("retry_parts", result.package_dir)

    def test_parse_build_retry_output_handles_no_chunks(self):
        result = parse_build_retry_output("No retry chunks needed.\n", exit_code=0)

        self.assertEqual(result.status, "empty")

    def test_summarize_retry_manifest_includes_scope_and_reasons(self):
        report = summarize_retry_manifest(RETRY_MANIFEST)

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.chunk_count, 2)
        self.assertEqual(report.item_count, 5)
        self.assertEqual(report.file_count, 2)
        self.assertEqual(report.parent_manifest_path, r"C:\pkg\manifest.json")
        self.assertTrue(any("response_missing_item_id" in line for line in report.detail_lines))
        self.assertTrue(any("script.rpy" in line for line in report.detail_lines))

    def test_retry_followup_allowed_before_and_after_confirmation(self):
        manifest_without_retry = {"last_check_summary": {"safety_level": "warn"}}
        manifest_with_retry = {
            "last_check_summary": {"safety_level": "warn"},
            "last_retry_manifest_path": r"C:\pkg\retry_parts\retry1\manifest.json",
        }
        parent_path = r"C:\pkg\manifest.json"

        self.assertTrue(
            retry_followup_allowed(
                manifest_without_retry,
                parent_manifest_path=parent_path,
                confirmed_parent_paths=set(),
            )
        )
        self.assertFalse(
            retry_followup_allowed(
                manifest_with_retry,
                parent_manifest_path=parent_path,
                confirmed_parent_paths=set(),
            )
        )
        self.assertTrue(
            retry_followup_allowed(
                manifest_with_retry,
                parent_manifest_path=parent_path,
                confirmed_parent_paths={parent_path},
            )
        )

    def test_build_retry_cli_args_only_build_retry(self):
        self.assertEqual(
            build_retry_cli_args(r"C:\pkg\manifest.json"),
            ["build-retry", r"C:\pkg\manifest.json"],
        )
        self.assertNotIn("submit", build_retry_cli_args(r"C:\pkg\manifest.json"))


if __name__ == "__main__":
    unittest.main()