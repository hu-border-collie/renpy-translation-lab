import json
import unittest

from gui_qt.apply_failure_report import (
    APPLY_REASON_SUGGESTIONS,
    apply_failure_report_available,
    apply_reason_code_label,
    build_apply_failure_report,
    parse_apply_failure_report_payload,
)

SAMPLE_REPORT = {
    "timestamp": "2026-06-25T13:00:00",
    "status": "block",
    "reason_code": "stale_check_fingerprint",
    "error": "Manifest or results changed after the last check. Run check again before apply.",
    "manifest_path": r"C:\pkg\manifest.json",
    "last_check_at": "2026-06-25T12:50:00",
    "last_check_safety_level": "safe",
    "current_check_fingerprint": {
        "id": "fp-123",
        "pending_files": 2,
        "pending_lines": 18,
        "failure_items": 1,
    },
    "summary": {
        "safety_level": "block",
        "pending_files": 2,
        "pending_lines": 18,
        "failure_items": 1,
        "source_mismatch_items": 1,
    },
    "failure_count": 1,
    "failures_path": r"C:\pkg\failures.jsonl",
}

SAMPLE_FAILURES_JSONL = "\n".join(
    [
        json.dumps(
            {
                "reason_code": "source_text_mismatch",
                "status": "block",
                "file_rel_path": "game/tl/schinese/script.rpy",
                "line": 12,
                "item_id": "item-1",
                "error": "Source text mismatch",
            },
            ensure_ascii=False,
        )
    ]
)


class GuiApplyFailureReportTests(unittest.TestCase):
    def test_apply_reason_code_label_covers_known_codes(self):
        self.assertEqual(
            apply_reason_code_label("stale_check_fingerprint"),
            "检查结果已过期",
        )
        self.assertIn(
            "重新检查",
            APPLY_REASON_SUGGESTIONS["stale_check_fingerprint"],
        )

    def test_parse_apply_failure_report_payload_normalizes_fields(self):
        parsed = parse_apply_failure_report_payload(SAMPLE_REPORT)

        self.assertEqual(parsed["reason_code"], "stale_check_fingerprint")
        self.assertEqual(parsed["last_check_safety_level"], "safe")
        self.assertEqual(parsed["failure_count"], 1)

    def test_build_apply_failure_report_from_text_and_failures(self):
        manifest = {
            "_manifest_path": r"C:\pkg\manifest.json",
            "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
        }

        report = build_apply_failure_report(
            manifest,
            manifest_path=r"C:\pkg\manifest.json",
            report_text=json.dumps(SAMPLE_REPORT, ensure_ascii=False),
            failures_text=SAMPLE_FAILURES_JSONL,
        )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.reason_code, "stale_check_fingerprint")
        self.assertEqual(report.failure_item_count, 1)
        self.assertTrue(any("stale_check_fingerprint" in line for line in report.detail_lines))
        self.assertTrue(any("script.rpy" in line for line in report.detail_lines))
        self.assertTrue(any("最近检查" in fact for fact in report.facts))

    def test_build_apply_failure_report_keeps_valid_report_when_failures_unreadable(self):
        manifest = {
            "_manifest_path": r"C:\pkg\manifest.json",
            "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
        }

        report = build_apply_failure_report(
            manifest,
            manifest_path=r"C:\pkg\manifest.json",
            report_text=json.dumps(SAMPLE_REPORT, ensure_ascii=False),
            failures_text="not valid jsonl",
        )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.reason_code, "stale_check_fingerprint")
        self.assertTrue(any("失败明细解析错误" in fact for fact in report.facts))
        self.assertTrue(
            any("无法解析 failures.jsonl" in line for line in report.detail_lines)
        )

    def test_build_apply_failure_report_missing_file(self):
        manifest = {
            "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
        }

        report = build_apply_failure_report(
            manifest,
            path_exists=lambda _path: False,
        )

        self.assertEqual(report.status, "missing_report")

    def test_apply_failure_report_available_respects_applied_manifest(self):
        manifest = {
            "applied_at": "2026-06-25T12:00:00",
            "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
        }

        self.assertFalse(
            apply_failure_report_available(
                manifest,
                path_exists=lambda _path: True,
            )
        )

    def test_apply_failure_report_available_when_report_exists(self):
        manifest = {
            "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
        }

        self.assertTrue(
            apply_failure_report_available(
                manifest,
                path_exists=lambda path: path.endswith("apply_failure_report.json"),
            )
        )


if __name__ == "__main__":
    unittest.main()