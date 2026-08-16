import json
import unittest

from gui_qt.quality_findings_report import (
    build_quality_findings_report,
    filter_quality_items,
    normalize_quality_finding,
    parse_quality_findings_jsonl,
    quality_issues_report_ready,
)


FINDING_1 = {
    "finding_id": "f1",
    "reason_code": "quality.typography.cjk_latin_spacing",
    "disposition": "warning",
    "severity": "medium",
    "item_id": "item-1",
    "file": "script.rpy",
    "line": 10,
    "source": "Hello",
    "translation": "你好iPhone",
    "evidence": "{\"token\": \"iPhone\"}",
    "suggestion": "Insert spacing.",
}

FINDING_2 = {
    "finding_id": "f2",
    "reason_code": "quality.renpy.wait_tag_inside_cjk",
    "disposition": "blocker",
    "severity": "high",
    "item_id": "item-2",
    "file": "day1.rpy",
    "line": 20,
    "source": "Hi",
    "translation": "你{w=0.5}好",
    "evidence": "{\"match\": \"你{w=0.5}好\"}",
    "suggestion": "Move tag.",
}

REPORT_TEXT = "\n".join(json.dumps(item, ensure_ascii=False) for item in (FINDING_1, FINDING_2)) + "\n"


class QualityFindingsReportTests(unittest.TestCase):
    def test_parse_and_normalize_report(self):
        entries = parse_quality_findings_jsonl(REPORT_TEXT)
        items = [normalize_quality_finding(entry) for entry in entries]

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].file_rel_path, "script.rpy")
        self.assertEqual(items[0].line, 10)
        self.assertEqual(items[1].disposition, "blocker")

    def test_build_report_from_text(self):
        manifest = {
            "_manifest_path": "C:\\pkg\\manifest.json",
            "last_check_summary": {
                "quality_gate": {
                    "decision": "needs_review",
                    "warning_count": 1,
                    "blocker_count": 1,
                }
            },
        }
        report = build_quality_findings_report(
            manifest,
            manifest_path="C:\\pkg\\manifest.json",
            report_text=REPORT_TEXT,
        )

        self.assertEqual(report.status, "ok")
        self.assertEqual(len(report.items), 2)
        self.assertEqual(report.reason_counts["quality.renpy.wait_tag_inside_cjk"], 1)
        self.assertIn("其中 1 条已被项目配置提升为 blocker", report.message)

    def test_quality_issues_report_ready_uses_manifest_gate(self):
        self.assertTrue(
            quality_issues_report_ready(
                {"last_check_summary": {"quality_gate": {"warning_count": 3}}}
            )
        )
        self.assertFalse(
            quality_issues_report_ready(
                {"last_check_summary": {"quality_gate": {"warning_count": 0}}}
            )
        )

    def test_filter_by_rule_file_and_severity(self):
        items = [
            normalize_quality_finding(FINDING_1),
            normalize_quality_finding(FINDING_2),
        ]

        self.assertEqual(
            [item.item_id for item in filter_quality_items(items, reason_code="quality.renpy.wait_tag_inside_cjk")],
            ["item-2"],
        )
        self.assertEqual(
            [item.item_id for item in filter_quality_items(items, file_text="day1")],
            ["item-2"],
        )
        self.assertEqual(
            [item.item_id for item in filter_quality_items(items, min_severity="high")],
            ["item-2"],
        )


if __name__ == "__main__":
    unittest.main()
