import json
import tempfile
import unittest
from pathlib import Path

from gui_qt.quality_findings_report import (
    acknowledged_finding_ids_from_manifest,
    build_quality_findings_report,
    filter_quality_items,
    persist_quality_acknowledgement,
    normalize_quality_finding,
    parse_quality_findings_jsonl,
    quality_issues_report_ready,
    resolve_quality_findings_path,
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

    def test_resolve_quality_paths_prefers_revision_apply_summary(self):
        manifest = {
            "last_revision_preview": {
                "quality_findings_path": "preview/quality_findings.jsonl",
            },
            "revision_apply_summary": {
                "quality_findings_path": "apply/quality_findings.apply.jsonl",
            },
        }
        self.assertEqual(
            resolve_quality_findings_path(manifest),
            "apply/quality_findings.apply.jsonl",
        )

        blocked_manifest = {
            "last_revision_preview": {
                "quality_findings_path": "preview/quality_findings.jsonl",
            },
            "last_revision_apply_summary": {
                "quality_findings_path": "blocked/quality_findings.apply.jsonl",
            },
        }
        self.assertEqual(
            resolve_quality_findings_path(blocked_manifest),
            "blocked/quality_findings.apply.jsonl",
        )

    def test_acknowledged_finding_ids_from_manifest(self):
        self.assertEqual(
            acknowledged_finding_ids_from_manifest(
                {"quality_acknowledged_finding_ids": ["f1", "f2", "", None]}
            ),
            {"f1", "f2"},
        )

    def test_persist_quality_acknowledgement_updates_manifest_only(self):
        base = Path(tempfile.mkdtemp())
        manifest_path = base / "manifest.json"
        report_path = base / "quality_findings.jsonl"
        report_path.write_text(REPORT_TEXT, encoding="utf-8")
        manifest = {
            "mode": "translation",
            "last_quality_findings_path": str(report_path),
            "last_check_summary": {
                "check_status": "ready_with_warnings",
                "writeback_gate": {
                    "decision": "allow",
                    "can_apply": True,
                    "blocker_count": 0,
                    "quality_blocker_count": 0,
                },
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        applied = persist_quality_acknowledgement(
            str(manifest_path),
            finding_ids=["f1"],
        )

        self.assertEqual(applied["manifest"]["quality_acknowledged_finding_ids"], ["f1"])
        self.assertEqual(applied["quality_gate"]["acknowledged_count"], 1)
        # The JSONL report itself is never rewritten.
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8").splitlines()[0]),
            FINDING_1,
        )
        updated = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["quality_acknowledged_finding_ids"], ["f1"])
        self.assertEqual(updated["last_check_summary"]["quality_gate"]["acknowledged_count"], 1)

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
