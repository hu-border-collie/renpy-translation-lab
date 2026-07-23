# -*- coding: utf-8 -*-
"""Tests for final-review campaign contract (#255 PR A)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import final_review as fr


def _items(*pairs: tuple[str, str, str]) -> list[dict]:
    """Build simple review items: (file, source, translation)."""
    out = []
    for index, (file_rel, source, translation) in enumerate(pairs):
        out.append(
            {
                "id": f"id-{index}-{file_rel}",
                "identity_v2": f"id-{index}-{file_rel}",
                "file_rel_path": file_rel,
                "source": source,
                "current_translation": translation,
                "line_number": index + 1,
            }
        )
    return out


class ReadinessTests(unittest.TestCase):
    def test_pending_blocks_start_with_actionable_reason(self):
        report = fr.evaluate_readiness(
            pending_task_count=3,
            pending_files=[
                {"file_rel_path": "script.rpy", "pending_task_count": 2},
                {"file_rel_path": "day2.rpy", "pending_task_count": 1},
            ],
            review_item_count=10,
            require_zero_pending=True,
            allow_pending=False,
        )
        self.assertFalse(report.ready)
        self.assertTrue(any("未完成" in r or "pending" in r.lower() for r in report.reasons))
        self.assertTrue(any("script.rpy" in r for r in report.reasons))
        with self.assertRaises(fr.FinalReviewReadinessError) as ctx:
            fr.require_readiness(report)
        self.assertGreaterEqual(len(ctx.exception.reasons), 1)

    def test_allow_pending_override_warns_but_ready(self):
        report = fr.evaluate_readiness(
            pending_task_count=2,
            review_item_count=5,
            require_zero_pending=True,
            allow_pending=True,
        )
        self.assertTrue(report.ready)
        self.assertTrue(any("allow-pending" in r or "警告" in r for r in report.reasons))

    def test_no_review_items_blocks(self):
        report = fr.evaluate_readiness(
            pending_task_count=0,
            review_item_count=0,
            require_zero_pending=True,
        )
        self.assertFalse(report.ready)
        self.assertTrue(any("已译" in r or "可审校" in r for r in report.reasons))

    def test_zero_pending_with_items_ready(self):
        report = fr.evaluate_readiness(
            pending_task_count=0,
            review_item_count=4,
            require_zero_pending=True,
        )
        self.assertTrue(report.ready)
        self.assertEqual(report.reasons, [])


class SnapshotAndDigestTests(unittest.TestCase):
    def test_translation_digest_stable_under_reorder(self):
        items = _items(
            ("b.rpy", "Hello", "你好"),
            ("a.rpy", "Bye", "再见"),
        )
        d1 = fr.digest_translation_items(items)
        d2 = fr.digest_translation_items(list(reversed(items)))
        self.assertEqual(d1, d2)

    def test_translation_change_changes_digest(self):
        a = _items(("script.rpy", "Hello", "你好"))
        b = _items(("script.rpy", "Hello", "您好"))
        self.assertNotEqual(fr.digest_translation_items(a), fr.digest_translation_items(b))

    def test_snapshot_reproducible_and_macro_affects_digest(self):
        items = _items(("script.rpy", "Hi", "嗨"))
        snap1 = fr.build_context_snapshot(
            translation_items=items,
            macro_setting_text="tone: casual",
            glossary_enabled=False,
        )
        snap2 = fr.build_context_snapshot(
            translation_items=items,
            macro_setting_text="tone: casual",
            glossary_enabled=False,
        )
        self.assertEqual(snap1["snapshot_digest"], snap2["snapshot_digest"])

        snap3 = fr.build_context_snapshot(
            translation_items=items,
            macro_setting_text="tone: formal",
            glossary_enabled=False,
        )
        self.assertNotEqual(snap1["snapshot_digest"], snap3["snapshot_digest"])

    def test_glossary_file_content_affects_snapshot(self):
        items = _items(("script.rpy", "Hi", "嗨"))
        with tempfile.TemporaryDirectory() as tmp:
            gloss = Path(tmp) / "glossary.json"
            gloss.write_text('{"preserve_terms": ["Foo"]}', encoding="utf-8")
            snap_a = fr.build_context_snapshot(
                translation_items=items,
                glossary_path=str(gloss),
                glossary_enabled=True,
            )
            gloss.write_text('{"preserve_terms": ["Bar"]}', encoding="utf-8")
            snap_b = fr.build_context_snapshot(
                translation_items=items,
                glossary_path=str(gloss),
                glossary_enabled=True,
            )
            self.assertNotEqual(snap_a["snapshot_digest"], snap_b["snapshot_digest"])

    def test_optional_project_analysis_only_when_injectable(self):
        items = _items(("script.rpy", "Hi", "嗨"))
        without_pa = fr.build_context_snapshot(
            translation_items=items,
            project_analysis_enabled=False,
        )
        with_pa_disabled_inject = fr.build_context_snapshot(
            translation_items=items,
            project_analysis_enabled=True,
            project_analysis_inject=False,
            project_analysis_fingerprint="fp-1",
            project_analysis_status="published",
        )
        # Fingerprint not included when inject is off → digest payload PA empty.
        self.assertFalse(
            with_pa_disabled_inject["digest_payload"]["project_analysis"]["included_in_digest"]
        )
        with_pa = fr.build_context_snapshot(
            translation_items=items,
            project_analysis_enabled=True,
            project_analysis_inject=True,
            project_analysis_fingerprint="fp-1",
            project_analysis_status="published",
            project_analysis_version=1,
        )
        self.assertTrue(with_pa["digest_payload"]["project_analysis"]["included_in_digest"])
        self.assertNotEqual(without_pa["snapshot_digest"], with_pa["snapshot_digest"])


class UnitDigestAndStatusTests(unittest.TestCase):
    def test_build_units_chunking_and_stable_digest(self):
        items = _items(
            ("a.rpy", "A1", "甲1"),
            ("a.rpy", "A2", "甲2"),
            ("a.rpy", "A3", "甲3"),
            ("b.rpy", "B1", "乙1"),
        )
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items,
            chunk_size=2,
            snapshot_digest=snap["snapshot_digest"],
            model="model-x",
            prompt_schema_version="final-review-v1",
        )
        # a.rpy: 2 units (2+1), b.rpy: 1 unit
        self.assertEqual(len(units), 3)
        for unit in units:
            self.assertEqual(unit["status"], fr.STATUS_PENDING)
            self.assertTrue(unit["input_digest"])
            self.assertEqual(unit["snapshot_digest"], snap["snapshot_digest"])

        units2 = fr.build_review_units(
            items,
            chunk_size=2,
            snapshot_digest=snap["snapshot_digest"],
            model="model-x",
            prompt_schema_version="final-review-v1",
        )
        digests1 = [u["input_digest"] for u in units]
        digests2 = [u["input_digest"] for u in units2]
        self.assertEqual(digests1, digests2)

    def test_translation_change_marks_done_unit_stale(self):
        items = _items(("script.rpy", "Hello", "你好"))
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items,
            chunk_size=8,
            snapshot_digest=snap["snapshot_digest"],
            model="m",
        )
        unit = fr.mark_unit_done(units[0], finding_count=0)
        self.assertEqual(unit["status"], fr.STATUS_DONE)

        changed = _items(("script.rpy", "Hello", "您好"))
        snap2 = fr.build_context_snapshot(translation_items=changed)
        reevaluated = fr.reevaluate_campaign_units(
            [unit],
            live_items_by_unit={unit["unit_id"]: changed},
            snapshot_digest=snap2["snapshot_digest"],
            model="m",
        )
        self.assertEqual(reevaluated[0]["status"], fr.STATUS_STALE)

    def test_skip_done_same_digest(self):
        items = _items(("script.rpy", "Hello", "你好"))
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items, chunk_size=4, snapshot_digest=snap["snapshot_digest"]
        )
        unit = fr.mark_unit_done(units[0], finding_count=1)
        self.assertTrue(
            fr.should_skip_unit(unit, live_input_digest=unit["input_digest"])
        )
        self.assertFalse(fr.should_skip_unit(unit, live_input_digest=unit["input_digest"], force=True))

    def test_failure_must_not_be_done(self):
        items = _items(("script.rpy", "Hello", "你好"))
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items, chunk_size=4, snapshot_digest=snap["snapshot_digest"]
        )
        failed = fr.mark_unit_failed(units[0], "parse_error")
        self.assertEqual(failed["status"], fr.STATUS_FAILED)
        self.assertEqual(failed["completed_at"], "")
        bad = dict(failed)
        bad["status"] = fr.STATUS_DONE
        with self.assertRaises(fr.FinalReviewSchemaError):
            fr.assert_failure_not_done(bad)

    def test_campaign_status_failed_not_done_with_zero_findings(self):
        items = _items(("script.rpy", "Hello", "你好"))
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items, chunk_size=4, snapshot_digest=snap["snapshot_digest"]
        )
        units = [fr.mark_unit_failed(units[0], "model_error")]
        counts = fr.summarize_unit_statuses(units)
        self.assertEqual(fr.derive_campaign_status(counts), fr.STATUS_FAILED)
        # Even with zero findings, status is failed — never "done".
        self.assertNotEqual(fr.derive_campaign_status(counts), fr.STATUS_DONE)


class PackageIoTests(unittest.TestCase):
    def test_write_load_status_export_roundtrip(self):
        items = _items(
            ("script.rpy", "Hello", "你好"),
            ("script.rpy", "World", "世界"),
        )
        readiness = fr.evaluate_readiness(
            pending_task_count=0,
            review_item_count=len(items),
        )
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items,
            chunk_size=1,
            snapshot_digest=snap["snapshot_digest"],
            model="test-model",
        )
        findings = [
            fr.normalize_finding(
                {
                    "identity_v2": items[0]["identity_v2"],
                    "file_rel_path": "script.rpy",
                    "source": "Hello",
                    "current_translation": "你好",
                    "finding_type": "mistranslation",
                    "severity": "high",
                    "reason": "tone mismatch",
                    "suggested_revision": "您好",
                    # Model must not be allowed to claim applied:
                    "revision_state": "applied",
                },
                review_unit_id=units[0]["unit_id"],
                review_unit_digest=units[0]["input_digest"],
                model="test-model",
            )
        ]
        self.assertEqual(findings[0]["revision_state"], fr.REVISION_STATE_NONE)

        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "campaign")
            manifest = fr.build_campaign_manifest(
                package_dir=package_dir,
                display_name="test-campaign",
                snapshot=snap,
                units=units,
                readiness=readiness,
                model="test-model",
                chunk_size=1,
            )
            paths = fr.write_campaign_package(
                package_dir,
                manifest=manifest,
                snapshot=snap,
                units=units,
                findings=findings,
            )
            self.assertTrue(os.path.isfile(paths["manifest"]))
            self.assertTrue(os.path.isfile(paths["snapshot"]))
            self.assertTrue(os.path.isfile(paths["review_units"]))
            self.assertTrue(os.path.isfile(paths["findings"]))
            self.assertTrue(os.path.isfile(paths["report"]))

            loaded = fr.load_campaign_package(paths["manifest"])
            self.assertEqual(loaded["manifest"]["mode"], fr.MANIFEST_MODE_FINAL_REVIEW)
            self.assertTrue(loaded["manifest"]["report_only"])
            self.assertFalse(loaded["manifest"]["autofix"])
            self.assertEqual(len(loaded["units"]), 2)
            self.assertEqual(len(loaded["findings"]), 1)

            status = fr.collect_campaign_status(package=loaded)
            self.assertEqual(status["finding_count"], 1)
            self.assertTrue(status["report_only"])
            self.assertEqual(status["mode"], fr.MANIFEST_MODE_FINAL_REVIEW)

            export = fr.export_findings(package_dir)
            self.assertEqual(export["finding_count"], 1)
            self.assertTrue(os.path.isfile(export["jsonl_path"]))
            self.assertTrue(os.path.isfile(export["markdown_path"]))
            with open(export["jsonl_path"], encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["finding_type"], "mistranslation")
            self.assertEqual(row["revision_state"], fr.REVISION_STATE_NONE)

    def test_write_rejects_failed_marked_done(self):
        items = _items(("script.rpy", "Hello", "你好"))
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items, chunk_size=4, snapshot_digest=snap["snapshot_digest"]
        )
        bad = fr.mark_unit_failed(units[0], "boom")
        bad["status"] = fr.STATUS_DONE
        readiness = fr.evaluate_readiness(review_item_count=1, pending_task_count=0)
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "bad")
            manifest = fr.build_campaign_manifest(
                package_dir=package_dir,
                display_name="bad",
                snapshot=snap,
                units=[bad],
                readiness=readiness,
            )
            with self.assertRaises(fr.FinalReviewSchemaError):
                fr.write_campaign_package(
                    package_dir,
                    manifest=manifest,
                    snapshot=snap,
                    units=[bad],
                )


class ConfigMergeTests(unittest.TestCase):
    def test_merge_final_review_config(self):
        cfg = fr.merge_final_review_config(
            {"chunk_size": 8, "require_zero_pending": False, "model": "m1"}
        )
        self.assertEqual(cfg["chunk_size"], 8)
        self.assertFalse(cfg["require_zero_pending"])
        self.assertEqual(cfg["model"], "m1")
        self.assertEqual(cfg["prompt_schema_version"], fr.PROMPT_SCHEMA_VERSION)


class CliStatusExportTests(unittest.TestCase):
    def test_final_review_status_and_export_cli(self):
        import gemini_translate_batch as batch_mod
        from unittest import mock

        items = _items(("script.rpy", "Hello", "你好"))
        readiness = fr.evaluate_readiness(pending_task_count=0, review_item_count=1)
        snap = fr.build_context_snapshot(translation_items=items)
        units = fr.build_review_units(
            items, chunk_size=4, snapshot_digest=snap["snapshot_digest"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "fr_pkg")
            manifest = fr.build_campaign_manifest(
                package_dir=package_dir,
                display_name="cli-test",
                snapshot=snap,
                units=units,
                readiness=readiness,
            )
            paths = fr.write_campaign_package(
                package_dir,
                manifest=manifest,
                snapshot=snap,
                units=units,
                findings=[],
            )
            with mock.patch.object(batch_mod, "load_batch_settings"), mock.patch(
                "translator_runtime.load_translator_settings"
            ), mock.patch("translator_runtime.load_glossary"), mock.patch(
                "builtins.print"
            ) as print_mock:
                batch_mod.main(["final-review-status", paths["manifest"], "--json"])
            printed = "\n".join(
                str(c.args[0]) for c in print_mock.call_args_list if c.args
            )
            start = printed.find("{")
            end = printed.rfind("}")
            self.assertGreaterEqual(start, 0, msg=printed)
            self.assertGreater(end, start, msg=printed)
            payload = json.loads(printed[start : end + 1])
            self.assertEqual(payload["mode"], fr.MANIFEST_MODE_FINAL_REVIEW)
            self.assertTrue(payload["report_only"])
            self.assertFalse(payload["autofix"])

            with mock.patch.object(batch_mod, "load_batch_settings"), mock.patch(
                "translator_runtime.load_translator_settings"
            ), mock.patch("translator_runtime.load_glossary"), mock.patch(
                "builtins.print"
            ):
                batch_mod.main(["final-review-export", paths["manifest"]])
            self.assertTrue(os.path.isfile(paths["findings"]))
            self.assertTrue(os.path.isfile(paths["report"]))


class DiagnosticsCommandTests(unittest.TestCase):
    def test_diagnostics_lists_final_review_commands(self):
        from gui_qt.diagnostics_context import build_cli_commands

        commands = build_cli_commands(
            python_exe="python",
            batch_script_path="gemini_translate_batch.py",
            manifest_path=r"C:\jobs\manifest.json",
            manifest={"mode": "final_review", "job_name": ""},
        )
        labels = [c.label for c in commands]
        self.assertIn("最终审校·构建 campaign", labels)
        self.assertIn("最终审校·状态", labels)
        self.assertIn("最终审校·导出 findings", labels)
        text = "\n".join(c.command for c in commands)
        self.assertIn("final-review-build", text)
        self.assertIn("final-review-status", text)
        self.assertIn("final-review-export", text)


if __name__ == "__main__":
    unittest.main()
