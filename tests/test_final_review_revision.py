"""Final-review finding selection to revision lifecycle tests (#255 PR C)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import final_review as fr
import final_review_revision as handoff
import gemini_translate_batch as batch


FIXTURE = Path(__file__).parent / "fixtures" / "golden_revision_minimal" / "tl"


class FinalReviewRevisionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tl_dir = self.root / "game" / "tl" / "schinese"
        self.tl_dir.parent.mkdir(parents=True)
        shutil.copytree(FIXTURE, self.tl_dir)
        self.old = {
            "base": batch.legacy.BASE_DIR,
            "tl": batch.legacy.TL_DIR,
            "files": set(batch.legacy.INCLUDE_FILES),
            "prefixes": set(batch.legacy.INCLUDE_PREFIXES),
            "log": batch.LOG_DIR,
            "jobs": batch.BATCH_JOBS_DIR,
            "latest": batch.LATEST_MANIFEST_FILE,
            "progress": batch.PROGRESS_LOG,
            "rag": batch.RAG_ENABLED,
            "story": batch.STORY_MEMORY_ENABLED,
        }
        log_dir = self.root / "logs"
        batch.legacy.BASE_DIR = str(self.root)
        batch.legacy.TL_DIR = str(self.tl_dir)
        batch.legacy.INCLUDE_FILES = set()
        batch.legacy.INCLUDE_PREFIXES = set()
        batch.LOG_DIR = str(log_dir)
        batch.BATCH_JOBS_DIR = str(log_dir / "batch_jobs")
        batch.LATEST_MANIFEST_FILE = str(log_dir / "batch_jobs" / "latest_manifest.txt")
        batch.PROGRESS_LOG = str(log_dir / "translation_progress_batch.json")
        batch.RAG_ENABLED = False
        batch.STORY_MEMORY_ENABLED = False

    def tearDown(self):
        batch.legacy.BASE_DIR = self.old["base"]
        batch.legacy.TL_DIR = self.old["tl"]
        batch.legacy.INCLUDE_FILES = self.old["files"]
        batch.legacy.INCLUDE_PREFIXES = self.old["prefixes"]
        batch.LOG_DIR = self.old["log"]
        batch.BATCH_JOBS_DIR = self.old["jobs"]
        batch.LATEST_MANIFEST_FILE = self.old["latest"]
        batch.PROGRESS_LOG = self.old["progress"]
        batch.RAG_ENABLED = self.old["rag"]
        batch.STORY_MEMORY_ENABLED = self.old["story"]
        self.temp.cleanup()

    def _campaign(self):
        item = next(
            item
            for job in batch.collect_revision_file_jobs()
            for item in job["items"]
            if item["source"] == "Void Gate"
        )
        review_item = {
            "identity_v2": item["id"],
            "file_rel_path": item["file_rel_path"],
            "source": item["source"],
            "current_translation": item["current_translation"],
            "line_number": item["line_number"],
            "start": item["start"],
            "end": item["end"],
        }
        snapshot = fr.build_context_snapshot(translation_items=[review_item])
        unit = fr.build_review_units(
            [review_item],
            context_digest=snapshot["context_digest"],
            snapshot_digest=snapshot["snapshot_digest"],
        )[0]
        unit = fr.mark_unit_done(unit, finding_count=1)
        finding = fr.normalize_finding(
            {
                "identity_v2": item["id"],
                "file_rel_path": item["file_rel_path"],
                "source": item["source"],
                "current_translation": item["current_translation"],
                "finding_type": "terminology",
                "severity": "high",
                "reason": "统一核心术语",
                "suggested_revision": "虚空之门",
            },
            review_unit_id=unit["unit_id"],
            review_unit_digest=unit["input_digest"],
        )
        package_dir = self.root / "campaign"
        manifest = fr.build_campaign_manifest(
            package_dir=str(package_dir),
            display_name="handoff-test",
            snapshot=snapshot,
            units=[unit],
            readiness=fr.evaluate_readiness(review_item_count=1, pending_task_count=0),
            base_dir=str(self.root),
            tl_dir=str(self.tl_dir),
        )
        paths = fr.write_campaign_package(
            package_dir,
            manifest=manifest,
            snapshot=snapshot,
            units=[unit],
            findings=[finding],
        )
        return paths["manifest"], finding["finding_id"]

    def test_machine_envelopes_cover_real_campaign_lifecycle(self):
        campaign, finding_id = self._campaign()
        target = str(campaign)
        args = SimpleNamespace(target=target, force=False)

        status = batch.build_machine_success_envelope(
            "final-review-status",
            batch.run_final_review_status(target),
            args,
        )
        self.assertTrue(status["ok"])
        self.assertEqual(status["command"], "final-review-status")
        self.assertEqual(status["status"], "done")
        self.assertEqual(status["result"]["unit_count"], 1)
        self.assertEqual(status["result"]["finding_count"], 1)
        self.assertEqual(status["artifacts"]["manifest"], target)

        export = batch.build_machine_success_envelope(
            "final-review-export",
            batch.run_final_review_export(target),
            args,
        )
        self.assertTrue(export["ok"])
        self.assertEqual(export["status"], "completed")
        self.assertEqual(export["result"]["finding_count"], 1)
        self.assertTrue(Path(export["artifacts"]["findings_jsonl"]).is_file())
        self.assertTrue(Path(export["artifacts"]["findings_markdown"]).is_file())

        handoff_manifest = batch.run_final_review_create_revisions(target, [finding_id])
        create = batch.build_machine_success_envelope(
            "final-review-create-revisions",
            handoff_manifest,
            args,
        )
        self.assertTrue(create["ok"])
        self.assertEqual(
            create["status"],
            handoff_manifest["last_revision_preview"]["check_status"],
        )
        self.assertTrue(
            Path(create["artifacts"]["revision_preview_jsonl"]).is_file()
        )
        self.assertEqual(
            create["result"]["final_review_source"]["manifest_path"],
            target,
        )

        resume = batch.build_machine_success_envelope(
            "final-review-resume",
            batch.run_final_review_resume(target),
            args,
        )
        self.assertTrue(resume["ok"])
        self.assertEqual(
            resume["result"]["run_count"] + resume["result"]["skip_count"],
            1,
        )
        self.assertIn(resume["status"], {"no_work", "rebuilt"})

    def test_selected_finding_previews_then_apply_marks_real_state(self):
        campaign, finding_id = self._campaign()
        manifest = handoff.create_revision_package(batch, campaign, [finding_id])
        self.assertEqual(manifest["execution"], "final_review_handoff")
        self.assertTrue(manifest["submit_disabled"])
        self.assertEqual(manifest["last_revision_preview"]["summary"]["valid_items"], 1)
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["selection_state"], fr.SELECTION_STATE_SELECTED)
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_PREVIEWED)

        applied = batch.apply_revisions(manifest["_manifest_path"])
        self.assertEqual(applied["revision_apply_summary"]["applied_lines"], 1)
        self.assertEqual(applied["revision_apply_state"], "applied")
        self.assertIn("revision_applied_at", applied)
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_APPLIED)
        text = next(self.tl_dir.rglob("*.rpy")).read_text(encoding="utf-8")
        self.assertIn("虚空之门", text)

    def test_unchanged_selection_reports_no_op_without_marking_finding_applied(self):
        campaign, finding_id = self._campaign()
        package = fr.load_campaign_package(campaign)
        finding = next(
            row
            for row in package["findings"]
            if str(row.get("finding_id") or "") == finding_id
        )
        finding["suggested_revision"] = finding["current_translation"]
        paths = package["paths"]
        manifest = dict(package["manifest"])
        manifest["summary"] = {**dict(manifest.get("summary") or {}), "finding_count": 1}
        fr.write_campaign_package(
            paths["package_dir"],
            manifest=manifest,
            snapshot=package["snapshot"],
            units=package["units"],
            findings=[finding],
        )

        revision_manifest = handoff.create_revision_package(batch, campaign, [finding_id])
        applied = batch.apply_revisions(revision_manifest["_manifest_path"])

        self.assertEqual(applied["revision_apply_state"], "no_op")
        self.assertNotIn("revision_applied_at", applied)
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_PREVIEWED)

    def test_blocked_apply_keeps_finding_previewed(self):
        campaign, finding_id = self._campaign()
        manifest = handoff.create_revision_package(batch, campaign, [finding_id])
        result_path = Path(manifest["_package_dir"]) / manifest["result_jsonl_path"]
        result_path.write_text(
            result_path.read_text(encoding="utf-8").rstrip() + '\n{"replaced": true}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SystemExit, "result JSONL changed since preview"):
            batch.apply_revisions(manifest["_manifest_path"])
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_PREVIEWED)

    def test_repreview_after_applied_keeps_finding_applied(self):
        campaign, finding_id = self._campaign()
        manifest = handoff.create_revision_package(batch, campaign, [finding_id])
        applied = batch.apply_revisions(manifest["_manifest_path"])
        self.assertEqual(applied["revision_apply_state"], "applied")

        refreshed = batch.preview_revisions(manifest["_manifest_path"])
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_APPLIED)

        no_op = batch.apply_revisions(refreshed["_manifest_path"])
        self.assertEqual(no_op["revision_apply_state"], "no_op")
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_APPLIED)

    def test_stale_translation_refuses_candidate_creation(self):
        campaign, finding_id = self._campaign()
        path = next(self.tl_dir.rglob("*.rpy"))
        path.write_text(path.read_text(encoding="utf-8").replace("虚空门", "虚空门改"), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "changed since review"):
            handoff.create_revision_package(batch, campaign, [finding_id])

    def test_empty_apply_identity_set_does_not_claim_applied(self):
        campaign, finding_id = self._campaign()
        manifest = handoff.create_revision_package(batch, campaign, [finding_id])
        handoff.sync_linked_findings(manifest, fr.REVISION_STATE_APPLIED, identity_ids=set())
        finding = fr.load_campaign_package(campaign)["findings"][0]
        self.assertEqual(finding["revision_state"], fr.REVISION_STATE_PREVIEWED)
    def test_local_candidate_manifest_cannot_be_submitted(self):
        from unittest import mock

        with mock.patch.object(batch, "load_manifest", return_value={
            "submit_disabled": True,
            "job_name": "",
        }):
            with self.assertRaisesRegex(SystemExit, "Submit disabled"):
                batch.submit_manifest("candidate-manifest.json")
    def test_cli_exposes_repeatable_explicit_selection(self):
        help_text = batch.build_arg_parser().format_help()
        self.assertIn("final-review-create-revisions", help_text)
        args = batch.build_arg_parser().parse_args([
            "final-review-create-revisions", "campaign", "--finding-id", "a", "--finding-id", "b"
        ])
        self.assertEqual(args.finding_id, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
