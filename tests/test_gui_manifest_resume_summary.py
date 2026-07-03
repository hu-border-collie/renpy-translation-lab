"""Tests for archived completed manifest display behavior."""
from __future__ import annotations

import unittest
from unittest import mock

from gui_qt.manifest_resume_summary import build_manifest_workflow_display
from gui_qt.work_modes import WORK_MODE_SPECS, WorkMode


class ManifestResumeSummaryTests(unittest.TestCase):
    def test_completed_translation_archives_when_idle(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "RESULTS_MERGED",
            "job_name": "batches/example",
            "summary": {"file_count": 5, "chunk_count": 42, "item_count": 800},
        }
        with mock.patch("gui_qt.manifest_resume_summary.resume_workflow") as resume_mock:
            workflow = mock.Mock()
            workflow.current_step.return_value = None
            resume_mock.return_value = workflow
            display = build_manifest_workflow_display(
                spec,
                "C:/logs/batch_jobs/20260615_Game_Example/manifest.json",
                manifest,
            )

        self.assertTrue(display.archive_when_idle)
        self.assertEqual(display.status, "done")

    def test_incomplete_translation_stays_active(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "JOB_STATE_SUCCEEDED",
            "job_name": "batches/example",
            "summary": {"file_count": 3, "chunk_count": 2, "item_count": 120},
        }
        with mock.patch("gui_qt.manifest_resume_summary.resume_workflow") as resume_mock:
            workflow = mock.Mock()
            step = mock.Mock()
            step.key = "check"
            workflow.current_step.return_value = step
            resume_mock.return_value = workflow
            display = build_manifest_workflow_display(
                spec,
                "C:/logs/batch_jobs/current/manifest.json",
                manifest,
            )

        self.assertFalse(display.archive_when_idle)
        self.assertEqual(display.status, "ready")

    def test_retry_manifest_with_safety_warn_stays_active(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "JOB_STATE_SUCCEEDED",
            "job_name": "batches/retry",
            "retry_of_manifest": "C:/logs/batch_jobs/parent/manifest.json",
            "last_check_summary": {"safety_level": "warn"},
        }
        with mock.patch("gui_qt.manifest_resume_summary.resume_workflow") as resume_mock:
            workflow = mock.Mock()
            workflow.current_step.return_value = None
            resume_mock.return_value = workflow
            display = build_manifest_workflow_display(
                spec,
                "C:/logs/batch_jobs/retry/manifest.json",
                manifest,
            )

        self.assertFalse(display.archive_when_idle)
        self.assertEqual(display.status, "stale")
        self.assertEqual(display.timeline_step_key, "check")
        self.assertEqual(display.heading, "补译结果仍需处理")

    def test_retry_manifest_completed_archives_when_idle(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "RESULTS_MERGED",
            "job_name": "batches/retry",
            "retry_of_manifest": "C:/logs/batch_jobs/parent/manifest.json",
        }
        with mock.patch("gui_qt.manifest_resume_summary.resume_workflow") as resume_mock:
            workflow = mock.Mock()
            workflow.current_step.return_value = None
            resume_mock.return_value = workflow
            display = build_manifest_workflow_display(
                spec,
                "C:/logs/batch_jobs/retry/manifest.json",
                manifest,
            )

        self.assertTrue(display.archive_when_idle)
        self.assertEqual(display.status, "done")
        self.assertEqual(display.heading, "补译任务已完成")


if __name__ == "__main__":
    unittest.main()