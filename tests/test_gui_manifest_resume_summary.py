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
            "summary": {"file_count": 17, "chunk_count": 285, "item_count": 11015},
        }
        with mock.patch("gui_qt.manifest_resume_summary.resume_workflow") as resume_mock:
            workflow = mock.Mock()
            workflow.current_step.return_value = None
            resume_mock.return_value = workflow
            display = build_manifest_workflow_display(
                spec,
                "C:/logs/batch_jobs/20260615_Game_GloryHounds/manifest.json",
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


if __name__ == "__main__":
    unittest.main()