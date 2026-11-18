"""Tests for archived completed manifest display behavior."""
from __future__ import annotations

import unittest

from gui_qt.reports.manifest_resume_summary import build_manifest_workflow_display
from gui_qt.work_modes import WORK_MODE_SPECS, WorkMode


class ManifestResumeSummaryTests(unittest.TestCase):
    def test_sync_keyword_pending_denominator_survives_reload_and_processing_failure(self):
        display = build_manifest_workflow_display(
            WORK_MODE_SPECS[WorkMode.SYNC_KEYWORD_EXTRACTION], 'C:/package/manifest.json', {
                'job_state': 'SYNC_PARTIAL',
                'sync_summary': {
                    'request_count': 768, 'successful_request_count': 767,
                    'failed_request_count': 1, 'checkpointed_request_count': 768,
                    'contract_final_complete_chunks': 766, 'contract_partial_requests': 1,
                    'restored_request_count': 655, 'pending_request_count': 113,
                    'processed_pending_request_count': 113,
                },
            })
        self.assertEqual((display.progress.current, display.progress.total), (113, 113))
        self.assertIn('本轮待补已处理 113/113', display.progress.label)
        self.assertEqual(display.status, 'warning')
        self.assertIn('完整有效请求块：766/768（99.7%）', display.facts)
        self.assertTrue(any('已复用有效结果：655' in fact for fact in display.facts))

    def test_sync_keyword_partial_progress_remains_visible_after_reload(self):
        display = build_manifest_workflow_display(
            WORK_MODE_SPECS[WorkMode.SYNC_KEYWORD_EXTRACTION], "C:/package/manifest.json", {
                "job_state": "SYNC_PARTIAL",
                "sync_summary": {
                    "request_count": 768, "successful_request_count": 559,
                    "failed_request_count": 209, "checkpointed_request_count": 768,
                    "contract_final_complete_chunks": 558, "contract_partial_requests": 1,
                },
                "keyword_export": {"summary": {
                    "candidate_count_deduped": 2712, "chunk_summary_count": 476,
                }},
            },
        )
        self.assertEqual(display.status, "warning")
        self.assertIn("部分完成", display.heading)
        self.assertFalse(display.archive_when_idle)
        self.assertEqual((display.progress.current, display.progress.total), (558, 768))
        self.assertTrue(display.progress.visible)
        self.assertIn("已保存处理记录：768/768", display.facts)
        self.assertIn("服务响应：成功 559 个；失败 209 个", display.facts)
        self.assertIn("关键词候选：2,712 个", display.facts)
        self.assertIn("剧情概要：476 条", display.facts)

    def test_completed_sync_keyword_progress_is_not_archived(self):
        display = build_manifest_workflow_display(
            WORK_MODE_SPECS[WorkMode.SYNC_KEYWORD_EXTRACTION], "C:/package/manifest.json", {
                "job_state": "SYNC_COMPLETED",
                "sync_summary": {
                    "request_count": 3, "successful_request_count": 3,
                    "failed_request_count": 0, "checkpointed_request_count": 3,
                    "contract_final_complete_chunks": 3,
                },
            },
        )
        self.assertEqual(display.status, "done")
        self.assertFalse(display.archive_when_idle)
        self.assertEqual((display.progress.current, display.progress.total), (3, 3))

    def test_interrupted_sync_keyword_progress_uses_saved_record_count(self):
        display = build_manifest_workflow_display(
            WORK_MODE_SPECS[WorkMode.SYNC_KEYWORD_EXTRACTION], "C:/package/manifest.json", {
                "job_state": "SYNC_LOCAL", "summary": {"chunk_count": 5},
                "_keyword_saved_request_count": 2,
            },
        )
        self.assertEqual(display.status, "ready")
        self.assertEqual((display.progress.current, display.progress.total), (2, 5))
        self.assertIn("已保存处理记录", display.progress.label)
        self.assertFalse(any("成功" in fact for fact in display.facts))

    def test_completed_translation_archives_when_idle(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "RESULTS_MERGED",
            "job_name": "batches/example",
            "last_check_summary": {"writeback_gate": {"decision": "allow"}},
            "summary": {"file_count": 5, "chunk_count": 42, "item_count": 800},
        }
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
        display = build_manifest_workflow_display(
            spec,
            "C:/logs/batch_jobs/retry/manifest.json",
            manifest,
        )

        self.assertFalse(display.archive_when_idle)
        self.assertEqual(display.status, "stale")
        self.assertEqual(display.timeline_step_key, "check")
        self.assertEqual(display.heading, "补译结果仍需处理")

    def test_resume_summary_shows_unacknowledged_quality_warnings(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "JOB_STATE_SUCCEEDED",
            "job_name": "batches/example",
            "last_check_summary": {
                "quality_gate": {
                    "decision": "needs_review",
                    "warning_count": 3,
                    "blocker_count": 0,
                    "acknowledged_count": 1,
                }
            },
        }
        display = build_manifest_workflow_display(
            spec,
            "C:/logs/batch_jobs/current/manifest.json",
            manifest,
        )

        quality_facts = [fact for fact in display.facts if fact.startswith("质量检查：")]
        self.assertTrue(quality_facts)
        self.assertIn("未确认 2", quality_facts[0])

    def test_merged_retry_manifest_stays_active_until_rechecked(self):
        spec = WORK_MODE_SPECS[WorkMode.BATCH_TRANSLATION]
        manifest = {
            "job_state": "RESULTS_MERGED",
            "job_name": "batches/retry",
            "retry_of_manifest": "C:/logs/batch_jobs/parent/manifest.json",
        }
        display = build_manifest_workflow_display(
            spec,
            "C:/logs/batch_jobs/retry/manifest.json",
            manifest,
        )

        self.assertFalse(display.archive_when_idle)
        self.assertEqual(display.status, "ready")
        self.assertEqual(display.timeline_step_key, "check")


if __name__ == "__main__":
    unittest.main()
