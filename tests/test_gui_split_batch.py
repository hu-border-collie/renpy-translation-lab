import unittest

from gui_qt.split_batch import entry_from_manifest, summarize_split_entries
from gui_qt.split_batch_workflow import SplitBatchQueueWorkflow


class GuiSplitBatchTests(unittest.TestCase):
    def test_split_entries_have_expected_status_and_actions(self):
        entries = [
            entry_from_manifest(
                r"C:\pkg\part01_of_03\manifest.json",
                {
                    "split_index": 1,
                    "split_total": 3,
                    "job_name": "batches/one",
                    "job_state": "JOB_STATE_RUNNING",
                    "summary": {"item_count": 10, "chunk_count": 2},
                },
            ),
            entry_from_manifest(
                r"C:\pkg\part02_of_03\manifest.json",
                {
                    "split_index": 2,
                    "split_total": 3,
                    "job_name": "batches/two",
                    "job_state": "JOB_STATE_SUCCEEDED",
                    "summary": {"item_count": 20, "chunk_count": 4},
                },
            ),
            entry_from_manifest(
                r"C:\pkg\part03_of_03\manifest.json",
                {
                    "split_index": 3,
                    "split_total": 3,
                    "job_name": "",
                    "job_state": "LOCAL_ONLY",
                    "summary": {"item_count": 30, "chunk_count": 6},
                },
            ),
        ]

        self.assertEqual([entry.part_label for entry in entries], ["part01/03", "part02/03", "part03/03"])
        self.assertFalse(entries[0].selectable)
        self.assertEqual(entries[0].status_kind, "running")
        self.assertTrue(entries[1].selectable)
        self.assertEqual(entries[1].status_kind, "succeeded")
        self.assertTrue(entries[2].needs_submit)
        self.assertEqual(entries[2].status_kind, "unsubmitted")
        self.assertIn("未提交 1 个", "\n".join(summarize_split_entries(entries)))

    def test_checked_and_applied_entries_have_expected_actions(self):
        safe_entry = entry_from_manifest(
            r"C:\pkg\part01_of_02\manifest.json",
            {
                "split_index": 1,
                "split_total": 2,
                "job_name": "batches/one",
                "job_state": "JOB_STATE_SUCCEEDED",
                "last_check_summary": {"safety_level": "safe"},
            },
        )
        applied_entry = entry_from_manifest(
            r"C:\pkg\part02_of_02\manifest.json",
            {
                "split_index": 2,
                "split_total": 2,
                "job_name": "batches/two",
                "job_state": "JOB_STATE_SUCCEEDED",
                "applied_at": "2026-06-30T12:00:00",
            },
        )

        self.assertEqual(safe_entry.status_label, "检查：可写回")
        self.assertEqual(safe_entry.status_kind, "checked_safe")
        self.assertTrue(safe_entry.selectable)
        self.assertEqual(applied_entry.status_label, "已写回")
        self.assertEqual(applied_entry.status_kind, "applied")
        self.assertFalse(applied_entry.selectable)

    def test_submit_remaining_queue_only_uses_unsubmitted_entries(self):
        part1 = r"C:\pkg\part01_of_02\manifest.json"
        part2 = r"C:\pkg\part02_of_02\manifest.json"
        entries = [
            entry_from_manifest(part1, {"split_index": 1, "split_total": 2, "job_name": "batches/one"}),
            entry_from_manifest(part2, {"split_index": 2, "split_total": 2, "job_name": ""}),
        ]

        workflow = SplitBatchQueueWorkflow.submit_remaining(entries, anchor_manifest_path=part1)

        self.assertEqual(workflow.current_step().args, ["submit", part2])
        update = workflow.complete_current_step(0, "Manifest: ignored\n")
        self.assertEqual(update.status, "waiting")
        self.assertEqual(update.timeline_step_key, "status")
        self.assertIsNone(workflow.current_step())
        self.assertEqual(workflow.restore_latest_manifest_path, part1)

    def test_refresh_status_queue_skips_unsubmitted_and_applied_entries(self):
        part1 = r"C:\pkg\part01_of_03\manifest.json"
        part2 = r"C:\pkg\part02_of_03\manifest.json"
        part3 = r"C:\pkg\part03_of_03\manifest.json"
        entries = [
            entry_from_manifest(part1, {"split_index": 1, "split_total": 3, "job_name": "batches/one"}),
            entry_from_manifest(part2, {"split_index": 2, "split_total": 3, "job_name": "batches/two", "applied_at": "x"}),
            entry_from_manifest(part3, {"split_index": 3, "split_total": 3, "job_name": ""}),
        ]

        workflow = SplitBatchQueueWorkflow.refresh_status(entries, anchor_manifest_path=part1)

        self.assertEqual(workflow.current_step().args, ["status", part1])
        update = workflow.complete_current_step(0, "State: JOB_STATE_RUNNING\n")
        self.assertEqual(update.status, "ready")
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()
