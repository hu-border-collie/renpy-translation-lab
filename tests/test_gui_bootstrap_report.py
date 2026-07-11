import unittest
from unittest import mock

from gui_qt.bootstrap_report import (
    BootstrapProgressState,
    BootstrapProgressTracker,
    coerce_bool,
    create_bootstrap_progress_state,
    create_bootstrap_progress_tracker,
    format_bootstrap_progress_bar_label,
    format_bootstrap_progress_facts,
    format_remaining_duration_zh,
    idle_bootstrap_summary,
    read_batch_context_flags,
    summarize_rag_bootstrap_output,
    summarize_source_index_bootstrap_output,
    update_bootstrap_progress_from_line,
)


class GuiBootstrapReportTests(unittest.TestCase):
    def test_coerce_bool_accepts_common_literals(self):
        self.assertTrue(coerce_bool("true", False))
        self.assertTrue(coerce_bool("ON", False))
        self.assertFalse(coerce_bool("off", True))
        self.assertTrue(coerce_bool(True, False))
        self.assertTrue(coerce_bool(1, False))
        self.assertFalse(coerce_bool(0, True))

    def test_read_batch_context_flags_defaults(self):
        flags = read_batch_context_flags({})
        self.assertFalse(flags["rag_enabled"])
        self.assertFalse(flags["source_index_enabled"])
        self.assertTrue(flags["bootstrap_on_build"])

    def test_read_batch_context_flags_reads_nested_config(self):
        flags = read_batch_context_flags(
            {
                "batch": {
                    "rag": {"enabled": True, "bootstrap_on_build": False},
                    "source_index": {"enabled": True},
                }
            }
        )
        self.assertTrue(flags["rag_enabled"])
        self.assertTrue(flags["source_index_enabled"])
        self.assertFalse(flags["bootstrap_on_build"])

    def test_read_batch_context_flags_propagates_resolver_failures(self):
        with mock.patch(
            "project_context_settings.resolve_batch_context_flags",
            side_effect=PermissionError("denied"),
        ):
            with self.assertRaises(PermissionError):
                read_batch_context_flags({})

    def test_summarize_rag_bootstrap_output_marks_disabled(self):
        summary = summarize_rag_bootstrap_output(
            "RAG is disabled. Enable batch.rag.enabled=true before bootstrapping.\n",
            0,
        )
        self.assertEqual(summary.status, "warning")
        self.assertEqual(summary.kind, "rag")

    def test_summarize_rag_bootstrap_output_marks_success(self):
        output = """
RAG bootstrap summary:
- store_dir: logs/rag_store/demo
- files_scanned: 2
- scanned: 5
- embedded: 3
- upserted: 3
- history_records_before: 0
- history_records_after: 3
"""
        summary = summarize_rag_bootstrap_output(output, 0)
        self.assertEqual(summary.status, "ready")
        self.assertIn("存储目录：logs/rag_store/demo", summary.facts)

    def test_summarize_rag_bootstrap_output_marks_empty_scan_as_warning(self):
        output = """
RAG bootstrap summary:
- files_scanned: 0
- scanned: 0
- upserted: 0
"""
        summary = summarize_rag_bootstrap_output(output, 0)
        self.assertEqual(summary.status, "warning")

    def test_summarize_source_index_bootstrap_output_marks_failure(self):
        summary = summarize_source_index_bootstrap_output("TL dir does not exist", 1)
        self.assertEqual(summary.status, "failed")

    def test_summarize_source_index_bootstrap_output_marks_success(self):
        output = """
Source Index bootstrap final summary:
- store_dir: logs/source_index_store/demo
- files_scanned: 1
- scanned: 4
- embedded: 2
- upserted: 2
- history_records_after: 4
"""
        summary = summarize_source_index_bootstrap_output(output, 0)
        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.kind, "source_index")

    def test_idle_bootstrap_summary_is_idle(self):
        summary = idle_bootstrap_summary()
        self.assertEqual(summary.status, "idle")

    def test_update_bootstrap_progress_from_pre_run_and_embedding_lines(self):
        state = create_bootstrap_progress_state("source_index")
        state = update_bootstrap_progress_from_line(
            "- Total segments scanned from files: 28886",
            state,
        )
        state = update_bootstrap_progress_from_line(
            "- Unchanged segments (reusing embeddings): 9409",
            state,
        )
        state = update_bootstrap_progress_from_line(
            "- New/updated segments (need embeddings): 19477",
            state,
        )
        state = update_bootstrap_progress_from_line(
            "Reused embeddings written: 9409.",
            state,
        )
        state = update_bootstrap_progress_from_line(
            "Source index embedding progress: 64/19477 scanned, 64 embedded, 9473 stored.",
            state,
        )

        self.assertEqual(state.total_segments, 28886)
        self.assertEqual(state.reused_embeddings, 9409)
        self.assertEqual(state.embedding_total, 19477)
        self.assertEqual(state.embedding_done, 64)
        self.assertEqual(state.stored_segments, 9473)

        facts = format_bootstrap_progress_facts(state)
        self.assertIn("入库进度：9473/28886 片段（32%）", facts)
        self.assertIn("本轮向量生成：64/19477", facts)

    def test_update_bootstrap_progress_uses_embedded_count_not_scanned(self):
        state = create_bootstrap_progress_state("source_index")
        state = update_bootstrap_progress_from_line(
            "- Total segments scanned from files: 100",
            state,
        )
        state = update_bootstrap_progress_from_line(
            "- New/updated segments (need embeddings): 100",
            state,
        )
        state = update_bootstrap_progress_from_line(
            "Source index embedding progress: 80/100 scanned, 64 embedded, 50 stored.",
            state,
        )

        self.assertEqual(state.embedding_done, 64)

    def test_format_remaining_duration_zh(self):
        self.assertEqual(format_remaining_duration_zh(0), "即将完成")
        self.assertEqual(format_remaining_duration_zh(45), "约剩 45 秒")
        self.assertEqual(format_remaining_duration_zh(125), "约剩 2 分 5 秒")
        self.assertEqual(format_remaining_duration_zh(3720), "约剩 1 小时 2 分")

    def test_bootstrap_progress_tracker_waits_for_second_sample_before_rate(self):
        tracker = create_bootstrap_progress_tracker()
        state = BootstrapProgressState(
            kind="source_index",
            total_segments=100,
            stored_segments=20,
        )
        tracker.observe(state, now=10.0)
        self.assertIsNone(tracker.seconds_per_segment)
        # Before a second sample, ETA may still use the elapsed/stored fallback rate.
        self.assertEqual(tracker.estimate_remaining_seconds(state, now=15.0), 20)

        state = BootstrapProgressState(
            kind="source_index",
            total_segments=100,
            stored_segments=36,
        )
        tracker.observe(state, now=20.0)
        self.assertIsNotNone(tracker.seconds_per_segment)

    def test_bootstrap_progress_tracker_estimates_remaining_seconds(self):
        tracker = create_bootstrap_progress_tracker()
        state = BootstrapProgressState(
            kind="source_index",
            total_segments=100,
            stored_segments=20,
        )
        tracker.observe(state, now=10.0)
        self.assertEqual(tracker.estimate_remaining_seconds(state, now=20.0), 40)

        state = BootstrapProgressState(
            kind="source_index",
            total_segments=100,
            stored_segments=36,
        )
        tracker.observe(state, now=20.0)
        self.assertEqual(tracker.estimate_remaining_seconds(state, now=20.0), 40)
        self.assertEqual(tracker.estimate_remaining_seconds(state, now=25.0), 35)

    def test_bootstrap_progress_tracker_accepts_zero_timestamp(self):
        tracker = create_bootstrap_progress_tracker()
        state = BootstrapProgressState(
            kind="source_index",
            total_segments=100,
            stored_segments=20,
        )
        tracker.observe(state, now=0.0)

        state = BootstrapProgressState(
            kind="source_index",
            total_segments=100,
            stored_segments=40,
        )
        tracker.observe(state, now=10.0)

        self.assertEqual(tracker.estimate_remaining_seconds(state, now=10.0), 30)

    def test_format_bootstrap_progress_bar_label_includes_eta(self):
        state = BootstrapProgressState(
            kind="source_index",
            total_segments=28886,
            stored_segments=9473,
        )
        label = format_bootstrap_progress_bar_label(state, 4525)
        self.assertIn("32%（9473/28886）", label)
        self.assertIn("约剩 1 小时 15 分", label)


if __name__ == "__main__":
    unittest.main()