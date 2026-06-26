import unittest

from gui_qt.bootstrap_report import (
    coerce_bool,
    idle_bootstrap_summary,
    read_batch_context_flags,
    summarize_rag_bootstrap_output,
    summarize_source_index_bootstrap_output,
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


if __name__ == "__main__":
    unittest.main()