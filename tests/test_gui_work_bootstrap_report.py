import unittest

from gui_qt.work_bootstrap_report import (
    summarize_work_bootstrap_output,
    work_bootstrap_to_doctor_summary,
)


BOOTSTRAP_OUTPUT = """
Work bootstrap summary:
- status: created
- project_root: C:\\Games\\Example
- work_dir: C:\\Games\\Example\\work
- source_game_dir: C:\\Games\\Example\\original\\game
- files_copied: 42
- game_root_updated: True
- message: Copied 42 files from original/game into work/game.
"""


class GuiWorkBootstrapReportTests(unittest.TestCase):
    def test_summarize_created_bootstrap_output(self):
        summary = summarize_work_bootstrap_output(BOOTSTRAP_OUTPUT, exit_code=0)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.work_dir, "C:\\Games\\Example\\work")
        self.assertTrue(summary.game_root_updated)
        self.assertTrue(any("复制文件数：42" in fact for fact in summary.facts))

    def test_summarize_skipped_bootstrap_output(self):
        output = BOOTSTRAP_OUTPUT.replace("status: created", "status: skipped").replace(
            "Copied 42 files",
            "work directory already exists and is not empty",
        )

        summary = summarize_work_bootstrap_output(output, exit_code=0)

        self.assertEqual(summary.status, "warning")
        self.assertIn("非空", summary.message)
        self.assertTrue(any(fact.startswith("注意：") for fact in summary.facts))
        self.assertFalse(any(fact.startswith("- ") for fact in summary.facts))

    def test_work_bootstrap_to_doctor_summary_preserves_heading(self):
        summary = summarize_work_bootstrap_output(BOOTSTRAP_OUTPUT, exit_code=0)
        doctor_summary = work_bootstrap_to_doctor_summary(summary)

        self.assertEqual(doctor_summary.heading, summary.heading)
        self.assertEqual(doctor_summary.facts, summary.facts)


if __name__ == "__main__":
    unittest.main()