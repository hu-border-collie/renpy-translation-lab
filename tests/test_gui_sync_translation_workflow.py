import unittest

from gui_qt.sync_translation_workflow import SyncTranslationWorkflow


class GuiSyncTranslationWorkflowTests(unittest.TestCase):
    def test_start_workflow_runs_sync_script(self):
        workflow = SyncTranslationWorkflow.start_new()

        step = workflow.current_step()

        self.assertEqual(step.script_basename, "gemini_translate.py")
        self.assertEqual(step.args, [])

    def test_successful_run_finishes(self):
        workflow = SyncTranslationWorkflow.start_new()
        output = "Found 1 files.\nProcessing: a.rpy\n  Found 1 lines to translate.\n  Done with a.rpy.\n"

        update = workflow.complete_current_step(0, output)

        self.assertEqual(update.status, "done")
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()