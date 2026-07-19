import unittest

from gui_qt.sync_translation_workflow import SyncTranslationWorkflow


class GuiSyncTranslationWorkflowTests(unittest.TestCase):
    def test_start_workflow_runs_sync_script(self):
        workflow = SyncTranslationWorkflow.start_new()

        step = workflow.current_step()

        self.assertEqual(step.script_basename, "gemini_translate.py")
        self.assertEqual(step.args, [])
        self.assertEqual(step.key, "preview")

    def test_successful_run_finishes(self):
        workflow = SyncTranslationWorkflow.start_new()
        output = (
            "Found 1 files.\nProcessing: a.rpy\n"
            "  Found 1 lines to translate.\n"
            "  Translated 1/1 items. (Received 8 chars of translation)\n"
            "  Previewed a.rpy.\n"
            "Sync preview manifest: C:/run/manifest.json\n"
            "Sync preview report: C:/run/preview.diff\n"
            "Preview files: 1\n"
            "Preview translations: 1\n"
            "Preview status: safe\n"
        )

        update = workflow.complete_current_step(0, output)

        self.assertEqual(update.status, "done")
        self.assertIn("预览", update.heading)
        self.assertEqual(workflow.manifest_path, "C:/run/manifest.json")
        self.assertIsNone(workflow.current_step())

    def test_apply_workflow_uses_explicit_manifest(self):
        workflow = SyncTranslationWorkflow.apply_existing("C:/run/manifest.json")

        step = workflow.current_step()
        self.assertEqual(step.key, "apply")
        self.assertEqual(step.args, ["--apply", "C:/run/manifest.json"])

        update = workflow.complete_current_step(
            0,
            "Sync apply manifest: C:/run/manifest.json\n"
            "Applied files: 1\n"
            "Sync translation apply complete.\n",
        )
        self.assertEqual(update.status, "done")
        self.assertIn("写回", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_empty_pending_steps_means_no_current_step(self):
        workflow = SyncTranslationWorkflow(pending_steps=[])

        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()
