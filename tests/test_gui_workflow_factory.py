import unittest

from gui_qt.sync_translation_workflow import SyncTranslationWorkflow
from gui_qt.translation_workflow import TranslationWorkflow
from gui_qt.work_modes import WorkMode
from gui_qt.workflow_factory import (
    create_workflow,
    resume_workflow,
    validate_resume_manifest,
)


class GuiWorkflowFactoryTests(unittest.TestCase):
    def test_create_workflow_returns_batch_translation_workflow(self):
        workflow = create_workflow(WorkMode.BATCH_TRANSLATION)

        self.assertIsInstance(workflow, TranslationWorkflow)
        self.assertEqual(workflow.current_step().args, ["build"])

    def test_create_workflow_returns_sync_translation_workflow(self):
        workflow = create_workflow(WorkMode.SYNC_TRANSLATION)

        self.assertIsInstance(workflow, SyncTranslationWorkflow)
        self.assertEqual(workflow.current_step().script_basename, "gemini_translate.py")

    def test_create_workflow_returns_none_for_unimplemented_modes(self):
        self.assertIsNone(create_workflow(WorkMode.KEYWORD_EXTRACTION))

    def test_validate_resume_manifest_rejects_mode_mismatch(self):
        with self.assertRaisesRegex(ValueError, "不是Batch 翻译任务"):
            validate_resume_manifest(
                WorkMode.BATCH_TRANSLATION,
                {"mode": "revision", "base_dir": "C:\\work"},
                game_root=None,
                normalized_path_text=lambda path: str(path),
            )

    def test_validate_resume_manifest_accepts_revision_for_revision_mode(self):
        validate_resume_manifest(
            WorkMode.REVISION,
            {"mode": "revision", "base_dir": "C:\\work"},
            game_root="C:\\work",
            normalized_path_text=lambda path: str(path),
        )

    def test_resume_workflow_returns_translation_resume(self):
        workflow = resume_workflow(
            WorkMode.BATCH_TRANSLATION,
            "C:\\package\\manifest.json",
            {"job_name": "batches/example"},
        )

        self.assertIsInstance(workflow, TranslationWorkflow)
        self.assertEqual(
            workflow.current_step().args,
            ["status", "C:\\package\\manifest.json"],
        )


if __name__ == "__main__":
    unittest.main()