import unittest

from gui_qt.keyword_workflow import KeywordBatchWorkflow
from gui_qt.sync_keyword_workflow import SyncKeywordWorkflow
from gui_qt.revision_workflow import RevisionBatchWorkflow
from gui_qt.sync_revision_workflow import SyncRevisionWorkflow
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

    def test_create_workflow_returns_keyword_batch_workflow(self):
        workflow = create_workflow(WorkMode.KEYWORD_EXTRACTION)

        self.assertIsInstance(workflow, KeywordBatchWorkflow)
        self.assertEqual(workflow.current_step().args, ["build-keywords"])

    def test_create_workflow_returns_sync_keyword_workflow(self):
        workflow = create_workflow(WorkMode.SYNC_KEYWORD_EXTRACTION)

        self.assertIsInstance(workflow, SyncKeywordWorkflow)
        self.assertEqual(workflow.current_step().args, ["sync-keywords"])

    def test_create_workflow_returns_revision_workflow(self):
        workflow = create_workflow(WorkMode.REVISION)

        self.assertIsInstance(workflow, RevisionBatchWorkflow)
        self.assertEqual(workflow.current_step().args, ["build-revisions"])

    def test_create_workflow_returns_sync_revision_workflow(self):
        workflow = create_workflow(WorkMode.SYNC_REVISION)

        self.assertIsInstance(workflow, SyncRevisionWorkflow)
        self.assertEqual(workflow.current_step().args, ["sync-revisions"])

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

    def test_resume_workflow_returns_revision_resume(self):
        workflow = resume_workflow(
            WorkMode.REVISION,
            "C:\\package\\manifest.json",
            {"job_name": "batches/example", "mode": "revision"},
        )

        self.assertIsInstance(workflow, RevisionBatchWorkflow)
        self.assertEqual(
            workflow.current_step().args,
            ["status", "C:\\package\\manifest.json"],
        )

    def test_resume_workflow_returns_none_for_sync_revision(self):
        self.assertIsNone(
            resume_workflow(
                WorkMode.SYNC_REVISION,
                "C:\\package\\manifest.json",
                {"mode": "revision"},
            )
        )

    def test_resume_workflow_returns_keyword_resume(self):
        workflow = resume_workflow(
            WorkMode.KEYWORD_EXTRACTION,
            "C:\\package\\manifest.json",
            {"job_name": "batches/example"},
        )

        self.assertIsInstance(workflow, KeywordBatchWorkflow)
        self.assertEqual(
            workflow.current_step().args,
            ["status", "C:\\package\\manifest.json"],
        )

    def test_validate_resume_manifest_accepts_keyword_extraction_mode(self):
        validate_resume_manifest(
            WorkMode.KEYWORD_EXTRACTION,
            {"mode": "keyword_extraction", "base_dir": "C:\\work"},
            game_root="C:\\work",
            normalized_path_text=lambda path: str(path),
        )


if __name__ == "__main__":
    unittest.main()