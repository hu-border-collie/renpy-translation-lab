import unittest

from gui_qt.work_modes import (
    TASK_CATEGORY_ORDER,
    TaskCategory,
    WorkMode,
    default_work_mode_for_category,
    normalize_work_mode,
    task_category_for_work_mode,
    work_mode_from_manifest_mode,
    work_mode_spec,
    work_modes_for_category,
)


class GuiWorkModesTests(unittest.TestCase):
    def test_task_category_order_starts_with_translation(self):
        self.assertEqual(TASK_CATEGORY_ORDER[0], TaskCategory.TRANSLATION)

    def test_translation_category_contains_batch_and_sync(self):
        modes = work_modes_for_category(TaskCategory.TRANSLATION)
        self.assertEqual(modes, (WorkMode.BATCH_TRANSLATION, WorkMode.SYNC_TRANSLATION))

    def test_analysis_prep_category_contains_keywords_and_bootstrap(self):
        modes = work_modes_for_category(TaskCategory.ANALYSIS_PREP)
        self.assertIn(WorkMode.KEYWORD_EXTRACTION, modes)
        self.assertIn(WorkMode.BOOTSTRAP_RAG, modes)
        self.assertIn(WorkMode.BOOTSTRAP_SOURCE_INDEX, modes)

    def test_batch_translation_is_implemented(self):
        spec = work_mode_spec(WorkMode.BATCH_TRANSLATION)
        self.assertTrue(spec.implemented)
        self.assertTrue(spec.supports_translation_writeback)

    def test_bootstrap_tasks_are_implemented(self):
        for mode in (WorkMode.BOOTSTRAP_RAG, WorkMode.BOOTSTRAP_SOURCE_INDEX):
            with self.subTest(mode=mode):
                spec = work_mode_spec(mode)
                self.assertTrue(spec.implemented)
                self.assertTrue(spec.is_bootstrap)

    def test_follow_up_workflow_modes_are_not_implemented_yet(self):
        for mode in (
            WorkMode.SYNC_TRANSLATION,
            WorkMode.KEYWORD_EXTRACTION,
            WorkMode.REVISION,
        ):
            with self.subTest(mode=mode):
                self.assertFalse(work_mode_spec(mode).implemented)

    def test_default_work_mode_for_category(self):
        self.assertEqual(
            default_work_mode_for_category(TaskCategory.TRANSLATION),
            WorkMode.BATCH_TRANSLATION,
        )
        self.assertEqual(
            default_work_mode_for_category(TaskCategory.ANALYSIS_PREP),
            WorkMode.BOOTSTRAP_RAG,
        )

    def test_task_category_for_work_mode(self):
        self.assertEqual(
            task_category_for_work_mode(WorkMode.BOOTSTRAP_RAG),
            TaskCategory.ANALYSIS_PREP,
        )

    def test_normalize_work_mode_defaults_to_batch_translation(self):
        self.assertEqual(normalize_work_mode(None), WorkMode.BATCH_TRANSLATION)
        self.assertEqual(normalize_work_mode(""), WorkMode.BATCH_TRANSLATION)

    def test_work_mode_from_manifest_mode_maps_known_modes(self):
        self.assertEqual(work_mode_from_manifest_mode("translation"), WorkMode.BATCH_TRANSLATION)
        self.assertEqual(work_mode_from_manifest_mode("revision"), WorkMode.REVISION)
        self.assertEqual(
            work_mode_from_manifest_mode("keyword_extraction"),
            WorkMode.KEYWORD_EXTRACTION,
        )
        self.assertIsNone(work_mode_from_manifest_mode("unknown"))


if __name__ == "__main__":
    unittest.main()