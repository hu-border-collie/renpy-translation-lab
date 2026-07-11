import unittest

from gui_qt.work_modes import (
    TASK_CATEGORY_ORDER,
    TaskCategory,
    WorkMode,
    WORK_MODE_SPECS,
    bootstrap_disabled_message,
    default_work_mode_for_category,
    normalize_work_mode,
    task_category_for_work_mode,
    work_mode_from_manifest_mode,
    work_mode_hint_texts,
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
        self.assertIn(WorkMode.SYNC_KEYWORD_EXTRACTION, modes)
        self.assertIn(WorkMode.BOOTSTRAP_RAG, modes)
        self.assertIn(WorkMode.BOOTSTRAP_SOURCE_INDEX, modes)
        keyword_index = modes.index(WorkMode.KEYWORD_EXTRACTION)
        sync_keyword_index = modes.index(WorkMode.SYNC_KEYWORD_EXTRACTION)
        self.assertLess(keyword_index, sync_keyword_index)

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

    def test_sync_translation_is_implemented(self):
        self.assertTrue(work_mode_spec(WorkMode.SYNC_TRANSLATION).implemented)

    def test_keyword_modes_are_implemented(self):
        self.assertTrue(work_mode_spec(WorkMode.KEYWORD_EXTRACTION).implemented)
        self.assertTrue(work_mode_spec(WorkMode.SYNC_KEYWORD_EXTRACTION).implemented)

    def test_maintenance_category_contains_revision_modes(self):
        modes = work_modes_for_category(TaskCategory.MAINTENANCE)
        self.assertEqual(modes, (WorkMode.REVISION, WorkMode.SYNC_REVISION))

    def test_revision_modes_are_implemented(self):
        for mode in (WorkMode.REVISION, WorkMode.SYNC_REVISION):
            with self.subTest(mode=mode):
                spec = work_mode_spec(mode)
                self.assertTrue(spec.implemented)
                self.assertFalse(spec.supports_translation_writeback)

    def test_revision_supports_resume_but_sync_revision_does_not(self):
        self.assertTrue(work_mode_spec(WorkMode.REVISION).supports_resume)
        self.assertFalse(work_mode_spec(WorkMode.SYNC_REVISION).supports_resume)

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

    def test_work_mode_hint_texts_include_idle_and_bootstrap_messages(self):
        texts = work_mode_hint_texts()
        for spec in WORK_MODE_SPECS.values():
            if spec.idle_workflow_message.strip():
                self.assertIn(spec.idle_workflow_message.strip(), texts)
        self.assertEqual(
            bootstrap_disabled_message("rag"),
            "请先在「设置 · 上下文」勾选「启用 RAG 记忆库（批量，当前项目）」，并点击「保存设置」。",
        )
        self.assertEqual(
            bootstrap_disabled_message("source_index"),
            "请先在「设置 · 上下文」勾选「启用原文索引（当前项目）」，并点击「保存设置」。",
        )
        self.assertIn(bootstrap_disabled_message("rag"), texts)
        self.assertIn(bootstrap_disabled_message("source_index"), texts)


if __name__ == "__main__":
    unittest.main()