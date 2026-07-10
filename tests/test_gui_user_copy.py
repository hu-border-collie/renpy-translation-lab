import unittest

import doctor_recommendations as doctor_rec

from gui_qt.user_copy import (
    doctor_mode_label,
    format_bootstrap_fact,
    format_manifest_path_fact,
    job_state_label,
    manifest_mode_label,
    recommendation_requires_attention,
    safety_level_label,
    translate_doctor_warning,
)


class GuiUserCopyTests(unittest.TestCase):
    def test_safety_level_label_maps_known_values(self):
        self.assertEqual(safety_level_label("safe"), "可写回")
        self.assertEqual(safety_level_label("warn"), "需处理")
        self.assertEqual(safety_level_label("block"), "禁止写回")

    def test_safety_level_label_case_insensitive(self):
        self.assertEqual(safety_level_label("Safe"), "可写回")
        self.assertEqual(safety_level_label("WARN"), "需处理")
        self.assertEqual(safety_level_label("Block"), "禁止写回")

    def test_doctor_mode_label_maps_known_values(self):
        self.assertEqual(doctor_mode_label("can_generate_template"), "可生成翻译模板")
        self.assertEqual(doctor_mode_label("existing_tl_only"), "已有翻译模板")
        self.assertEqual(doctor_mode_label("blocked_missing_template"), "缺少模板且无法生成")

    def test_job_state_label_maps_known_values(self):
        self.assertEqual(job_state_label("JOB_STATE_SUCCEEDED"), "已完成")
        self.assertEqual(job_state_label("JOB_STATE_FAILED"), "失败")
        self.assertEqual(job_state_label("JOB_STATE_CANCELLED"), "已取消")
        self.assertEqual(job_state_label("JOB_STATE_EXPIRED"), "已过期")
        self.assertEqual(job_state_label("JOB_STATE_PENDING"), "排队中")
        self.assertEqual(job_state_label("JOB_STATE_RUNNING"), "处理中")

    def test_format_manifest_path_fact_uses_chinese_label(self):
        self.assertEqual(
            format_manifest_path_fact(r"C:\jobs\manifest.json"),
            r"任务记录：C:\jobs\manifest.json",
        )

    def test_translate_doctor_warning_maps_known_message(self):
        translated = translate_doctor_warning(
            "old/new line counts differ; string translation blocks may be malformed."
        )
        self.assertIn("界面字符串块", translated)

    def test_manifest_mode_label_falls_back_for_empty_and_unknown(self):
        self.assertEqual(manifest_mode_label(""), "未知")
        self.assertEqual(manifest_mode_label("custom_mode"), "custom_mode")

    def test_format_bootstrap_fact_passthrough_unknown_key(self):
        self.assertEqual(format_bootstrap_fact("unknown_key", "42"), "unknown_key：42")  # noqa: RUF001

    def test_optional_doctor_recommendations_do_not_require_attention(self):
        optional_codes = [
            doctor_rec.BOOTSTRAP_RAG_OR_WARM_ON_BUILD,
            doctor_rec.ENABLE_RAG_FOR_CONSISTENCY,
            doctor_rec.ENABLE_SOURCE_INDEX_FOR_NEW_PROJECT,
        ]

        for code in optional_codes:
            with self.subTest(code=code):
                self.assertFalse(recommendation_requires_attention([code]))
        self.assertTrue(
            recommendation_requires_attention([doctor_rec.BOOTSTRAP_SOURCE_INDEX])
        )

    def test_ready_doctor_recommendation_codes_do_not_require_attention(self):
        ready_codes = [
            doctor_rec.START_INCREMENTAL_BATCH,
            doctor_rec.START_PENDING_BATCH,
            doctor_rec.SUBSTANTIALLY_COMPLETE,
            doctor_rec.NO_PENDING_LINES,
        ]
        for code in ready_codes:
            with self.subTest(code=code):
                self.assertFalse(recommendation_requires_attention([code]))

    def test_workflow_state_messages_match_primary_recommendation_copy(self):
        from gui_qt.user_copy import (
            DOCTOR_RECOMMENDATION_PRIMARY_MESSAGES,
            DOCTOR_WORKFLOW_STATE_MESSAGES,
        )

        for code, message in DOCTOR_WORKFLOW_STATE_MESSAGES.items():
            with self.subTest(code=code):
                self.assertEqual(message, DOCTOR_RECOMMENDATION_PRIMARY_MESSAGES[code])


if __name__ == "__main__":
    unittest.main()
