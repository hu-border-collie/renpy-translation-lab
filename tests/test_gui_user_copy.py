import unittest

from gui_qt.user_copy import (
    doctor_mode_label,
    format_bootstrap_fact,
    format_manifest_path_fact,
    job_state_label,
    manifest_mode_label,
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
        self.assertEqual(doctor_mode_label("existing_tl_only"), "仅处理已有翻译文件")
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


if __name__ == "__main__":
    unittest.main()