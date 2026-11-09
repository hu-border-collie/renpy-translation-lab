"""Saved translation context display without providers or game scans."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from foundation.project_asset_paths import canonical_abs_path
from foundation.project_context_settings import save_project_context_settings
from gui_qt.reports.translation_preprocessing import summarize_translation_preprocessing


class TranslationPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "glossary.json").write_text("{}", encoding="utf-8")
        (self.root / "macro_setting.md").write_text("Use a calm tone.", encoding="utf-8")

    def summary(self, config, strategy="gemini_batch"):
        return summarize_translation_preprocessing(config, game_root=str(self.root), strategy=strategy)

    def test_execution_scope_and_asset_paths_follow_saved_settings(self):
        config = {
            "batch": {"source_index": {"enabled": True}, "macro_setting_file": "macro_setting.md"},
            "sync": {"rag": {"enabled": True}, "story_memory": {"enabled": True}},
        }
        original = copy.deepcopy(config)
        batch = self.summary(config)
        sync = self.summary(config, "sync")
        self.assertEqual(batch.text.splitlines()[0], "已开启：原文索引")
        self.assertEqual(sync.text.splitlines()[0], "已开启：历史译文检索（RAG）、剧情记忆")
        self.assertIn("术语表：已找到文件（按命中使用）", batch.text)
        self.assertIn("风格设定：已找到文件", batch.text)
        # Display uses canonical long paths. Windows temp dirs are often 8.3
        # short names (RUNNER~1) until resolution expands them.
        self.assertIn(canonical_abs_path(self.root / "glossary.json"), batch.details)
        self.assertIn(canonical_abs_path(self.root / "macro_setting.md"), sync.details)
        self.assertIn("开启不代表已完成预建或本次已命中", batch.details)
        self.assertEqual(config, original)

    def test_project_overrides_win_without_changing_execution_scope(self):
        save_project_context_settings(self.root, {
            "rag_enabled": False, "source_index_enabled": True,
            "project_analysis_enabled": True, "project_analysis_inject_enabled": False,
            "sync_source_index_enabled": False, "sync_project_analysis_inject_enabled": True,
        })
        config = {
            "batch": {"rag": {"enabled": True}, "project_analysis": {"inject_published_brief": True}},
            "sync": {"source_index": {"enabled": True}},
        }
        batch = self.summary(config)
        sync = self.summary(config, "sync")
        self.assertEqual(batch.text.splitlines()[0], "已开启：原文索引")
        self.assertIn("未开启：历史译文检索（RAG）、剧情记忆、项目摘要注入", batch.text)
        self.assertIn("项目分析预建：已开启", batch.details)
        self.assertEqual(sync.text.splitlines()[0], "已开启：项目摘要注入")

    def test_mv_unsupported_flags_do_not_appear_as_enabled_features(self):
        config = {
            "engine": "rpgmaker_mv",
            "batch": {
                "rag": {"enabled": True}, "story_memory": {"enabled": True},
                "project_analysis": {"inject_published_brief": True},
            },
            "sync": {"story_memory": {"enabled": True}},
        }
        for strategy in ("gemini_batch", "sync"):
            with self.subTest(strategy=strategy):
                text = self.summary(config, strategy).text
                self.assertNotIn("剧情记忆", text.splitlines()[0])
                self.assertNotIn("项目摘要注入", text.splitlines()[0])
                self.assertIn("MV 暂不支持：剧情记忆（配置已开启，需关闭）", text)
                self.assertIn("项目摘要注入", text.splitlines()[2])

    def test_false_strings_and_missing_assets_are_not_reported_as_active(self):
        config = {
            "glossary_file": "absent.json",
            "batch": {
                "macro_setting_file": "absent.md",
                "rag": {"enabled": "false"}, "source_index": {"enabled": "off"},
                "story_memory": {"enabled": "0"},
            },
        }
        text = self.summary(config).text
        self.assertIn("已开启：无可选增强", text)
        self.assertIn("术语表：未找到文件", text)
        self.assertIn("风格设定：未找到文件", text)

    def test_batch_inline_macro_and_unconfigured_macro_are_distinct(self):
        self.assertIn("风格设定：未配置", self.summary({}).text)
        self.assertIn("风格设定：已配置文本", self.summary({"batch": {"macro_setting": "Style guide"}}).text)
        self.assertIn("风格设定：已配置文本", self.summary({"batch_macro_setting": "Legacy guide"}).text)
        # Sync only consumes its file, including the default file.
        self.assertIn("风格设定：已找到文件", self.summary({"sync": {"macro_setting": "Ignored"}}, "sync").text)

    def test_missing_project_or_unreadable_config_keeps_status_unknown(self):
        self.assertIn("选择项目后", summarize_translation_preprocessing(
            {}, game_root="", strategy="sync").text)
        self.assertIn("无法读取", self.summary(None).text)
        self.assertIn("无法读取", self.summary({}, "invalid").text)
        self.assertIn("已开启：无可选增强", self.summary({"batch": [], "sync": None}).text)


if __name__ == "__main__":
    unittest.main()
