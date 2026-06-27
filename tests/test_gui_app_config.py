import unittest
from pathlib import Path

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    # Skip when PySide6 is missing or Qt system libraries are unavailable (e.g. headless CI).
    MainWindow = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiAppConfigHelperTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow.__new__(MainWindow)

    def _fake_timeline(self):
        class FakeTimeline:
            def __init__(self):
                self.steps = [
                    ("status", "云端执行"),
                    ("download", "下载结果"),
                    ("check", "安全校验"),
                ]
                self.visible = None
                self.current_step_key = None
                self.status = None

            def set_current_step(self, step_key, status="running"):
                self.current_step_key = step_key
                self.status = status

            def setVisible(self, visible):
                self.visible = visible

        return FakeTimeline()

    def _make_resume_state(
        self,
        manifest=None,
        *,
        latest_manifest=Path("C:/dummy/manifest.json"),
        game_root=Path("C:/dummy/project"),
        load_error=None,
        batch_script_path=Path("C:/tool/gemini_translate_batch.py"),
    ):
        manifest_data = dict(manifest or {})

        class FakeState:
            def get_game_root(self):
                return game_root

            def get_latest_manifest_path_for_mode(self, game_root, mode):
                return latest_manifest

            def load_resume_manifest(self, path, work_mode):
                if load_error is not None:
                    raise load_error
                return dict(manifest_data)

            def get_batch_script_path(self):
                return batch_script_path

        return FakeState()

    def _prepare_resume_refresh(
        self,
        *,
        work_mode,
        manifest=None,
        latest_manifest=Path("C:/dummy/manifest.json"),
        load_error=None,
    ):
        self.window.kill_btn = type("FakeBtn", (), {"isEnabled": lambda _self: False})()
        self.window._current_work_mode = lambda: work_mode
        self.window.state = self._make_resume_state(
            manifest,
            latest_manifest=latest_manifest,
            load_error=load_error,
        )
        self.window.timeline = self._fake_timeline()
        summary_calls = []
        self.window._set_workflow_summary = (
            lambda status, heading, message, facts=None: summary_calls.append(
                (status, heading, message, facts)
            )
        )
        return summary_calls

    def test_sync_models_for_save_preserves_fallback_models(self):
        models = self.window._sync_models_for_save(
            ["gemini-primary", "gemini-fallback", "gemini-primary", "", 123],
            "gemini-new",
        )

        self.assertEqual(models, ["gemini-new", "gemini-primary", "gemini-fallback"])

    def test_sync_models_for_save_keeps_existing_models_when_selection_is_empty(self):
        models = self.window._sync_models_for_save(
            ["gemini-primary", "gemini-fallback"],
            "",
        )

        self.assertEqual(models, ["gemini-primary", "gemini-fallback"])

    def test_missing_batch_thinking_uses_cli_default_for_supported_model(self):
        thinking_level = self.window._batch_thinking_value_for_load(
            {},
            "gemini-3.5-flash",
        )

        self.assertEqual(thinking_level, "minimal")

    def test_explicit_empty_batch_thinking_overrides_supported_model_default(self):
        thinking_level = self.window._batch_thinking_value_for_load(
            {"thinking_level": ""},
            "gemini-3.5-flash",
        )

        self.assertEqual(thinking_level, "")

    def test_empty_batch_thinking_is_saved_after_user_change_for_supported_model(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-3.5-flash",
            "",
            True,
        )

        self.assertTrue(should_save)

    def test_empty_batch_thinking_is_not_saved_for_implicit_supported_model_switch(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-3.5-flash",
            "",
            False,
        )

        self.assertFalse(should_save)

    def test_empty_batch_thinking_is_not_added_for_unsupported_model(self):
        should_save = self.window._should_save_batch_thinking_level(
            {},
            "gemini-2.5-flash",
            "",
            True,
        )

        self.assertFalse(should_save)

    def test_supported_model_switch_defaults_empty_implicit_selection_to_minimal(self):
        thinking_level = self.window._batch_thinking_value_for_model_change(
            "gemini-3.5-flash",
            "",
            False,
            False,
        )

        self.assertEqual(thinking_level, "minimal")

    def test_supported_model_switch_preserves_explicit_empty_selection(self):
        thinking_level = self.window._batch_thinking_value_for_model_change(
            "gemini-3.5-flash",
            "",
            False,
            True,
        )

        self.assertIsNone(thinking_level)

    def _prepare_bootstrap_window(self, config):
        class FakeRunner:
            def __init__(self):
                self.calls = []

            def run(self, script, args):
                self.calls.append((script, args))

        class FakeState:
            def get_game_root(self):
                return Path("C:/Game/work")

            def get_batch_script_path(self):
                return Path("C:/tool/gemini_translate_batch.py")

            def load_translator_config(self):
                return config

        self.window.state = FakeState()
        self.window.runner = FakeRunner()
        self.window.log_view = type("FakeLogView", (), {"clear": lambda _self: None})()
        self.window._focus_log_tab = lambda: None
        self.window._focus_workbench_status_tab = lambda _index: None
        self.window._set_workflow_from_bootstrap_summary = lambda _summary: None
        self.window._append_log = lambda _text: None
        self.window._set_task_running = lambda _running: None
        return self.window.runner

    def test_bootstrap_rag_uses_skip_prepare(self):
        runner = self._prepare_bootstrap_window(
            {"batch": {"rag": {"enabled": True}}}
        )

        self.window._start_bootstrap_task("rag")

        self.assertEqual(
            runner.calls,
            [
                (
                    Path("C:/tool/gemini_translate_batch.py"),
                    ["bootstrap-rag", "--skip-prepare"],
                )
            ],
        )

    def test_bootstrap_source_index_uses_skip_prepare(self):
        runner = self._prepare_bootstrap_window(
            {"batch": {"source_index": {"enabled": True}}}
        )

        self.window._start_bootstrap_task("source_index")

        self.assertEqual(
            runner.calls,
            [
                (
                    Path("C:/tool/gemini_translate_batch.py"),
                    ["bootstrap-source-index", "--skip-prepare"],
                )
            ],
        )

    def test_saved_batch_context_flags_reads_persisted_config(self):
        class FakeState:
            def load_translator_config(self):
                return {
                    "batch": {
                        "rag": {"enabled": True},
                        "source_index": {"enabled": False},
                    }
                }

        self.window.state = FakeState()

        flags = self.window._saved_batch_context_flags()

        self.assertTrue(flags["rag_enabled"])
        self.assertFalse(flags["source_index_enabled"])

    def test_bootstrap_task_ready_uses_saved_config(self):
        from gui_qt.work_modes import WorkMode, work_mode_spec

        class FakeState:
            def load_translator_config(self):
                return {
                    "batch": {
                        "rag": {"enabled": True},
                        "source_index": {"enabled": False},
                    }
                }

        self.window.state = FakeState()

        rag_spec = work_mode_spec(WorkMode.BOOTSTRAP_RAG)
        source_spec = work_mode_spec(WorkMode.BOOTSTRAP_SOURCE_INDEX)

        self.assertTrue(self.window._bootstrap_task_ready(rag_spec))
        self.assertFalse(self.window._bootstrap_task_ready(source_spec))

    def test_refresh_workflow_from_latest_manifest_no_manifest_shows_idle(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            latest_manifest=None,
        )

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(summary_calls[0][0], "idle")

    def test_refresh_workflow_from_latest_manifest_load_error_shows_idle(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            load_error=ValueError("Dummy load error"),
        )

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(summary_calls[0][0], "idle")

    def test_refresh_workflow_from_latest_manifest_running_job_state(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            manifest={
                "job_name": "jobs/12345",
                "job_state": "JOB_STATE_RUNNING",
                "summary": {
                    "file_count": 3,
                    "chunk_count": 10,
                    "item_count": 100,
                },
            },
        )

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "waiting")
        self.assertIn("任务进行中", heading)
        self.assertIn("云端任务：jobs/12345", facts)
        self.assertIn("任务状态：处理中", facts)
        self.assertIn("扫描文件：3 个", facts)
        self.assertTrue(self.window.timeline.visible)
        self.assertEqual(self.window.timeline.current_step_key, "status")
        self.assertEqual(self.window.timeline.status, "waiting")

    def test_refresh_workflow_from_latest_manifest_failed_job_state(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            manifest={
                "job_state": "JOB_STATE_FAILED",
                "job_error": "Quota exceeded",
            },
        )

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "failed")
        self.assertIn("任务已失败", heading)
        self.assertIn("Quota exceeded", message)

    def test_refresh_workflow_from_latest_manifest_cancelled_job_state_is_terminal(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            manifest={"job_state": "JOB_STATE_CANCELLED"},
        )

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "failed")
        self.assertIn("无法继续", heading)
        self.assertIn("已取消", message)
        self.assertTrue(self.window.timeline.visible)
        self.assertEqual(self.window.timeline.current_step_key, "status")
        self.assertEqual(self.window.timeline.status, "failed")

    def test_refresh_workflow_from_latest_manifest_resumable_job_state(self):
        from gui_qt.work_modes import WorkMode
        from unittest.mock import patch, MagicMock

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            manifest={"job_state": "JOB_STATE_SUCCEEDED"},
        )

        mock_workflow = MagicMock()
        mock_workflow.current_step.return_value = MagicMock(key="download")

        with patch("gui_qt.app.resume_workflow", return_value=mock_workflow):
            self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "ready")
        self.assertIn("可继续最新", heading)
        self.assertTrue(self.window.timeline.visible)
        self.assertEqual(self.window.timeline.current_step_key, "download")
        self.assertEqual(self.window.timeline.status, "ready")

    def test_refresh_workflow_from_latest_manifest_completed_job_state(self):
        from gui_qt.work_modes import WorkMode
        from unittest.mock import patch, MagicMock

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            manifest={"job_state": "done"},
        )

        mock_workflow = MagicMock()
        mock_workflow.current_step.return_value = None

        with patch("gui_qt.app.resume_workflow", return_value=mock_workflow):
            self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "done")
        self.assertIn("任务已完成", heading)
        self.assertTrue(self.window.timeline.visible)
        self.assertIsNone(self.window.timeline.current_step_key)
        self.assertEqual(self.window.timeline.status, "done")

    def test_resume_completed_keyword_manifest_shows_result_summary(self):
        from gui_qt.work_modes import WorkMode
        from unittest.mock import MagicMock, patch

        self.window._current_work_mode = lambda: WorkMode.KEYWORD_EXTRACTION
        self.window.state = self._make_resume_state(
            {
                "mode": "keyword_extraction",
                "base_dir": "C:/dummy/project",
                "job_name": "batches/keywords",
                "job_state": "JOB_STATE_SUCCEEDED",
                "keyword_export": {
                    "markdown_path": "C:/dummy/keyword_candidates.md",
                },
            }
        )
        self.window.timeline = self._fake_timeline()

        summary_calls = []
        self.window._set_workflow_summary = lambda status, heading, message, facts=None: summary_calls.append((status, heading, message, facts))
        self.window._refresh_diagnostics_context = MagicMock()
        self.window._refresh_writeback_from_latest_manifest = MagicMock()
        self.window._current_writeback_summary = lambda: type("Summary", (), {"status": "safe"})()
        focus_calls = []
        self.window._focus_workbench_status_tab = lambda index: focus_calls.append(index)

        class FakeStatusBar:
            def __init__(self):
                self.messages = []
            def showMessage(self, text, timeout):
                self.messages.append((text, timeout))
        status_bar = FakeStatusBar()
        self.window.statusBar = lambda: status_bar

        mock_workflow = MagicMock()
        mock_workflow.current_step.return_value = None

        with patch("gui_qt.app.resume_workflow", return_value=mock_workflow):
            self.window._on_resume_translation()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "done")
        self.assertIn("任务已完成", heading)
        self.assertTrue(self.window.timeline.visible)
        self.assertIsNone(self.window.timeline.current_step_key)
        self.assertEqual(self.window.timeline.status, "done")
        self.assertTrue(any("任务记录" in fact for fact in facts))
        self.window._refresh_diagnostics_context.assert_called_once()
        self.window._refresh_writeback_from_latest_manifest.assert_called_once()
        self.assertEqual(focus_calls, [2])
        self.assertIn("任务已完成", status_bar.messages[-1][0])

    def test_resume_after_successful_status_query_runs_download_next(self):
        from gui_qt.work_modes import WorkMode

        manifest_path = Path("C:/dummy/manifest.json")
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window.timeline = self._fake_timeline()
        self.window.state = self._make_resume_state(
            {
                "mode": "translation",
                "base_dir": "C:/dummy/project",
                "job_name": "batches/translation",
                "job_state": "JOB_STATE_SUCCEEDED",
            },
            latest_manifest=manifest_path,
        )

        class FakeRunner:
            def __init__(self):
                self.calls = []
            def run(self, script_path, args):
                self.calls.append((script_path, args))

        class FakeLogView:
            def clear(self):
                pass

        class FakeResumeButton:
            def text(self):
                return "继续翻译"

        self.window.runner = FakeRunner()
        self.window.log_view = FakeLogView()
        self.window.resume_btn = FakeResumeButton()
        self.window._focus_log_tab = lambda: None
        self.window._focus_workbench_status_tab = lambda index: None
        self.window._refresh_diagnostics_context = lambda: None
        self.window._append_log = lambda message: None
        self.window._set_task_running = lambda running: None
        summary_calls = []
        self.window._set_workflow_summary = lambda status, heading, message, facts=None: summary_calls.append((status, heading, message, facts))

        self.window._on_resume_translation()

        self.assertEqual(len(self.window.runner.calls), 1)
        script_path, args = self.window.runner.calls[0]
        self.assertEqual(script_path, Path("C:/tool/gemini_translate_batch.py"))
        self.assertEqual(args, ["download", str(manifest_path)])
        self.assertEqual(summary_calls[-1][0], "running")
        self.assertTrue(self.window.timeline.visible)
        self.assertEqual(self.window.timeline.current_step_key, "download")

    def test_keyword_export_finish_refreshes_keyword_result_summary(self):
        from gui_qt.translation_workflow import WorkflowStep, WorkflowUpdate
        from gui_qt.work_modes import WorkMode
        from unittest.mock import MagicMock

        class FakeWorkflow:
            manifest_path = "C:/dummy/manifest.json"

            def current_step(self):
                return WorkflowStep(
                    "export-keywords",
                    ["export-keywords", self.manifest_path],
                    "正在导出关键词报告",
                    "正在整理候选术语与剧情概要报告。",
                )

            def complete_current_step(self, exit_code, output):
                return WorkflowUpdate(
                    status="done",
                    heading="关键词提取完成",
                    message="done",
                    facts=[],
                    should_continue=False,
                )

        class FakeStatusBar:
            def __init__(self):
                self.messages = []
            def showMessage(self, text, timeout):
                self.messages.append((text, timeout))

        self.window._workflow = FakeWorkflow()
        self.window._workflow_step_output_lines = ["Keyword candidates: 3 deduped from 5 raw"]
        self.window._current_work_mode = lambda: WorkMode.KEYWORD_EXTRACTION
        self.window._copy_keyword_reports_to_game_parent = MagicMock()
        self.window._uses_revision_writeback = lambda: False
        self.window._set_workflow_update = MagicMock()
        self.window._refresh_diagnostics_context = MagicMock()
        self.window._set_task_running = MagicMock()
        self.window._refresh_writeback_from_latest_manifest = MagicMock()
        self.window._set_writeback_summary = MagicMock()
        status_bar = FakeStatusBar()
        self.window.statusBar = lambda: status_bar

        self.window._on_workflow_step_finished(0)

        self.window._copy_keyword_reports_to_game_parent.assert_called_once_with("C:/dummy/manifest.json")
        self.window._refresh_writeback_from_latest_manifest.assert_called_once()
        self.window._set_writeback_summary.assert_not_called()
        self.assertIn("关键词提取完成", status_bar.messages[-1][0])

    def test_update_resume_btn_text_uses_current_status_without_disk_lookup(self):
        from gui_qt.work_modes import WorkMode
        from unittest.mock import MagicMock

        self.window._workflow = None
        self.window._current_work_mode = lambda: WorkMode.KEYWORD_EXTRACTION
        self.window.resume_btn = MagicMock()
        self.window.workflow_status_label = MagicMock()
        self.window.workflow_status_label.property.return_value = "waiting"

        self.window._update_resume_btn_text()

        self.window.resume_btn.setText.assert_called_once_with("查询云端状态")

    def test_refresh_writeback_from_latest_keyword_manifest_shows_report_summary(self):
        from gui_qt.work_modes import WorkMode

        self.window._current_work_mode = lambda: WorkMode.KEYWORD_EXTRACTION

        class FakeState:
            def get_game_root(self):
                return Path("C:/dummy/project")
            def get_latest_manifest_path_for_mode(self, game_root, mode):
                return Path("C:/dummy/manifest.json")
            def load_resume_manifest(self, path, work_mode):
                return {
                    "_manifest_path": str(path),
                    "mode": "keyword_extraction",
                    "base_dir": "C:/dummy/project",
                    "keyword_export": {
                        "markdown_path": "C:/dummy/keyword_candidates.md",
                        "jsonl_path": "C:/dummy/keyword_candidates.jsonl",
                        "summary": {
                            "candidate_count_deduped": 3,
                            "candidate_count_raw": 5,
                            "chunk_summary_count": 2,
                        },
                    },
                }
        self.window.state = FakeState()

        summaries = []
        self.window._set_writeback_summary = lambda summary: summaries.append(summary)

        self.window._refresh_writeback_from_latest_manifest()

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.status, "safe")
        self.assertFalse(summary.can_apply)
        self.assertIn("关键词报告已生成", summary.heading)
        self.assertTrue(any("候选 Markdown" in fact for fact in summary.facts))

    def test_keyword_result_hides_writeback_action_buttons(self):
        from gui_qt.check_report import WritebackSummary
        from gui_qt.work_modes import WorkMode

        self.window._current_work_mode = lambda: WorkMode.KEYWORD_EXTRACTION

        class FakeButton:
            def __init__(self):
                self.visible = True
                self.enabled = True
            def setVisible(self, visible):
                self.visible = visible
            def setEnabled(self, enabled):
                self.enabled = enabled

        button_names = [
            "apply_btn",
            "apply_revision_btn",
            "check_issues_btn",
            "retry_btn",
            "apply_failure_btn",
            "remediation_btn",
        ]
        for name in button_names:
            setattr(self.window, name, FakeButton())

        self.window._update_writeback_action_buttons(
            WritebackSummary(
                status="safe",
                heading="关键词报告已生成",
                message="done",
                facts=[],
                findings=[],
                can_apply=False,
                manifest_path="C:/dummy/manifest.json",
            ),
            running=False,
        )

        for name in button_names:
            button = getattr(self.window, name)
            self.assertFalse(button.visible, name)
            self.assertFalse(button.enabled, name)

    def test_copy_keyword_reports_to_game_parent_copies_successfully(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            game_root = tmp_path / "Game" / "work"
            game_root.mkdir(parents=True, exist_ok=True)

            manifest_dir = tmp_path / "logs" / "batch_jobs" / "job_keywords"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            # Create dummy report files
            (manifest_dir / "keyword_candidates.md").write_text("candidates", encoding="utf-8")
            (manifest_dir / "keyword_chunk_summaries.md").write_text("summaries", encoding="utf-8")

            class FakeState:
                def get_game_root(self):
                    return game_root
            self.window.state = FakeState()

            logged_messages = []
            self.window._append_log = lambda msg: logged_messages.append(msg)

            self.window._copy_keyword_reports_to_game_parent(str(manifest_path))

            target_dir = tmp_path / "Game" / "extracted_keywords"
            self.assertTrue(target_dir.exists())
            self.assertTrue((target_dir / "keyword_candidates.md").exists())
            self.assertTrue((target_dir / "keyword_chunk_summaries.md").exists())
            self.assertEqual((target_dir / "keyword_candidates.md").read_text(encoding="utf-8"), "candidates")

            self.assertTrue(any("已将关键词提取报告复制一份至" in msg for msg in logged_messages))


if __name__ == "__main__":
    unittest.main()
