import copy
import unittest
from unittest import mock
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

    def test_on_cli_line_ready_updates_workflow_progress_bar(self):
        from gui_qt.workflow_progress import create_workflow_progress_state

        class FakeProgressBar:
            def __init__(self):
                self.visible = False
                self.range_values = None
                self.value = None
                self.format = ""

            def setRange(self, minimum, maximum):
                self.range_values = (minimum, maximum)

            def setValue(self, value):
                self.value = value

            def setFormat(self, value):
                self.format = value

            def setVisible(self, visible):
                self.visible = visible

        class FakeLabel:
            def __init__(self):
                self.text = ""

            def setText(self, text):
                self.text = text

        self.window._active_command = "translation_workflow"
        self.window._workflow_step_output_lines = []
        self.window._workflow_progress = create_workflow_progress_state("sync_requests")
        self.window._workflow_progress_base_facts = ["Manifest: C:/dummy/manifest.json"]
        self.window.workflow_progress_bar = FakeProgressBar()
        self.window.workflow_facts_label = FakeLabel()
        self.window._append_log = lambda text: None
        self.window._progress_flush_timer = None

        self.window._on_cli_line_ready("[2/4] sync-key")

        self.assertEqual(self.window._workflow_step_output_lines, ["[2/4] sync-key"])
        self.assertTrue(self.window.workflow_progress_bar.visible)
        self.assertEqual(self.window.workflow_progress_bar.range_values, (0, 4))
        self.assertEqual(self.window.workflow_progress_bar.value, 2)
        self.assertEqual(self.window.workflow_progress_bar.format, "请求 2/4")
        self.assertEqual(
            self.window.workflow_facts_label.text,
            "Manifest: C:/dummy/manifest.json\n当前请求：sync-key",
        )

    def test_clear_log_view_flushes_pending_buffer(self):
        class FakeLogView:
            def __init__(self):
                self.lines: list[str] = []

            def append(self, text: str) -> None:
                self.lines.append(text)

            def clear(self) -> None:
                self.lines.clear()

            def verticalScrollBar(self):
                class Bar:
                    def setValue(self, _value: int) -> None:
                        return None

                    @property
                    def maximum(self) -> int:
                        return 0

                return Bar()

        self.window.log_view = FakeLogView()
        self.window._pending_log_lines = ["stale line"]
        self.window._log_flush_timer = None

        self.window._clear_log_view()

        self.assertEqual(self.window._pending_log_lines, [])
        self.assertEqual(self.window.log_view.lines, [])

    def test_short_job_name_compacts_batch_ids(self):
        job_name = "batches/l10v1nppy30lvv2cbzbhedje4oqnqlzedlve"

        self.assertEqual(self.window._short_job_name(job_name), "batches/l10v1nppy3...")
        self.assertEqual(self.window._short_job_name("batches/short"), "batches/short")

    def test_split_submit_button_only_shows_when_remaining_packages_need_submit(self):
        from gui_qt.work_modes import WorkMode

        class FakeButton:
            def __init__(self):
                self.visible = None
                self.enabled = None
                self.text = ""

            def setVisible(self, visible):
                self.visible = visible

            def setEnabled(self, enabled):
                self.enabled = enabled

            def setText(self, text):
                self.text = text

        class FakeKillButton:
            def __init__(self, enabled=False):
                self.enabled = enabled

            def isEnabled(self):
                return self.enabled

        class FakeEntry:
            def __init__(self, needs_submit):
                self.needs_submit = needs_submit

        button = FakeButton()
        self.window.split_submit_btn = button
        self.window.kill_btn = FakeKillButton()
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._split_status_entries = []

        self.window._update_split_submit_btn([FakeEntry(False), FakeEntry(False)])

        self.assertFalse(button.visible)
        self.assertFalse(button.enabled)

        self.window._update_split_submit_btn([FakeEntry(False), FakeEntry(True)])

        self.assertTrue(button.visible)
        self.assertTrue(button.enabled)
        self.assertEqual(button.text, "提交剩余包")

        self.window.kill_btn = FakeKillButton(enabled=True)
        self.window._update_split_submit_btn([FakeEntry(False), FakeEntry(True)], running=False)

        self.assertTrue(button.enabled)

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
        self.window._completed_manifest_snapshot = None
        self.window._viewing_completed_manifest = False
        self.window._split_status_entries = []
        self.window._split_status_selected_manifest_path = ""
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
        self.window._render_split_status_entries = lambda *_args, **_kwargs: None
        self.window._update_completed_manifest_entry_ui = lambda: None
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
        self.window._show_workbench_log_drawer = lambda: None
        self.window._focus_workbench_status_tab = lambda _index: None
        self.window._set_workflow_from_bootstrap_summary = lambda _summary: None
        self.window._append_log = lambda _text: None
        self.window._set_task_running = lambda _running: None
        # Avoid per-project project_context_settings.json on a real path overriding
        # the FakeState translator_config flags used by these unit tests.
        self.window._game_root_str_for_flags = lambda: None
        self.window._confirm_unsaved_config_before_workflow = lambda: True
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

    def test_bootstrap_summary_syncs_timeline_for_terminal_states(self):
        from gui_qt.bootstrap_report import BootstrapSummary, running_bootstrap_summary

        class FakeTimeline:
            def __init__(self):
                self.steps = [("run", "同步运行")]
                self.visible = None
                self.current_step_key = "unset"
                self.status = "unset"

            def set_current_step(self, step_key, status="running"):
                self.current_step_key = step_key
                self.status = status

            def setVisible(self, visible):
                self.visible = visible

        class FakeProgressBar:
            def __init__(self):
                self.visible = True

            def setVisible(self, visible):
                self.visible = visible

        self.window.timeline = FakeTimeline()
        self.window.workflow_progress_bar = FakeProgressBar()
        self.window._bootstrap_progress = object()
        summary_calls = []
        self.window._set_workflow_summary = (
            lambda status, heading, message, facts: summary_calls.append(
                (status, heading, message, facts)
            )
        )

        self.window._set_workflow_from_bootstrap_summary(running_bootstrap_summary("rag"))
        self.assertEqual(self.window.timeline.current_step_key, "run")
        self.assertEqual(self.window.timeline.status, "running")
        self.assertFalse(self.window.timeline.visible)

        self.window._set_workflow_from_bootstrap_summary(
            BootstrapSummary(
                kind="rag",
                status="failed",
                heading="预建失败",
                message="出错",
                facts=[],
                findings=[],
            )
        )
        self.assertEqual(self.window.timeline.current_step_key, "run")
        self.assertEqual(self.window.timeline.status, "failed")
        self.assertFalse(self.window.timeline.visible)
        self.assertIsNone(self.window._bootstrap_progress)
        self.assertFalse(self.window.workflow_progress_bar.visible)
        self.assertEqual(summary_calls[-1][0], "failed")

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

    def test_save_config_persists_context_storage_and_rolls_back_on_project_failure(self):
        from gui_qt.work_modes import WorkMode

        saved_configs = []
        config = {
            "sync": {"rag": {}},
            "batch": {"rag": {"enabled": False}, "source_index": {"enabled": True}, "model": "gemini-3.1-flash-lite"},
        }

        class FakeState:
            def get_game_root(self):
                return Path("C:/Game/work")
            def normalize_game_root(self, path):
                return Path("C:/Game/work"), False
            def load_translator_config(self):
                return config
            def save_translator_config(self, saved_config):
                saved_configs.append(copy.deepcopy(saved_config))

        class FakeCheckBox:
            def __init__(self, checked):
                self._checked = checked
            def isChecked(self):
                return self._checked

        class FakeCombo:
            def __init__(self, text="", data=""):
                self._text = text
                self._data = data
            def currentText(self):
                return self._text
            def currentData(self):
                return self._data

        class FakeStatusBar:
            def showMessage(self, text, timeout):
                pass

        self.window.state = FakeState()
        self.window.rag_enabled_cb = FakeCheckBox(True)
        self.window.source_index_enabled_cb = FakeCheckBox(True)
        self.window.bootstrap_on_build_cb = FakeCheckBox(False)
        self.window.context_storage_game_cb = FakeCheckBox(True)
        self.window.sync_model_combo = FakeCombo("gemini-sync")
        self.window.batch_model_combo = FakeCombo("gemini-3.1-flash-lite")
        self.window.sync_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_thinking_combo = FakeCombo(data="minimal")
        self.window._batch_thinking_user_changed = False
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._append_log = lambda _text: None
        self.window._refresh_project_label = lambda: None
        self.window.statusBar = lambda: FakeStatusBar()

        with mock.patch("project_context_settings.save_project_context_settings") as save_project:
            saved = self.window._on_save_config()

        self.assertTrue(saved)
        self.assertEqual(len(saved_configs), 1)
        self.assertEqual(saved_configs[0]["context_storage"]["location"], "game")
        self.assertEqual(saved_configs[0]["context_storage"]["game_dir_name"], "translation_context")
        save_project.assert_called_once_with(
            Path("C:/Game/work"),
            {
                "rag_enabled": True,
                "source_index_enabled": True,
                "bootstrap_on_build": False,
            },
        )
        # Legacy global values remain fallback defaults for projects without
        # project_context_settings.json; this project's values live in save_project.
        self.assertFalse(saved_configs[0]["batch"]["rag"]["enabled"])
        self.assertTrue(saved_configs[0]["batch"]["source_index"]["enabled"])

        saved_configs.clear()
        self.window.context_storage_game_cb = FakeCheckBox(False)
        with (
            mock.patch(
                "project_context_settings.save_project_context_settings",
                side_effect=OSError("project write failed"),
            ),
            mock.patch("gui_qt.app.QMessageBox.warning"),
        ):
            saved = self.window._on_save_config()
        self.assertFalse(saved)
        self.assertEqual(saved_configs[0]["context_storage"]["location"], "tool")
        self.assertEqual(len(saved_configs), 2)
        self.assertEqual(saved_configs[1]["context_storage"]["location"], "game")

    def test_save_config_preserves_legacy_context_storage_dir_name(self):
        from gui_qt.work_modes import WorkMode

        saved_configs = []
        config = {
            "context_storage": {"directory_name": "my_context"},
            "sync": {"rag": {}},
            "batch": {"rag": {}, "source_index": {}, "model": "gemini-3.1-flash-lite"},
        }

        class FakeState:
            def get_game_root(self):
                return Path("C:/Game/work")
            def normalize_game_root(self, path):
                return Path("C:/Game/work"), False
            def load_translator_config(self):
                return config
            def save_translator_config(self, saved_config):
                saved_configs.append(saved_config)

        class FakeCheckBox:
            def __init__(self, checked):
                self._checked = checked
            def isChecked(self):
                return self._checked

        class FakeCombo:
            def __init__(self, text="", data=""):
                self._text = text
                self._data = data
            def currentText(self):
                return self._text
            def currentData(self):
                return self._data

        class FakeStatusBar:
            def showMessage(self, text, timeout):
                pass

        self.window.state = FakeState()
        self.window.rag_enabled_cb = FakeCheckBox(True)
        self.window.source_index_enabled_cb = FakeCheckBox(False)
        self.window.bootstrap_on_build_cb = FakeCheckBox(False)
        self.window.context_storage_game_cb = FakeCheckBox(True)
        self.window.sync_model_combo = FakeCombo("gemini-sync")
        self.window.batch_model_combo = FakeCombo("gemini-3.1-flash-lite")
        self.window.sync_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_thinking_combo = FakeCombo(data="minimal")
        self.window._batch_thinking_user_changed = False
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._append_log = lambda _text: None
        self.window._refresh_project_label = lambda: None
        self.window.statusBar = lambda: FakeStatusBar()

        with mock.patch("project_context_settings.save_project_context_settings"):
            saved = self.window._on_save_config()

        self.assertTrue(saved)
        self.assertEqual(saved_configs[0]["context_storage"]["game_dir_name"], "my_context")

    def test_save_config_persists_advanced_settings_and_theme(self):
        from gui_qt.work_modes import WorkMode

        saved_configs = []
        config = {
            "sync": {"rag": {}, "custom": 1},
            "batch": {
                "rag": {},
                "source_index": {},
                "model": "gemini-3.1-flash-lite",
                "custom": "keep",
            },
        }

        class FakeState:
            def get_game_root(self):
                return Path("C:/Game/work")
            def normalize_game_root(self, path):
                return Path("C:/Game/work"), False
            def load_translator_config(self):
                return config
            def save_translator_config(self, saved_config):
                saved_configs.append(saved_config)

        class FakeCheckBox:
            def __init__(self, checked):
                self._checked = checked
            def isChecked(self):
                return self._checked

        class FakeCombo:
            def __init__(self, text="", data=""):
                self._text = text
                self._data = data
            def currentText(self):
                return self._text
            def currentData(self):
                return self._data

        class FakeText:
            def __init__(self, text):
                self._text = text
                self.focused = False
            def text(self):
                return self._text
            def setFocus(self):
                self.focused = True

        class FakeValue:
            def __init__(self, value):
                self._value = value
            def value(self):
                return self._value
            def setFocus(self):
                pass

        class FakeStatusBar:
            def showMessage(self, text, timeout):
                pass

        self.window.state = FakeState()
        self.window.rag_enabled_cb = FakeCheckBox(True)
        self.window.source_index_enabled_cb = FakeCheckBox(True)
        self.window.bootstrap_on_build_cb = FakeCheckBox(False)
        self.window.context_storage_game_cb = FakeCheckBox(True)
        self.window.sync_model_combo = FakeCombo("gemini-sync")
        self.window.batch_model_combo = FakeCombo("gemini-3.1-flash-lite")
        self.window.sync_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_thinking_combo = FakeCombo(data="minimal")
        self.window.theme_combo = FakeCombo(data="dark")
        self.window._refresh_project_label = lambda: None
        self.window._advanced_setting_widgets = {
            "batch_chunk_size": FakeValue(12),
            "batch_temperature": FakeValue(0.4),
            "context_storage_game_dir_name": FakeText("custom_context"),
            "include_files": FakeText("chapter01.rpy\nchapter02.rpy"),
            "prepare_unpack_command": FakeText("[\"python\", \"unpack.py\"]"),
            "batch_safety_settings": FakeText("relaxed_adult"),
            "batch_macro_setting": FakeText("Use a concise voice.\nKeep honorifics."),
            "batch_source_index_store_dir": FakeText("C:/ctx/source"),
        }
        self.window._advanced_setting_error_labels = {}
        self.window._settings_nav_rows = {}
        self.window._batch_thinking_user_changed = False
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._append_log = lambda _text: None
        self.window.statusBar = lambda: FakeStatusBar()

        with mock.patch("project_context_settings.save_project_context_settings"):
            saved = self.window._on_save_config()

        self.assertTrue(saved)
        self.assertEqual(len(saved_configs), 1)
        saved_config = saved_configs[0]
        self.assertEqual(saved_config["gui"]["theme"], "dark")
        self.assertEqual(saved_config["sync"]["custom"], 1)
        self.assertEqual(saved_config["batch"]["custom"], "keep")
        self.assertEqual(saved_config["game_root"], str(Path("C:/Game/work")))
        self.assertEqual(saved_config["batch"]["chunk_size"], 12)
        self.assertEqual(saved_config["batch"]["temperature"], 0.4)
        self.assertEqual(saved_config["context_storage"]["game_dir_name"], "custom_context")
        self.assertEqual(saved_config["include_files"], ["chapter01.rpy", "chapter02.rpy"])
        self.assertEqual(saved_config["prepare"]["unpack_command"], ["python", "unpack.py"])
        self.assertEqual(saved_config["batch"]["safety_settings"], "relaxed_adult")
        self.assertEqual(saved_config["batch"]["macro_setting"], "Use a concise voice.\nKeep honorifics.")
        self.assertEqual(saved_config["batch"]["source_index"]["store_dir"], "C:/ctx/source")

    def test_save_config_validation_error_blocks_write(self):
        from gui_qt.work_modes import WorkMode

        saved_configs = []
        config = {
            "sync": {"rag": {}},
            "batch": {"rag": {}, "source_index": {}, "model": "gemini-3.1-flash-lite"},
        }

        class FakeState:
            def get_game_root(self):
                return Path("C:/Game/work")
            def load_translator_config(self):
                return config
            def save_translator_config(self, saved_config):
                saved_configs.append(saved_config)

        class FakeCheckBox:
            def __init__(self, checked):
                self._checked = checked
            def isChecked(self):
                return self._checked

        class FakeCombo:
            def __init__(self, text="", data=""):
                self._text = text
                self._data = data
            def currentText(self):
                return self._text
            def currentData(self):
                return self._data

        class FakeText:
            def __init__(self, text):
                self._text = text
                self.focused = False
            def text(self):
                return self._text
            def setFocus(self):
                self.focused = True

        class FakeLabel:
            def __init__(self):
                self.text = ""
            def setText(self, value):
                self.text = value

        class FakeNav:
            def __init__(self):
                self.row = None
            def setCurrentRow(self, row):
                self.row = row

        class FakeStatusBar:
            def __init__(self):
                self.messages = []
            def showMessage(self, text, timeout):
                self.messages.append((text, timeout))

        invalid_widget = FakeText("")
        error_label = FakeLabel()
        nav = FakeNav()
        status_bar = FakeStatusBar()
        self.window.state = FakeState()
        self.window.rag_enabled_cb = FakeCheckBox(True)
        self.window.source_index_enabled_cb = FakeCheckBox(False)
        self.window.bootstrap_on_build_cb = FakeCheckBox(True)
        self.window.context_storage_game_cb = FakeCheckBox(True)
        self.window.sync_model_combo = FakeCombo("gemini-sync")
        self.window.batch_model_combo = FakeCombo("gemini-3.1-flash-lite")
        self.window.sync_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_embedding_combo = FakeCombo("gemini-embedding-001")
        self.window.batch_thinking_combo = FakeCombo(data="minimal")
        self.window.theme_combo = FakeCombo(data="system")
        self.window._advanced_setting_widgets = {
            "context_storage_game_dir_name": invalid_widget,
        }
        self.window._advanced_setting_error_labels = {
            "context_storage_game_dir_name": error_label,
        }
        self.window._settings_nav_rows = {"workspace": 0, "project": 1, "advanced": 6}
        self.window.settings_nav = nav
        self.window._batch_thinking_user_changed = False
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._append_log = lambda _text: None
        self.window.statusBar = lambda: status_bar

        with mock.patch("project_context_settings.save_project_context_settings") as save_project:
            saved = self.window._on_save_config()

        self.assertFalse(saved)
        self.assertEqual(saved_configs, [])
        save_project.assert_not_called()
        self.assertIn("不能为空", error_label.text)
        self.assertEqual(nav.row, 6)
        self.assertTrue(invalid_widget.focused)
        self.assertEqual(status_bar.messages[-1][0], "高级设置有无效字段，未保存。")

    def test_settings_workspace_nav_disables_save_buttons(self):
        class FakeNav:
            def __init__(self):
                self.row = None

            def currentRow(self):
                return self.row

            def setCurrentRow(self, row):
                self.row = row

        class FakeBtn:
            def __init__(self):
                self.enabled = None

            def setEnabled(self, value):
                self.enabled = value

        class FakeStack:
            def setCurrentIndex(self, _index):
                pass

        nav = FakeNav()
        save_btn = FakeBtn()
        restore_btn = FakeBtn()
        reload_btn = FakeBtn()
        self.window.settings_nav = nav
        self.window.settings_stack = FakeStack()
        self.window._settings_nav_rows = {"workspace": 0, "project": 1}
        self.window.save_config_btn = save_btn
        self.window.restore_defaults_btn = restore_btn
        self.window.reload_config_btn = reload_btn
        self.window._task_running = False

        self.window._on_settings_nav_row_changed(0)
        self.assertFalse(save_btn.enabled)
        self.assertFalse(restore_btn.enabled)
        self.assertTrue(reload_btn.enabled)

        self.window._on_settings_nav_row_changed(1)
        self.assertTrue(save_btn.enabled)
        self.assertTrue(restore_btn.enabled)

    def test_settings_workspace_nav_skips_registry_activate_before_config_tab_open(self):
        activated = []

        class FakeNav:
            def currentRow(self):
                return 0

        class FakeStack:
            def setCurrentIndex(self, _index):
                pass

        class FakePanel:
            def set_current_game_root(self, _root):
                pass

            def activate_section(self):
                activated.append(True)

        config_tab = object()
        other_tab = object()
        self.window._config_tab = config_tab
        self.window.tab_widget = type(
            "FakeTabs",
            (),
            {"currentWidget": lambda self: other_tab},
        )()
        self.window.settings_nav = FakeNav()
        self.window.settings_stack = FakeStack()
        self.window._settings_nav_rows = {"workspace": 0, "project": 1}
        self.window._games_registry_panel = FakePanel()
        self.window._task_running = False
        self.window._sync_settings_action_bar_enabled = lambda **_kwargs: None

        self.window._on_settings_nav_row_changed(0)

        self.assertEqual(activated, [])

    def test_settings_workspace_nav_activates_registry_when_config_tab_open(self):
        activated = []

        class FakeNav:
            def currentRow(self):
                return 0

        class FakeStack:
            def setCurrentIndex(self, _index):
                pass

        class FakePanel:
            def set_current_game_root(self, _root):
                pass

            def activate_section(self):
                activated.append(True)

        config_tab = object()
        self.window._config_tab = config_tab
        self.window.tab_widget = type(
            "FakeTabs",
            (),
            {"currentWidget": lambda self: config_tab},
        )()
        self.window.settings_nav = FakeNav()
        self.window.settings_stack = FakeStack()
        self.window._settings_nav_rows = {"workspace": 0, "project": 1}
        self.window._games_registry_panel = FakePanel()
        self.window.state = type("FakeState", (), {"get_game_root": lambda self: Path("C:/Game/work")})()
        self.window._task_running = False
        self.window._sync_settings_action_bar_enabled = lambda **_kwargs: None

        self.window._on_settings_nav_row_changed(0)

        self.assertEqual(activated, [True])

    def test_opening_config_tab_activates_workspace_registry_section(self):
        activated = []

        class FakeNav:
            def __init__(self):
                self.row = 0

            def currentRow(self):
                return self.row

        class FakePanel:
            def set_current_game_root(self, _root):
                pass

            def activate_section(self):
                activated.append(True)

        config_tab = object()
        other_tab = object()

        class FakeTabs:
            def __init__(self):
                self.widgets = {0: other_tab, 1: config_tab}
                self.current_index = 0

            def widget(self, index):
                return self.widgets[index]

            def setCurrentIndex(self, index):
                self.current_index = index

            def currentWidget(self):
                return self.widgets[self.current_index]

        self.window._config_tab = config_tab
        self.window._diagnostics_tab = object()
        self.window.tab_widget = FakeTabs()
        self.window.settings_nav = FakeNav()
        self.window._settings_nav_rows = {"workspace": 0, "project": 1}
        self.window._games_registry_panel = FakePanel()
        self.window.state = type("FakeState", (), {"get_game_root": lambda self: Path("C:/Game/work")})()
        self.window._handling_config_tab_leave = False
        self.window._last_main_tab_index = 0
        self.window._refresh_api_status = lambda: None
        self.window._refresh_diagnostics_context = lambda: None

        self.window._on_tab_changed(1)

        self.assertEqual(activated, [True])
        self.assertEqual(self.window._last_main_tab_index, 1)

    def test_on_registry_switch_project_stays_on_workspace_section(self):
        switched = []
        focused = []
        panel_roots = []

        class FakePanel:
            def set_current_game_root(self, root):
                panel_roots.append(root)

        self.window._switch_game_root = lambda target: switched.append(target) or True
        self.window._focus_settings_section = lambda key: focused.append(key)
        self.window._show_settings_status = lambda *_args, **_kwargs: None
        self.window._confirm_unsaved_config_before_registry_switch = lambda: True
        self.window._games_registry_panel = FakePanel()
        self.window.state = type(
            "FakeState",
            (),
            {"get_game_root": lambda self: Path("C:/Game/work")},
        )()

        result = self.window._on_registry_switch_project("C:/Game/work")

        self.assertTrue(result)
        self.assertEqual(switched, ["C:/Game/work"])
        # Stay on 项目列表 — do not auto-jump to 项目 settings.
        self.assertEqual(focused, [])
        self.assertEqual(panel_roots, [Path("C:/Game/work")])

    def test_on_registry_switch_project_blocked_when_unsaved_changes_cancelled(self):
        from unittest.mock import patch

        switched = []

        class FakeMessageBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            shown = False

            def __init__(self, *_args, **_kwargs):
                self._cancel_btn = object()

            def setIcon(self, _icon):
                pass

            def setWindowTitle(self, _title):
                pass

            def setText(self, _text):
                pass

            def setInformativeText(self, _text):
                pass

            def addButton(self, text, _role):
                return self._cancel_btn if text == "取消" else object()

            def setDefaultButton(self, _button):
                pass

            def exec(self):
                FakeMessageBox.shown = True

            def clickedButton(self):
                return self._cancel_btn

        self.window._loading_config_to_ui = False
        self.window._config_ui_saved_snapshot = {"dirty": False}
        self.window._current_config_ui_snapshot = lambda: {"dirty": True}
        self.window._switch_game_root = lambda target: switched.append(target) or True

        with patch("gui_qt.app.QMessageBox", FakeMessageBox):
            result = self.window._on_registry_switch_project("C:/Game/work")

        self.assertFalse(result)
        self.assertEqual(switched, [])
        self.assertTrue(FakeMessageBox.shown)

    def test_on_registry_switch_project_discards_unsaved_changes_when_confirmed(self):
        from unittest.mock import patch

        switched = []

        class FakeMessageBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            shown = False

            def __init__(self, *_args, **_kwargs):
                self._discard_btn = object()

            def setIcon(self, _icon):
                pass

            def setWindowTitle(self, _title):
                pass

            def setText(self, _text):
                pass

            def setInformativeText(self, _text):
                pass

            def addButton(self, text, _role):
                return self._discard_btn if text == "不保存切换" else object()

            def setDefaultButton(self, _button):
                pass

            def exec(self):
                FakeMessageBox.shown = True

            def clickedButton(self):
                return self._discard_btn

        self.window._loading_config_to_ui = False
        self.window._config_ui_saved_snapshot = {"dirty": False}
        self.window._current_config_ui_snapshot = lambda: {"dirty": True}
        self.window._switch_game_root = lambda target: switched.append(target) or True
        self.window._focus_settings_section = lambda _key: None
        self.window._show_settings_status = lambda *_args, **_kwargs: None
        self.window._games_registry_panel = type(
            "FakePanel",
            (),
            {"set_current_game_root": lambda self, _root: None},
        )()
        self.window.state = type("FakeState", (), {"get_game_root": lambda self: Path("C:/Game/work")})()

        with patch("gui_qt.app.QMessageBox", FakeMessageBox):
            result = self.window._on_registry_switch_project("C:/Game/work")

        self.assertTrue(result)
        self.assertEqual(switched, ["C:/Game/work"])
        self.assertTrue(FakeMessageBox.shown)

    def test_on_registry_switch_project_saves_before_switch_when_requested(self):
        from unittest.mock import patch

        switched = []
        saved = []

        class FakeMessageBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            def __init__(self, *_args, **_kwargs):
                self._save_btn = object()

            def setIcon(self, _icon):
                pass

            def setWindowTitle(self, _title):
                pass

            def setText(self, _text):
                pass

            def setInformativeText(self, _text):
                pass

            def addButton(self, text, _role):
                return self._save_btn if text == "保存并切换" else object()

            def setDefaultButton(self, _button):
                pass

            def exec(self):
                pass

            def clickedButton(self):
                return self._save_btn

        self.window._loading_config_to_ui = False
        self.window._config_ui_saved_snapshot = {"dirty": False}
        self.window._current_config_ui_snapshot = lambda: {"dirty": True}
        self.window._on_save_config = lambda: saved.append(True) or True
        self.window._switch_game_root = lambda target: switched.append(target) or True
        self.window._focus_settings_section = lambda _key: None
        self.window._show_settings_status = lambda *_args, **_kwargs: None
        self.window._games_registry_panel = type(
            "FakePanel",
            (),
            {"set_current_game_root": lambda self, _root: None},
        )()
        self.window.state = type("FakeState", (), {"get_game_root": lambda self: Path("C:/Game/work")})()

        with patch("gui_qt.app.QMessageBox", FakeMessageBox):
            result = self.window._on_registry_switch_project("C:/Game/work")

        self.assertTrue(result)
        self.assertEqual(saved, [True])
        self.assertEqual(switched, ["C:/Game/work"])

    def test_settings_workspace_panel_enables_auto_discover_on_show(self):
        from unittest.mock import MagicMock, patch

        fake_page = MagicMock()
        fake_layout = MagicMock()
        with patch.object(MainWindow, "_settings_page", return_value=(fake_page, fake_layout)):
            with patch("gui_qt.app.GamesRegistryPanel") as panel_cls:
                with patch("gui_qt.app.QLabel", return_value=MagicMock()):
                    self.window.state = type(
                        "FakeState",
                        (),
                        {
                            "get_tool_root": lambda self: Path("C:/tool"),
                            "get_game_root": lambda self: Path("C:/Game/work"),
                            "get_workspace_root": lambda self: Path("C:/workspace"),
                        },
                    )()
                    self.window._current_registry_doctor_report = lambda: None
                    self.window._on_registry_switch_project = lambda _target: True
                    self.window._on_workspace_changed = lambda _path: None

                    self.window._build_settings_workspace_page()

        panel_cls.assert_called_once()
        kwargs = panel_cls.call_args.kwargs
        self.assertEqual(kwargs.get("workspace_root"), Path("C:/workspace"))
        self.assertNotIn(
            "auto_discover_on_show",
            kwargs,
            "workspace panel should use default auto_discover_on_show=True",
        )

    def test_focus_advanced_setting_navigates_to_workspace_for_game_root(self):
        class FakeNav:
            def __init__(self):
                self.row = None

            def setCurrentRow(self, row):
                self.row = row

        nav = FakeNav()
        self.window.settings_nav = nav
        self.window._settings_nav_rows = {"workspace": 0, "project": 1, "advanced": 6}

        self.window._focus_advanced_setting("game_root")

        self.assertEqual(nav.row, 0)

    def test_focus_advanced_setting_navigates_to_project_page_for_project_fields(self):
        class FakeNav:
            def __init__(self):
                self.row = None

            def setCurrentRow(self, row):
                self.row = row

        class FakeText:
            def __init__(self):
                self.focused = False

            def setFocus(self):
                self.focused = True

        nav = FakeNav()
        widget = FakeText()
        self.window.settings_nav = nav
        self.window._settings_nav_rows = {"workspace": 0, "project": 1, "advanced": 6}
        self.window._advanced_setting_widgets = {"glossary_file": widget}

        self.window._focus_advanced_setting("glossary_file")

        self.assertEqual(nav.row, 1)
        self.assertTrue(widget.focused)

    def test_sync_state_game_root_from_settings_switches_root_when_changed(self):
        class FakeState:
            def __init__(self):
                self._game_root = Path("C:/Game/old_work")

            def get_game_root(self):
                return self._game_root

            def normalize_game_root(self, path):
                return Path(str(path).replace("\\", "/")), False

        switched = []
        self.window.state = FakeState()
        self.window._switch_game_root = lambda directory: switched.append(directory) or True

        result = self.window._sync_state_game_root_from_settings("C:/Game/new_work")

        self.assertTrue(result)
        self.assertEqual(switched, ["C:/Game/new_work"])

    def test_sync_state_game_root_from_settings_skips_switch_when_unchanged(self):
        class FakeState:
            def __init__(self):
                self._game_root = Path("C:/Game/work")

            def get_game_root(self):
                return self._game_root

            def normalize_game_root(self, path):
                return Path("C:/Game/work"), False

        switched = []
        refreshed = []
        self.window.state = FakeState()
        self.window._switch_game_root = lambda directory: switched.append(directory) or True
        self.window._refresh_project_label = lambda: refreshed.append(True)

        result = self.window._sync_state_game_root_from_settings("C:/Game/work")

        self.assertTrue(result)
        self.assertEqual(switched, [])
        self.assertEqual(refreshed, [True])

    def test_restore_recommended_config_updates_basic_and_advanced_widgets(self):
        class FakeCheckBox:
            def __init__(self):
                self.checked = None
            def setChecked(self, value):
                self.checked = value
            def isChecked(self):
                return bool(self.checked)

        class FakeCombo:
            def __init__(self):
                self.items = []
                self.current_index = -1
                self.current_data = ""
            def findText(self, text):
                try:
                    return self.items.index((text, text))
                except ValueError:
                    return -1
            def addItem(self, text, data=None):
                self.items.append((text, text if data is None else data))
            def count(self):
                return len(self.items)
            def setCurrentIndex(self, index):
                self.current_index = index
                if 0 <= index < len(self.items):
                    self.current_data = self.items[index][1]
            def findData(self, data):
                for index, (_text, item_data) in enumerate(self.items):
                    if item_data == data:
                        return index
                return -1
            def currentText(self):
                if 0 <= self.current_index < len(self.items):
                    return self.items[self.current_index][0]
                return ""
            def currentData(self):
                return self.current_data

        class FakeValue:
            def __init__(self):
                self.saved = None
            def setValue(self, value):
                self.saved = value
            def value(self):
                return self.saved

        class FakeText:
            def __init__(self):
                self.saved = None
            def setText(self, value):
                self.saved = value
            def text(self):
                return self.saved or ""

        class FakeLabel:
            def __init__(self):
                self.text = "stale"
            def setText(self, value):
                self.text = value

        class FakeStatusBar:
            def __init__(self):
                self.messages = []
            def showMessage(self, text, timeout):
                self.messages.append((text, timeout))

        self.window.rag_enabled_cb = FakeCheckBox()
        self.window.source_index_enabled_cb = FakeCheckBox()
        self.window.bootstrap_on_build_cb = FakeCheckBox()
        self.window.context_storage_game_cb = FakeCheckBox()
        self.window.sync_model_combo = FakeCombo()
        self.window.batch_model_combo = FakeCombo()
        self.window.sync_embedding_combo = FakeCombo()
        self.window.batch_embedding_combo = FakeCombo()
        self.window.batch_thinking_combo = FakeCombo()
        for text, data in (("（不启用）", ""), ("最小", "minimal")):
            self.window.batch_thinking_combo.addItem(text, data)
        self.window.theme_combo = FakeCombo()
        for text, data in (("跟随系统", "system"), ("浅色", "light"), ("深色", "dark")):
            self.window.theme_combo.addItem(text, data)
        batch_chunk = FakeValue()
        context_dir = FakeText()
        error_label = FakeLabel()
        self.window._advanced_setting_widgets = {
            "batch_chunk_size": batch_chunk,
            "context_storage_game_dir_name": context_dir,
        }
        self.window._advanced_setting_error_labels = {
            "batch_chunk_size": error_label,
        }
        self.window._qt_app = None
        status_bar = FakeStatusBar()
        self.window.statusBar = lambda: status_bar

        self.window._on_restore_recommended_config()

        self.assertTrue(self.window.rag_enabled_cb.checked)
        self.assertFalse(self.window.source_index_enabled_cb.checked)
        self.assertEqual(self.window.sync_model_combo.currentText(), "gemini-3.1-flash-lite")
        self.assertEqual(self.window.batch_embedding_combo.currentText(), "gemini-embedding-001")
        self.assertEqual(self.window.batch_thinking_combo.currentData(), "minimal")
        self.assertEqual(self.window.theme_combo.currentData(), "system")
        self.assertEqual(batch_chunk.saved, 60)
        self.assertEqual(context_dir.saved, "translation_context")
        self.assertEqual(error_label.text, "")
        self.assertEqual(status_bar.messages[-1][0], "已恢复推荐值，保存后生效。")

    def test_download_recommended_fonts_starts_background_worker(self):
        class FakeWidget:
            def __init__(self):
                self.enabled = True
                self.text = ""
                self.visible = False

            def setEnabled(self, value):
                self.enabled = value

            def setText(self, value):
                self.text = value

            def setVisible(self, value):
                self.visible = value

        class FakeSignal:
            def __init__(self):
                self.callback = None

            def connect(self, callback):
                self.callback = callback

        class FakeWorker:
            def __init__(self, parent):
                self.parent = parent
                self.completed = FakeSignal()
                self.started = False

            def start(self):
                self.started = True

        button = FakeWidget()
        label = FakeWidget()
        progress = FakeWidget()
        self.window._font_install_worker = None
        self.window.download_fonts_btn = button
        self.window.font_install_status_label = label
        self.window.font_install_progress = progress

        from gui_qt.app import QMessageBox

        with (
            mock.patch(
                "gui_qt.app.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            mock.patch("gui_qt.app.FontInstallWorker", FakeWorker),
        ):
            self.window._on_download_recommended_fonts()

        worker = self.window._font_install_worker
        self.assertIsInstance(worker, FakeWorker)
        self.assertTrue(worker.started)
        self.assertEqual(worker.completed.callback, self.window._on_recommended_fonts_downloaded)
        self.assertFalse(button.enabled)
        self.assertEqual(button.text, "正在下载…")
        self.assertTrue(progress.visible)
        self.assertIn("正在从字体发布者", label.text)

    def test_recommended_fonts_download_failure_keeps_system_fallback(self):
        class FakeWidget:
            def __init__(self):
                self.text = ""
                self.visible = True

            def setText(self, value):
                self.text = value

            def setVisible(self, value):
                self.visible = value

        class FakeWorker:
            def __init__(self):
                self.deleted = False

            def deleteLater(self):
                self.deleted = True

        worker = FakeWorker()
        label = FakeWidget()
        progress = FakeWidget()
        messages = []
        self.window._font_install_worker = worker
        self.window.font_install_status_label = label
        self.window.font_install_progress = progress
        self.window._refresh_font_install_status = lambda: None
        self.window._show_settings_status = lambda message, timeout: messages.append(
            (message, timeout)
        )

        from gui_qt.font_worker import FontInstallResult

        self.window._on_recommended_fonts_downloaded(
            FontInstallResult(False, error="network down")
        )

        self.assertIsNone(self.window._font_install_worker)
        self.assertTrue(worker.deleted)
        self.assertFalse(progress.visible)
        self.assertIn("系统字体回退", label.text)
        self.assertIn("network down", label.text)
        self.assertIn("下载失败", messages[-1][0])

    def test_theme_change_previews_without_persisting(self):
        class FakeCombo:
            def currentData(self):
                return "dark"

        calls = []
        self.window._loading_config_to_ui = False
        self.window._loading_theme_to_ui = False
        self.window.theme_combo = FakeCombo()
        self.window._set_theme_preference = lambda preference, persist: calls.append(
            (preference, persist)
        )

        self.window._on_theme_changed(0)

        self.assertEqual(calls, [("dark", False)])

    def test_load_config_reads_legacy_context_storage_location(self):
        config = {
            "context_storage_location": "game",
            "sync": {"rag": {}},
            "batch": {"rag": {}, "source_index": {}, "model": "gemini-3.1-flash-lite"},
        }

        class FakeState:
            def load_translator_config(self):
                return config

        class FakeCheckBox:
            def __init__(self):
                self._checked = False
            def isChecked(self):
                return self._checked
            def setChecked(self, checked):
                self._checked = checked

        class FakeCombo:
            def __init__(self, text=""):
                self._text = text
                self._index = -1
            def findData(self, _data):
                return 0
            def findText(self, _text):
                return -1
            def setCurrentIndex(self, index):
                self._index = index
            def addItem(self, text, _data=None):
                self._text = text
            def count(self):
                return 1
            def currentText(self):
                return self._text
            def currentData(self):
                return ""
            def setEnabled(self, _enabled):
                pass

        self.window.state = FakeState()
        self.window._loading_config_to_ui = False
        self.window._batch_thinking_user_changed = False
        self.window._updating_batch_thinking_combo = False
        self.window.rag_enabled_cb = FakeCheckBox()
        self.window.source_index_enabled_cb = FakeCheckBox()
        self.window.bootstrap_on_build_cb = FakeCheckBox()
        self.window.context_storage_game_cb = FakeCheckBox()
        self.window.theme_combo = FakeCombo()
        self.window.sync_model_combo = FakeCombo()
        self.window.batch_model_combo = FakeCombo()
        self.window.sync_embedding_combo = FakeCombo()
        self.window.batch_embedding_combo = FakeCombo()
        self.window.batch_thinking_combo = FakeCombo()

        self.window._load_config_to_ui()

        self.assertTrue(self.window.context_storage_game_cb.isChecked())

    def test_load_config_normalizes_context_storage_location_aliases(self):
        config = {
            "context_storage": {"location": "game_dir"},
            "sync": {"rag": {}},
            "batch": {"rag": {}, "source_index": {}, "model": "gemini-3.1-flash-lite"},
        }

        class FakeState:
            def load_translator_config(self):
                return config

        class FakeCheckBox:
            def __init__(self):
                self._checked = False
            def isChecked(self):
                return self._checked
            def setChecked(self, checked):
                self._checked = checked

        class FakeCombo:
            def __init__(self, text=""):
                self._text = text
                self._index = -1
            def findData(self, _data):
                return 0
            def findText(self, _text):
                return -1
            def setCurrentIndex(self, index):
                self._index = index
            def addItem(self, text, _data=None):
                self._text = text
            def count(self):
                return 1
            def currentText(self):
                return self._text
            def currentData(self):
                return ""
            def setEnabled(self, _enabled):
                pass

        self.window.state = FakeState()
        self.window._loading_config_to_ui = False
        self.window._batch_thinking_user_changed = False
        self.window._updating_batch_thinking_combo = False
        self.window.rag_enabled_cb = FakeCheckBox()
        self.window.source_index_enabled_cb = FakeCheckBox()
        self.window.bootstrap_on_build_cb = FakeCheckBox()
        self.window.context_storage_game_cb = FakeCheckBox()
        self.window.theme_combo = FakeCombo()
        self.window.sync_model_combo = FakeCombo()
        self.window.batch_model_combo = FakeCombo()
        self.window.sync_embedding_combo = FakeCombo()
        self.window.batch_embedding_combo = FakeCombo()
        self.window.batch_thinking_combo = FakeCombo()

        self.window._load_config_to_ui()

        self.assertTrue(self.window.context_storage_game_cb.isChecked())

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

    def test_refresh_workflow_from_latest_retry_warn_shows_actionable_state(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.BATCH_TRANSLATION,
            manifest={
                "job_name": "batches/retry",
                "job_state": "JOB_STATE_SUCCEEDED",
                "retry_of_manifest": r"C:\package\manifest.json",
                "last_check_summary": {"safety_level": "warn"},
            },
            latest_manifest=Path(r"C:\package\retry_parts\retry1\manifest.json"),
        )
        self.window._split_entries_for_manifest = lambda *_args, **_kwargs: []
        self.window._render_split_status_entries = lambda *_args, **_kwargs: None

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        self.assertEqual(status, "stale")
        self.assertIn("补译结果仍需处理", heading)
        self.assertIn("暂不能合并", message)
        self.assertIn(r"父任务：C:\package\manifest.json", facts)
        self.assertTrue(self.window.timeline.visible)
        self.assertEqual(self.window.timeline.current_step_key, "check")
        self.assertEqual(self.window.timeline.status, "stale")

    def test_refresh_workflow_from_latest_retry_safe_prompts_merge(self):
        from gui_qt.work_modes import WorkMode

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.BATCH_TRANSLATION,
            manifest={
                "job_name": "batches/retry",
                "job_state": "JOB_STATE_SUCCEEDED",
                "retry_of_manifest": r"C:\package\manifest.json",
                "last_check_summary": {"safety_level": "safe"},
            },
            latest_manifest=Path(r"C:\package\retry_parts\retry1\manifest.json"),
        )
        self.window._split_entries_for_manifest = lambda *_args, **_kwargs: []
        self.window._render_split_status_entries = lambda *_args, **_kwargs: None

        self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, _facts = summary_calls[0]
        self.assertEqual(status, "ready")
        self.assertIn("补译后续处理", heading)
        self.assertIn("继续补译", message)
        self.assertEqual(self.window.timeline.current_step_key, "merge-retry")

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
        from gui_qt.work_modes import WorkMode, work_mode_spec
        from unittest.mock import patch, MagicMock

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.KEYWORD_EXTRACTION,
            manifest={"job_state": "done"},
        )

        mock_workflow = MagicMock()
        mock_workflow.current_step.return_value = None

        with patch("gui_qt.manifest_resume_summary.resume_workflow", return_value=mock_workflow):
            self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        spec = work_mode_spec(WorkMode.KEYWORD_EXTRACTION)
        self.assertEqual(status, "idle")
        self.assertEqual(heading, spec.idle_workflow_heading)
        self.assertTrue(any("已完成" in fact for fact in (facts or [])))
        self.assertIsNotNone(self.window._completed_manifest_snapshot)
        self.assertFalse(self.window.timeline.visible)

    def test_refresh_workflow_from_latest_manifest_warn_stops_at_check_step(self):
        from gui_qt.work_modes import WorkMode
        from unittest.mock import MagicMock, patch

        summary_calls = self._prepare_resume_refresh(
            work_mode=WorkMode.BATCH_TRANSLATION,
            manifest={
                "job_name": "batches/example",
                "job_state": "JOB_STATE_SUCCEEDED",
                "last_check_summary": {"safety_level": "warn"},
            },
        )
        self.window._split_entries_for_manifest = lambda *_args, **_kwargs: []
        self.window._render_split_status_entries = lambda *_args, **_kwargs: None
        mock_workflow = MagicMock()
        mock_workflow.current_step.return_value = None

        with patch("gui_qt.workflow_factory.resume_workflow", return_value=mock_workflow):
            self.window._refresh_workflow_from_latest_manifest()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, _facts = summary_calls[0]
        self.assertEqual(status, "stale")
        self.assertIn("需要先处理问题", heading)
        self.assertIn("暂不能写回", message)
        self.assertTrue(self.window.timeline.visible)
        self.assertEqual(self.window.timeline.current_step_key, "check")
        self.assertEqual(self.window.timeline.status, "stale")

    def test_resume_completed_keyword_manifest_shows_result_summary(self):
        from gui_qt.work_modes import WorkMode, work_mode_spec
        from unittest.mock import MagicMock, patch

        self.window.kill_btn = type("FakeBtn", (), {"isEnabled": lambda _self: False})()
        self.window._completed_manifest_snapshot = None
        self.window._viewing_completed_manifest = False
        self.window._split_status_entries = []
        self.window._split_status_selected_manifest_path = ""
        self.window._split_entries_for_manifest = lambda *_args, **_kwargs: []
        self.window._render_split_status_entries = lambda *_args, **_kwargs: None
        self.window._update_completed_manifest_entry_ui = lambda: None
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

        with (
            patch("gui_qt.app.resume_workflow", return_value=mock_workflow),
            patch("gui_qt.manifest_resume_summary.resume_workflow", return_value=mock_workflow),
        ):
            self.window._on_resume_translation()

        self.assertEqual(len(summary_calls), 1)
        status, heading, message, facts = summary_calls[0]
        spec = work_mode_spec(WorkMode.KEYWORD_EXTRACTION)
        self.assertEqual(status, "idle")
        self.assertEqual(heading, spec.idle_workflow_heading)
        self.assertTrue(any("已完成" in fact for fact in (facts or [])))
        self.assertIsNotNone(self.window._completed_manifest_snapshot)
        self.assertFalse(self.window.timeline.visible)
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
        self.window._show_workbench_log_drawer = lambda: None
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
        self.window.kill_btn = type("FakeBtn", (), {"isEnabled": lambda _self: False})()
        self.window._refresh_workflow_from_latest_manifest = MagicMock()
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

    def test_sync_keyword_finish_copies_reports_from_sync_run_output(self):
        from gui_qt.translation_workflow import WorkflowStep, WorkflowUpdate
        from gui_qt.work_modes import WorkMode
        from unittest.mock import MagicMock

        class FakeWorkflow:
            manifest_path = ""

            def current_step(self):
                return WorkflowStep(
                    "sync-keywords",
                    ["sync-keywords"],
                    "正在同步提取关键词",
                    "正在扫描翻译文本并生成术语与剧情报告。",
                )

            def complete_current_step(self, exit_code, output):
                return WorkflowUpdate(
                    status="done",
                    heading="同步关键词提取完成",
                    message="done",
                    facts=[],
                    should_continue=False,
                )

        class FakeStatusBar:
            def __init__(self):
                self.messages = []
            def showMessage(self, text, timeout):
                self.messages.append((text, timeout))

        output = "Sync keyword run: C:/dummy/sync_keywords\nKeyword candidates: 2 deduped from 3 raw"
        self.window._workflow = FakeWorkflow()
        self.window._workflow_step_output_lines = output.splitlines()
        self.window._current_work_mode = lambda: WorkMode.SYNC_KEYWORD_EXTRACTION
        self.window.kill_btn = type("FakeBtn", (), {"isEnabled": lambda _self: False})()
        self.window._refresh_workflow_from_latest_manifest = MagicMock()
        self.window._copy_sync_keyword_reports_to_game_parent = MagicMock()
        self.window._copy_keyword_reports_to_game_parent = MagicMock()
        self.window._uses_revision_writeback = lambda: False
        self.window._set_workflow_update = MagicMock()
        self.window._refresh_diagnostics_context = MagicMock()
        self.window._set_task_running = MagicMock()
        self.window._set_writeback_summary = MagicMock()
        status_bar = FakeStatusBar()
        self.window.statusBar = lambda: status_bar

        self.window._on_workflow_step_finished(0)

        self.window._copy_sync_keyword_reports_to_game_parent.assert_called_once_with(output)
        self.window._copy_keyword_reports_to_game_parent.assert_not_called()
        self.window._set_writeback_summary.assert_called_once()
        self.assertIn("同步关键词提取完成", status_bar.messages[-1][0])

    def test_leaving_dirty_config_tab_can_stay_on_config(self):
        from unittest.mock import patch

        class FakeMessageBox:
            class Icon:
                Warning = object()
            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            shown = False

            def __init__(self, parent=None):
                self._stay_btn = object()
                self._buttons = []
            def setIcon(self, icon):
                pass
            def setWindowTitle(self, title):
                pass
            def setText(self, text):
                pass
            def setInformativeText(self, text):
                pass
            def addButton(self, text, role):
                button = self._stay_btn if text == "留在设置页" else object()
                self._buttons.append((text, button))
                return button
            def setDefaultButton(self, button):
                pass
            def exec(self):
                FakeMessageBox.shown = True
            def clickedButton(self):
                return self._stay_btn

        class FakeTabs:
            def __init__(self, config_tab, other_tab):
                self.config_tab = config_tab
                self.other_tab = other_tab
                self.current_index = 1
            def widget(self, index):
                return self.config_tab if index == 1 else self.other_tab
            def setCurrentIndex(self, index):
                self.current_index = index

        config_tab = object()
        other_tab = object()
        self.window._config_tab = config_tab
        self.window._diagnostics_tab = object()
        self.window.tab_widget = FakeTabs(config_tab, other_tab)
        self.window._last_main_tab_index = 1
        self.window._handling_config_tab_leave = False
        self.window._loading_config_to_ui = False
        self.window._config_ui_saved_snapshot = {"dirty": False}
        self.window._current_config_ui_snapshot = lambda: {"dirty": True}
        self.window._refresh_api_status = lambda: None
        self.window._refresh_diagnostics_context = lambda: None

        with patch("gui_qt.app.QMessageBox", FakeMessageBox):
            self.window._on_tab_changed(0)

        self.assertTrue(FakeMessageBox.shown)
        self.assertEqual(self.window.tab_widget.current_index, 1)
        self.assertEqual(self.window._last_main_tab_index, 1)

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

    def test_refresh_writeback_from_latest_translation_manifest_syncs_split_status(self):
        from gui_qt.work_modes import WorkMode

        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION

        class FakeState:
            def get_game_root(self):
                return Path("C:/dummy/project")

            def get_latest_manifest_path_for_mode(self, game_root, mode):
                return Path("C:/dummy/split_parts/part02_of_10/manifest.json")

            def load_resume_manifest(self, path, work_mode):
                return {
                    "_manifest_path": str(path),
                    "last_check_summary": {
                        "safety_level": "warn",
                        "pending_files": 12,
                        "pending_lines": 11813,
                        "failure_items": 12,
                    },
                }

        self.window.state = FakeState()
        summaries = []
        split_refreshes = []
        self.window._set_writeback_summary = lambda summary: summaries.append(summary)
        self.window._refresh_split_status_ui = (
            lambda **kwargs: split_refreshes.append(kwargs)
        )

        self.window._refresh_writeback_from_latest_manifest()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].status, "warn")
        self.assertEqual(len(split_refreshes), 1)
        self.assertEqual(
            split_refreshes[0]["manifest_path"],
            str(Path("C:/dummy/split_parts/part02_of_10/manifest.json")),
        )

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
            "recheck_btn",
            "check_issues_btn",
            "retry_btn",
            "retry_followup_btn",
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

    def test_resolve_keyword_merge_candidates_path_uses_latest_keyword_manifest(self):
        from pathlib import Path
        from unittest.mock import patch

        from gui_qt.work_modes import WorkMode
        from project_asset_paths import canonical_abs_path

        jsonl_path = "C:/dummy/game/logs/keyword_candidates.jsonl"
        keyword_manifest_path = "C:/dummy/game/logs/keyword_manifest.json"
        keyword_manifest = {
            "mode": "keyword_extraction",
            "keyword_export": {"jsonl_path": jsonl_path},
        }
        expected_manifest_path = canonical_abs_path(keyword_manifest_path).lower()

        class FakeState:
            def get_game_root(self):
                return Path("C:/dummy/game")

            def get_latest_manifest_path_for_mode(self, game_root, mode):
                if mode == WorkMode.KEYWORD_EXTRACTION:
                    return Path(keyword_manifest_path)
                return None

        def load_manifest(path):
            if (
                path
                and canonical_abs_path(path).lower() == expected_manifest_path
            ):
                return keyword_manifest
            return None

        self.window.state = FakeState()
        self.window._keyword_merge_candidates_path = ""
        self.window._writeback_manifest_path = ""
        self.window._workflow = None
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._load_diagnostics_manifest = load_manifest

        with patch("os.path.isfile", return_value=True):
            resolved = self.window._resolve_keyword_merge_candidates_path()

        self.assertEqual(resolved, jsonl_path)

    def test_resolve_keyword_merge_ignores_stale_cached_path_from_other_project(self):
        from pathlib import Path
        from unittest.mock import patch

        from gui_qt.work_modes import WorkMode
        from project_asset_paths import canonical_abs_path

        jsonl_path = "C:/dummy/game/logs/keyword_candidates.jsonl"
        keyword_manifest_path = "C:/dummy/game/logs/keyword_manifest.json"
        keyword_manifest = {
            "mode": "keyword_extraction",
            "keyword_export": {"jsonl_path": jsonl_path},
        }
        expected_manifest_path = canonical_abs_path(keyword_manifest_path).lower()

        class FakeState:
            def get_game_root(self):
                return Path("C:/dummy/game")

            def get_latest_manifest_path_for_mode(self, game_root, mode):
                if mode == WorkMode.KEYWORD_EXTRACTION:
                    return Path(keyword_manifest_path)
                return None

        def load_manifest(path):
            if (
                path
                and canonical_abs_path(path).lower() == expected_manifest_path
            ):
                return keyword_manifest
            return None

        self.window.state = FakeState()
        self.window._keyword_merge_candidates_path = "C:/other/project/keyword_candidates.jsonl"
        self.window._writeback_manifest_path = ""
        self.window._workflow = None
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._load_diagnostics_manifest = load_manifest

        with patch("os.path.isfile", return_value=True):
            resolved = self.window._resolve_keyword_merge_candidates_path()

        self.assertEqual(resolved, jsonl_path)
        self.assertEqual(self.window._keyword_merge_candidates_path, "")

    def test_recheck_button_enabled_for_batch_translation_with_manifest(self):
        from gui_qt.check_report import WritebackSummary
        from gui_qt.work_modes import WorkMode

        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._uses_revision_writeback = lambda *_args, **_kwargs: False
        self.window._writeback_manifest_path = "C:/dummy/manifest.json"
        self.window._retry_followup_confirmed = set()
        self.window._load_writeback_manifest = lambda: None

        class FakeButton:
            def __init__(self):
                self.visible = False
                self.enabled = False

            def setVisible(self, visible):
                self.visible = visible

            def setEnabled(self, enabled):
                self.enabled = enabled

            def setText(self, text):
                pass

            def setToolTip(self, text):
                pass

        for name in (
            "apply_btn",
            "apply_revision_btn",
            "recheck_btn",
            "check_issues_btn",
            "retry_btn",
            "retry_followup_btn",
            "apply_failure_btn",
            "remediation_btn",
        ):
            setattr(self.window, name, FakeButton())

        self.window._update_writeback_action_buttons(
            WritebackSummary(
                status="warn",
                heading="需要先处理问题",
                message="warn",
                facts=[],
                findings=[],
                can_apply=False,
                manifest_path="C:/dummy/manifest.json",
            ),
            running=False,
        )

        self.assertTrue(self.window.recheck_btn.visible)
        self.assertTrue(self.window.recheck_btn.enabled)
        self.assertTrue(self.window.check_issues_btn.enabled)

    def test_retry_followup_button_enabled_after_confirmed_preview(self):
        from gui_qt.check_report import WritebackSummary
        from gui_qt.work_modes import WorkMode

        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._uses_revision_writeback = lambda *_args, **_kwargs: False
        self.window._writeback_manifest_path = "C:/dummy/manifest.json"
        self.window._retry_followup_confirmed = {"C:/dummy/manifest.json"}

        class FakeState:
            def load_manifest_file(self, path):
                if path == "C:/dummy/retry/manifest.json":
                    return {
                        "retry_of_manifest": "C:/dummy/manifest.json",
                        "job_name": "",
                    }
                return {
                    "last_check_summary": {"safety_level": "warn"},
                    "last_retry_manifest_path": "C:/dummy/retry/manifest.json",
                }

        self.window.state = FakeState()

        class FakeButton:
            def __init__(self):
                self.visible = False
                self.enabled = False
                self.text = ""
                self.tool_tip = ""

            def setVisible(self, visible):
                self.visible = visible

            def setEnabled(self, enabled):
                self.enabled = enabled

            def setText(self, text):
                self.text = text

            def setToolTip(self, text):
                self.tool_tip = text

        for name in (
            "apply_btn",
            "apply_revision_btn",
            "recheck_btn",
            "check_issues_btn",
            "retry_btn",
            "retry_followup_btn",
            "apply_failure_btn",
            "remediation_btn",
        ):
            setattr(self.window, name, FakeButton())

        self.window._load_writeback_manifest = lambda: FakeState().load_manifest_file(
            "C:/dummy/manifest.json"
        )

        self.window._update_writeback_action_buttons(
            WritebackSummary(
                status="warn",
                heading="需要先处理问题",
                message="warn",
                facts=[],
                findings=[],
                can_apply=False,
                manifest_path="C:/dummy/manifest.json",
            ),
            running=False,
        )

        self.assertTrue(self.window.retry_followup_btn.visible)
        self.assertTrue(self.window.retry_followup_btn.enabled)
        self.assertEqual(self.window.retry_followup_btn.text, "提交补译任务")

    def test_recheck_button_disabled_while_running(self):
        from gui_qt.check_report import WritebackSummary
        from gui_qt.work_modes import WorkMode

        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._uses_revision_writeback = lambda *_args, **_kwargs: False
        self.window._writeback_manifest_path = "C:/dummy/manifest.json"
        self.window._retry_followup_confirmed = set()
        self.window._load_writeback_manifest = lambda: None

        class FakeButton:
            def __init__(self):
                self.visible = False
                self.enabled = False

            def setVisible(self, visible):
                self.visible = visible

            def setEnabled(self, enabled):
                self.enabled = enabled

            def setText(self, text):
                pass

            def setToolTip(self, text):
                pass

        self.window.recheck_btn = FakeButton()
        self.window.apply_btn = FakeButton()
        self.window.apply_revision_btn = FakeButton()
        self.window.check_issues_btn = FakeButton()
        self.window.retry_btn = FakeButton()
        self.window.retry_followup_btn = FakeButton()
        self.window.apply_failure_btn = FakeButton()
        self.window.remediation_btn = FakeButton()

        self.window._update_writeback_action_buttons(
            WritebackSummary(
                status="safe",
                heading="可以写回翻译",
                message="safe",
                facts=[],
                findings=[],
                can_apply=True,
                manifest_path="C:/dummy/manifest.json",
            ),
            running=True,
        )

        self.assertTrue(self.window.recheck_btn.visible)
        self.assertFalse(self.window.recheck_btn.enabled)

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

    def test_copy_sync_keyword_reports_to_game_parent_copies_successfully(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            game_root = tmp_path / "Game" / "work"
            game_root.mkdir(parents=True, exist_ok=True)

            sync_dir = tmp_path / "logs" / "sync_runs" / "job_sync_keywords"
            sync_dir.mkdir(parents=True, exist_ok=True)
            (sync_dir / "keyword_candidates.md").write_text("candidates", encoding="utf-8")
            (sync_dir / "keyword_chunk_summaries.md").write_text("summaries", encoding="utf-8")

            class FakeState:
                def get_game_root(self):
                    return game_root
            self.window.state = FakeState()

            logged_messages = []
            self.window._append_log = lambda msg: logged_messages.append(msg)

            self.window._copy_sync_keyword_reports_to_game_parent(
                f"Sync keyword run: {sync_dir}\n"
                "Keyword candidates: 2 deduped from 3 raw\n"
            )

            target_dir = tmp_path / "Game" / "extracted_keywords"
            self.assertTrue((target_dir / "keyword_candidates.md").exists())
            self.assertTrue((target_dir / "keyword_chunk_summaries.md").exists())
            self.assertEqual((target_dir / "keyword_candidates.md").read_text(encoding="utf-8"), "candidates")
            self.assertTrue(any("已将关键词提取报告复制一份至" in msg for msg in logged_messages))

    def test_sync_layout_sizes_skips_scrollable_doctor_labels(self):
        class RaisingLabel:
            def __init__(self, text):
                self._text = text

            def text(self):
                return self._text

            def isVisible(self):
                return True

            def fontMetrics(self):
                raise AssertionError("scrollable doctor labels should not be measured")

            def minimumHeight(self):
                return 0

            def setMinimumHeight(self, height):
                raise AssertionError("scrollable doctor labels should not be resized")

        class FakeRect:
            def height(self):
                return 20

        class FakeMetrics:
            def boundingRect(self, *args):
                return FakeRect()

        class MeasuredLabel:
            def __init__(self, text=""):
                self._text = text
                self.minimum_height = 0
                self.measured = False

            def text(self):
                return self._text

            def isVisible(self):
                return bool(self._text)

            def fontMetrics(self):
                self.measured = True
                return FakeMetrics()

            def minimumHeight(self):
                return self.minimum_height

            def setMinimumHeight(self, height):
                self.minimum_height = height

        class FakeLayout:
            def __init__(self):
                self.invalidated = False

            def invalidate(self):
                self.invalidated = True

        class FakeWidget:
            def __init__(self):
                self._layout = FakeLayout()

            def layout(self):
                return self._layout

        class FakeTabs:
            def __init__(self):
                self.widget = FakeWidget()
                self.updated = False

            def width(self):
                return 320

            def currentWidget(self):
                return self.widget

            def updateGeometry(self):
                self.updated = True

        tabs = FakeTabs()
        self.window.workbench_status_tabs = tabs
        self.window.doctor_message_label = RaisingLabel("doctor message")
        self.window.doctor_facts_label = RaisingLabel("doctor facts")
        self.window.doctor_details_label = RaisingLabel("doctor details")
        self.window.workflow_message_label = MeasuredLabel("workflow message")
        self.window.workflow_facts_label = MeasuredLabel()
        self.window.writeback_message_label = MeasuredLabel()
        self.window.writeback_facts_label = MeasuredLabel()
        self.window.writeback_details_label = MeasuredLabel()

        self.window._sync_layout_sizes()

        self.assertTrue(self.window.workflow_message_label.measured)
        self.assertEqual(self.window.workflow_message_label.minimum_height, 24)
        self.assertTrue(tabs.widget.layout().invalidated)
        self.assertTrue(tabs.updated)

    def test_set_work_mode_clears_completed_manifest_snapshot(self):
        from gui_qt.work_modes import WorkMode

        self.window._work_mode = WorkMode.BATCH_TRANSLATION
        self.window._mode_sessions = {}
        self.window._workflow = None
        self.window._workflow_step_output_lines = []
        self.window._writeback_manifest_path = ""
        self.window._keyword_merge_candidates_path = ""
        self.window._completed_manifest_snapshot = {"manifest_path": "C:/dummy/manifest.json"}
        self.window._viewing_completed_manifest = True
        cleared = []
        self.window._clear_completed_manifest_snapshot = lambda: cleared.append(True)
        self.window._apply_work_mode_ui = lambda **_kwargs: None

        self.window._set_work_mode(WorkMode.KEYWORD_EXTRACTION, refresh_manifest_writeback=False)

        # Switching into a mode without a prior session clears active completed snapshot.
        self.assertEqual(cleared, [True])

if __name__ == "__main__":
    unittest.main()
