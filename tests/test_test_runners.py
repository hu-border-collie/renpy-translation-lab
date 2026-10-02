"""Smoke tests for CLI/GUI unittest discovery helpers."""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import run_gui_tests
from gui_test_support import GuiTestModalGuard
from run_cli_tests import build_suite as build_cli_suite
from run_gui_tests import build_suite as build_gui_suite


def _iter_cases(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def _module_names(test_ids: set[str]) -> set[str]:
    return {test_id.split(".", 1)[0] for test_id in test_ids}


class TestDiscoveryRunners(unittest.TestCase):
    def test_runner_isolates_developer_translator_config(self):
        from runtime_test_isolation import (
            isolate_developer_translator_config,
            isolated_translator_config_path,
        )
        import translator_runtime as runtime
        from test_runner_common import repo_root

        isolate_developer_translator_config()
        isolated = isolated_translator_config_path()
        self.assertEqual(pathlib.Path(runtime.TRANSLATOR_CONFIG), isolated)
        self.assertNotEqual(isolated, repo_root() / "translator_config.json")
        self.assertEqual(isolated.read_text(encoding="utf-8").strip(), "{}")

    def test_cli_and_gui_suites_cover_every_case_exactly_once(self):
        cli_ids = [case.id() for case in _iter_cases(build_cli_suite())]
        gui_ids = [case.id() for case in _iter_cases(build_gui_suite())]
        full_suite = unittest.TestLoader().discover(str(_TESTS_DIR), pattern="test_*.py")
        full_ids = [case.id() for case in _iter_cases(full_suite)]

        self.assertTrue(cli_ids)
        self.assertTrue(gui_ids)
        self.assertFalse(any("_FailedTest" in test_id for test_id in full_ids))
        self.assertEqual(len(cli_ids + gui_ids), len(set(cli_ids + gui_ids)))
        self.assertEqual(set(cli_ids) | set(gui_ids), set(full_ids))

        # Widget pages use Qt; their contracts and persistence helpers do not.
        gui_modules = _module_names(set(gui_ids))
        cli_modules = _module_names(set(cli_ids))
        self.assertIn("test_gui_translation_workflow", gui_modules)
        self.assertIn("test_settings_models_page", gui_modules)
        self.assertIn("test_settings_workspace_page", gui_modules)
        for module in (
            "test_settings_page_contract",
            "test_settings_registry",
            "test_settings_coordinator",
            "test_settings_save_apply",
        ):
            with self.subTest(module=module):
                self.assertIn(module, cli_modules)

    def test_gui_runner_fails_when_modal_guard_rejects_a_dialog(self):
        guard = mock.Mock(rejected_dialogs=("QMessageBox title='unexpected'",))
        manager = mock.MagicMock()
        manager.__enter__.return_value = guard
        manager.__exit__.return_value = False
        with (
            mock.patch(
                "gui_test_support.guarded_gui_test_environment",
                return_value=manager,
            ),
            mock.patch.object(run_gui_tests, "build_suite", return_value=unittest.TestSuite()),
            mock.patch.object(run_gui_tests, "run_discovered_suite", return_value=0),
            mock.patch(
                "gui_test_support.shutdown_gui_test_runtime",
                return_value=True,
            ),
        ):
            self.assertEqual(run_gui_tests.main([]), 1)

    def test_gui_runner_reads_rejected_dialogs_after_guard_cleanup(self):
        guard = mock.Mock(rejected_dialogs=())
        manager = mock.MagicMock()
        manager.__enter__.return_value = guard

        def reject_during_cleanup(*_args):
            guard.rejected_dialogs = ("QDialog title='teardown'",)
            return False

        manager.__exit__.side_effect = reject_during_cleanup
        with (
            mock.patch(
                "gui_test_support.guarded_gui_test_environment",
                return_value=manager,
            ),
            mock.patch.object(
                run_gui_tests,
                "build_suite",
                return_value=unittest.TestSuite(),
            ),
            mock.patch.object(run_gui_tests, "run_discovered_suite", return_value=0),
            mock.patch(
                "gui_test_support.shutdown_gui_test_runtime",
                return_value=True,
            ),
        ):
            self.assertEqual(run_gui_tests.main([]), 1)

    def test_gui_runner_shuts_down_qt_runtime(self):
        guard = mock.Mock(rejected_dialogs=())
        manager = mock.MagicMock()
        manager.__enter__.return_value = guard
        manager.__exit__.return_value = False
        with (
            mock.patch(
                "gui_test_support.guarded_gui_test_environment",
                return_value=manager,
            ),
            mock.patch(
                "gui_test_support.shutdown_gui_test_runtime",
                return_value=True,
            ) as shutdown,
            mock.patch.object(
                run_gui_tests,
                "build_suite",
                return_value=unittest.TestSuite(),
            ),
            mock.patch.object(run_gui_tests, "run_discovered_suite", return_value=0),
        ):
            self.assertEqual(run_gui_tests.main([]), 0)

        shutdown.assert_called_once_with()

    def test_gui_script_path_skips_qt_teardown_before_hard_exit(self):
        guard = mock.Mock(rejected_dialogs=())
        manager = mock.MagicMock()
        manager.__enter__.return_value = guard
        manager.__exit__.return_value = False
        with (
            mock.patch(
                "gui_test_support.guarded_gui_test_environment",
                return_value=manager,
            ) as guarded,
            mock.patch(
                "gui_test_support.shutdown_gui_test_runtime",
                return_value=True,
            ) as shutdown,
            mock.patch.object(
                run_gui_tests,
                "build_suite",
                return_value=unittest.TestSuite(),
            ),
            mock.patch.object(
                run_gui_tests,
                "run_discovered_suite",
                return_value=0,
            ),
        ):
            self.assertEqual(
                run_gui_tests.main([], shutdown_runtime=False),
                0,
            )

        shutdown.assert_not_called()
        guarded.assert_called_once_with(process_events=False)

    def test_gui_runner_fails_when_qt_pool_does_not_stop(self):
        guard = mock.Mock(rejected_dialogs=())
        manager = mock.MagicMock()
        manager.__enter__.return_value = guard
        manager.__exit__.return_value = False
        with (
            mock.patch(
                "gui_test_support.guarded_gui_test_environment",
                return_value=manager,
            ),
            mock.patch(
                "gui_test_support.shutdown_gui_test_runtime",
                return_value=False,
            ),
            mock.patch.object(
                run_gui_tests,
                "build_suite",
                return_value=unittest.TestSuite(),
            ),
            mock.patch.object(run_gui_tests, "run_discovered_suite", return_value=0),
        ):
            self.assertEqual(run_gui_tests.main([]), 1)

    def test_gui_script_exit_flushes_output_and_preserves_status(self):
        stdout = mock.Mock()
        stderr = mock.Mock()
        with (
            mock.patch.object(run_gui_tests.sys, "stdout", stdout),
            mock.patch.object(run_gui_tests.sys, "stderr", stderr),
            mock.patch.object(run_gui_tests.os, "_exit") as exit_process,
        ):
            run_gui_tests._terminate_process(7)

        stdout.flush.assert_called_once_with()
        stderr.flush.assert_called_once_with()
        exit_process.assert_called_once_with(7)


class GuiTestModalGuardTests(unittest.TestCase):
    def test_rejects_each_modal_once_without_recording_body(self):
        dialog = mock.Mock()
        dialog.windowTitle.return_value = "未保存设置"
        dialog.objectName.return_value = "confirm_close"
        app = mock.Mock()
        app.activeModalWidget.return_value = dialog
        guard = GuiTestModalGuard(app)
        guard.set_current_test("test_gui_example.ExampleTests.test_modal")

        guard.reject_active_modal()
        guard.reject_active_modal()

        dialog.reject.assert_called_once_with()
        self.assertEqual(len(guard.rejected_dialogs), 1)
        self.assertIn("未保存设置", guard.rejected_dialogs[0])
        self.assertIn("test_gui_example", guard.rejected_dialogs[0])
        self.assertNotIn("secret body", guard.rejected_dialogs[0])

    def test_cleanup_hides_and_deletes_leaked_top_level_widgets(self):
        first = mock.Mock()
        second = mock.Mock()
        app = mock.Mock()
        app.topLevelWidgets.return_value = [first, second]
        guard = GuiTestModalGuard(app)

        guard.cleanup_top_levels()

        first.hide.assert_called_once_with()
        first.deleteLater.assert_called_once_with()
        second.hide.assert_called_once_with()
        second.deleteLater.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
