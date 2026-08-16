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
    def test_cli_suite_excludes_gui_modules(self):
        names = {case.id() for case in _iter_cases(build_cli_suite())}
        self.assertTrue(names)
        self.assertFalse(
            any(module.startswith("test_gui_") for module in _module_names(names))
        )

    def test_gui_suite_only_includes_gui_modules(self):
        names = {case.id() for case in _iter_cases(build_gui_suite())}
        self.assertTrue(names)
        self.assertTrue(
            all(module.startswith("test_gui_") for module in _module_names(names))
        )

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
