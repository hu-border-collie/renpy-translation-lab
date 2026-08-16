"""Application shutdown coordinator regression tests."""
from __future__ import annotations

import unittest
from unittest import mock

from tests import gui_test_support

try:
    from gui_qt.lifecycle import CallbackShutdownParticipant, ShutdownCoordinator
except ImportError as exc:
    ShutdownCoordinator = None  # type: ignore[assignment,misc]
    CallbackShutdownParticipant = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(ShutdownCoordinator is None, IMPORT_ERROR)
class ShutdownCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = ShutdownCoordinator()

    def _register(self, key: str, label: str, state: dict[str, object]) -> None:
        self.coordinator.register(
            CallbackShutdownParticipant(
                key=key,
                label=label,
                active_callback=lambda: bool(state["active"]),
                shutdown_callback=lambda: state.__setitem__(
                    "requests",
                    int(state["requests"]) + 1,
                ),
            )
        )

    def test_begin_requests_each_active_participant_and_settles_after_real_finish(self):
        first = {"active": True, "requests": 0}
        second = {"active": False, "requests": 0}
        self._register("first", "环境检查", first)
        self._register("second", "模型加载", second)
        settled = []
        self.coordinator.settled.connect(lambda: settled.append(True))

        self.assertTrue(self.coordinator.begin(timeout_ms=5000))

        self.assertEqual(first["requests"], 1)
        self.assertEqual(second["requests"], 0)
        self.assertTrue(self.coordinator.in_progress)
        self.assertEqual(self.coordinator.active_labels(), ("环境检查",))

        first["active"] = False
        self.coordinator.check_now()
        self.assertEqual(settled, [True])
        self.assertFalse(self.coordinator.in_progress)

    def test_deadline_reports_stall_but_does_not_fake_completion(self):
        state = {"active": True, "requests": 0}
        self._register("worker", "字体下载", state)
        stalled = []
        settled = []
        self.coordinator.stalled.connect(stalled.append)
        self.coordinator.settled.connect(lambda: settled.append(True))

        self.coordinator.begin(timeout_ms=5000)
        self.coordinator._report_stalled()

        self.assertEqual(stalled, [("字体下载",)])
        self.assertEqual(settled, [])
        self.assertTrue(self.coordinator.in_progress)

        state["active"] = False
        self.coordinator.check_now()
        self.assertEqual(settled, [True])

    def test_participant_activated_during_shutdown_is_cancelled(self):
        first = {"active": True, "requests": 0}
        second = {"active": False, "requests": 0}
        self._register("first", "当前任务", first)
        self._register("second", "后续任务", second)

        self.coordinator.begin(timeout_ms=5000)
        first["active"] = False
        second["active"] = True
        self.coordinator.check_now()

        self.assertEqual(second["requests"], 1)
        self.assertTrue(self.coordinator.in_progress)

        second["active"] = False
        self.coordinator.check_now()
        self.assertFalse(self.coordinator.in_progress)

    def test_failed_cancellation_retries_without_repeating_error_signal(self):
        state = {"active": True, "requests": 0}

        def request_shutdown() -> None:
            state["requests"] += 1
            if state["requests"] == 1:
                raise RuntimeError("temporary failure")

        self.coordinator.register(
            CallbackShutdownParticipant(
                key="worker",
                label="后台任务",
                active_callback=lambda: bool(state["active"]),
                shutdown_callback=request_shutdown,
            )
        )
        failures = []
        self.coordinator.cancellation_failed.connect(
            lambda label, error: failures.append((label, error))
        )

        self.coordinator.begin(timeout_ms=5000)
        self.assertEqual(state["requests"], 1)
        self.assertEqual(failures, [("后台任务", "temporary failure")])

        self.coordinator.check_now()
        self.assertEqual(state["requests"], 2)
        self.assertEqual(failures, [("后台任务", "temporary failure")])

        state["active"] = False
        self.coordinator.check_now()
        self.assertFalse(self.coordinator.in_progress)

    def test_unexpected_probe_failure_reports_once_and_fails_closed(self):
        state = {"raises": True, "active": False, "requests": 0}

        def is_active() -> bool:
            if state["raises"]:
                raise ValueError("probe failed")
            return bool(state["active"])

        self.coordinator.register(
            CallbackShutdownParticipant(
                key="worker",
                label="后台任务",
                active_callback=is_active,
                shutdown_callback=lambda: state.__setitem__(
                    "requests",
                    int(state["requests"]) + 1,
                ),
            )
        )
        failures = []
        self.coordinator.cancellation_failed.connect(
            lambda label, error: failures.append((label, error))
        )

        self.coordinator.begin(timeout_ms=5000)
        self.coordinator.check_now()
        self.coordinator.check_now()

        self.assertTrue(self.coordinator.in_progress)
        self.assertEqual(state["requests"], 1)
        self.assertEqual(failures, [("后台任务", "probe failed")])

        state["raises"] = False
        self.coordinator.check_now()
        self.assertFalse(self.coordinator.in_progress)

    def test_disappeared_qt_wrapper_probe_errors_are_inactive(self):
        for error_type in (RuntimeError, TypeError):
            with self.subTest(error_type=error_type.__name__):
                coordinator = ShutdownCoordinator()
                requests = []
                failures = []
                coordinator.register(
                    CallbackShutdownParticipant(
                        key="worker",
                        label="后台任务",
                        active_callback=mock.Mock(side_effect=error_type("wrapper deleted")),
                        shutdown_callback=lambda requests=requests: requests.append(True),
                    )
                )
                coordinator.cancellation_failed.connect(
                    lambda label, error, failures=failures: failures.append(
                        (label, error)
                    )
                )

                self.assertTrue(coordinator.begin(timeout_ms=5000))

                self.assertFalse(coordinator.in_progress)
                self.assertEqual(requests, [])
                self.assertEqual(failures, [])

    def test_duplicate_participant_key_is_rejected(self):
        state = {"active": False, "requests": 0}
        self._register("same", "一", state)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._register("same", "二", state)


@gui_test_support.skip_unless_gui(ShutdownCoordinator is None, IMPORT_ERROR)
class CloseMainWindowHelperTests(unittest.TestCase):
    @mock.patch("PySide6.QtWidgets.QApplication.instance")
    def test_synchronous_close_does_not_drain_pending_focus_events(self, instance):
        app = instance.return_value
        window = mock.Mock()
        window._shutdown_close_ready = False
        window._shutdown_coordinator.in_progress = False

        gui_test_support.close_main_window(window)

        window.close.assert_called_once_with()
        app.processEvents.assert_not_called()

    @mock.patch("PySide6.QtWidgets.QApplication.instance")
    def test_asynchronous_close_drains_terminal_close_callback(self, instance):
        app = instance.return_value
        window = mock.Mock()
        window._shutdown_close_ready = False
        window._shutdown_coordinator.in_progress = False

        def schedule_terminal_close():
            window._shutdown_close_ready = True

        window.close.side_effect = schedule_terminal_close

        gui_test_support.close_main_window(window)

        self.assertEqual(app.processEvents.call_count, 3)


@gui_test_support.skip_unless_gui(ShutdownCoordinator is None, IMPORT_ERROR)
class GuiTestRuntimeShutdownTests(unittest.TestCase):
    @mock.patch("PySide6.QtCore.QCoreApplication.sendPostedEvents")
    @mock.patch("PySide6.QtCore.QThreadPool.globalInstance")
    def test_runtime_shutdown_drains_pool_and_deferred_deletes(
        self,
        global_pool,
        send_posted_events,
    ):
        pool = global_pool.return_value
        pool.waitForDone.return_value = True
        widget = mock.Mock()
        app = mock.Mock()
        app.topLevelWidgets.return_value = [widget]

        stopped = gui_test_support.shutdown_gui_test_runtime(app, wait_ms=321)

        self.assertTrue(stopped)
        pool.clear.assert_called_once_with()
        pool.waitForDone.assert_called_once_with(321)
        widget.hide.assert_called_once_with()
        widget.deleteLater.assert_called_once_with()
        send_posted_events.assert_called_once()
        app.processEvents.assert_called_once_with()
        app.shutdown.assert_not_called()

    @mock.patch("PySide6.QtCore.QCoreApplication.sendPostedEvents")
    @mock.patch("PySide6.QtCore.QThreadPool.globalInstance")
    def test_runtime_shutdown_reports_unfinished_pool(
        self,
        global_pool,
        _send_posted_events,
    ):
        global_pool.return_value.waitForDone.return_value = False
        app = mock.Mock()
        app.topLevelWidgets.return_value = []

        self.assertFalse(gui_test_support.shutdown_gui_test_runtime(app, wait_ms=0))


if __name__ == "__main__":
    unittest.main()
