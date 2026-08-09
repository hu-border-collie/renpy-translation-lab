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


if __name__ == "__main__":
    unittest.main()
