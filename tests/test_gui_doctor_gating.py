import unittest
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.doctor_worker import DoctorWorker
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiDoctorGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_invalidate_doctor_worker_clears_task_running(self):
        window = MainWindow()
        window._active_command = "doctor"
        window._set_task_running(True)
        worker = mock.Mock(spec=DoctorWorker)
        worker.isRunning.return_value = False
        window._doctor_worker = worker

        window._invalidate_doctor_worker()

        self.assertIsNone(window._doctor_worker)
        self.assertEqual(window._active_command, "")
        self.assertFalse(window.kill_btn.isEnabled())

    def test_kill_during_doctor_check_cancels_worker_and_unlocks_ui(self):
        window = MainWindow()
        window._active_command = "doctor"
        window._set_task_running(True)
        worker = mock.Mock(spec=DoctorWorker)
        worker.isRunning.return_value = True
        window._doctor_worker = worker

        window._on_kill()

        self.assertIsNone(window._doctor_worker)
        self.assertFalse(window._doctor_check_completed)
        self.assertFalse(window.kill_btn.isEnabled())