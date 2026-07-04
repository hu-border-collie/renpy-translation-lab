"""Tests for in-process doctor worker."""
from __future__ import annotations

import unittest
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.doctor_worker import DoctorWorker, DoctorWorkerResult, run_doctor_check
except ImportError as exc:
    DoctorWorker = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(DoctorWorker is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiDoctorWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_run_doctor_check_returns_report(self):
        report = {"mode": "existing_tl_only", "layout_status": "ready", "counts": {}}
        with mock.patch(
            "gemini_translate_batch.collect_doctor_report",
            return_value=report,
        ), mock.patch(
            "gemini_translate_batch.load_batch_settings",
        ), mock.patch(
            "gemini_translate_batch.print_doctor_report",
        ), mock.patch(
            "translator_runtime.load_translator_settings",
        ), mock.patch(
            "translator_runtime.load_glossary",
        ):
            result = run_doctor_check()
        self.assertTrue(result.ok)
        self.assertEqual(result.report, report)

    def test_run_doctor_check_returns_error_when_collect_fails(self):
        with mock.patch(
            "gemini_translate_batch.collect_doctor_report",
            side_effect=RuntimeError("boom"),
        ), mock.patch(
            "gemini_translate_batch.load_batch_settings",
        ), mock.patch(
            "translator_runtime.load_translator_settings",
        ), mock.patch(
            "translator_runtime.load_glossary",
        ):
            result = run_doctor_check()
        self.assertFalse(result.ok)
        self.assertIn("boom", result.error)

    def test_worker_delegates_to_run_doctor_check(self):
        worker = DoctorWorker()
        payload = DoctorWorkerResult(True, {"mode": "ready"}, "log")
        with mock.patch(
            "gui_qt.doctor_worker.run_doctor_check",
            return_value=payload,
        ) as run_mock:
            worker.run()
        run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()