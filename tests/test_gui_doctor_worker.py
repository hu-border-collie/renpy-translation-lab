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

    def test_run_doctor_check_surfaces_systemexit_from_settings(self):
        with mock.patch(
            "translator_runtime.load_translator_settings",
            side_effect=SystemExit("ERROR: Invalid tl_subdir configuration."),
        ):
            result = run_doctor_check()
        self.assertFalse(result.ok)
        self.assertIn("Invalid tl_subdir", result.error)

    def test_run_doctor_check_applies_explicit_config_without_disk_reload(self):
        import translator_runtime as runtime

        report = {"mode": "ready"}
        cfg = runtime.default_runtime_config()
        cfg.prep_language = "korean"
        cfg.tl_subdir = "game/tl/korean"
        cfg.base_dir = "C:/games/Example/work"
        cfg.tl_dir = "C:/games/Example/work/game/tl/korean"
        cfg.work_game_dir = "C:/games/Example/work/game"

        with mock.patch(
            "gemini_translate_batch.collect_doctor_report",
            return_value=report,
        ), mock.patch(
            "gemini_translate_batch.load_batch_settings",
        ), mock.patch(
            "gemini_translate_batch.print_doctor_report",
        ), mock.patch(
            "translator_runtime.load_translator_settings",
        ) as load_settings, mock.patch(
            "translator_runtime.load_glossary",
        ):
            result = run_doctor_check(cfg)

        self.assertTrue(result.ok)
        load_settings.assert_not_called()

    def test_run_doctor_check_restores_previous_globals(self):
        import translator_runtime as runtime

        previous_lang = runtime.PREP_LANGUAGE
        previous_subdir = runtime.TL_SUBDIR
        report = {"mode": "ready"}
        cfg = runtime.default_runtime_config()
        cfg.prep_language = "japanese"
        cfg.tl_subdir = "game/tl/japanese"
        cfg.base_dir = runtime.BASE_DIR
        cfg.tl_dir = runtime.TL_DIR
        cfg.work_game_dir = runtime.WORK_GAME_DIR

        try:
            with mock.patch(
                "gemini_translate_batch.collect_doctor_report",
                return_value=report,
            ), mock.patch(
                "gemini_translate_batch.load_batch_settings",
            ), mock.patch(
                "gemini_translate_batch.print_doctor_report",
            ), mock.patch(
                "translator_runtime.load_glossary",
            ):
                result = run_doctor_check(cfg)
            self.assertTrue(result.ok)
            self.assertEqual(runtime.PREP_LANGUAGE, previous_lang)
            self.assertEqual(runtime.TL_SUBDIR, previous_subdir)
        finally:
            runtime.PREP_LANGUAGE = previous_lang
            runtime.TL_SUBDIR = previous_subdir

    def test_worker_passes_config_to_run_doctor_check(self):
        import translator_runtime as runtime

        cfg = runtime.default_runtime_config()
        worker = DoctorWorker(config=cfg)
        payload = DoctorWorkerResult(True, {"mode": "ready"}, "log")
        with mock.patch(
            "gui_qt.doctor_worker.run_doctor_check",
            return_value=payload,
        ) as run_mock:
            worker.run()
        run_mock.assert_called_once_with(cfg)


if __name__ == "__main__":
    unittest.main()
