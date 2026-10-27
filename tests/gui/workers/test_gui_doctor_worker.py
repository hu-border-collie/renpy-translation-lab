"""Tests for in-process doctor worker."""
from __future__ import annotations

import unittest
from queue import Empty
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.workers.doctor_worker import (
        DoctorWorker, DoctorWorkerResult, run_doctor_check,
        run_doctor_check_in_subprocess,
    )
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
        worker = DoctorWorker(config=cfg, isolate_process=False)
        payload = DoctorWorkerResult(True, {"mode": "ready"}, "log")
        with mock.patch(
            "gui_qt.workers.doctor_worker.run_doctor_check",
            return_value=payload,
        ) as run_mock:
            worker.run()
        run_mock.assert_called_once_with(cfg)

    def test_worker_uses_subprocess_isolation_by_default(self):
        import translator_runtime as runtime

        cfg = runtime.default_runtime_config()
        worker = DoctorWorker(config=cfg)
        payload = DoctorWorkerResult(True, {"mode": "ready"}, "log")
        with mock.patch(
            "gui_qt.workers.doctor_worker.run_doctor_check_in_subprocess",
            return_value=payload,
        ) as run_mock, mock.patch(
            "gui_qt.workers.doctor_worker.run_doctor_check",
        ) as inproc_mock:
            worker.run()
        run_mock.assert_called_once()
        self.assertIs(run_mock.call_args.args[0], cfg)
        inproc_mock.assert_not_called()

    def test_subprocess_drains_large_report_before_waiting_for_child_exit(self):
        payload = DoctorWorkerResult(True, {"mode": "fixture"}, "x" * 500_000)
        process, queue, context = mock.Mock(), mock.Mock(), mock.Mock()
        alive = [True]
        process.is_alive.side_effect = lambda: alive[0]
        process.join.side_effect = AssertionError("Child cannot exit until its report is drained")

        def receive(*, timeout):
            self.assertTrue(alive[0])
            alive[0] = False
            return payload

        queue.get.side_effect = receive
        context.Queue.return_value = queue
        context.Process.return_value = process
        with mock.patch("gui_qt.workers.doctor_worker.mp.get_context", return_value=context):
            result = run_doctor_check_in_subprocess()

        self.assertIs(result, payload)
        queue.get.assert_called_once_with(timeout=0.05)
        process.join.assert_not_called()
        process.terminate.assert_not_called()
        queue.close.assert_called_once()
        queue.join_thread.assert_called_once()

    def test_subprocess_cancel_while_waiting_still_terminates_child(self):
        process, queue, context = mock.Mock(), mock.Mock(), mock.Mock()
        alive = [True]
        process.is_alive.side_effect = lambda: alive[0]
        process.terminate.side_effect = lambda: alive.__setitem__(0, False)
        queue.get.side_effect = Empty
        context.Queue.return_value = queue
        context.Process.return_value = process
        cancelled = mock.Mock(side_effect=[False, True])
        with mock.patch("gui_qt.workers.doctor_worker.mp.get_context", return_value=context):
            result = run_doctor_check_in_subprocess(should_cancel=cancelled)

        self.assertFalse(result.ok)
        self.assertIn("已取消", result.error)
        process.terminate.assert_called_once()
        queue.close.assert_called_once()
        queue.join_thread.assert_called_once()

    def test_subprocess_exit_without_report_still_returns_error(self):
        process, queue, context = mock.Mock(), mock.Mock(), mock.Mock()
        process.is_alive.return_value = False
        process.exitcode = 7
        queue.get.side_effect = Empty
        context.Queue.return_value = queue
        context.Process.return_value = process
        with mock.patch("gui_qt.workers.doctor_worker.mp.get_context", return_value=context):
            result = run_doctor_check_in_subprocess()

        self.assertFalse(result.ok)
        self.assertIn("code=7", result.error)
        queue.close.assert_called_once()
        queue.join_thread.assert_called_once()

    def test_worker_falls_back_to_inprocess_when_isolation_fails(self):
        import translator_runtime as runtime

        cfg = runtime.default_runtime_config()
        worker = DoctorWorker(config=cfg)
        payload = DoctorWorkerResult(True, {"mode": "ready"}, "log")
        with mock.patch(
            "gui_qt.workers.doctor_worker.run_doctor_check_in_subprocess",
            side_effect=RuntimeError("spawn failed"),
        ), mock.patch(
            "gui_qt.workers.doctor_worker.run_doctor_check",
            return_value=payload,
        ) as inproc_mock:
            worker.run()
        inproc_mock.assert_called_once_with(cfg)

    def test_run_doctor_check_reloads_without_persisting_game_root(self):
        with mock.patch(
            "gemini_translate_batch.collect_doctor_report",
            return_value={"mode": "ready"},
        ), mock.patch(
            "gemini_translate_batch.load_batch_settings",
        ), mock.patch(
            "gemini_translate_batch.print_doctor_report",
        ), mock.patch(
            "translator_runtime.load_translator_settings",
        ) as load_settings, mock.patch(
            "translator_runtime.load_glossary",
        ):
            result = run_doctor_check()
        self.assertTrue(result.ok)
        load_settings.assert_called_once_with(
            persist_corrected_game_root=False,
            tolerate_routing_errors=True,
        )

    def test_run_doctor_check_keeps_config_bytes_unchanged(self):
        import json
        import tempfile
        from pathlib import Path as P

        import translator_runtime as runtime

        with tempfile.TemporaryDirectory() as tmp:
            root_dir = P(tmp)
            project = root_dir / "Game_Example"
            (project / "original" / "game").mkdir(parents=True)
            work = project / "work"
            (work / "game" / "tl" / "schinese").mkdir(parents=True)
            (work / "game" / "script.rpy").write_text(
                "label start:\n    return\n", encoding="utf-8"
            )
            config_path = root_dir / "translator_config.json"
            config_path.write_text(
                json.dumps(
                    {"game_root": str(project), "tl_subdir": "game/tl/schinese"},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            before = config_path.read_bytes()

            with (
                mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(config_path)),
                mock.patch.object(runtime, "persist_game_root") as persist_mock,
                mock.patch(
                    "gemini_translate_batch.collect_doctor_report",
                    return_value={"mode": "ready", "counts": {}},
                ),
                mock.patch("gemini_translate_batch.load_batch_settings"),
                mock.patch("gemini_translate_batch.print_doctor_report"),
                mock.patch("translator_runtime.load_glossary"),
            ):
                result = run_doctor_check()

            self.assertTrue(result.ok)
            self.assertEqual(config_path.read_bytes(), before)
            persist_mock.assert_not_called()

if __name__ == "__main__":
    unittest.main()
