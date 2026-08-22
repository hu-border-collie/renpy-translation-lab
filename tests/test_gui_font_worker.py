import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    from gui_qt.font_worker import (
        FontInstallResult,
        FontInstallWorker,
        run_font_install,
        run_font_install_in_subprocess,
    )
except ImportError as exc:
    FontInstallResult = None  # type: ignore[assignment,misc]
    FontInstallWorker = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


class _FakeResultQueue:
    def __init__(self, payload=None):
        self._payload = payload

    def get(self, timeout=None):
        if self._payload is None:
            raise TimeoutError("no payload")
        return self._payload


class _FakeProcess:
    """Stand-in child process; joins flip liveness only when configured."""

    def __init__(self, *, exit_after_joins=None):
        self.exit_after_joins = exit_after_joins
        self.alive = False
        self.joins = 0
        self.terminated = False
        self.killed = False
        self.exitcode = 1

    def start(self):
        self.alive = True

    def join(self, timeout=None):
        self.joins += 1
        if self.exit_after_joins is not None and self.joins >= self.exit_after_joins:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False


class _FakeMpContext:
    def __init__(self, proc, queue):
        self._proc = proc
        self._queue = queue

    def Queue(self, maxsize=1):
        return self._queue

    def Process(self, target=None, args=(), daemon=False):
        return self._proc


@unittest.skipIf(FontInstallResult is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class FontWorkerTests(unittest.TestCase):
    def test_run_font_install_returns_installed_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir)
            installed = [destination / "ui.ttf", destination / "mono.ttf"]
            with mock.patch(
                "gui_qt.font_worker.install_fonts", return_value=installed
            ) as install:
                result = run_font_install(destination)

        self.assertEqual(result, FontInstallResult(True, tuple(installed)))
        install.assert_called_once_with(destination)

    def test_run_font_install_reports_error(self):
        with mock.patch(
            "gui_qt.font_worker.install_fonts", side_effect=RuntimeError("network down")
        ):
            result = run_font_install(Path("fonts"))

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "network down")

    def test_font_worker_uses_subprocess_isolation_by_default(self):
        worker = FontInstallWorker()
        payload = FontInstallResult(True, (Path("a.ttf"),))
        with mock.patch(
            "gui_qt.font_worker.run_font_install_in_subprocess",
            return_value=payload,
        ) as isolated, mock.patch(
            "gui_qt.font_worker.run_font_install",
        ) as inproc:
            worker.run()
        isolated.assert_called_once_with(should_cancel=mock.ANY)
        inproc.assert_not_called()

    def test_font_worker_passes_live_cancel_probe_to_subprocess_runner(self):
        worker = FontInstallWorker()
        captured: dict = {}

        def fake_subprocess(**kwargs):
            captured.update(kwargs)
            return FontInstallResult(True)

        with mock.patch(
            "gui_qt.font_worker.run_font_install_in_subprocess",
            side_effect=fake_subprocess,
        ):
            worker.run()

        should_cancel = captured.get("should_cancel")
        self.assertTrue(callable(should_cancel))
        self.assertFalse(should_cancel())
        worker.request_cancel()
        self.assertTrue(should_cancel())

    def test_worker_cancel_before_run_emits_cancelled_without_installing(self):
        worker = FontInstallWorker()
        worker.request_cancel()
        emitted = []
        worker.completed.connect(emitted.append)
        with mock.patch(
            "gui_qt.font_worker.run_font_install_in_subprocess"
        ) as isolated, mock.patch(
            "gui_qt.font_worker.run_font_install"
        ) as inproc:
            worker.run()
        isolated.assert_not_called()
        inproc.assert_not_called()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].cancelled)
        self.assertFalse(emitted[0].ok)

    def test_worker_cancel_during_install_relabels_failure_as_cancelled(self):
        worker = FontInstallWorker(isolate_process=False)
        emitted = []
        worker.completed.connect(emitted.append)

        def install(destination=None):
            worker.request_cancel()
            return FontInstallResult(False, error="boom")

        with mock.patch("gui_qt.font_worker.run_font_install", side_effect=install):
            worker.run()

        self.assertTrue(emitted[0].cancelled)
        self.assertFalse(emitted[0].ok)
        self.assertEqual(emitted[0].error, "boom")

    def test_worker_spawn_failure_after_cancel_never_falls_back_in_process(self):
        worker = FontInstallWorker()
        emitted = []
        worker.completed.connect(emitted.append)

        def failing_subprocess(**kwargs):
            worker.request_cancel()
            raise RuntimeError("spawn failed")

        with mock.patch(
            "gui_qt.font_worker.run_font_install_in_subprocess",
            side_effect=failing_subprocess,
        ) as isolated, mock.patch(
            "gui_qt.font_worker.run_font_install"
        ) as inproc:
            worker.run()
        isolated.assert_called_once()
        inproc.assert_not_called()
        self.assertTrue(emitted[0].cancelled)
        self.assertFalse(emitted[0].ok)

    def test_worker_spawn_failure_without_cancel_falls_back_in_process(self):
        worker = FontInstallWorker()
        emitted = []
        worker.completed.connect(emitted.append)

        with mock.patch(
            "gui_qt.font_worker.run_font_install_in_subprocess",
            side_effect=RuntimeError("spawn failed"),
        ), mock.patch(
            "gui_qt.font_worker.run_font_install",
            return_value=FontInstallResult(True, (Path("a.ttf"),)),
        ) as inproc:
            worker.run()
        inproc.assert_called_once_with()
        self.assertTrue(emitted[0].ok)
        self.assertFalse(emitted[0].cancelled)

    def test_worker_success_is_not_relabelled_when_cancel_arrives_late(self):
        worker = FontInstallWorker(isolate_process=False)
        emitted = []
        worker.completed.connect(emitted.append)

        def install(destination=None):
            worker.request_cancel()
            return FontInstallResult(True, (Path("a.ttf"),))

        with mock.patch("gui_qt.font_worker.run_font_install", side_effect=install):
            worker.run()

        self.assertTrue(emitted[0].ok)
        self.assertFalse(emitted[0].cancelled)


@unittest.skipIf(FontInstallResult is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class FontSubprocessCancelTests(unittest.TestCase):
    def _run(self, proc, queue, **kwargs):
        with mock.patch(
            "gui_qt.font_worker.mp.get_context",
            return_value=_FakeMpContext(proc, queue),
        ):
            return run_font_install_in_subprocess(**kwargs)

    def test_cancel_poll_terminates_child_and_reports_cancelled(self):
        proc = _FakeProcess()
        cancel_states = iter([False, False, True])
        result = self._run(
            proc,
            _FakeResultQueue(),
            should_cancel=lambda: next(cancel_states),
            poll_interval_s=0.001,
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.cancelled)
        self.assertIn("已取消", result.error)
        self.assertTrue(proc.terminated)
        self.assertFalse(proc.killed)

    def test_deadline_terminates_child_without_cancel_label(self):
        proc = _FakeProcess()
        result = self._run(
            proc,
            _FakeResultQueue(),
            total_timeout_s=0.0,
            poll_interval_s=0.001,
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.cancelled)
        self.assertIn("超时", result.error)
        self.assertTrue(proc.terminated)

    def test_completed_child_returns_payload_without_termination(self):
        proc = _FakeProcess(exit_after_joins=1)
        payload = FontInstallResult(True, (Path("a.ttf"),))
        result = self._run(proc, _FakeResultQueue(payload), poll_interval_s=0.001)
        self.assertEqual(result, payload)
        self.assertFalse(proc.terminated)

    def test_exited_child_without_payload_reports_crash(self):
        proc = _FakeProcess(exit_after_joins=1)
        result = self._run(proc, _FakeResultQueue(), poll_interval_s=0.001)
        self.assertFalse(result.ok)
        self.assertIn("异常退出", result.error)


@unittest.skipIf(FontInstallWorker is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class FontInstallShutdownOwnershipTests(unittest.TestCase):
    """The running font worker is owned and cancelled by app shutdown (#297 P3)."""

    @classmethod
    def setUpClass(cls):
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as exc:  # pragma: no cover - guarded by skipIf
            raise unittest.SkipTest(f"GUI dependencies are unavailable: {exc}")
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self):
        from gui_qt.app import MainWindow

        self.window = MainWindow()

    def tearDown(self):
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_running_font_worker_is_cancelled_by_background_shutdown(self):
        started = threading.Event()
        window = self.window
        worker = FontInstallWorker(window)

        def blocked_install(**kwargs):
            should_cancel = kwargs.get("should_cancel")
            started.set()
            while not (callable(should_cancel) and should_cancel()):
                time.sleep(0.01)
            return FontInstallResult(False, error="字体下载已取消。", cancelled=True)

        worker.completed.connect(window._on_recommended_fonts_downloaded)
        window._font_install_worker = worker
        try:
            with mock.patch(
                "gui_qt.font_worker.run_font_install_in_subprocess",
                side_effect=blocked_install,
            ):
                worker.start()
                self.assertTrue(started.wait(timeout=2))
                self.assertIn(worker, window._owned_background_threads())
                self.assertIn(
                    "后台下载、检查或列表任务",
                    window._shutdown_coordinator.active_labels(),
                )

                settled_events = []
                window._shutdown_coordinator.settled.connect(
                    lambda: settled_events.append(True)
                )
                # Drive the close path: begin() cancels via the participant.
                self.assertTrue(
                    window._shutdown_coordinator.begin(timeout_ms=10_000)
                )
                self.assertTrue(window._shutdown_coordinator.in_progress)
                self.assertTrue(worker.is_cancel_requested())

                # Bounded shutdown without faking completion: the coordinator
                # settles only after the real thread reached its terminal state.
                self.assertTrue(worker.wait(10_000))
                # Drain the queued completed delivery before checking state.
                self._app.processEvents()
                window._shutdown_coordinator.check_now()
                self.assertFalse(window._shutdown_coordinator.in_progress)
                self.assertEqual(len(settled_events), 1)
                self.assertIsNone(window._font_install_worker)
        finally:
            worker.request_cancel()
            worker.wait(2000)


if __name__ == "__main__":
    unittest.main()
