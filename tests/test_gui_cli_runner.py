"""Channel buffering and finished-drain behavior for Gui CLI runner."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tests import gui_test_support

try:
    from PySide6.QtCore import QProcess

    from gui_qt.cli_runner import CliRunner
except ImportError as exc:
    CliRunner = None  # type: ignore[assignment,misc]
    QProcess = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FakeProcess:
    def __init__(self, *, stdout=(), stderr=()):
        self.stdout_chunks = list(stdout)
        self.stderr_chunks = list(stderr)

    def readAllStandardOutput(self):
        return self.stdout_chunks.pop(0) if self.stdout_chunks else b""

    def readAllStandardError(self):
        return self.stderr_chunks.pop(0) if self.stderr_chunks else b""


class _FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _FakeTimer:
    def __init__(self):
        self.started_with = []
        self.stop_count = 0

    def start(self, interval):
        self.started_with.append(interval)

    def stop(self):
        self.stop_count += 1


class _LifecycleFakeProcess(_FakeProcess):
    ProcessState = QProcess.ProcessState if QProcess is not None else None
    ProcessChannelMode = QProcess.ProcessChannelMode if QProcess is not None else None
    ProcessError = QProcess.ProcessError if QProcess is not None else None
    ExitStatus = QProcess.ExitStatus if QProcess is not None else None

    def __init__(self, _parent=None, *, state=None, stdout=(), stderr=()):
        super().__init__(stdout=stdout, stderr=stderr)
        self._state = state or self.ProcessState.Starting
        self.started = _FakeSignal()
        self.readyReadStandardOutput = _FakeSignal()
        self.readyReadStandardError = _FakeSignal()
        self.finished = _FakeSignal()
        self.errorOccurred = _FakeSignal()
        self.start_args = None
        self.terminate_count = 0
        self.kill_count = 0
        self.deleted = False

    def setProcessChannelMode(self, _mode):
        return None

    def setProcessEnvironment(self, _environment):
        return None

    def start(self, executable, args):
        self.start_args = (executable, args)

    def state(self):
        return self._state

    def terminate(self):
        self.terminate_count += 1

    def kill(self):
        self.kill_count += 1

    def errorString(self):
        return "fake error"

    def deleteLater(self):
        self.deleted = True


@gui_test_support.skip_unless_gui(CliRunner is None, IMPORT_ERROR)
class CliRunnerChannelTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.stdout_lines = []
        self.stderr_lines = []
        self.all_lines = []
        self.finished_codes = []
        self.errors = []
        self.runner.stdout_line_ready.connect(self.stdout_lines.append)
        self.runner.stderr_line_ready.connect(self.stderr_lines.append)
        self.runner.line_ready.connect(self.all_lines.append)
        self.runner.finished.connect(self.finished_codes.append)
        self.runner.error.connect(self.errors.append)

    def test_channels_stay_separate_across_chunks_and_final_tail(self):
        self.runner._proc = _FakeProcess(
            stdout=(b'{\r', b'\n  "ok": true\r\n', b'}'),
            stderr=(b'progress 1\r\npartial diagnostic',),
        )

        self.runner._on_stdout_ready()
        self.runner._on_stdout_ready()
        self.runner._on_stdout_ready()
        self.runner._on_stderr_ready()
        self.runner._on_finished(0, QProcess.ExitStatus.NormalExit)

        self.assertEqual(self.stdout_lines, ['{', '  "ok": true', '}'])
        self.assertEqual(self.stderr_lines, ['progress 1', 'partial diagnostic'])
        self.assertEqual(
            self.all_lines,
            self.stdout_lines[:-1] + ['progress 1', '}', 'partial diagnostic'],
        )
        self.assertEqual(self.finished_codes, [0])
        self.assertIsNone(self.runner._proc)

    def test_finished_drains_unread_process_bytes_before_flushing(self):
        self.runner._proc = _FakeProcess(
            stdout=(b'{"status":"completed"}',),
            stderr=(b'final diagnostic',),
        )

        self.runner._on_finished(7, QProcess.ExitStatus.NormalExit)

        self.assertEqual(self.stdout_lines, ['{"status":"completed"}'])
        self.assertEqual(self.stderr_lines, ['final diagnostic'])
        self.assertEqual(self.finished_codes, [7])

    def test_run_requests_start_without_waiting_for_started(self):
        start_timer = _FakeTimer()
        self.runner._start_timeout_timer = start_timer

        with mock.patch("gui_qt.cli_runner.QProcess", _LifecycleFakeProcess):
            started = self.runner.run(Path(__file__), ["--example"])

        self.assertTrue(started)
        self.assertEqual(start_timer.started_with, [3000])
        self.assertIsNotNone(self.runner._proc.start_args)
        # The fake intentionally has no waitForStarted method: reaching here
        # proves run() did not synchronously wait in the caller thread.

    def test_stop_uses_async_terminate_then_kill_fallback(self):
        process = _LifecycleFakeProcess(state=QProcess.ProcessState.Running)
        stop_timer = _FakeTimer()
        self.runner._proc = process
        self.runner._start_timeout_timer = _FakeTimer()
        self.runner._stop_timeout_timer = stop_timer

        self.assertTrue(self.runner.request_stop(grace_ms=25))

        self.assertEqual(process.terminate_count, 1)
        self.assertEqual(process.kill_count, 0)
        self.assertEqual(stop_timer.started_with, [25])
        self.runner._force_kill()
        self.assertEqual(process.kill_count, 1)

    def test_stale_process_completion_cannot_finish_current_process(self):
        stale = _LifecycleFakeProcess(state=QProcess.ProcessState.NotRunning)
        current = _LifecycleFakeProcess(state=QProcess.ProcessState.Running)
        self.runner._proc = current

        self.runner._on_process_finished(
            stale,
            0,
            QProcess.ExitStatus.NormalExit,
        )

        self.assertIs(self.runner._proc, current)
        self.assertEqual(self.finished_codes, [])

    def test_finished_callback_can_install_next_process_without_being_clobbered(self):
        completed = _LifecycleFakeProcess(state=QProcess.ProcessState.NotRunning)
        replacement = _LifecycleFakeProcess(state=QProcess.ProcessState.Running)
        self.runner._proc = completed
        self.runner.finished.connect(lambda _code: setattr(self.runner, "_proc", replacement))

        self.runner._on_process_finished(
            completed,
            0,
            QProcess.ExitStatus.NormalExit,
        )

        self.assertIs(self.runner._proc, replacement)

    def test_failed_to_start_reports_error_and_terminal_state_once(self):
        process = _LifecycleFakeProcess(state=QProcess.ProcessState.NotRunning)
        self.runner._proc = process
        self.runner._start_timeout_timer = _FakeTimer()
        self.runner._stop_timeout_timer = _FakeTimer()

        self.runner._on_process_error(process, QProcess.ProcessError.FailedToStart)
        self.runner._on_process_error(process, QProcess.ProcessError.FailedToStart)

        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.finished_codes, [-1])
        self.assertIsNone(self.runner._proc)

    def test_start_timeout_keeps_process_owned_until_finished(self):
        process = _LifecycleFakeProcess(state=QProcess.ProcessState.Starting)
        stop_timer = _FakeTimer()
        self.runner._proc = process
        self.runner._stop_timeout_timer = stop_timer

        self.runner._on_start_timeout()

        self.assertEqual(self.errors, ["启动进程失败（超时）"])
        self.assertEqual(process.kill_count, 1)
        self.assertEqual(stop_timer.started_with, [1000])
        self.assertIs(self.runner._proc, process)


if __name__ == "__main__":
    unittest.main()
