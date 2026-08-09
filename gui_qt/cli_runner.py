"""CLI Runner for the GUI.

Wraps QProcess to invoke the existing batch CLI using argument lists only
(no shell strings). Streams stdout/stderr line-by-line and supports kill.

This keeps the GUI as a pure shell layer per the first version plan in #42.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class CliRunner(QObject):
    """Manages one long-running CLI invocation.

    Signals:
        line_ready(str): A decoded line of output (stdout or stderr).
        stdout_line_ready(str): A decoded stdout line for structured consumers.
        stderr_line_ready(str): A decoded stderr line for diagnostics.
        finished(int): Process exited with this code.
        error(str): Fatal error message.
    """

    line_ready = Signal(str)
    stdout_line_ready = Signal(str)
    stderr_line_ready = Signal(str)
    started = Signal()
    stopping = Signal()
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._stdout_pending_buffer = b""
        self._stderr_pending_buffer = b""
        self._error_reported = False
        self._stop_requested = False

        self._start_timeout_timer = QTimer(self)
        self._start_timeout_timer.setSingleShot(True)
        self._start_timeout_timer.timeout.connect(self._on_start_timeout)
        self._stop_timeout_timer = QTimer(self)
        self._stop_timeout_timer.setSingleShot(True)
        self._stop_timeout_timer.timeout.connect(self._force_kill)

    def run(self, script_path: str | Path, args: list[str]) -> bool:
        """Start the CLI command.

        Example:
            runner.run("/path/to/gemini_translate_batch.py", ["doctor"])
            runner.run(..., ["build", "--display-name", "foo"])

        Returns ``True`` once the asynchronous start request has been issued.
        Process startup success or failure is reported by Qt signals; this
        method never waits in the caller (normally the GUI) thread.
        """
        if self.is_active():
            self.error.emit("已有命令行任务正在运行，请先停止后再重试。")
            return False
        # Timers are shared by consecutive invocations. Clear any defensive
        # residue before installing a new process so an older grace timeout can
        # never act on the new owner.
        self._start_timeout_timer.stop()
        self._stop_timeout_timer.stop()
        self._error_reported = False

        script = Path(script_path).resolve()
        if not script.exists():
            self._fail(f"找不到命令行脚本：{script}")
            return False

        python_exe = sys.executable

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)

        # Ensure UTF-8 on Windows
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        self._proc.setProcessEnvironment(env)

        process = self._proc
        process.started.connect(lambda current=process: self._on_started(current))
        process.readyReadStandardOutput.connect(
            lambda current=process: self._on_stdout_ready(current)
        )
        process.readyReadStandardError.connect(
            lambda current=process: self._on_stderr_ready(current)
        )
        process.finished.connect(
            lambda exit_code, exit_status, current=process: self._on_process_finished(
                current,
                exit_code,
                exit_status,
            )
        )
        process.errorOccurred.connect(
            lambda error, current=process: self._on_process_error(current, error)
        )

        # Use list of arguments - never shell
        cmd_args = [str(script)] + args
        self._stdout_pending_buffer = b""
        self._stderr_pending_buffer = b""
        self._stop_requested = False

        self.line_ready.emit(f"[GUI] 正在启动：{python_exe} {script} {' '.join(args)}\n")
        self._start_timeout_timer.start(3000)
        process.start(python_exe, cmd_args)
        return True

    def is_active(self) -> bool:
        """Return True until the owned process delivers its terminal signal.

        ``QProcess.state()`` can become ``NotRunning`` before the queued
        ``finished`` callback is handled. Keeping ownership through that gap
        prevents a replacement process from inheriting an older stop timer.
        """
        return self._proc is not None

    def is_running(self) -> bool:
        """Backward-compatible alias for :meth:`is_active`."""
        return self.is_active()

    def kill(self) -> None:
        """Request non-blocking termination of the active process, if any."""
        self.request_stop()

    def request_stop(self, *, grace_ms: int = 2000) -> bool:
        """Ask the child to terminate, then asynchronously fall back to kill."""
        process = self._proc
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return False
        if self._stop_requested:
            return True
        self._stop_requested = True
        self._start_timeout_timer.stop()
        self.line_ready.emit("\n[GUI] 正在停止本地进程...\n")
        self.stopping.emit()
        self._stop_timeout_timer.start(max(1, int(grace_ms)))
        process.terminate()
        return True

    def _on_started(self, process: QProcess) -> None:
        if process is not self._proc:
            return
        self._start_timeout_timer.stop()
        self.started.emit()

    def _on_start_timeout(self) -> None:
        process = self._proc
        if process is None or process.state() != QProcess.ProcessState.Starting:
            return
        self._emit_error_once("启动进程失败（超时）")
        self._stop_requested = True
        self._stop_timeout_timer.start(1000)
        process.kill()

    def _force_kill(self) -> None:
        process = self._proc
        if process is None:
            return
        if process.state() == QProcess.ProcessState.NotRunning:
            # The state can change before Qt delivers the queued ``finished``
            # signal. Complete ownership now so a missed/delayed callback cannot
            # leave the runner permanently active; the stale signal is ignored.
            self._on_process_finished(
                process,
                -1,
                QProcess.ExitStatus.CrashExit,
            )
            return
        self.line_ready.emit("[GUI] 本地进程未及时停止，正在强制终止...\n")
        process.kill()

    def _on_stdout_ready(self, process: QProcess | None = None):
        process = process or self._proc
        if process is None or process is not self._proc:
            return
        self._stdout_pending_buffer = self._emit_complete_lines(
            self._stdout_pending_buffer + bytes(process.readAllStandardOutput()),
            self.stdout_line_ready,
        )

    def _on_stderr_ready(self, process: QProcess | None = None):
        process = process or self._proc
        if process is None or process is not self._proc:
            return
        self._stderr_pending_buffer = self._emit_complete_lines(
            self._stderr_pending_buffer + bytes(process.readAllStandardError()),
            self.stderr_line_ready,
        )

    def _emit_complete_lines(self, buffer: bytes, channel_signal) -> bytes:
        while True:
            indexes = [idx for idx in (buffer.find(b"\n"), buffer.find(b"\r")) if idx != -1]
            if not indexes:
                break
            pos = min(indexes)
            line_bytes = buffer[:pos]
            buffer = buffer[pos + (2 if buffer[pos : pos + 2] == b"\r\n" else 1) :]
            line = line_bytes.decode("utf-8", errors="replace")
            if line:
                channel_signal.emit(line)
                self.line_ready.emit(line)
        return buffer

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        process = self._proc
        if process is None:
            return
        self._on_process_finished(process, exit_code, exit_status)

    def _on_process_finished(
        self,
        process: QProcess,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        if process is not self._proc:
            return
        self._start_timeout_timer.stop()
        self._stop_timeout_timer.stop()
        # QProcess may still hold final bytes when finished is delivered.
        self._on_stdout_ready(process)
        self._on_stderr_ready(process)
        for buffer_name, channel_signal in (
            ("_stdout_pending_buffer", self.stdout_line_ready),
            ("_stderr_pending_buffer", self.stderr_line_ready),
        ):
            buffer = getattr(self, buffer_name)
            if buffer:
                leftover = buffer.decode("utf-8", errors="replace").strip()
                if leftover:
                    channel_signal.emit(leftover)
                    self.line_ready.emit(leftover)
                setattr(self, buffer_name, b"")

        self._proc = None
        delete_later = getattr(process, "deleteLater", None)
        if callable(delete_later):
            delete_later()
        # Clear ownership before notifying consumers: workflow callbacks may
        # synchronously start the next process from ``finished``.
        self.finished.emit(exit_code)

    def _on_error(self, error: QProcess.ProcessError):
        process = self._proc
        if process is not None:
            self._on_process_error(process, error)

    def _on_process_error(
        self,
        process: QProcess,
        error: QProcess.ProcessError,
    ) -> None:
        if process is not self._proc:
            return
        msg = f"进程错误：{error}"
        msg += f" - {process.errorString()}"
        self._emit_error_once(msg)
        if (
            error == QProcess.ProcessError.FailedToStart
            or process.state() == QProcess.ProcessState.NotRunning
        ):
            self._on_process_finished(
                process,
                -1,
                QProcess.ExitStatus.CrashExit,
            )

    def _emit_error_once(self, message: str) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        self.error.emit(message)

    def _fail(self, message: str) -> None:
        self._emit_error_once(message)
        self.finished.emit(-1)
