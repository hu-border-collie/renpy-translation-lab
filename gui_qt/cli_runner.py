"""CLI Runner for the GUI.

Wraps QProcess to invoke the existing batch CLI using argument lists only
(no shell strings). Streams stdout/stderr line-by-line and supports kill.

This keeps the GUI as a pure shell layer per the first version plan in #42.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


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
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._stdout_pending_buffer = b""
        self._stderr_pending_buffer = b""

    def run(self, script_path: str | Path, args: list[str]) -> None:
        """Start the CLI command.

        Example:
            runner.run("/path/to/gemini_translate_batch.py", ["doctor"])
            runner.run(..., ["build", "--display-name", "foo"])
        """
        if self._proc is not None and self._proc.state() == QProcess.ProcessState.Running:
            self.kill()

        script = Path(script_path).resolve()
        if not script.exists():
            self._fail(f"找不到命令行脚本：{script}")
            return

        python_exe = sys.executable

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)

        # Ensure UTF-8 on Windows
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        self._proc.setProcessEnvironment(env)

        self._proc.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._proc.readyReadStandardError.connect(self._on_stderr_ready)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        # Use list of arguments - never shell
        cmd_args = [str(script)] + args
        self._stdout_pending_buffer = b""
        self._stderr_pending_buffer = b""

        self.line_ready.emit(f"[GUI] 正在启动：{python_exe} {script} {' '.join(args)}\n")
        self._proc.start(python_exe, cmd_args)

        if not self._proc.waitForStarted(3000):
            self._fail("启动进程失败（超时）")
            return

    def is_running(self) -> bool:
        """Return True while a CLI subprocess is active."""
        return self._proc is not None and self._proc.state() == QProcess.ProcessState.Running

    def kill(self) -> None:
        """Terminate the running process if any."""
        if self._proc and self._proc.state() == QProcess.ProcessState.Running:
            self.line_ready.emit("\n[GUI] 正在停止进程...\n")
            self._proc.kill()
            self._proc.waitForFinished(2000)

    def _on_stdout_ready(self):
        if not self._proc:
            return
        self._stdout_pending_buffer = self._emit_complete_lines(
            self._stdout_pending_buffer + bytes(self._proc.readAllStandardOutput()),
            self.stdout_line_ready,
        )

    def _on_stderr_ready(self):
        if not self._proc:
            return
        self._stderr_pending_buffer = self._emit_complete_lines(
            self._stderr_pending_buffer + bytes(self._proc.readAllStandardError()),
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
        if self._proc is None:
            return
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

        self.finished.emit(exit_code)
        self._proc = None

    def _on_error(self, error: QProcess.ProcessError):
        msg = f"进程错误：{error}"
        if self._proc:
            msg += f" - {self._proc.errorString()}"
        self._fail(msg)

    def _fail(self, message: str) -> None:
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
        self._proc = None
        self.error.emit(message)
        self.finished.emit(-1)
