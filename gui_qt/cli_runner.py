"""CLI Runner for the GUI.

Wraps QProcess to invoke the existing batch CLI using argument lists only
(no shell strings). Streams stdout/stderr line-by-line and supports kill.

This keeps the GUI as a pure shell layer per the first version plan in #42.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QObject, Signal, QProcessEnvironment


class CliRunner(QObject):
    """Manages one long-running CLI invocation.

    Signals:
        line_ready(str): A decoded line of output (stdout or stderr).
        finished(int): Process exited with this code.
        error(str): Fatal error message.
    """

    line_ready = Signal(str)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._pending_buffer = b""

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
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        # Ensure UTF-8 on Windows
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        self._proc.setProcessEnvironment(env)

        self._proc.readyReadStandardOutput.connect(self._on_ready_read)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        # Use list of arguments - never shell
        cmd_args = [str(script)] + args
        self._pending_buffer = b""

        self.line_ready.emit(f"[GUI] 正在启动：{python_exe} {script} {' '.join(args)}\n")
        self._proc.start(python_exe, cmd_args)

        if not self._proc.waitForStarted(3000):
            self._fail("启动进程失败（超时）")
            return

    def is_running(self) -> bool:
        """Return True while a CLI subprocess is active."""
        return (
            self._proc is not None
            and self._proc.state() == QProcess.ProcessState.Running
        )

    def kill(self) -> None:
        """Terminate the running process if any."""
        if self._proc and self._proc.state() == QProcess.ProcessState.Running:
            self.line_ready.emit("\n[GUI] 正在停止进程...\n")
            self._proc.kill()
            self._proc.waitForFinished(2000)

    def _on_ready_read(self):
        if not self._proc:
            return
        data = self._proc.readAllStandardOutput()
        if not data:
            return
        self._pending_buffer += bytes(data)

        # Split on the earliest newline marker, including mixed line endings.
        while True:
            idx_n = self._pending_buffer.find(b"\n")
            idx_r = self._pending_buffer.find(b"\r")
            candidates = [idx for idx in (idx_n, idx_r) if idx != -1]
            if not candidates:
                break
            pos = min(candidates)

            line_bytes = self._pending_buffer[:pos]
            if self._pending_buffer[pos:pos + 2] == b"\r\n":
                self._pending_buffer = self._pending_buffer[pos + 2:]
            else:
                self._pending_buffer = self._pending_buffer[pos + 1:]

            try:
                line = line_bytes.decode("utf-8", errors="replace")
            except Exception:
                line = line_bytes.decode("utf-8", errors="replace")

            if line:
                self.line_ready.emit(line)

        # Flush any remaining partial line on process end (handled in finished too)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        if self._proc is None:
            return
        # Flush any leftover data
        if self._pending_buffer:
            try:
                leftover = self._pending_buffer.decode("utf-8", errors="replace")
                if leftover.strip():
                    self.line_ready.emit(leftover.strip())
            except Exception:
                pass
            self._pending_buffer = b""

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
