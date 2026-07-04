"""Background worker for in-process doctor checks."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class DoctorWorkerResult:
    ok: bool
    report: dict[str, Any] | None
    log_text: str
    error: str = ""


def run_doctor_check() -> DoctorWorkerResult:
    """Collect and format a doctor report for the current translator settings."""
    try:
        import gemini_translate_batch as batch_mod
        import translator_runtime as legacy
    except Exception as exc:
        return DoctorWorkerResult(False, None, "", f"无法加载 doctor 模块：{exc}")

    try:
        with legacy.locked_runtime_state():
            legacy.load_translator_settings()
            legacy.load_glossary()
            batch_mod.load_batch_settings()
            report = batch_mod.collect_doctor_report()
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                batch_mod.print_doctor_report(report)
        return DoctorWorkerResult(True, report, buffer.getvalue())
    except Exception as exc:
        return DoctorWorkerResult(False, None, "", f"环境检查失败：{exc}")


class DoctorWorker(QThread):
    """Run collect_doctor_report() off the UI thread."""

    completed = Signal(object)

    def run(self) -> None:
        self.completed.emit(run_doctor_check())