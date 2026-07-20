"""Background worker for in-process doctor checks."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

if TYPE_CHECKING:
    from translator_runtime import RuntimeConfig


@dataclass(frozen=True)
class DoctorWorkerResult:
    ok: bool
    report: dict[str, Any] | None
    log_text: str
    error: str = ""


def run_doctor_check(config: RuntimeConfig | None = None) -> DoctorWorkerResult:
    """Collect and format a doctor report for a frozen project configuration.

    When ``config`` is provided it is applied for the duration of the check and
    the previous process-global runtime state is restored afterwards. When
    omitted, translator settings are reloaded from disk under the same scoped
    restore semantics (issue #216 phase 2).
    """
    try:
        import gemini_translate_batch as batch_mod
        import translator_runtime as legacy
    except Exception as exc:
        return DoctorWorkerResult(False, None, "", f"无法加载 doctor 模块：{exc}")

    try:
        with legacy.runtime_config_scope(
            config,
            reload_translator_settings=config is None,
        ):
            legacy.load_glossary()
            batch_mod.load_batch_settings()
            report = batch_mod.collect_doctor_report()
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                batch_mod.print_doctor_report(report)
        return DoctorWorkerResult(True, report, buffer.getvalue())
    except SystemExit as exc:
        # load_translator_settings / invalid config raise SystemExit for hard
        # config errors (e.g. tl_subdir escaping the project root).
        detail = exc.code if isinstance(exc.code, str) else (str(exc) or "配置错误")
        return DoctorWorkerResult(False, None, "", f"环境检查失败：{detail}")
    except Exception as exc:
        return DoctorWorkerResult(False, None, "", f"环境检查失败：{exc}")


class DoctorWorker(QThread):
    """Run collect_doctor_report() off the UI thread against a config snapshot."""

    completed = Signal(object)

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # Freeze whatever the UI/host passed; do not re-read UI state in run().
        self._config = config

    def run(self) -> None:
        self.completed.emit(run_doctor_check(self._config))
