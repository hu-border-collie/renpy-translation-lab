"""Background worker for doctor checks.

Heavy TL scans used to run in a QThread inside the GUI process. That still
starves the UI: pure-Python file walks hold the GIL, so page switches during
环境检查 feel frozen. Prefer a short-lived child process so the GUI keeps a
responsive event loop; fall back to in-process execution when isolation is
disabled (tests) or process start fails.
"""
from __future__ import annotations

import io
import multiprocessing as mp
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


def _doctor_process_entry(
    config: RuntimeConfig | None,
    result_queue: Any,
) -> None:
    """Top-level entry for spawn workers (must stay picklable)."""
    try:
        result_queue.put(run_doctor_check(config))
    except BaseException as exc:  # pragma: no cover - defensive process boundary
        result_queue.put(
            DoctorWorkerResult(False, None, "", f"环境检查失败：{exc}")
        )


def run_doctor_check_in_subprocess(
    config: RuntimeConfig | None = None,
    *,
    should_cancel=None,
    join_timeout_s: float = 0.05,
) -> DoctorWorkerResult:
    """Run :func:`run_doctor_check` in a child process.

    ``should_cancel`` is an optional zero-arg callable polled while waiting; when
    it returns true the child is terminated and a cancelled result is returned.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(1)
    proc = ctx.Process(
        target=_doctor_process_entry,
        args=(config, result_queue),
        daemon=True,
    )
    proc.start()
    try:
        while proc.is_alive():
            if should_cancel is not None and should_cancel():
                _terminate_process(proc)
                return DoctorWorkerResult(False, None, "", "环境检查已取消。")
            proc.join(timeout=join_timeout_s)
        try:
            return result_queue.get(timeout=1.0)
        except Exception:
            code = proc.exitcode
            return DoctorWorkerResult(
                False,
                None,
                "",
                f"环境检查进程异常退出（code={code}）。",
            )
    finally:
        if proc.is_alive():
            _terminate_process(proc)


def _terminate_process(proc: mp.Process) -> None:
    proc.terminate()
    proc.join(timeout=2.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1.0)


class DoctorWorker(QThread):
    """Run collect_doctor_report() without starving the GUI event loop."""

    completed = Signal(object)

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        parent=None,
        *,
        isolate_process: bool = True,
    ) -> None:
        super().__init__(parent)
        # Freeze whatever the UI/host passed; do not re-read UI state in run().
        self._config = config
        # Child process keeps the GUI process GIL free during TL scans.
        self._isolate_process = isolate_process

    def run(self) -> None:
        if self._isolate_process:
            try:
                result = run_doctor_check_in_subprocess(
                    self._config,
                    should_cancel=self.isInterruptionRequested,
                )
            except Exception:
                # Spawn/import failures should not brick 环境检查 entirely.
                if self.isInterruptionRequested():
                    result = DoctorWorkerResult(False, None, "", "环境检查已取消。")
                else:
                    result = run_doctor_check(self._config)
        else:
            result = run_doctor_check(self._config)
        if not self.isInterruptionRequested():
            self.completed.emit(result)
