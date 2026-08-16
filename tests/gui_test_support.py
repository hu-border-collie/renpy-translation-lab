"""Helpers for GUI tests that may require PySide6.

Use in test modules::

    try:
        from PySide6.QtWidgets import QApplication
        from gui_qt.foo import WidgetUnderTest
    except ImportError as exc:
        WidgetUnderTest = None  # type: ignore[assignment,misc]
        IMPORT_ERROR = exc
    else:
        IMPORT_ERROR = None

    @gui_test_support.skip_unless_gui(WidgetUnderTest is None, IMPORT_ERROR)
    class WidgetUnderTestTests(unittest.TestCase):
        ...
"""
from __future__ import annotations

import os
import sys
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar
from unittest import mock

_T = TypeVar("_T", bound=type)

# Headless GUI tests must not spin up the background LiteLLM import warmup
# thread: the import takes ~10s on machines with the optional dependency and a
# QThread destroyed while still importing aborts the process.
os.environ.setdefault("RTL_DISABLE_LITELLM_WARMUP", "1")


class GuiTestModalGuard:
    """Reject unexpected modal widgets so unattended GUI tests cannot hang."""

    def __init__(self, app: Any, *, interval_ms: int = 25) -> None:
        self._app = app
        self._interval_ms = max(1, int(interval_ms))
        self._timer: Any = None
        self._seen_modal_ids: set[int] = set()
        self._rejected_dialogs: list[str] = []
        self._current_test_id = ""

    @property
    def rejected_dialogs(self) -> tuple[str, ...]:
        return tuple(self._rejected_dialogs)

    def start(self) -> None:
        """Poll from the Qt event loop, including nested ``dialog.exec()`` loops."""
        if self._timer is not None:
            return
        from PySide6.QtCore import QTimer

        self._timer = QTimer(self._app)
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self.reject_active_modal)
        self._timer.start()

    def set_current_test(self, test_id: str) -> None:
        self._current_test_id = str(test_id or "").strip()

    def stop(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is None:
            return
        timer.stop()
        timer.deleteLater()

    def reject_active_modal(self) -> None:
        """Reject the current modal once, without exposing its message body."""
        try:
            widget = self._app.activeModalWidget()
        except RuntimeError:
            return
        if widget is None:
            return
        identity = id(widget)
        if identity in self._seen_modal_ids:
            return
        self._seen_modal_ids.add(identity)
        title = str(getattr(widget, "windowTitle", lambda: "")() or "").strip()
        object_name = str(getattr(widget, "objectName", lambda: "")() or "").strip()
        label = type(widget).__name__
        if title:
            label += f" title={title!r}"
        if object_name:
            label += f" object={object_name!r}"
        if self._current_test_id:
            label += f" test={self._current_test_id!r}"
        self._rejected_dialogs.append(label)
        reject = getattr(widget, "reject", None)
        if callable(reject):
            reject()
            return
        close = getattr(widget, "close", None)
        if callable(close):
            close()

    def cleanup_top_levels(self) -> None:
        """Hide and schedule deletion of widgets leaked by completed tests."""
        try:
            widgets = tuple(self._app.topLevelWidgets())
        except RuntimeError:
            return
        for widget in widgets:
            try:
                widget.hide()
                widget.deleteLater()
            except RuntimeError:
                continue


def _modal_guard_enabled() -> bool:
    value = os.environ.get("RENPY_TRANSLATION_LAB_GUI_TEST_MODAL_GUARD", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def shutdown_gui_test_runtime(
    app: Any = None,
    *,
    wait_ms: int = 30000,
    cleanup_widgets: bool = True,
) -> bool:
    """Stop test-owned Qt work and optionally drain deferred widget deletion.

    Do not call ``QApplication.shutdown()`` here.  PySide6 can segfault while
    destroying the offscreen platform plugin on Linux even after every test
    passed.  The script runner uses the pool-only mode before ``os._exit``;
    embedded callers may request the bounded widget cleanup without explicitly
    destroying the application.
    """
    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return True

    app = app or QApplication.instance()
    if app is None:
        return True

    pool = QThreadPool.globalInstance()
    try:
        pool.clear()
        pool_finished = bool(pool.waitForDone(max(0, int(wait_ms))))
    except RuntimeError:
        pool_finished = True

    if not cleanup_widgets:
        return pool_finished

    try:
        for widget in tuple(app.topLevelWidgets()):
            try:
                widget.hide()
                widget.deleteLater()
            except RuntimeError:
                continue
        QCoreApplication.sendPostedEvents(
            None,
            QEvent.Type.DeferredDelete,
        )
        app.processEvents()
    except RuntimeError:
        pass
    return pool_finished


def guarded_test_result_class(guard: GuiTestModalGuard | None):
    """Build a unittest result class that tells the guard which test is active."""

    class GuardedGuiTestResult(unittest.TextTestResult):
        def startTest(self, test) -> None:
            if guard is not None:
                guard.set_current_test(test.id())
            super().startTest(test)

    return GuardedGuiTestResult


@contextmanager
def guarded_gui_test_environment():
    """Reject accidental dialogs without changing the caller's Qt platform."""
    try:
        from PySide6.QtCore import QCoreApplication, Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        yield None
        return

    app = QApplication.instance()
    if app is None:
        QCoreApplication.setAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs,
            True,
        )
        app = QApplication([])

    guard = GuiTestModalGuard(app) if _modal_guard_enabled() else None
    if guard is not None:
        guard.start()
    try:
        yield guard
    finally:
        if guard is not None:
            guard.reject_active_modal()
            guard.stop()
            guard.cleanup_top_levels()
        try:
            app.processEvents()
        except RuntimeError:
            pass
        if guard is not None:
            if guard.rejected_dialogs:
                summary = "\n".join(
                    f"  - {label}" for label in guard.rejected_dialogs
                )
                sys.stderr.write(
                    "\nGUI modal guard auto-rejected unexpected dialogs:\n"
                    f"{summary}\n"
                )


def skip_unless_gui(
    unavailable: bool,
    import_error: BaseException | None,
) -> Callable[[_T], _T]:
    """Skip a test class when PySide6 or Qt platform plugins are unavailable."""
    message = "GUI dependencies are unavailable"
    if import_error is not None:
        message = f"{message}: {import_error}"
    return unittest.skipIf(unavailable, message)  # type: ignore[return-value]


def close_main_window(window: Any) -> None:
    """Close a MainWindow in tests without opening an interactive prompt."""
    with (
        mock.patch.object(
            window,
            "_confirm_unsaved_config_before_close",
            return_value=True,
        ),
        mock.patch.object(
            window,
            "_confirm_active_tasks_before_close",
            return_value=True,
        ),
    ):
        window.close()
        coordinator = getattr(window, "_shutdown_coordinator", None)
        asynchronous_close = bool(
            getattr(window, "_shutdown_close_ready", False)
            or getattr(coordinator, "in_progress", False)
        )
        if not asynchronous_close:
            return
        # Active-task cleanup intentionally ignores the first close event and
        # schedules a terminal second close after the coordinator settles.
        # Drain a few zero-delay callbacks only for that asynchronous path so
        # ordinary teardown cannot run a hidden window's pending focus callback
        # before the caller queues deleteLater().
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                for _ in range(3):
                    app.processEvents()
        except (ImportError, RuntimeError):
            pass
