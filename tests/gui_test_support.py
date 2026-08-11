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
import unittest
from collections.abc import Callable
from typing import Any, TypeVar
from unittest import mock

_T = TypeVar("_T", bound=type)

# Headless GUI tests must not spin up the background LiteLLM import warmup
# thread: the import takes ~10s on machines with the optional dependency and a
# QThread destroyed while still importing aborts the process.
os.environ.setdefault("RTL_DISABLE_LITELLM_WARMUP", "1")


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
