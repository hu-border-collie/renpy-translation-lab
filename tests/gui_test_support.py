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

import unittest
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T", bound=type)


def skip_unless_gui(
    unavailable: bool,
    import_error: BaseException | None,
) -> Callable[[_T], _T]:
    """Skip a test class when PySide6 or Qt platform plugins are unavailable."""
    message = "GUI dependencies are unavailable"
    if import_error is not None:
        message = f"{message}: {import_error}"
    return unittest.skipIf(unavailable, message)  # type: ignore[return-value]