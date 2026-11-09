"""Application artwork and native desktop identity for the GUI workbench."""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys

from PySide6.QtGui import QIcon


WINDOWS_APP_ID = "RenPyTranslationLab.Workbench"


def application_icon(resources_dir: Path) -> QIcon:
    """Load the bundled, theme-independent application icon."""
    return QIcon(str(resources_dir / "branding" / "app-icon.png"))


def set_windows_app_id() -> bool:
    """Give source launches their own taskbar identity before creating Qt UI.

    This is process-local and optional: unsupported platforms or a failed
    shell API must not prevent the workbench from starting.
    """
    if sys.platform != "win32":
        return False
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        set_app_id = shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        return set_app_id(WINDOWS_APP_ID) == 0
    except (AttributeError, OSError):
        return False
