"""Optional PySide6 GUI workbench for renpy-translation-lab.

This package is completely isolated:
- Never imported by core CLI or tests unless explicitly launching the GUI.
- Depends on PySide6 (see requirements-gui.txt).
- All actions ultimately delegate to the existing CLI via QProcess.
"""
__version__ = "0.1.0-dev"
