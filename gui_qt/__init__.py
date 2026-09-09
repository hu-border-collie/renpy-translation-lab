"""Optional PySide6 GUI workbench for renpy-translation-lab.

This package is completely isolated:
- Never imported by core CLI or tests unless explicitly launching the GUI.
- Depends on PySide6 (see requirements-gui.txt).
- Long-running translation and doctor jobs delegate to the existing CLI through
  QProcess or a short-lived child process. GUI-local background work uses
  QThread/QThreadPool, while immediate GUI-local actions (theme preview,
  keyring/registry writes, dialogs) run in-process. The GUI does not reimplement
  translation or writeback semantics.
"""
__version__ = "0.1.0-dev"
