import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from gui_qt.font_worker import (
        FontInstallResult,
        FontInstallWorker,
        run_font_install,
    )
except ImportError as exc:
    FontInstallResult = None  # type: ignore[assignment,misc]
    FontInstallWorker = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(FontInstallResult is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class FontWorkerTests(unittest.TestCase):
    def test_run_font_install_returns_installed_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir)
            installed = [destination / "ui.ttf", destination / "mono.ttf"]
            with mock.patch(
                "gui_qt.font_worker.install_fonts", return_value=installed
            ) as install:
                result = run_font_install(destination)

        self.assertEqual(result, FontInstallResult(True, tuple(installed)))
        install.assert_called_once_with(destination)

    def test_run_font_install_reports_error(self):
        with mock.patch(
            "gui_qt.font_worker.install_fonts", side_effect=RuntimeError("network down")
        ):
            result = run_font_install(Path("fonts"))

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "network down")

    def test_font_worker_uses_subprocess_isolation_by_default(self):
        worker = FontInstallWorker()
        payload = FontInstallResult(True, (Path("a.ttf"),))
        with mock.patch(
            "gui_qt.font_worker.run_font_install_in_subprocess",
            return_value=payload,
        ) as isolated, mock.patch(
            "gui_qt.font_worker.run_font_install",
        ) as inproc:
            worker.run()
        isolated.assert_called_once_with()
        inproc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
