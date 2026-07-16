import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui_qt.font_worker import FontInstallResult, run_font_install


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


if __name__ == "__main__":
    unittest.main()
