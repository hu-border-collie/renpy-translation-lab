import unittest
from pathlib import Path

from gui_qt.font_helpers import (
    GuiFontFamilies,
    MONO_FONT_FILENAME,
    UI_FONT_FILENAME,
    build_font_stylesheet,
    bundled_fonts_dir,
    load_gui_fonts,
)


class GuiFontHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources_dir = Path(__file__).resolve().parent.parent / "gui_qt" / "resources"

    def test_bundled_fonts_dir_points_under_resources(self):
        self.assertEqual(bundled_fonts_dir(self.resources_dir), self.resources_dir / "fonts")

    def test_bundled_font_files_exist(self):
        fonts_dir = bundled_fonts_dir(self.resources_dir)
        self.assertTrue((fonts_dir / UI_FONT_FILENAME).is_file())
        self.assertTrue((fonts_dir / MONO_FONT_FILENAME).is_file())
        self.assertTrue((fonts_dir / "HarmonyOS_Sans_LICENSE.txt").is_file())
        self.assertTrue((fonts_dir / "LXGW_WenKai_OFL.txt").is_file())

    def test_build_font_stylesheet_empty_when_no_families(self):
        self.assertEqual(build_font_stylesheet(GuiFontFamilies()), "")

    def test_build_font_stylesheet_includes_loaded_families(self):
        stylesheet = build_font_stylesheet(
            GuiFontFamilies(
                ui_family="HarmonyOS Sans SC",
                mono_family="LXGW WenKai Mono GB",
            )
        )
        self.assertIn("HarmonyOS Sans SC", stylesheet)
        self.assertIn("LXGW WenKai Mono GB", stylesheet)
        self.assertIn("QTextEdit#log_view", stylesheet)
        self.assertNotIn("QTextEdit#workbench_log_view", stylesheet)
        self.assertIn("QLineEdit#global_project_path_edit", stylesheet)

    def test_load_gui_fonts_reads_bundled_files(self):
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 is not installed")

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        families = load_gui_fonts(self.resources_dir)
        self.assertEqual(families.ui_family, "HarmonyOS Sans SC")
        self.assertEqual(families.mono_family, "LXGW WenKai Mono GB")


if __name__ == "__main__":
    unittest.main()
