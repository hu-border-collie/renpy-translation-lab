import unittest
from pathlib import Path
from unittest import mock

from gui_qt.font_helpers import (
    GuiFontFamilies,
    build_font_stylesheet,
    bundled_fonts_dir,
    load_gui_fonts,
    user_fonts_dir,
)


class GuiFontHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources_dir = Path(__file__).resolve().parent.parent / "gui_qt" / "resources"

    def test_bundled_fonts_dir_points_under_resources(self):
        self.assertEqual(bundled_fonts_dir(self.resources_dir), self.resources_dir / "fonts")

    def test_font_license_files_exist(self):
        fonts_dir = bundled_fonts_dir(self.resources_dir)
        self.assertTrue((fonts_dir / "HarmonyOS_Sans_LICENSE.txt").is_file())
        self.assertTrue((fonts_dir / "LXGW_WenKai_OFL.txt").is_file())

    def test_user_fonts_dir_honors_override(self):
        with mock.patch.dict(
            "os.environ", {"RENPY_TRANSLATION_LAB_FONT_DIR": "custom-fonts"}
        ):
            self.assertEqual(user_fonts_dir(), Path("custom-fonts"))

    def test_user_fonts_dir_treats_empty_cache_environment_as_unset(self):
        cases = (
            ("win32", "LOCALAPPDATA", Path("home/AppData/Local")),
            ("linux", "XDG_CACHE_HOME", Path("home/.cache")),
        )
        for platform, environment_key, expected_root in cases:
            with self.subTest(platform=platform), mock.patch.dict(
                "os.environ",
                {
                    "RENPY_TRANSLATION_LAB_FONT_DIR": "",
                    environment_key: "",
                },
            ), mock.patch("gui_qt.font_helpers.sys.platform", platform), mock.patch(
                "gui_qt.font_helpers.Path.home", return_value=Path("home")
            ):
                self.assertEqual(
                    user_fonts_dir(),
                    expected_root / "renpy-translation-lab" / "fonts",
                )

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

    def test_load_gui_fonts_falls_back_cleanly_when_optional_fonts_are_missing(self):
        with mock.patch("gui_qt.font_helpers._load_font_family", return_value="") as load:
            families = load_gui_fonts(self.resources_dir)
        self.assertEqual(families, GuiFontFamilies())
        self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
