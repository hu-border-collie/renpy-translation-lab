from pathlib import Path
import unittest

from gui_qt.appearance.theme_helpers import (
    DEFAULT_THEME_PREFERENCE,
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    clear_stylesheet_cache,
    load_theme_stylesheet,
    normalize_theme_preference,
    read_gui_theme_from_config,
    resolve_effective_theme,
    theme_stylesheet_filename,
    write_gui_theme_to_config,
)
from gui_qt.appearance.theme_tokens import DARK_TOKENS, LIGHT_TOKENS


class GuiThemeHelperTests(unittest.TestCase):
    def test_normalize_theme_preference_defaults_to_system(self):
        self.assertEqual(normalize_theme_preference(None), THEME_SYSTEM)
        self.assertEqual(normalize_theme_preference("  "), THEME_SYSTEM)
        self.assertEqual(normalize_theme_preference("invalid"), THEME_SYSTEM)

    def test_normalize_theme_preference_accepts_known_values(self):
        self.assertEqual(normalize_theme_preference("LIGHT"), THEME_LIGHT)
        self.assertEqual(normalize_theme_preference(" dark "), THEME_DARK)
        self.assertEqual(normalize_theme_preference("system"), THEME_SYSTEM)

    def test_resolve_effective_theme_honors_explicit_preference(self):
        self.assertEqual(resolve_effective_theme(THEME_LIGHT, system_is_dark=True), THEME_LIGHT)
        self.assertEqual(resolve_effective_theme(THEME_DARK, system_is_dark=False), THEME_DARK)

    def test_resolve_effective_theme_follows_system_when_requested(self):
        self.assertEqual(resolve_effective_theme(THEME_SYSTEM, system_is_dark=True), THEME_DARK)
        self.assertEqual(resolve_effective_theme(THEME_SYSTEM, system_is_dark=False), THEME_LIGHT)
        self.assertEqual(resolve_effective_theme(THEME_SYSTEM, system_is_dark=None), THEME_LIGHT)

    def test_theme_stylesheet_filename_maps_effective_theme(self):
        self.assertEqual(theme_stylesheet_filename(THEME_LIGHT), "app.qss")
        self.assertEqual(theme_stylesheet_filename(THEME_DARK), "app_dark.qss")

    def test_read_gui_theme_from_config_defaults_to_system(self):
        self.assertEqual(read_gui_theme_from_config({}), DEFAULT_THEME_PREFERENCE)
        self.assertEqual(read_gui_theme_from_config({"gui": "bad"}), DEFAULT_THEME_PREFERENCE)

    def test_write_gui_theme_to_config_preserves_other_gui_fields(self):
        config = {"gui": {"theme": "light", "other": 1}, "sync": {"model": "x"}}
        write_gui_theme_to_config(config, " dark ")
        self.assertEqual(config["gui"]["theme"], THEME_DARK)
        self.assertEqual(config["gui"]["other"], 1)
        self.assertEqual(config["sync"]["model"], "x")

    def test_light_and_dark_tokens_share_identical_keys(self):
        self.assertEqual(set(LIGHT_TOKENS.keys()), set(DARK_TOKENS.keys()))

    def test_rendered_stylesheet_resolves_all_tokens_and_uses_solid_surfaces(self):
        clear_stylesheet_cache()
        resources_dir = Path(__file__).resolve().parents[3] / "gui_qt" / "resources"
        for theme in (THEME_LIGHT, THEME_DARK):
            with self.subTest(theme=theme):
                css = load_theme_stylesheet(resources_dir, theme)
                self.assertNotIn("${", css)
                self.assertNotIn("qlineargradient", css)
                self.assertIn("QFrame#translation_target_section", css)
                self.assertIn('QLabel#task_page_notice[tone="info"]', css)
                self.assertIn('QLabel#doctor_status_label[status="idle"]', css)


if __name__ == "__main__":
    unittest.main()
