import os
import tempfile
import unittest
from pathlib import Path

from gui_qt.font_helpers import (
    GuiFontFamilies,
    build_font_stylesheet,
    resolve_workspace_fonts_dir,
)


class GuiFontHelpersTests(unittest.TestCase):
    def test_resolve_workspace_fonts_dir_from_sibling(self):
        tool_root = Path(r"C:\RenPy_Workspace\renpy-translation-lab")
        resolved = resolve_workspace_fonts_dir(tool_root)
        self.assertEqual(resolved, tool_root.parent / "_fonts")

    def test_resolve_workspace_fonts_dir_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("RENPY_TRANSLATION_LAB_FONTS_DIR")
            os.environ["RENPY_TRANSLATION_LAB_FONTS_DIR"] = tmp
            try:
                resolved = resolve_workspace_fonts_dir(Path(r"C:\RenPy_Workspace\renpy-translation-lab"))
                self.assertEqual(resolved, Path(tmp))
            finally:
                if previous is None:
                    os.environ.pop("RENPY_TRANSLATION_LAB_FONTS_DIR", None)
                else:
                    os.environ["RENPY_TRANSLATION_LAB_FONTS_DIR"] = previous

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


if __name__ == "__main__":
    unittest.main()