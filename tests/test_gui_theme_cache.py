import tempfile
import unittest
from pathlib import Path

from gui_qt.theme_helpers import _render_template, clear_stylesheet_cache, load_theme_stylesheet

_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "gui_qt" / "resources" / "app_template.qss"


class GuiThemeCacheTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_stylesheet_cache()

    def test_clear_stylesheet_cache_allows_stylesheet_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = Path(tmp)
            (resources / "app.qss").write_text("QWidget { color: red; }", encoding="utf-8")

            first = load_theme_stylesheet(resources, "light")
            clear_stylesheet_cache()
            (resources / "app.qss").write_text("QWidget { color: blue; }", encoding="utf-8")
            second = load_theme_stylesheet(resources, "light")

            self.assertIn("red", first)
            self.assertIn("blue", second)

    def test_template_tokens_fully_resolved(self):
        import re

        unresolved = re.compile(r"\$\{[a-z_]+\}")
        for theme in ("light", "dark"):
            rendered = _render_template(_TEMPLATE_PATH, theme)
            self.assertEqual([], unresolved.findall(rendered))


if __name__ == "__main__":
    unittest.main()