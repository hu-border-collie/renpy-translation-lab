import tempfile
import unittest
from pathlib import Path

from gui_qt.theme import clear_theme_caches, load_theme_stylesheet


class GuiThemeCacheTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_theme_caches()

    def test_clear_theme_caches_allows_stylesheet_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = Path(tmp)
            (resources / "app.qss").write_text("QWidget { color: red; }", encoding="utf-8")

            first = load_theme_stylesheet(resources, "light")
            clear_theme_caches()
            (resources / "app.qss").write_text("QWidget { color: blue; }", encoding="utf-8")
            second = load_theme_stylesheet(resources, "light")

            self.assertIn("red", first)
            self.assertIn("blue", second)


if __name__ == "__main__":
    unittest.main()